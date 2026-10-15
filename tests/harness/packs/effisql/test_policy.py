"""Tests for the v0 Effi-SQL policy: the loop, the fallback, the prompt and config.yaml."""

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from goaly.contracts.gen.environment_pb2 import SqlResult
from goaly.contracts.gen.task_pb2 import EffiSqlTaskInput, SweTaskInput, TaskInput
from goaly.contracts.gen.trajectory_pb2 import ChatRole, StopReason
from goaly.contracts.interfaces import Policy, SqlEnvironment
from goaly.harness.core.agent import AgentContext
from goaly.harness.core.budgets import Budgets
from goaly.harness.core.context_window import ContextWindow
from goaly.harness.core.llm_models import ModelParams
from goaly.harness.packs import effisql
from goaly.harness.packs.effisql.policy import (
    EffiSqlPolicy,
    PackConfig,
    build_policy,
    fallback,
    task_prompt,
)
from tests.fakes import FakeSqlEnvironment
from tests.harness.core.agent_fakes import FakeLLM, reply, tool_call

pytestmark = pytest.mark.anyio

PACK = Path(effisql.__file__).parent
BASE_SQL = "SELECT count(*) FROM orders WHERE o_orderkey IN (SELECT o_orderkey FROM orders)"
FASTER = "SELECT count(*) FROM orders"
TASK = TaskInput(
    effisql=EffiSqlTaskInput(
        db="tpch",
        base_sql=BASE_SQL,
        schema="CREATE TABLE orders (o_orderkey integer PRIMARY KEY);",
        base_plan='{"Node Type": "Aggregate"}',
        difficulty="easy",
    )
)


def config_file() -> dict[str, object]:
    loaded = yaml.safe_load((PACK / "config.yaml").read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def policy() -> EffiSqlPolicy:
    pack = config_file()["pack"]
    assert isinstance(pack, dict)
    return build_policy(pack, (PACK / "prompt.md").read_text(encoding="utf-8"))


V0 = Budgets(steps=30, wall_clock_s=600)


def context(llm: FakeLLM, budgets: Budgets = V0) -> AgentContext:
    return AgentContext(llm, budgets=budgets)


class TestSolve:
    async def test_a_submit_is_the_prediction(self) -> None:
        env = FakeSqlEnvironment({FASTER: SqlResult(columns=["count"], row_count=1)})
        llm = FakeLLM(
            [
                reply(tool_call("execute_sql", {"sql": FASTER})),
                reply(tool_call("submit", {"sql": FASTER})),
            ]
        )
        ctx = context(llm)

        prediction = await policy().solve(TASK, env, ctx)

        assert prediction.effisql.sql == FASTER
        assert not prediction.fallback
        assert env.executed == [(FASTER, 120.0)]
        assert ctx.trajectory().stop_reason == StopReason.STOP_REASON_SUBMITTED

    async def test_the_prompt_opens_the_chat(self) -> None:
        llm = FakeLLM([reply(tool_call("submit", {"sql": FASTER}))])

        await policy().solve(TASK, FakeSqlEnvironment(), context(llm))

        system, user = llm.requests[0].messages
        assert system.content == (PACK / "prompt.md").read_text(encoding="utf-8")
        assert user.content == task_prompt(
            TASK.effisql, PackConfig(), Budgets(steps=30, wall_clock_s=600)
        )
        sent: Any = json.loads(json.dumps(llm.requests[0].tools))
        assert [tool["function"]["name"] for tool in sent] == [
            "execute_sql",
            "verify_equivalence",
            "submit",
        ]

    async def test_the_model_can_verify_a_rewrite_before_submitting_it(self) -> None:
        env = FakeSqlEnvironment()  # Nothing canned: the diff statement comes back as an error.
        llm = FakeLLM(
            [
                reply(
                    tool_call("verify_equivalence", {"base_sql": BASE_SQL, "candidate_sql": FASTER})
                ),
                reply(tool_call("submit", {"sql": BASE_SQL})),
            ]
        )

        prediction = await policy().solve(TASK, env, context(llm))

        assert prediction.effisql.sql == BASE_SQL
        assert not prediction.fallback
        assert len(env.executed) == 1  # The one fixed statement, with the statement timeout.
        statement, timeout = env.executed[0]
        assert timeout == 120.0
        assert statement.count("EXCEPT ALL") == 2
        assert statement.count(f"({FASTER})") == 2
        assert statement.count(f"({BASE_SQL})") == 2
        assert statement.startswith("SELECT count(*) AS diffs FROM (")
        assert statement.endswith(") d")
        verdict = next(
            message.content
            for message in llm.requests[1].messages
            if message.role == ChatRole.CHAT_ROLE_TOOL
        )
        assert verdict.startswith("no canned result for")  # The database's error, as it is.

    async def test_running_out_of_steps_submits_base_sql_unchanged(self) -> None:
        env = FakeSqlEnvironment({FASTER: SqlResult(row_count=1)})
        llm = FakeLLM([reply(tool_call("execute_sql", {"sql": FASTER})) for _ in range(2)])
        ctx = context(llm, Budgets(steps=2, wall_clock_s=600))

        prediction = await policy().solve(TASK, env, ctx)

        assert prediction.fallback
        assert prediction.effisql.sql == BASE_SQL
        stop = ctx.trajectory().stop_reason
        assert stop == StopReason.STOP_REASON_STEP_BUDGET_EXHAUSTED

    async def test_running_out_of_time_submits_base_sql_unchanged(self) -> None:
        env = FakeSqlEnvironment({FASTER: SqlResult(elapsed_s=10)})
        llm = FakeLLM([reply(tool_call("execute_sql", {"sql": FASTER}))])
        ctx = context(llm, Budgets(steps=30, wall_clock_s=0.05))

        prediction = await policy().solve(TASK, env, ctx)

        assert prediction.fallback
        assert prediction.effisql.sql == BASE_SQL
        stop = ctx.trajectory().stop_reason
        assert stop == StopReason.STOP_REASON_TIME_BUDGET_EXHAUSTED
        assert env.stopped == [FASTER]  # The running statement was cancelled.

    async def test_a_whitespace_submit_is_no_submit_so_the_fallback_applies(self) -> None:
        llm = FakeLLM([reply(tool_call("submit", {"sql": "  \n\t "})) for _ in range(2)])
        ctx = context(llm, Budgets(steps=2, wall_clock_s=600))

        prediction = await policy().solve(TASK, FakeSqlEnvironment(), ctx)

        assert prediction.fallback
        assert prediction.effisql.sql == BASE_SQL

    async def test_a_task_of_another_benchmark_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Effi-SQL tasks only"):
            await policy().solve(
                TaskInput(swe=SweTaskInput(repo="r")), FakeSqlEnvironment(), context(FakeLLM([]))
            )

    def test_the_fallback_is_base_sql(self) -> None:
        assert fallback(TASK.effisql).effisql.sql == BASE_SQL


class TestPrompt:
    def test_the_system_prompt_states_the_rules(self) -> None:
        text = " ".join((PACK / "prompt.md").read_text(encoding="utf-8").split())

        assert "must compute its result from the tables" in text
        assert "same result as the original query on any data" in text
        assert "embeds result values" in text
        assert "is invalid" in text
        assert "begins with SELECT or WITH" in text

    def test_the_task_prompt_shows_the_task_and_the_limits(self) -> None:
        text = task_prompt(TASK.effisql, PackConfig(), Budgets(steps=30, wall_clock_s=600))

        for part in (BASE_SQL, "tpch", "easy", TASK.effisql.schema, TASK.effisql.base_plan):
            assert part in text
        assert "after 120 s" in text
        assert "30 steps and 10 minutes" in text
        assert "submitted unchanged" in text

    def test_a_missing_difficulty_reads_unknown(self) -> None:
        task = EffiSqlTaskInput(db="tpch", base_sql=BASE_SQL)

        assert "rated unknown" in task_prompt(task, PackConfig(), Budgets(steps=1, wall_clock_s=1))


class TestConfig:
    def test_config_yaml_holds_the_v0_values(self) -> None:
        config = config_file()

        assert ModelParams.model_validate(config["model"]) == ModelParams(
            enable_thinking=True, temperature=0.6, top_p=0.95, top_k=20, max_tokens=16000
        )
        assert Budgets.model_validate(config["budgets"]) == Budgets(steps=30, wall_clock_s=600)
        assert ContextWindow.model_validate(config["context"]) == ContextWindow()
        assert PackConfig.model_validate(config["pack"]) == PackConfig()
        assert set(config) == {"model", "budgets", "context", "pack"}

    def test_the_defaults_are_v0s(self) -> None:
        execute_sql = PackConfig().execute_sql

        assert execute_sql.statement_timeout_s == 120
        assert execute_sql.rows_shown == 100
        assert execute_sql.value_chars == 2000

    @pytest.mark.parametrize(
        "config",
        [
            {"execute_sql": {"statement_timeout": 5}},
            {"unknown": 1},
            {"execute_sql": {"statement_timeout_s": 0}},
            {"execute_sql": {"statement_timeout_s": 120.5}},
            {"execute_sql": {"statement_timeout_s": 1e9}},
            {"execute_sql": {"rows_shown": -1}},
            {"execute_sql": {"value_chars": 0}},
            {"submit_timeout_s": 0},
        ],
    )
    def test_invalid_pack_settings_are_refused(self, config: dict[str, object]) -> None:
        with pytest.raises(ValidationError):
            build_policy(config, "prompt")  # type: ignore[arg-type]

    async def test_a_pack_may_lower_the_statement_timeout(self) -> None:
        env = FakeSqlEnvironment({FASTER: SqlResult(row_count=1)})
        llm = FakeLLM(
            [
                reply(tool_call("execute_sql", {"sql": FASTER})),
                reply(tool_call("submit", {"sql": FASTER})),
            ]
        )
        lowered = build_policy({"execute_sql": {"statement_timeout_s": 30}}, "prompt")

        await lowered.solve(TASK, env, context(llm))

        assert env.executed == [(FASTER, 30.0)]

    def test_the_policy_is_a_policy(self) -> None:
        typed: Policy[SqlEnvironment, AgentContext] = policy()

        assert typed is not None
