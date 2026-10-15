"""Tests for execute_sql's two budget guards: the runaway cancel and the one-statement rule.

The runaway guard stops a call that has run past its share of the time the task has left, so a
slow probe can no longer run until the wall budget ends the task mid-statement; the batch guard
runs only the first statement of a string that holds several, which the server would refuse.
"""

import asyncio
import json

import pytest
from google.protobuf.struct_pb2 import ListValue

from goaly.contracts.gen.environment_pb2 import SqlResult
from goaly.contracts.interfaces import SqlEnvironment
from goaly.harness.core.budgets import Budgets
from goaly.harness.core.tools import ToolRegistry, bind
from goaly.harness.packs.effisql.policy import PackConfig, registry
from goaly.harness.packs.effisql.tools import (
    BATCH_NOTE,
    ExecuteSql,
    ExecuteSqlConfig,
    first_statement,
    split_statements,
)
from tests.fakes import FakeSqlEnvironment
from tests.harness.core.agent_fakes import tool_call

pytestmark = pytest.mark.anyio

V0 = Budgets(steps=30, wall_clock_s=600)
QUERY = "SELECT n_name FROM nation"


def row(*values: object) -> ListValue:
    listed = ListValue()
    listed.extend(values)  # type: ignore[arg-type]  # Test data of JSON-like values.
    return listed


def tools(
    config: ExecuteSqlConfig | None = None, remaining: float | None = None
) -> ToolRegistry[SqlEnvironment]:
    """The pack's tools, with a task that has ``remaining`` seconds of wall budget left."""
    settings = PackConfig() if config is None else PackConfig(execute_sql=config)
    source = None if remaining is None else (lambda: remaining)
    return registry(settings, V0, source)


class TestBatchGuard:
    async def test_only_the_first_statement_runs_and_the_model_is_told(self) -> None:
        first = "SELECT n_name FROM nation"
        env = FakeSqlEnvironment({first: SqlResult(columns=["n_name"], row_count=1)})

        result = await tools().execute(tool_call("execute_sql", {"sql": f"{first}; SELECT 1"}), env)

        assert not result.is_error
        assert env.executed == [(first, 120.0)]
        assert result.content.endswith(BATCH_NOTE)
        assert "Columns: n_name" in result.content

    async def test_a_trailing_semicolon_is_still_one_statement(self) -> None:
        env = FakeSqlEnvironment({QUERY: SqlResult(row_count=1)})

        result = await tools().execute(tool_call("execute_sql", {"sql": f"{QUERY};"}), env)

        assert not result.is_error
        assert env.executed == [(f"{QUERY};", 120.0)]  # Nothing was cut off.
        assert BATCH_NOTE not in result.content

    async def test_quotes_and_comments_hide_a_semicolon(self) -> None:
        sql = "SELECT ';' AS a, $$x;y$$ AS b, \"c;d\" FROM nation -- ; not a separator"
        env = FakeSqlEnvironment({sql: SqlResult(row_count=0)})

        result = await tools().execute(tool_call("execute_sql", {"sql": sql}), env)

        assert env.executed == [(sql, 120.0)]
        assert BATCH_NOTE not in result.content

    async def test_a_batch_that_also_runs_too_long_says_both(self) -> None:
        config = ExecuteSqlConfig(statement_timeout_s=120, guard_fraction=0.5, guard_min_s=0.05)
        env = FakeSqlEnvironment({"SELECT pg_sleep(30)": SqlResult(elapsed_s=30)})

        result = await tools(config, remaining=1).execute(
            tool_call("execute_sql", {"sql": "SELECT pg_sleep(30); SELECT 1"}), env
        )

        assert not result.is_error
        assert result.content.startswith("cancelled after 0.5 s - too slow")
        assert result.content.endswith(BATCH_NOTE)
        assert env.stopped == ["SELECT pg_sleep(30)"]  # The call was cancelled, not left running.

    @pytest.mark.parametrize(
        ("sql", "statements"),
        [
            ("SELECT 1", ["SELECT 1"]),
            ("  SELECT 1 ;  ", ["SELECT 1"]),
            ("", []),
            (";", []),
            ("; ; SELECT 1", ["SELECT 1"]),
            ("SELECT 1; SELECT 2; SELECT 3", ["SELECT 1", "SELECT 2", "SELECT 3"]),
            (
                "SELECT 1 -- a comment ; here\n; SELECT 2",
                ["SELECT 1 -- a comment ; here", "SELECT 2"],
            ),
            ("SELECT 1 /* ; */ ; SELECT 2", ["SELECT 1 /* ; */", "SELECT 2"]),
            ('SELECT "a;b"; SELECT 2', ['SELECT "a;b"', "SELECT 2"]),
            ("SELECT $$a;b$$; SELECT 2", ["SELECT $$a;b$$", "SELECT 2"]),
            ("SELECT $tag$a $$ b$tag$; SELECT 2", ["SELECT $tag$a $$ b$tag$", "SELECT 2"]),
            ("SELECT 1 $$unterminated; SELECT 2", ["SELECT 1 $$unterminated; SELECT 2"]),
            ("SELECT a$b; SELECT 2", ["SELECT a$b", "SELECT 2"]),  # $ that opens no dollar quote.
        ],
    )
    def test_splitting_follows_postgresql(self, sql: str, statements: list[str]) -> None:
        assert split_statements(sql) == statements

    def test_first_statement_says_whether_there_were_more(self) -> None:
        assert first_statement("SELECT 1") == ("SELECT 1", False)
        assert first_statement("SELECT 1;") == ("SELECT 1;", False)
        assert first_statement("SELECT 1; SELECT 2") == ("SELECT 1", True)


class TestRunawayGuard:
    def test_the_limit_is_half_of_what_is_left(self) -> None:
        tool = ExecuteSql(ExecuteSqlConfig(), lambda: 100)

        assert tool.guard_limit_s() == 50

    def test_the_limit_is_never_below_the_minimum(self) -> None:
        assert ExecuteSql(ExecuteSqlConfig(), lambda: 4).guard_limit_s() == 10
        assert ExecuteSql(ExecuteSqlConfig(), lambda: 0).guard_limit_s() == 10

    def test_no_deadline_leaves_the_fixed_cap(self) -> None:
        assert ExecuteSql(ExecuteSqlConfig()).guard_limit_s() == 30
        assert ExecuteSql(ExecuteSqlConfig(), lambda: float("inf")).guard_limit_s() == 30

    def test_a_tighter_statement_timeout_needs_no_guard(self) -> None:
        # The environment cancels at the statement timeout, so the guard would add nothing.
        config = ExecuteSqlConfig(statement_timeout_s=5)

        assert ExecuteSql(config, lambda: 600).guard_limit_s() is None

    async def test_a_call_past_its_share_is_cancelled_and_the_model_moves_on(self) -> None:
        config = ExecuteSqlConfig(guard_min_s=0.05)
        env = FakeSqlEnvironment({QUERY: SqlResult(elapsed_s=30)})

        result = await tools(config, remaining=1).execute(
            tool_call("execute_sql", {"sql": QUERY}), env
        )

        assert not result.is_error  # An ordinary result, as a statement timeout is.
        assert result.content == (
            "cancelled after 0.5 s - too slow; do not re-run this statement, it cannot be your "
            "submission; try a cheaper probe or submit your best candidate"
        )
        assert env.stopped == [QUERY]
        assert result.elapsed_s < 30

    async def test_a_call_within_its_share_runs_to_the_statement_timeout(self) -> None:
        config = ExecuteSqlConfig(statement_timeout_s=0.05, guard_min_s=0.05)
        env = FakeSqlEnvironment({QUERY: SqlResult(elapsed_s=30)})

        result = await tools(config, remaining=600).execute(
            tool_call("execute_sql", {"sql": QUERY}), env
        )

        assert not result.is_error
        assert result.content.startswith(
            "The query failed: canceling statement due to statement timeout"
        )
        assert env.stopped == []

    async def test_the_environment_is_usable_after_a_cancel(self) -> None:
        config = ExecuteSqlConfig(guard_min_s=0.05)
        env = FakeSqlEnvironment(
            {
                QUERY: SqlResult(elapsed_s=30),
                "SELECT 1": SqlResult(columns=["?column?"], rows=[row(1.0)]),
            }
        )
        bound = bind(ExecuteSql(config, lambda: 1), timeout_s=600)

        cancelled = await bound.invoke(json.dumps({"sql": QUERY}), env)
        again = await bound.invoke(json.dumps({"sql": "SELECT 1"}), env)

        assert cancelled.content.startswith("cancelled after 0.5 s")
        assert "Columns: ?column?" in again.content
        assert env.stopped == [QUERY]

    async def test_a_cancel_never_hides_a_failure_of_the_environment(self) -> None:
        from goaly.contracts.errors import EnvironmentFailure

        class Failing(FakeSqlEnvironment):
            async def execute(self, sql: str, *, timeout_s: float) -> SqlResult:
                raise EnvironmentFailure("the database died")

        bound = bind(ExecuteSql(ExecuteSqlConfig(guard_min_s=0.05), lambda: 1), timeout_s=600)

        with pytest.raises(EnvironmentFailure):
            await bound.invoke(json.dumps({"sql": QUERY}), Failing())


class TestWiring:
    async def test_the_tool_runs_with_the_time_the_task_has_left(self) -> None:
        # What the policy hands the tool shrinks as the task spends its wall budget, so the
        # guard tightens with it; the clock is the loop's, which core's wall budget uses too.
        from goaly.harness.packs.effisql.policy import wall_clock_remaining

        budgets = Budgets(steps=30, wall_clock_s=40)
        started = asyncio.get_running_loop().time()
        remaining = wall_clock_remaining(budgets, started)

        assert remaining() <= 40
        await asyncio.sleep(0.01)
        assert remaining() < 40
        limit = ExecuteSql(ExecuteSqlConfig(), remaining).guard_limit_s()
        assert limit is not None  # Half of the time the task has left, so a guard, not a cap.
        assert limit < 20

    async def test_the_policy_passes_its_clock_to_the_tool(self) -> None:
        from goaly.contracts.gen.task_pb2 import EffiSqlTaskInput, TaskInput
        from goaly.harness.core.agent import AgentContext
        from goaly.harness.packs.effisql.policy import build_policy
        from tests.harness.core.agent_fakes import FakeLLM, reply

        base = "SELECT count(*) FROM orders"
        task = TaskInput(
            effisql=EffiSqlTaskInput(
                db="tpch", base_sql=base, schema="", base_plan="{}", difficulty=""
            )
        )
        slow = "SELECT pg_sleep(30)"
        env = FakeSqlEnvironment({slow: SqlResult(elapsed_s=30)})
        llm = FakeLLM([reply(tool_call("execute_sql", {"sql": slow}))])

        ctx = AgentContext(llm, budgets=Budgets(steps=1, wall_clock_s=4))
        prediction = await build_policy({"execute_sql": {"guard_min_s": 0.05}}, "p").solve(
            task, env, ctx
        )

        assert prediction.fallback  # The step budget ran out, not the wall clock mid-statement.
        assert prediction.effisql.sql == base
        assert env.stopped == [slow]  # The guard cancelled the call, not the task's time.
