"""Tests for the v0 Effi-SQL tools: execute_sql's arguments, timeouts and output, and submit."""

import asyncio
import json
from typing import Any

import pytest
from google.protobuf.struct_pb2 import ListValue

from goaly.bench.effisql import sql_env
from goaly.contracts.gen.environment_pb2 import SqlResult
from goaly.contracts.gen.trajectory_pb2 import ToolResult
from goaly.contracts.interfaces import SqlEnvironment
from goaly.harness.core.budgets import Budgets
from goaly.harness.core.tools import ToolRegistry, bind
from goaly.harness.packs.effisql.policy import PackConfig, registry
from goaly.harness.packs.effisql.tools import (
    STATEMENT_TIMEOUT,
    ExecuteSql,
    ExecuteSqlConfig,
    Submit,
    VerifyEquivalence,
    VerifyEquivalenceArgs,
    cell,
    render,
    verify_equivalence,
)
from tests.bench.effisql.test_sql_env import Scenario
from tests.bench.effisql.test_sql_env import _environment as trusted_environment
from tests.fakes import STATEMENT_TIMEOUT as FAKE_STATEMENT_TIMEOUT
from tests.fakes import FakeSqlEnvironment
from tests.harness.core.agent_fakes import tool_call

pytestmark = pytest.mark.anyio

V0 = Budgets(steps=30, wall_clock_s=600)
QUERY = "SELECT n_name FROM nation"
BASE = "SELECT n_name FROM nation"
CANDIDATE = "SELECT n_name FROM nation ORDER BY 1"


def diff_statement(base: str, candidate: str) -> str:
    """The one statement verify_equivalence runs: the bidirectional EXCEPT ALL diff."""
    return (
        "SELECT count(*) AS diffs FROM ("  # noqa: S608  # The two queries under test.
        f"(SELECT * FROM ({candidate}) c EXCEPT ALL SELECT * FROM ({base}) b)"
        f" UNION ALL (SELECT * FROM ({base}) b EXCEPT ALL SELECT * FROM ({candidate}) c)"
        ") d"
    )


def rows(*values: list[object]) -> list[ListValue]:
    out = []
    for row in values:
        listed = ListValue()
        listed.extend(row)  # type: ignore[arg-type]  # Test data of JSON-like values.
        out.append(listed)
    return out


def tools(env_config: ExecuteSqlConfig | None = None) -> ToolRegistry[SqlEnvironment]:
    config = PackConfig() if env_config is None else PackConfig(execute_sql=env_config)
    return registry(config, V0)


class TestArguments:
    def test_the_model_sees_each_tool_and_its_arguments(self) -> None:
        listed: Any = json.loads(json.dumps(tools().json_schemas()))
        schemas = {entry["function"]["name"]: entry["function"] for entry in listed}

        assert list(schemas) == ["execute_sql", "verify_equivalence", "submit"]
        for name in ("execute_sql", "submit"):
            parameters = schemas[name]["parameters"]
            assert parameters["required"] == ["sql"]
            assert list(parameters["properties"]) == ["sql"]
        assert schemas["execute_sql"]["parameters"]["properties"]["sql"]["minLength"] == 1
        assert schemas["submit"]["parameters"]["properties"]["sql"]["pattern"] == r"\S"
        verify = schemas["verify_equivalence"]["parameters"]
        assert verify["required"] == ["base_sql", "candidate_sql"]
        assert list(verify["properties"]) == ["base_sql", "candidate_sql"]
        for field in verify["properties"].values():
            assert field["pattern"] == r"\S"

    @pytest.mark.parametrize(
        ("name", "arguments"),
        [
            ("execute_sql", "{}"),
            ("execute_sql", '{"sql": ""}'),
            ("execute_sql", '{"sql": " \\n "}'),
            ("execute_sql", '{"sql": 1}'),
            ("execute_sql", "not json"),
            ("verify_equivalence", "{}"),
            ("verify_equivalence", '{"base_sql": "SELECT 1"}'),
            ("verify_equivalence", '{"candidate_sql": "SELECT 1"}'),
            ("verify_equivalence", '{"base_sql": "", "candidate_sql": "SELECT 1"}'),
            ("verify_equivalence", '{"base_sql": " ", "candidate_sql": "SELECT 1"}'),
            ("submit", "{}"),
            ("submit", '{"sql": ""}'),
            ("submit", '{"sql": " \\n "}'),
            ("submit", '{"sql": 1}'),
            ("submit", "not json"),
        ],
    )
    async def test_invalid_arguments_are_an_error_and_run_nothing(
        self, name: str, arguments: str
    ) -> None:
        env = FakeSqlEnvironment()

        result = await tools().execute(tool_call(name, arguments), env)

        assert result.is_error
        assert not result.HasField("submission")
        assert env.executed == []

    async def test_submit_hands_in_the_query_without_running_it(self) -> None:
        env = FakeSqlEnvironment()

        result = await tools().execute(tool_call("submit", {"sql": QUERY}), env)

        assert not result.is_error
        assert result.submission.effisql.sql == QUERY
        assert not result.submission.fallback
        assert env.executed == []


class TestTimeouts:
    async def test_the_statement_runs_with_the_120_s_statement_timeout(self) -> None:
        env = FakeSqlEnvironment({QUERY: SqlResult(columns=["n_name"], row_count=0)})

        await tools().execute(tool_call("execute_sql", {"sql": QUERY}), env)

        assert env.executed == [(QUERY, 120.0)]

    def test_the_executor_backstop_is_the_tasks_wall_budget(self) -> None:
        timeouts = {tool.name: tool.timeout_s for tool in tools()}

        assert timeouts == {
            "execute_sql": 600.0,
            "verify_equivalence": 600.0,
            "submit": 10.0,
        }
        short = registry(PackConfig(), Budgets(steps=3, wall_clock_s=42))
        assert next(iter(short)).timeout_s == 42

    async def test_a_statement_past_its_timeout_is_a_normal_result(self) -> None:
        config = ExecuteSqlConfig(statement_timeout_s=0.01)
        env = FakeSqlEnvironment({QUERY: SqlResult(elapsed_s=5)})

        result = await tools(config).execute(tool_call("execute_sql", {"sql": QUERY}), env)

        assert not result.is_error
        assert result.content.startswith(f"The query failed: {STATEMENT_TIMEOUT}\n")
        assert "stopped at the 0.01 s statement timeout" in result.content
        assert FAKE_STATEMENT_TIMEOUT == STATEMENT_TIMEOUT

    def test_the_timeout_message_is_the_real_environments(self) -> None:
        # The pack cannot import bench, so it keeps its own copy; this keeps the copies one.
        assert sql_env.STATEMENT_TIMEOUT == STATEMENT_TIMEOUT
        assert FAKE_STATEMENT_TIMEOUT == STATEMENT_TIMEOUT

    async def test_the_backstop_cancels_a_statement_the_environment_never_ends(self) -> None:
        class Stuck(FakeSqlEnvironment):
            async def execute(self, sql: str, *, timeout_s: float) -> SqlResult:
                await asyncio.Event().wait()
                raise AssertionError("unreachable")

        bound = bind(ExecuteSql(ExecuteSqlConfig()), timeout_s=0.05)

        result = await bound.invoke(json.dumps({"sql": QUERY}), Stuck())

        assert result.is_error
        assert "timed out after 0.05 s" in result.content


class TestVerifyEquivalence:
    def canned(self, statement: str, result: SqlResult) -> FakeSqlEnvironment:
        return FakeSqlEnvironment({statement: result})

    async def call(
        self, env: FakeSqlEnvironment, base: str = BASE, candidate: str = CANDIDATE
    ) -> ToolResult:
        return await tools().execute(
            tool_call("verify_equivalence", {"base_sql": base, "candidate_sql": candidate}), env
        )

    async def test_it_runs_the_one_fixed_diff_statement_once(self) -> None:
        statement = diff_statement(BASE, CANDIDATE)
        env = self.canned(statement, SqlResult(columns=["diffs"], rows=rows([0.0]), row_count=1))

        result = await self.call(env)

        assert not result.is_error
        assert not result.HasField("submission")
        assert env.executed == [(statement, 120.0)]

    async def test_no_differing_rows_is_equivalent(self) -> None:
        env = self.canned(
            diff_statement(BASE, CANDIDATE),
            SqlResult(columns=["diffs"], rows=rows([0.0]), row_count=1),
        )

        result = await self.call(env)

        assert result.content == "EQUIVALENT"

    async def test_differing_rows_are_counted(self) -> None:
        env = self.canned(
            diff_statement(BASE, CANDIDATE),
            SqlResult(columns=["diffs"], rows=rows([3.0]), row_count=1),
        )

        result = await self.call(env)

        assert result.content == "NOT EQUIVALENT: 3 differing rows"

    async def test_a_count_past_exact_floats_still_reads(self) -> None:
        env = self.canned(
            diff_statement(BASE, CANDIDATE),
            SqlResult(columns=["diffs"], rows=rows(["9007199254740993"]), row_count=1),
        )

        result = await self.call(env)

        assert result.content == "NOT EQUIVALENT: 9007199254740993 differing rows"

    async def test_a_database_error_is_the_answer(self) -> None:
        error = "each EXCEPT ALL query must have the same number of columns"
        env = self.canned(diff_statement(BASE, CANDIDATE), SqlResult(error=error, elapsed_s=0.1))

        result = await self.call(env)

        assert not result.is_error  # A statement that ran and failed is a normal result.
        assert result.content == error

    async def test_a_statement_timeout_is_an_ordinary_result(self) -> None:
        env = self.canned(diff_statement(BASE, CANDIDATE), SqlResult(error=STATEMENT_TIMEOUT))

        result = await self.call(env)

        assert not result.is_error
        assert result.content == STATEMENT_TIMEOUT

    async def test_a_missing_count_is_reported_not_guessed(self) -> None:
        env = self.canned(
            diff_statement(BASE, CANDIDATE), SqlResult(columns=["diffs"], row_count=0)
        )

        result = await self.call(env)

        assert result.content == "the database returned no count, so equivalence is unknown"

    async def test_trailing_semicolons_and_blank_ends_are_trimmed(self) -> None:
        statement = diff_statement(BASE, CANDIDATE)
        env = self.canned(statement, SqlResult(columns=["diffs"], rows=rows([0.0]), row_count=1))

        result = await self.call(env, base=f"\n{BASE} ; \n", candidate=f"{CANDIDATE};")

        assert result.content == "EQUIVALENT"
        assert env.executed == [(statement, 120.0)]

    async def test_the_function_reports_the_verdict_with_its_own_timeout(self) -> None:
        statement = diff_statement(BASE, CANDIDATE)
        env = self.canned(statement, SqlResult(columns=["diffs"], rows=rows([2.0]), row_count=1))

        verdict = await verify_equivalence(BASE, CANDIDATE, env, 30.0)

        assert verdict == "NOT EQUIVALENT: 2 differing rows"
        assert env.executed == [(statement, 30.0)]

    async def test_the_tool_is_named_described_and_instructed_for_the_model(self) -> None:
        assert VerifyEquivalence.name == "verify_equivalence"
        assert "EXCEPT ALL" in VerifyEquivalence.description
        assert "EQUIVALENT" in VerifyEquivalence.description
        docstring = " ".join((VerifyEquivalenceArgs.__doc__ or "").split())
        assert docstring == (
            "Check a rewrite against the original query before you submit it. Call this once "
            "with your final rewrite before submitting. If the result is not EQUIVALENT, or "
            "the call errors or times out, submit the base query unchanged instead."
        )


class TestOutput:
    def test_rows_count_and_timing_with_the_queue_wait_apart(self) -> None:
        result = SqlResult(
            columns=["n_name", "n_regionkey"],
            rows=rows(["ALGERIA", 0.0], ["ARGENTINA", 1.0]),
            row_count=25,
            truncated=True,
            elapsed_s=0.0123,
            queue_wait_s=1.5,
        )

        text = render(result, ExecuteSqlConfig())

        assert text == (
            "Columns: n_name, n_regionkey\n"
            "Rows: 25 rows, the first 2 shown.\n"
            "ALGERIA | 0\n"
            "ARGENTINA | 1\n"
            "Execution time: 0.012 s (it waited 1.500 s for a statement slot before that)."
        )

    def test_only_rows_shown_rows_reach_the_model(self) -> None:
        result = SqlResult(columns=["x"], rows=rows([1.0], [2.0], [3.0]), row_count=3)

        text = render(result, ExecuteSqlConfig(rows_shown=1))

        assert "Rows: 3 rows, the first 1 shown.\n1\n" in text
        assert "\n2\n" not in text

    def test_a_statement_without_columns(self) -> None:
        text = render(SqlResult(row_count=3), ExecuteSqlConfig())

        assert text.startswith("Columns: (none)\nRows: 3 rows.\n")

    def test_a_single_row_and_no_rows(self) -> None:
        one = render(SqlResult(columns=["x"], rows=rows([7.0]), row_count=1), ExecuteSqlConfig())
        none = render(SqlResult(columns=["x"], row_count=0), ExecuteSqlConfig())

        assert "Rows: 1 row.\n7\n" in one
        assert "Rows: 0 rows.\nExecution time" in none

    def test_an_error_shows_the_message_and_the_timing(self) -> None:
        result = SqlResult(error='relation "x" does not exist', elapsed_s=0.001, queue_wait_s=0)

        assert render(result, ExecuteSqlConfig()) == (
            'The query failed: relation "x" does not exist\n'
            "Execution time: 0.001 s (it waited 0.000 s for a statement slot before that)."
        )

    @pytest.mark.parametrize(
        ("value", "text"),
        [
            (None, "NULL"),
            (True, "true"),
            (False, "false"),
            (3.0, "3"),
            (2.5, "2.5"),
            ("12345678901234567890.5", "12345678901234567890.5"),
            ("abcdef", "abc…"),
        ],
    )
    def test_values(self, value: object, text: str) -> None:
        assert cell(value, 3 if text.endswith("…") else 100) == text

    async def test_execute_sql_shows_the_rendered_result(self) -> None:
        canned = SqlResult(columns=["x"], rows=rows(["a"]), row_count=1, queue_wait_s=0.25)
        env = FakeSqlEnvironment({QUERY: canned})

        result = await ExecuteSql(ExecuteSqlConfig()).run(ExecuteSql.args_model(sql=QUERY), env)

        assert result.content == render(canned, ExecuteSqlConfig())
        assert "waited 0.250 s" in result.content

    async def test_submit_is_named_and_described_for_the_model(self) -> None:
        assert Submit.name == "submit"
        assert "same result" in Submit.description


class TestTrustedCap:
    async def test_a_pack_asking_for_1e9_s_stops_at_the_environments_cap(self) -> None:
        # Pack code is untrusted: it may drop the le=120 bound, as model_construct does here.
        lifted = ExecuteSqlConfig.model_construct(statement_timeout_s=1e9)
        scenario = Scenario("cancellable")
        env = trusted_environment(scenario, timeout_s=0.2)  # The trusted side's cap.
        await env.open()
        bound = bind(ExecuteSql(lifted), timeout_s=600)

        result = await bound.invoke(json.dumps({"sql": "SELECT pg_sleep(1e9)"}), env)

        assert not result.is_error
        assert result.content.startswith(f"The query failed: {STATEMENT_TIMEOUT}\n")
        assert result.elapsed_s < 1  # The environment's 0.2 s, not the 1e9 s asked for.
        await env.close()
