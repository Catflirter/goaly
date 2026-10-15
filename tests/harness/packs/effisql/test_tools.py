"""Tests for the v0 Effi-SQL tools: execute_sql's arguments, timeouts and output, and submit."""

import asyncio
import json
from typing import Any

import pytest
from google.protobuf.struct_pb2 import ListValue

from goaly.contracts.gen.environment_pb2 import SqlResult
from goaly.contracts.interfaces import SqlEnvironment
from goaly.harness.core.budgets import Budgets
from goaly.harness.core.tools import ToolRegistry, bind
from goaly.harness.packs.effisql.policy import PackConfig, registry
from goaly.harness.packs.effisql.tools import (
    STATEMENT_TIMEOUT,
    ExecuteSql,
    ExecuteSqlConfig,
    Submit,
    cell,
    render,
)
from tests.fakes import STATEMENT_TIMEOUT as FAKE_STATEMENT_TIMEOUT
from tests.fakes import FakeSqlEnvironment
from tests.harness.core.agent_fakes import tool_call

pytestmark = pytest.mark.anyio

V0 = Budgets(steps=30, wall_clock_s=600)
QUERY = "SELECT n_name FROM nation"


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
    def test_the_model_sees_one_required_sql_argument_for_each_tool(self) -> None:
        listed: Any = json.loads(json.dumps(tools().json_schemas()))
        schemas = {entry["function"]["name"]: entry["function"] for entry in listed}

        assert list(schemas) == ["execute_sql", "submit"]
        for function in schemas.values():
            parameters = function["parameters"]
            assert parameters["required"] == ["sql"]
            assert list(parameters["properties"]) == ["sql"]
            assert parameters["properties"]["sql"]["minLength"] == 1

    @pytest.mark.parametrize("arguments", ["{}", '{"sql": ""}', '{"sql": 1}', "not json"])
    async def test_invalid_arguments_are_an_error_and_run_nothing(self, arguments: str) -> None:
        env = FakeSqlEnvironment()

        for name in ("execute_sql", "submit"):
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

        assert timeouts == {"execute_sql": 600.0, "submit": 10.0}
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

    async def test_the_backstop_cancels_a_statement_the_environment_never_ends(self) -> None:
        class Stuck(FakeSqlEnvironment):
            async def execute(self, sql: str, *, timeout_s: float) -> SqlResult:
                await asyncio.Event().wait()
                raise AssertionError("unreachable")

        bound = bind(ExecuteSql(ExecuteSqlConfig()), timeout_s=0.05)

        result = await bound.invoke(json.dumps({"sql": QUERY}), Stuck())

        assert result.is_error
        assert "timed out after 0.05 s" in result.content


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
