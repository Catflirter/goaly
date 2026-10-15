"""Tests for goaly.harness.core.tools: binding, the registry and the executor (spec §5.1).

The tools here run in the fake environments of tests/fakes.py, so a statement that sleeps,
fails or kills the environment is a canned result.
"""

import asyncio
import json
import logging
import math
from collections.abc import Callable

import pytest
from pydantic import BaseModel
from pydantic.errors import PydanticInvalidForJsonSchema

from goaly.contracts.errors import EnvironmentFailure
from goaly.contracts.gen.environment_pb2 import SqlResult
from goaly.contracts.gen.task_pb2 import EffiSqlPrediction, Prediction
from goaly.contracts.gen.trajectory_pb2 import ToolCall, ToolResult
from goaly.contracts.interfaces import Environment, JsonValue, SqlEnvironment
from goaly.harness.core.tools import BoundTool, ToolRegistry, bind
from tests.fakes import EchoTool, FakeSqlEnvironment, SubmitTool

pytestmark = pytest.mark.anyio


class QueryArgs(BaseModel):
    """QueryTool's arguments."""

    sql: str


class QueryTool:
    """Runs one statement, as Effi-SQL's execute_sql does: a failed statement is a result."""

    name = "query"
    description = "Run a SQL statement."
    args_model = QueryArgs

    async def run(self, args: QueryArgs, env: SqlEnvironment) -> ToolResult:
        result = await env.execute(args.sql, timeout_s=120.0)
        return ToolResult(content=result.error or f"rows: {result.row_count}")


class SearchArgs(BaseModel):
    """SearchTool's arguments."""

    pattern: str
    paths: list[str]
    limit: int = 10


class SearchTool:
    """A tool with several arguments, so that a call can get several of them wrong."""

    name = "search"
    description = "Search the files for a pattern."
    args_model = SearchArgs

    async def run(self, args: SearchArgs, env: Environment) -> ToolResult:
        return ToolResult(content=f"no match for {args.pattern}")


class NoArgs(BaseModel):
    """No arguments."""


class CannedTool:
    """Returns the same result object on every call."""

    name = "canned"
    description = "Answer with a canned result."
    args_model = NoArgs

    def __init__(self, result: ToolResult) -> None:
        self.result = result

    async def run(self, args: NoArgs, env: Environment) -> ToolResult:
        return self.result


class RaisingTool:
    """Raises the exception it was given."""

    name = "fragile"
    description = "Fail."
    args_model = NoArgs

    def __init__(self, error: Exception) -> None:
        self.error = error

    async def run(self, args: NoArgs, env: Environment) -> ToolResult:
        raise self.error


class PairArgs(BaseModel):
    """PairTool's arguments."""

    first: str
    second: str


class PairTool:
    """Runs two statements at once, in a TaskGroup."""

    name = "pair"
    description = "Run two SQL statements at once."
    args_model = PairArgs

    async def run(self, args: PairArgs, env: SqlEnvironment) -> ToolResult:
        async with asyncio.TaskGroup() as group:
            for sql in (args.first, args.second):
                group.create_task(env.execute(sql, timeout_s=120.0))
        return ToolResult(content="Both ran.")


class CallbackArgs(BaseModel):
    """Arguments that no JSON Schema can describe."""

    callback: Callable[[], None]


class CallbackTool:
    """A tool the model could never call: its arguments have no JSON Schema."""

    name = "callback"
    description = "Call back."
    args_model = CallbackArgs

    async def run(self, args: CallbackArgs, env: Environment) -> ToolResult:
        args.callback()
        return ToolResult(content="called")


SLEEP_60 = '{"sql": "SELECT pg_sleep(60)"}'


def sleepy_environment() -> FakeSqlEnvironment:
    """An environment in which ``SELECT pg_sleep(60)`` takes a minute."""
    return FakeSqlEnvironment({"SELECT pg_sleep(60)": SqlResult(elapsed_s=60.0)})


def without_elapsed(result: ToolResult) -> ToolResult:
    """A copy of ``result`` without its elapsed_s, which no test can predict exactly."""
    copied = ToolResult()
    copied.CopyFrom(result)
    copied.ClearField("elapsed_s")
    return copied


def member(value: JsonValue, *keys: str) -> dict[str, JsonValue]:
    """The JSON object found by following ``keys`` down from ``value``."""
    for key in keys:
        assert isinstance(value, dict)
        value = value[key]
    assert isinstance(value, dict)
    return value


def test_a_bound_tool_describes_itself_as_an_openai_tools_entry() -> None:
    assert bind(EchoTool()).json_schema() == {
        "type": "function",
        "function": {
            "name": "echo",
            "description": "Reply with the given text.",
            "parameters": {
                "description": "EchoTool's arguments.",
                "properties": {"text": {"title": "Text", "type": "string"}},
                "required": ["text"],
                "title": "EchoArgs",
                "type": "object",
            },
        },
    }


def test_changing_a_tools_entry_changes_no_later_entry() -> None:
    tool = bind(EchoTool())

    member(tool.json_schema(), "function", "parameters", "properties", "text")["type"] = "integer"

    text = member(tool.json_schema(), "function", "parameters", "properties", "text")
    assert text["type"] == "string"


def test_binding_a_tool_whose_arguments_have_no_json_schema_fails() -> None:
    with pytest.raises(PydanticInvalidForJsonSchema):
        bind(CallbackTool())


def test_a_tool_bound_without_a_timeout_gets_60_seconds() -> None:
    assert bind(EchoTool()).timeout_s == 60.0
    assert bind(EchoTool(), timeout_s=0.5).timeout_s == 0.5


@pytest.mark.parametrize("timeout_s", [0.0, -1.0, math.inf, math.nan])
def test_binding_with_a_timeout_that_is_not_a_positive_number_fails(timeout_s: float) -> None:
    with pytest.raises(ValueError, match="timeout_s must be a positive, finite number of seconds"):
        bind(EchoTool(), timeout_s=timeout_s)


def test_the_registry_lists_its_tools_for_the_model_in_registration_order() -> None:
    submit, echo = bind(SubmitTool()), bind(EchoTool())

    registry = ToolRegistry([submit, echo])

    assert registry.json_schemas() == [submit.json_schema(), echo.json_schema()]


def test_a_registry_takes_tools_that_run_in_a_wider_environment() -> None:
    # A bound tool is contravariant in its environment: echo runs in any environment, so it
    # fits a registry for SQL tools. make typecheck proves it.
    registry: ToolRegistry[SqlEnvironment] = ToolRegistry([bind(QueryTool())])

    registry.register(bind(EchoTool()))

    assert [tool.name for tool in registry] == ["query", "echo"]


def test_registering_a_second_tool_with_a_taken_name_fails() -> None:
    registry = ToolRegistry([bind(EchoTool())])

    with pytest.raises(ValueError, match="a tool named 'echo' is already registered"):
        registry.register(bind(EchoTool(), timeout_s=5.0))
    assert [tool.timeout_s for tool in registry] == [60.0]


def test_building_a_registry_with_two_tools_of_the_same_name_fails() -> None:
    with pytest.raises(ValueError, match="a tool named 'submit' is already registered"):
        ToolRegistry([bind(SubmitTool()), bind(EchoTool()), bind(SubmitTool())])


async def test_a_call_runs_its_tool_and_the_result_answers_the_call() -> None:
    registry = ToolRegistry([bind(EchoTool())])
    call = ToolCall(id="call-1", name="echo", arguments_json='{"text": "hello"}')

    result = await registry.execute(call, FakeSqlEnvironment())

    assert without_elapsed(result) == ToolResult(
        tool_call_id="call-1", name="echo", content="hello"
    )


async def test_elapsed_s_measures_the_call() -> None:
    env = FakeSqlEnvironment({"SELECT pg_sleep(0.05)": SqlResult(row_count=1, elapsed_s=0.05)})

    result = await bind(QueryTool()).invoke('{"sql": "SELECT pg_sleep(0.05)"}', env)

    assert result.content == "rows: 1"
    assert result.elapsed_s >= 0.05


async def test_a_statement_that_ran_and_failed_is_a_normal_result() -> None:
    env = FakeSqlEnvironment()  # It has no canned results, so every statement fails.

    result = await bind(QueryTool()).invoke('{"sql": "SELEC 1"}', env)

    assert without_elapsed(result) == ToolResult(
        name="query", content="no canned result for 'SELEC 1'"
    )


async def test_a_submission_passes_through_unchanged() -> None:
    result = await bind(SubmitTool()).invoke('{"sql": "SELECT 1"}', FakeSqlEnvironment())

    assert without_elapsed(result) == ToolResult(
        name="submit",
        content="Submitted.",
        submission=Prediction(effisql=EffiSqlPrediction(sql="SELECT 1")),
    )


async def test_the_executor_fills_the_fields_the_tool_does_not_own() -> None:
    canned = ToolResult(
        tool_call_id="mine",
        name="other",
        content="done",
        is_error=True,
        truncated=True,
        elapsed_s=99.0,
    )
    registry = ToolRegistry([bind(CannedTool(canned))])
    call = ToolCall(id="call-1", name="canned", arguments_json="{}")

    result = await registry.execute(call, FakeSqlEnvironment())

    assert without_elapsed(result) == ToolResult(
        tool_call_id="call-1", name="canned", content="done"
    )
    assert result.elapsed_s < 99.0


async def test_the_executor_leaves_the_tools_own_result_unchanged() -> None:
    submission = Prediction(effisql=EffiSqlPrediction(sql="SELECT 1"))
    canned = ToolResult(content="x" * 9000, submission=submission)
    tool = bind(CannedTool(canned))

    first = await tool.invoke("{}", FakeSqlEnvironment())
    second = await tool.invoke("{}", FakeSqlEnvironment())

    assert canned == ToolResult(content="x" * 9000, submission=submission)
    assert without_elapsed(first) == without_elapsed(second)
    assert (first.truncated, first.submission) == (True, submission)


async def test_invalid_arguments_give_an_error_result_and_the_tool_does_not_run() -> None:
    env = FakeSqlEnvironment()

    result = await bind(QueryTool()).invoke('{"query": "SELECT 1"}', env)

    assert without_elapsed(result) == ToolResult(
        name="query",
        is_error=True,
        content="Error: invalid arguments for query:\n- sql: Field required",
    )
    assert result.elapsed_s > 0
    assert env.executed == []


async def test_the_error_names_each_invalid_argument_with_pydantics_message() -> None:
    arguments_json = '{"paths": ["src", 3], "limit": "all"}'

    result = await bind(SearchTool()).invoke(arguments_json, FakeSqlEnvironment())

    assert result.is_error
    assert result.content == (
        "Error: invalid arguments for search:\n"
        "- pattern: Field required\n"
        "- paths.1: Input should be a valid string\n"
        "- limit: Input should be a valid integer, unable to parse string as an integer"
    )


@pytest.mark.parametrize(
    ("arguments_json", "problem"),
    [
        ('{"text": "hi"', "Invalid JSON: EOF while parsing an object at line 1 column 13"),
        ("", "Invalid JSON: EOF while parsing a value at line 1 column 0"),
        ('["hi"]', "Input should be an object"),
    ],
)
async def test_arguments_that_are_not_a_json_object_give_an_error_result(
    arguments_json: str, problem: str
) -> None:
    result = await bind(EchoTool()).invoke(arguments_json, FakeSqlEnvironment())

    assert without_elapsed(result) == ToolResult(
        name="echo", is_error=True, content=f"Error: invalid arguments for echo:\n- {problem}"
    )


@pytest.mark.parametrize(
    ("tools", "listed"),
    [
        ([bind(EchoTool()), bind(SubmitTool())], "The tools are: echo, submit."),
        ([], "There are no tools."),
    ],
)
async def test_a_call_to_an_unknown_tool_gives_an_error_result_that_lists_the_tools(
    tools: list[BoundTool[Environment]], listed: str
) -> None:
    registry = ToolRegistry(tools)
    call = ToolCall(id="call-1", name="execute_sql", arguments_json='{"sql": "SELECT 1"}')

    result = await registry.execute(call, FakeSqlEnvironment())

    # No tool ran, so no time was spent running one.
    assert result == ToolResult(
        tool_call_id="call-1",
        name="execute_sql",
        is_error=True,
        content=f'Error: there is no tool named "execute_sql". {listed}',
    )


async def test_a_call_runs_with_its_repaired_arguments_when_it_has_them() -> None:
    registry = ToolRegistry([bind(EchoTool())])
    call = ToolCall(
        id="call-1",
        name="echo",
        arguments_json='{"text": "hi",',
        repaired_arguments_json='{"text": "hi"}',
    )

    result = await registry.execute(call, FakeSqlEnvironment())

    assert without_elapsed(result) == ToolResult(tool_call_id="call-1", name="echo", content="hi")


async def test_a_call_that_runs_past_its_timeout_is_stopped_and_gives_an_error_result() -> None:
    env = sleepy_environment()
    tool = bind(QueryTool(), timeout_s=0.05)

    async with asyncio.timeout(10):  # Fails the test, rather than hanging it, if the call runs on.
        result = await tool.invoke(SLEEP_60, env)

    assert without_elapsed(result) == ToolResult(
        name="query", is_error=True, content="Error: query timed out after 0.05 s and was stopped."
    )
    assert result.elapsed_s >= 0.05
    assert env.stopped == ["SELECT pg_sleep(60)"]


async def test_a_timeout_error_the_tool_raises_itself_is_not_the_calls_timeout() -> None:
    tool = bind(RaisingTool(TimeoutError("the lock wait timed out")))

    result = await tool.invoke("{}", FakeSqlEnvironment())

    assert result.content == "Error: fragile failed with TimeoutError: the lock wait timed out"


@pytest.mark.parametrize(
    ("error", "content"),
    [
        (ValueError("no column named price"), "ValueError: no column named price"),
        (KeyError("rows"), "KeyError: 'rows'"),
        (RuntimeError(), "RuntimeError"),
        (
            ExceptionGroup("two checks failed", [ValueError("a"), KeyError("b")]),
            "ExceptionGroup: two checks failed (2 sub-exceptions)",
        ),
    ],
)
async def test_an_exception_becomes_an_error_result_with_its_type_and_message(
    error: Exception, content: str
) -> None:
    result = await bind(RaisingTool(error)).invoke("{}", FakeSqlEnvironment())

    assert without_elapsed(result) == ToolResult(
        name="fragile", is_error=True, content=f"Error: fragile failed with {content}"
    )


async def test_a_long_exception_message_keeps_only_its_start_and_end() -> None:
    error = ValueError("a" * 250 + "b" * 100 + "c" * 250)

    result = await bind(RaisingTool(error)).invoke("{}", FakeSqlEnvironment())

    assert result.content == (
        "Error: fragile failed with ValueError: "
        + "a" * 250
        + "\n[... 100 of 600 characters cut ...]\n"
        + "c" * 250
    )


async def test_the_traceback_of_an_exception_goes_to_the_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    error = ValueError("no column named price")

    with caplog.at_level(logging.WARNING, logger="goaly.harness.core.tools"):
        await bind(RaisingTool(error)).invoke("{}", FakeSqlEnvironment())

    [record] = caplog.records
    assert record.exc_info is not None
    assert record.exc_info[1] is error
    assert vars(record)["tool"] == "fragile"


async def test_an_environment_failure_propagates_rather_than_becoming_a_result() -> None:
    died = EnvironmentFailure("the database connection died")
    registry = ToolRegistry([bind(QueryTool())])
    call = ToolCall(id="call-1", name="query", arguments_json='{"sql": "SELECT 1"}')

    with pytest.raises(EnvironmentFailure) as raised:
        await registry.execute(call, FakeSqlEnvironment({"SELECT 1": died}))

    assert raised.value is died


async def test_an_environment_failure_inside_an_exception_group_propagates() -> None:
    died = EnvironmentFailure("the database connection died")
    env = FakeSqlEnvironment({"SELECT pg_sleep(60)": SqlResult(elapsed_s=60.0), "SELECT 1": died})
    arguments_json = '{"first": "SELECT pg_sleep(60)", "second": "SELECT 1"}'

    with pytest.raises(ExceptionGroup) as raised:
        await bind(PairTool()).invoke(arguments_json, env)

    assert raised.value.exceptions == (died,)
    assert env.stopped == ["SELECT pg_sleep(60)"]  # The TaskGroup cancelled the other statement.


async def test_a_cancelled_call_stops_its_tool_and_raises_cancelled_error() -> None:
    env = sleepy_environment()
    call = asyncio.create_task(bind(QueryTool()).invoke(SLEEP_60, env))
    async with asyncio.timeout(10):
        while not env.executed:  # Until the statement runs.
            await asyncio.sleep(0)

    call.cancel()

    with pytest.raises(asyncio.CancelledError):
        await call
    assert env.stopped == ["SELECT pg_sleep(60)"]


async def test_a_call_cancelled_from_outside_propagates_the_cancellation() -> None:
    env = sleepy_environment()
    tool = bind(QueryTool(), timeout_s=60.0)

    # The outer timeout raises TimeoutError only if its cancellation came back out of invoke.
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.05):
            await tool.invoke(SLEEP_60, env)

    assert env.stopped == ["SELECT pg_sleep(60)"]


@pytest.mark.parametrize("length", [0, 7999, 8000])
async def test_content_of_up_to_8000_characters_stays_whole(length: int) -> None:
    text = "x" * length

    result = await bind(EchoTool()).invoke(json.dumps({"text": text}), FakeSqlEnvironment())

    assert (result.content, result.truncated) == (text, False)


@pytest.mark.parametrize(
    ("middle", "marker"),
    [
        ("m", "[... 1 of 8001 characters cut ...]"),
        ("m" * 5000, "[... 5000 of 13000 characters cut ...]"),
    ],
    ids=["just-over", "far-over"],
)
async def test_longer_content_keeps_its_first_and_last_4000_characters(
    middle: str, marker: str
) -> None:
    text = "h" * 4000 + middle + "t" * 4000

    result = await bind(EchoTool()).invoke(json.dumps({"text": text}), FakeSqlEnvironment())

    assert without_elapsed(result) == ToolResult(
        name="echo", content="h" * 4000 + f"\n{marker}\n" + "t" * 4000, truncated=True
    )


async def test_a_long_error_result_is_cut_down_like_any_other() -> None:
    arguments_json = json.dumps({"pattern": "x", "paths": list(range(500))})

    result = await bind(SearchTool()).invoke(arguments_json, FakeSqlEnvironment())

    assert (result.is_error, result.truncated) == (True, True)
    assert result.content.startswith(
        "Error: invalid arguments for search:\n- paths.0: Input should be a valid string\n"
    )
    assert "\n[... " in result.content
    assert result.content.endswith("\n- paths.499: Input should be a valid string")
