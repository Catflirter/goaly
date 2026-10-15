"""Tests for goaly.harness.core.tool_calls: reading tool calls out of a reply, and repairing them.

The Qwen fixtures in qwen_replies.json are real replies of qwen3.6-35b-a3b through the gateway,
sanitized to the assistant message; each says how it was produced.
"""

import enum
import json
import re
import time
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import BaseModel

from goaly.contracts.gen.environment_pb2 import SqlResult
from goaly.contracts.gen.trajectory_pb2 import (
    CHAT_ROLE_ASSISTANT,
    TOOL_CALL_SOURCE_CONTENT_XML,
    TOOL_CALL_SOURCE_REASONING_XML,
    TOOL_CALL_SOURCE_STANDARD,
    ChatMessage,
    ToolCall,
    ToolResult,
)
from goaly.contracts.interfaces import Environment, JsonValue, SqlEnvironment
from goaly.harness.core.tool_calls import parse_tool_calls, repair_arguments
from goaly.harness.core.tools import ToolRegistry, bind
from tests.fakes import FakeSqlEnvironment, SubmitTool

pytestmark = pytest.mark.anyio


class ExecuteSqlArgs(BaseModel):
    """The arguments of execute_sql."""

    sql: str
    limit: int = 20


class ExecuteSql:
    """Effi-SQL's execute_sql, as v0 has it: it runs one statement."""

    name = "execute_sql"
    description = "Run one SQL statement and return the rows or the error."
    args_model = ExecuteSqlArgs

    async def run(self, args: ExecuteSqlArgs, env: SqlEnvironment) -> ToolResult:
        result = await env.execute(args.sql, timeout_s=120.0)
        return ToolResult(content=result.error or f"rows: {result.row_count}")


class Mode(enum.Enum):
    FAST = "fast"
    SAFE = "safe"


class Window(BaseModel):
    """A range of rows."""

    start: int
    end: int


class ConfigureArgs(BaseModel):
    """Arguments of every kind a pydantic model gives a JSON Schema for."""

    text: str
    count: int
    ratio: float
    flag: bool
    tags: list[str]
    window: Window
    mode: Mode
    note: str | None = None
    limit: int | None = None


class Configure:
    """A tool whose arguments have every kind of JSON Schema type."""

    name = "configure"
    description = "Configure something."
    args_model = ConfigureArgs

    async def run(self, args: ConfigureArgs, env: Environment) -> ToolResult:
        return ToolResult(content="configured")


# The tools offered to the model, as the LLM client sends them.
REGISTRY: ToolRegistry[SqlEnvironment] = ToolRegistry([bind(ExecuteSql()), bind(SubmitTool())])
OFFERED = [*REGISTRY.json_schemas(), bind(Configure()).json_schema()]

CALL_ID = re.compile(r"call_[0-9a-f]{24}")


def assistant(
    content: str = "", reasoning: str = "", calls: Sequence[ToolCall] = ()
) -> ChatMessage:
    return ChatMessage(
        role=CHAT_ROLE_ASSISTANT, content=content, reasoning_content=reasoning, tool_calls=calls
    )


def xml_call(name: str, **parameters: str) -> str:
    """A call in Qwen3-Coder's XML form, laid out as Qwen writes it."""
    body = "".join(
        f"<parameter={key}>\n{value}\n</parameter>\n" for key, value in parameters.items()
    )
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def arguments(call: ToolCall) -> object:
    return json.loads(call.arguments_json)


QWEN_REPLIES = json.loads(Path(__file__).with_name("qwen_replies.json").read_text("utf-8"))


def qwen_reply(name: str) -> ChatMessage:
    """A reply from qwen_replies.json, as the LLM client returns it."""
    reply = QWEN_REPLIES["replies"][name]
    calls = [
        ToolCall(id=call["id"], name=call["name"], arguments_json=call["arguments"])
        for call in reply["tool_calls"]
    ]
    return assistant(content=reply["content"], reasoning=reply["reasoning_content"], calls=calls)


# The query of the task the Qwen replies answer, and the plan Qwen asked for.
BASE_SQL = (
    "SELECT count(*) FROM orders WHERE o_custkey IN "
    "(SELECT c_custkey FROM customer WHERE c_nationkey = 7);"
)
EXPLAIN = f"EXPLAIN {BASE_SQL}"


@pytest.mark.parametrize(
    ("text", "repaired"),
    [
        ('{"sql": "SELECT 1",}', '{"sql": "SELECT 1"}'),
        ('{"columns": ["a", "b",], "limit": 5,}', '{"columns": ["a", "b"], "limit": 5}'),
        ('{"columns": ["a", "b" ,\n ], "limit": 5 , }', '{"columns": ["a", "b"], "limit": 5}'),
        ('```json\n{"sql": "SELECT 1"}\n```', '{"sql": "SELECT 1"}'),
        ('```\n{"sql": "SELECT 1"}\n```\n', '{"sql": "SELECT 1"}'),
        ('```{"sql": "SELECT 1"}```', '{"sql": "SELECT 1"}'),
        ('```json\n{"sql": "SELECT 1"', '{"sql": "SELECT 1"}'),
        ('{"sql": "SELECT 1"', '{"sql": "SELECT 1"}'),
        ('{"sql": "SELECT a FROM t WHERE x =', '{"sql": "SELECT a FROM t WHERE x ="}'),
        ('{"columns": ["a", "b"', '{"columns": ["a", "b"]}'),
        ('{"filter": {"a": [1, {"b": "c', '{"filter": {"a": [1, {"b": "c"}]}}'),
        ('{"sql": "SELECT \'\\', '{"sql": "SELECT \'"}'),
        ('{"sql": "SELECT 1", ', '{"sql": "SELECT 1"}'),
        ('{"sql": "SELECT a,\n  b FROM t"}', '{"sql": "SELECT a,\\n  b FROM t"}'),
        ('{"sql": "say \\"hi\\", then }"', '{"sql": "say \\"hi\\", then }"}'),
        ("", "{}"),
        (" \n", "{}"),
    ],
    ids=[
        "trailing comma",
        "trailing commas inside",
        "trailing commas among spaces",
        "json code fence",
        "bare code fence",
        "fence on one line",
        "fence cut off",
        "open brace",
        "open string",
        "open bracket",
        "open nesting",
        "cut escape",
        "cut after a comma",
        "raw newline in a string",
        "brackets inside a string",
        "empty",
        "blank",
    ],
)
def test_repair_fixes_json_it_can_fix(text: str, repaired: str) -> None:
    assert repair_arguments(text) == repaired


@pytest.mark.parametrize(
    "text",
    ['{"sql": ', '{"sql"', '{"sq', '{"sql": "a"}}', "[1, 2]", '"SELECT 1"', "SELECT 1"],
    ids=[
        "key with no value",
        "key with no colon",
        "cut inside a key",
        "extra closing brace",
        "an array",
        "a string",
        "not JSON",
    ],
)
def test_repair_gives_up_on_json_it_cannot_make_an_object(text: str) -> None:
    assert repair_arguments(text) is None


def test_standard_tool_calls_come_first_and_are_kept_as_they_are() -> None:
    standard = ToolCall(id="call_a", name="execute_sql", arguments_json='{"sql": "SELECT 1"}')
    message = assistant(
        content=xml_call("submit", sql="SELECT 2"),
        reasoning=xml_call("submit", sql="SELECT 3"),
        calls=[standard],
    )

    calls = parse_tool_calls(message, OFFERED)

    assert calls == [
        ToolCall(
            id="call_a",
            name="execute_sql",
            arguments_json='{"sql": "SELECT 1"}',
            source=TOOL_CALL_SOURCE_STANDARD,
        )
    ]


def test_with_no_standard_calls_the_xml_calls_in_the_content_come_next() -> None:
    message = assistant(
        content="Let me look.\n\n" + xml_call("execute_sql", sql="SELECT 1"),
        reasoning=xml_call("submit", sql="SELECT 2"),
    )

    [call] = parse_tool_calls(message, OFFERED)

    assert (call.name, arguments(call), call.source) == (
        "execute_sql",
        {"sql": "SELECT 1"},
        TOOL_CALL_SOURCE_CONTENT_XML,
    )


def test_with_no_calls_elsewhere_the_xml_calls_in_the_reasoning_are_used() -> None:
    message = assistant(
        content="\n\n", reasoning="I will submit.\n" + xml_call("submit", sql="SELECT 2")
    )

    [call] = parse_tool_calls(message, OFFERED)

    assert (call.name, arguments(call), call.source) == (
        "submit",
        {"sql": "SELECT 2"},
        TOOL_CALL_SOURCE_REASONING_XML,
    )


def test_a_reply_with_no_calls_has_none() -> None:
    message = assistant(content="The rewrite is done.", reasoning="Nothing left to check.")

    assert parse_tool_calls(message, OFFERED) == []


def test_a_call_with_no_id_gets_one_that_depends_only_on_the_call() -> None:
    message = assistant(
        calls=[
            ToolCall(name="execute_sql", arguments_json='{"sql": "SELECT 1"}'),
            ToolCall(name="execute_sql", arguments_json='{"sql": "SELECT 1"}'),
        ]
    )

    first, second = parse_tool_calls(message, OFFERED)

    assert CALL_ID.fullmatch(first.id)
    assert CALL_ID.fullmatch(second.id)
    assert first.id != second.id  # The second call of the same reply is a different call.
    assert [call.id for call in parse_tool_calls(message, OFFERED)] == [first.id, second.id]


def test_each_xml_call_gets_an_id_and_they_come_in_order() -> None:
    message = assistant(
        content=xml_call("execute_sql", sql="SELECT 1") + "\n" + xml_call("submit", sql="SELECT 1")
    )

    calls = parse_tool_calls(message, OFFERED)

    assert [(call.name, arguments(call)) for call in calls] == [
        ("execute_sql", {"sql": "SELECT 1"}),
        ("submit", {"sql": "SELECT 1"}),
    ]
    assert all(CALL_ID.fullmatch(call.id) for call in calls)
    assert calls[0].id != calls[1].id


def test_a_standard_call_with_broken_json_gets_repaired_arguments() -> None:
    written = '```json\n{"sql": "SELECT 1", "limit": 5,}\n```'
    message = assistant(calls=[ToolCall(id="call_a", name="execute_sql", arguments_json=written)])

    [call] = parse_tool_calls(message, OFFERED)

    assert call.arguments_json == written  # As the model wrote it.
    assert call.repaired_arguments_json == '{"sql": "SELECT 1", "limit": 5}'


def test_a_standard_call_whose_json_cannot_be_repaired_has_no_repaired_arguments() -> None:
    message = assistant(
        calls=[ToolCall(id="call_a", name="execute_sql", arguments_json='{"sql": ')]
    )

    [call] = parse_tool_calls(message, OFFERED)

    assert (call.arguments_json, call.repaired_arguments_json) == ('{"sql": ', "")


def test_xml_parameters_take_the_types_the_tools_json_schema_gives_them() -> None:
    message = assistant(
        content=xml_call(
            "configure",
            text=" two words ",
            count="3",
            ratio="0.5",
            flag="True",
            tags='["a", "b"]',
            window='{"start": 1, "end": 2}',
            mode="fast",
            note="null",
            limit="null",
            extra="7",
        )
    )

    [call] = parse_tool_calls(message, OFFERED)

    assert arguments(call) == {
        "text": " two words ",
        "count": 3,
        "ratio": 0.5,
        "flag": True,
        "tags": ["a", "b"],
        "window": {"start": 1, "end": 2},
        "mode": "fast",
        "note": None,
        "limit": None,
        "extra": "7",  # Not a parameter of the tool, so it stays a string.
    }
    ConfigureArgs.model_validate_json(call.arguments_json)  # The tool accepts them.


def test_an_xml_value_that_does_not_parse_as_its_type_stays_a_string() -> None:
    message = assistant(content=xml_call("configure", count="three", tags="a, b", flag="maybe"))

    [call] = parse_tool_calls(message, OFFERED)

    assert arguments(call) == {"count": "three", "tags": "a, b", "flag": "maybe"}


def test_the_parameters_of_an_unknown_tool_stay_strings() -> None:
    message = assistant(content=xml_call("explain_query", sql="SELECT 1", analyze="true"))

    [call] = parse_tool_calls(message, OFFERED)

    assert (call.name, arguments(call)) == ("explain_query", {"sql": "SELECT 1", "analyze": "true"})


def test_an_xml_call_cut_off_inside_a_value_keeps_what_was_written() -> None:
    message = assistant(
        content="<tool_call>\n<function=execute_sql>\n<parameter=limit>\n5\n</parameter>\n"
        "<parameter=sql>\nSELECT a FROM t WHERE"
    )

    [call] = parse_tool_calls(message, OFFERED)

    assert (call.name, arguments(call)) == (
        "execute_sql",
        {"limit": 5, "sql": "SELECT a FROM t WHERE"},
    )


def test_an_xml_call_with_unclosed_parameters_splits_them_at_the_next_tag() -> None:
    message = assistant(
        content="<tool_call><function=execute_sql><parameter=sql>SELECT 1"
        "<parameter=limit>5</function></tool_call>"
    )

    [call] = parse_tool_calls(message, OFFERED)

    assert arguments(call) == {"sql": "SELECT 1", "limit": 5}


def test_a_function_block_without_a_tool_call_tag_is_still_a_call() -> None:
    message = assistant(
        reasoning="I'll run it:\n"
        "<function=execute_sql>\n<parameter=sql>\nSELECT 1\n</parameter>\n</function>"
    )

    [call] = parse_tool_calls(message, OFFERED)

    assert (call.name, arguments(call), call.source) == (
        "execute_sql",
        {"sql": "SELECT 1"},
        TOOL_CALL_SOURCE_REASONING_XML,
    )


def test_names_in_the_xml_are_read_without_surrounding_spaces() -> None:
    message = assistant(
        content="<tool_call><function= submit >"
        "<parameter= sql >SELECT 1</parameter></function></tool_call>"
    )

    [call] = parse_tool_calls(message, OFFERED)

    assert (call.name, arguments(call)) == ("submit", {"sql": "SELECT 1"})


def test_a_qwen_standard_call_is_kept_as_the_gateway_parsed_it() -> None:
    [call] = parse_tool_calls(qwen_reply("standard_call"), OFFERED)

    assert call == ToolCall(
        id="call_c91540942d574baabf65e259",
        name="execute_sql",
        arguments_json='{"sql": "EXPLAIN SELECT count(*) FROM orders WHERE o_custkey IN '
        '(SELECT c_custkey FROM customer WHERE c_nationkey = 7);"}',
        source=TOOL_CALL_SOURCE_STANDARD,
    )


def test_qwens_two_standard_calls_come_in_their_order() -> None:
    calls = parse_tool_calls(qwen_reply("two_standard_calls"), OFFERED)

    assert [(call.id, call.name, call.source) for call in calls] == [
        ("call_033e27ba0fb743ce80d3c353", "execute_sql", TOOL_CALL_SOURCE_STANDARD),
        ("call_4ede598186de471495f243d8", "execute_sql", TOOL_CALL_SOURCE_STANDARD),
    ]
    assert [call.repaired_arguments_json for call in calls] == ["", ""]


async def test_a_qwen_call_cut_off_by_max_tokens_runs_its_cut_off_sql() -> None:
    # The gateway closed the cut-off call's JSON itself, so nothing marks it as cut off: the
    # statement runs, and its SQL error is a normal result the model can act on.
    cut_sql = (
        "EXPLAIN SELECT count(*) FROM orders WHERE o_custkey IN "
        "(SELECT c_custkey FROM customer WHERE c_nationkey ="
    )
    env = FakeSqlEnvironment({cut_sql: SqlResult(error="syntax error at end of input")})

    [call] = parse_tool_calls(qwen_reply("truncated_standard_call"), OFFERED)
    result = await REGISTRY.execute(call, env)

    assert (call.source, call.repaired_arguments_json) == (TOOL_CALL_SOURCE_STANDARD, "")
    assert (result.content, result.is_error) == ("syntax error at end of input", False)


def test_a_qwen_xml_call_in_the_content_is_read_with_typed_parameters() -> None:
    [call] = parse_tool_calls(qwen_reply("xml_in_content"), OFFERED)

    assert (call.name, arguments(call), call.source) == (
        "execute_sql",
        {"sql": EXPLAIN, "limit": 20},
        TOOL_CALL_SOURCE_CONTENT_XML,
    )
    assert CALL_ID.fullmatch(call.id)


def test_a_qwen_xml_call_cut_off_in_the_content_keeps_what_was_written() -> None:
    [call] = parse_tool_calls(qwen_reply("truncated_xml_in_content"), OFFERED)

    assert (call.name, arguments(call), call.source) == (
        "execute_sql",
        {
            "sql": "EXPLAIN SELECT count(*) FROM orders WHERE o_custkey IN "
            "(SELECT c_custkey FROM customer WHERE c_nationkey"
        },
        TOOL_CALL_SOURCE_CONTENT_XML,
    )


async def test_a_qwen_call_to_an_unknown_tool_gets_an_error_that_lists_the_tools() -> None:
    [call] = parse_tool_calls(qwen_reply("unknown_tool_in_content"), OFFERED)
    result = await REGISTRY.execute(call, FakeSqlEnvironment())

    assert (call.name, arguments(call)) == ("explain_query", {"sql": BASE_SQL})
    assert result == ToolResult(
        tool_call_id=call.id,
        name="explain_query",
        is_error=True,
        content='Error: there is no tool named "explain_query". '
        "The tools are: execute_sql, submit.",
    )


def test_a_qwen_call_written_into_the_reasoning_is_read_from_there() -> None:
    [call] = parse_tool_calls(qwen_reply("call_in_reasoning"), OFFERED)

    assert (call.name, arguments(call), call.source) == (
        "execute_sql",
        {"sql": EXPLAIN, "limit": 20},
        TOOL_CALL_SOURCE_REASONING_XML,
    )


async def test_qwen_arguments_cut_off_inside_a_string_are_repaired_and_run() -> None:
    cut_sql = (
        "EXPLAIN SELECT count(*) FROM orders WHERE o_custkey IN "
        "(SELECT c_custkey FROM customer WHERE c_nation"
    )
    env = FakeSqlEnvironment({cut_sql: SqlResult(error="syntax error at end of input")})

    [call] = parse_tool_calls(qwen_reply("invalid_json_arguments"), OFFERED)
    result = await REGISTRY.execute(call, env)

    assert CALL_ID.fullmatch(call.id)
    assert json.loads(call.repaired_arguments_json) == {"sql": cut_sql}
    assert env.executed == [(cut_sql, 120.0)]
    assert (result.tool_call_id, result.is_error) == (call.id, False)


async def test_a_call_whose_json_cannot_be_repaired_gets_an_error_with_an_excerpt() -> None:
    written = '{"sql": "SELECT a FROM t", "limit":'
    message = assistant(calls=[ToolCall(id="call_a", name="execute_sql", arguments_json=written)])

    [call] = parse_tool_calls(message, OFFERED)
    result = await REGISTRY.execute(call, FakeSqlEnvironment())

    assert result.content == (
        "Error: the arguments for execute_sql are not valid JSON "
        "(EOF while parsing a value at line 1 column 35):\n"
        '{"sql": "SELECT a FROM t", "limit":'
    )
    assert (result.tool_call_id, result.is_error) == ("call_a", True)


async def test_the_excerpt_of_long_invalid_arguments_keeps_their_start_and_end() -> None:
    written = "{" + "x" * 300
    message = assistant(calls=[ToolCall(id="call_a", name="execute_sql", arguments_json=written)])

    [call] = parse_tool_calls(message, OFFERED)
    result = await REGISTRY.execute(call, FakeSqlEnvironment())

    assert result.content == (
        "Error: the arguments for execute_sql are not valid JSON "
        "(key must be a string at line 1 column 2):\n"
        + "{"
        + "x" * 99
        + "\n[... 101 of 301 characters cut ...]\n"
        + "x" * 100
    )


# Nested deeper than Python's json module can parse without running out of recursion.
TOO_DEEP = "[" * 200_000 + "]" * 200_000


def test_json_nested_too_deep_to_parse_cannot_be_repaired() -> None:
    assert repair_arguments(TOO_DEEP) is None


def test_a_standard_call_whose_arguments_nest_too_deep_gets_no_repaired_arguments() -> None:
    message = assistant(calls=[ToolCall(id="call_a", name="execute_sql", arguments_json=TOO_DEEP)])

    [call] = parse_tool_calls(message, OFFERED)

    assert call.repaired_arguments_json == ""


def test_an_xml_value_that_nests_too_deep_stays_a_string() -> None:
    message = assistant(content=xml_call("configure", tags=TOO_DEEP))

    [call] = parse_tool_calls(message, OFFERED)

    assert arguments(call) == {"tags": TOO_DEEP}


@pytest.mark.parametrize(
    "content",
    ["<function=" * 10_000, "<function=x>" + "<parameter=" * 10_000],
    ids=["unclosed function tags", "unclosed parameter tags"],
)
def test_a_degenerate_reply_full_of_unclosed_tags_is_read_in_linear_time(content: str) -> None:
    started = time.perf_counter()

    parse_tool_calls(assistant(content=content), OFFERED)

    assert time.perf_counter() - started < 1.0  # A quadratic scan takes several seconds.


def test_a_degenerate_fence_full_of_blank_lines_is_repaired_in_linear_time() -> None:
    started = time.perf_counter()

    assert repair_arguments("```" + "\n" * 32_000 + "x") is None

    assert time.perf_counter() - started < 1.0  # A quadratic scan takes several seconds.


def test_a_malformed_tools_list_leaves_what_it_cannot_read_as_strings() -> None:
    odd: dict[str, JsonValue] = {
        "properties": {
            "count": {"anyOf": ["integer", {"type": "integer"}]},
            "loop": {"$ref": "#/$defs/Loop"},
            "lost": {"$ref": "#/$defs/Missing"},
        },
        "$defs": {"Loop": {"$ref": "#/$defs/Loop"}},  # A definition that refers to itself.
    }
    tools: list[dict[str, JsonValue]] = [
        {"type": "function"},
        {"type": "function", "function": {"name": 5, "parameters": {}}},
        {"type": "function", "function": {"name": "odd", "parameters": odd}},
    ]
    message = assistant(content=xml_call("odd", count="3", loop="1", lost="2"))

    [call] = parse_tool_calls(message, tools)

    assert arguments(call) == {"count": 3, "loop": "1", "lost": "2"}
