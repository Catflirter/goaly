"""Tests for goaly.harness.core.tool_calls: reading tool calls out of a reply, and repairing them.

The Qwen fixtures in qwen_replies.json are replies of qwen3.6-35b-a3b through the gateway,
sanitized to the assistant message: six as Qwen gave them, and two built from real ones for
failures that never happened. Each says how it was produced.
"""

import enum
import json
import re
import time
from collections.abc import Callable, Sequence
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
from goaly.harness.core.tool_calls import (
    parse_tool_calls,
    repair_arguments,
    strip_tool_call_markup,
)
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
    columns: list[str] | None = None
    frame: Window | None = None


class Configure:
    """A tool whose arguments have every kind of JSON Schema type."""

    name = "configure"
    description = "Configure something."
    args_model = ConfigureArgs

    async def run(self, args: ConfigureArgs, env: Environment) -> ToolResult:
        return ToolResult(content="configured")


class NoParams(BaseModel):
    """No parameters."""


class ListTables:
    """A tool with no parameters."""

    name = "list_tables"
    description = "List the tables."
    args_model = NoParams

    async def run(self, args: NoParams, env: Environment) -> ToolResult:
        return ToolResult(content="customer, orders")


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
        ('{"sql": "caf\\u00', '{"sql": "caf"}'),
        ('{"sql": "caf\\u', '{"sql": "caf"}'),
        ('{"sql": "caf\\u00e9', '{"sql": "café"}'),
        ('{"path": "C:\\\\u00', '{"path": "C:\\\\u00"}'),
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
        "cut inside a unicode escape",
        "cut right after \\u",
        "unicode escape complete",
        "escaped backslash before u",
        "empty",
        "blank",
    ],
)
def test_repair_fixes_json_it_can_fix(text: str, repaired: str) -> None:
    assert repair_arguments(text) == repaired


@pytest.mark.parametrize(
    "text",
    [
        '{"sql": ',
        '{"sql"',
        '{"sq',
        '{"sql": "a"}}',
        "[1, 2]",
        '"SELECT 1"',
        "SELECT 1",
        '{"sql": "caf\\uZZ',
    ],
    ids=[
        "key with no value",
        "key with no colon",
        "cut inside a key",
        "extra closing brace",
        "an array",
        "a string",
        "not JSON",
        "cut after a \\u that is no escape",
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


@pytest.mark.parametrize(
    ("parameter", "written", "value"),
    [
        ("limit", "5", 5),
        ("limit", "null", None),
        ("columns", '["a", "b"]', ["a", "b"]),
        ("columns", "null", None),
        ("window", '{"start": 1, "end": 2}', {"start": 1, "end": 2}),
        ("frame", '{"start": 1, "end": 2}', {"start": 1, "end": 2}),
        ("frame", "null", None),
    ],
    ids=[
        "int | None, a number",
        "int | None, null",
        "list[str] | None, a list",
        "list[str] | None, null",
        "nested model",
        "nested model | None, an object",
        "nested model | None, null",
    ],
)
def test_optional_and_nested_parameters_are_typed_through_any_of_and_ref(
    parameter: str, written: str, value: object
) -> None:
    # pydantic writes X | None as anyOf [X, null], and a nested model as a $ref into $defs.
    message = assistant(content=xml_call("configure", **{parameter: written}))

    [call] = parse_tool_calls(message, OFFERED)

    assert arguments(call) == {parameter: value}


def test_an_xml_value_that_does_not_parse_as_its_type_stays_a_string() -> None:
    message = assistant(content=xml_call("configure", count="three", tags="a, b", flag="maybe"))

    [call] = parse_tool_calls(message, OFFERED)

    assert arguments(call) == {"count": "three", "tags": "a, b", "flag": "maybe"}


def test_the_parameters_of_an_unknown_tool_stay_strings() -> None:
    message = assistant(content=xml_call("explain_query", sql="SELECT 1", analyze="true"))

    [call] = parse_tool_calls(message, OFFERED)

    assert (call.name, arguments(call)) == ("explain_query", {"sql": "SELECT 1", "analyze": "true"})


# A draft cut off before its closing tag, at the end of the text.
UNTERMINATED_BLOCK = "<tool_call>\n<function=submit>\n<parameter=sql>\nSELECT 1 WHERE"
UNTERMINATED_BARE = "<function=submit>\n<parameter=sql>\nSELECT 1\n</parameter>\n"


@pytest.mark.parametrize(
    "content",
    [
        "Let me submit.\n" + UNTERMINATED_BLOCK,
        "Let me submit.\n" + UNTERMINATED_BARE,
        # Its function block is closed, but the <tool_call> block is not.
        "<tool_call>\n<function=submit>\n<parameter=sql>\nSELECT 1\n</parameter>\n</function>\n",
    ],
    ids=["tool_call block", "bare function block", "closed function, open tool_call"],
)
def test_unterminated_xml_in_the_content_is_no_call(content: str) -> None:
    # Whatever the finish reason: a draft cut off at the end must never become the answer.
    assert parse_tool_calls(assistant(content=content), OFFERED) == []


def test_unterminated_xml_in_the_reasoning_is_no_call() -> None:
    message = assistant(reasoning="I could submit now:\n" + UNTERMINATED_BLOCK)

    assert parse_tool_calls(message, OFFERED) == []


def test_a_terminated_block_after_an_unterminated_draft_is_the_only_call() -> None:
    message = assistant(content=UNTERMINATED_BLOCK + "\n" + xml_call("execute_sql", sql="SELECT 1"))

    [call] = parse_tool_calls(message, OFFERED)

    assert (call.name, arguments(call)) == ("execute_sql", {"sql": "SELECT 1"})


def test_an_xml_call_with_unclosed_parameters_splits_them_at_the_next_tag() -> None:
    message = assistant(
        content="<tool_call><function=execute_sql><parameter=sql>SELECT 1"
        "<parameter=limit>5</function></tool_call>"
    )

    [call] = parse_tool_calls(message, OFFERED)

    assert arguments(call) == {"sql": "SELECT 1", "limit": 5}


BARE_CALL = (
    "I'll run it:\n<function=execute_sql>\n<parameter=sql>\nSELECT 1\n</parameter>\n</function>"
)


def test_a_function_block_in_the_content_without_a_tool_call_tag_is_still_a_call() -> None:
    [call] = parse_tool_calls(assistant(content=BARE_CALL), OFFERED)

    assert (call.name, arguments(call), call.source) == (
        "execute_sql",
        {"sql": "SELECT 1"},
        TOOL_CALL_SOURCE_CONTENT_XML,
    )


def test_a_function_block_in_the_reasoning_without_a_tool_call_tag_is_no_call() -> None:
    # Thinking can mention a call in prose; only a <tool_call> block there is a call.
    assert parse_tool_calls(assistant(reasoning=BARE_CALL), OFFERED) == []


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


def test_a_qwen_xml_call_cut_off_in_the_content_is_no_call() -> None:
    assert parse_tool_calls(qwen_reply("truncated_xml_in_content"), OFFERED) == []


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


@pytest.mark.parametrize("written", ["", " \n"], ids=["empty", "blank"])
async def test_empty_arguments_to_a_tool_with_no_parameters_run_it(written: str) -> None:
    registry = ToolRegistry([bind(ListTables())])
    message = assistant(calls=[ToolCall(id="call_a", name="list_tables", arguments_json=written)])

    [call] = parse_tool_calls(message, registry.json_schemas())
    result = await registry.execute(call, FakeSqlEnvironment())

    assert call.repaired_arguments_json == "{}"
    assert (result.content, result.is_error) == ("customer, orders", False)


async def test_an_xml_call_with_no_parameters_has_empty_arguments() -> None:
    registry = ToolRegistry([bind(ListTables())])
    message = assistant(content="<tool_call>\n<function=list_tables>\n</function>\n</tool_call>")

    [call] = parse_tool_calls(message, registry.json_schemas())
    result = await registry.execute(call, FakeSqlEnvironment())

    assert call.arguments_json == "{}"
    assert (result.content, result.is_error) == ("customer, orders", False)


def test_strip_removes_a_call_block_and_leaves_the_rest_as_it_was() -> None:
    text = "Let me look.\n\n" + xml_call("execute_sql", sql="SELECT 1")

    assert strip_tool_call_markup(text) == "Let me look.\n\n"


def test_strip_removes_each_of_two_call_blocks() -> None:
    first, second = xml_call("execute_sql", sql="SELECT 1"), xml_call("submit", sql="SELECT 1")
    text = f"First:\n{first}\nthen:\n{second}\nDone."

    assert strip_tool_call_markup(text) == "First:\n\nthen:\n\nDone."


def test_strip_removes_a_call_written_into_the_reasoning() -> None:
    reasoning = QWEN_REPLIES["replies"]["call_in_reasoning"]["reasoning_content"]

    stripped = strip_tool_call_markup(reasoning, reasoning=True)

    assert stripped == reasoning[: reasoning.index("<tool_call>")]
    assert stripped.endswith("Let's do that.\n")


def test_strip_leaves_text_with_no_call_as_it_was() -> None:
    text = "I will run `EXPLAIN SELECT 1`; a <function> tag with no name is not a call.\n"

    assert strip_tool_call_markup(text) == text


def test_strip_removes_an_unterminated_block_to_the_end_though_it_is_no_call() -> None:
    content = QWEN_REPLIES["replies"]["truncated_xml_in_content"]["content"]

    assert strip_tool_call_markup(content) == (
        "Let me first look at the query plan for the original query to understand how it's "
        "being executed.\n\n"
    )


@pytest.mark.parametrize(
    ("text", "reasoning"),
    [
        ("Let me submit.\n" + UNTERMINATED_BLOCK, False),
        ("Let me submit.\n" + UNTERMINATED_BARE, False),
        ("Let me submit.\n" + UNTERMINATED_BLOCK, True),
    ],
    ids=[
        "tool_call block in the content",
        "bare block in the content",
        "tool_call block in the reasoning",
    ],
)
def test_strip_removes_unterminated_markup_too(text: str, reasoning: bool) -> None:
    assert strip_tool_call_markup(text, reasoning=reasoning) == "Let me submit.\n"


def test_strip_removes_a_block_that_holds_no_call_as_the_parser_skips_it() -> None:
    text = "Before <tool_call>not a call</tool_call> after"

    assert parse_tool_calls(assistant(content=text), OFFERED) == []
    assert strip_tool_call_markup(text) == "Before  after"


def test_strip_removes_bare_function_blocks_only_where_the_parser_reads_them() -> None:
    bare = (
        "I'll run it:\n<function=execute_sql>\n<parameter=sql>\nSELECT 1\n</parameter>\n</function>"
    )
    # With a <tool_call> block in the text, the parser reads only the blocks, so a bare function
    # block outside them is not a call, and stays.
    outside = "\n<function=execute_sql>\n</function>"

    mixed = xml_call("submit", sql="SELECT 1") + outside

    assert strip_tool_call_markup(bare + "\nThen check.") == "I'll run it:\n\nThen check."
    assert strip_tool_call_markup(mixed) == outside
    [call] = parse_tool_calls(assistant(content=mixed), OFFERED)
    assert (call.name, arguments(call)) == ("submit", {"sql": "SELECT 1"})


def test_strip_keeps_a_bare_function_block_in_the_reasoning_which_is_no_call() -> None:
    text = BARE_CALL + "\nThen check."

    assert strip_tool_call_markup(text, reasoning=True) == text
    assert (
        strip_tool_call_markup(
            "Plan.\n" + xml_call("submit", sql="SELECT 1") + "\n" + text, reasoning=True
        )
        == "Plan.\n\n" + text
    )


@pytest.mark.parametrize(
    "name",
    ["xml_in_content", "truncated_xml_in_content", "unknown_tool_in_content", "call_in_reasoning"],
)
def test_after_strip_no_call_is_left_in_a_qwen_reply(name: str) -> None:
    reply = QWEN_REPLIES["replies"][name]
    content = strip_tool_call_markup(reply["content"])
    reasoning = strip_tool_call_markup(reply["reasoning_content"], reasoning=True)

    assert parse_tool_calls(assistant(content=content, reasoning=reasoning), OFFERED) == []


def deepest(fits: Callable[[int], bool]) -> int:
    """The largest depth, below 100,000, at which nesting still ``fits``."""
    low, high = 1, 100_000
    while low < high:
        middle = (low + high + 1) // 2
        low, high = (middle, high) if fits(middle) else (low, middle - 1)
    return low


def nested(depth: int) -> str:
    return "[" * depth + "]" * depth


def loads_fits(depth: int) -> bool:
    try:
        json.loads(nested(depth))
    except RecursionError:
        return False
    return True


def dumps_one_deeper_fits(depth: int) -> bool:
    try:
        json.dumps([json.loads(nested(depth))])
    except RecursionError:
        return False
    return True


def json_limits() -> tuple[int, int]:
    """The deepest lists json.loads parses, and the deepest it parses that json.dumps can still
    write one level deeper, as inside a call's arguments.

    They depend on how much C stack the caller has used, so a test measures them where it calls
    the parser: about 9,997 and 9,996 on CPython 3.12.
    """
    return deepest(loads_fits), deepest(
        lambda depth: loads_fits(depth) and dumps_one_deeper_fits(depth)
    )


@pytest.mark.parametrize(
    "where",
    ["writable", "parses but is unwritable one level deeper", "deepest parsed", "too deep"],
)
def test_an_xml_value_at_the_json_depth_limits_is_typed_or_kept_and_never_raises(
    where: str,
) -> None:
    loads_limit, write_limit = json_limits()
    assert write_limit < loads_limit  # Else this test would not reach the boundary.
    depth = {
        "writable": write_limit,
        "parses but is unwritable one level deeper": write_limit + 1,
        "deepest parsed": loads_limit,
        "too deep": loads_limit + 1,
    }[where]
    message = assistant(content=xml_call("configure", tags=nested(depth)))

    [call] = parse_tool_calls(message, OFFERED)

    if depth <= write_limit:
        assert call.arguments_json == '{"tags": ' + nested(depth) + "}"
    else:
        assert call.arguments_json == '{"tags": "' + nested(depth) + '"}'


@pytest.mark.parametrize("where", ["one below the limit", "at the limit", "past the limit"])
def test_repair_at_the_json_depth_limit_repairs_or_gives_up_and_never_raises(where: str) -> None:
    loads_limit, _ = json_limits()
    depth = loads_limit + {"one below the limit": -1, "at the limit": 0, "past the limit": 1}[where]
    text = '{"tags": ' + nested(depth - 1) + ",}"  # The object is one level of the depth.

    repaired = repair_arguments(text)

    if depth <= loads_limit:
        assert repaired == '{"tags": ' + nested(depth - 1) + "}"
    else:
        assert repaired is None


def test_a_repaired_lone_surrogate_is_escaped_so_the_call_can_hold_it() -> None:
    written = '{"sql": "SELECT \'\\ud800\'",}'  # A \ud800 escape: a lone surrogate.
    message = assistant(calls=[ToolCall(id="call_a", name="execute_sql", arguments_json=written)])

    [call] = parse_tool_calls(message, OFFERED)

    assert call.repaired_arguments_json == '{"sql": "SELECT \'\\ud800\'"}'
    assert json.loads(call.repaired_arguments_json) == {"sql": "SELECT '\ud800'"}


def test_an_xml_value_with_a_lone_surrogate_is_escaped_so_the_call_can_hold_it() -> None:
    message = assistant(content=xml_call("configure", tags='["caf\\u00e9", "\\ud800"]'))

    [call] = parse_tool_calls(message, OFFERED)

    assert call.arguments_json == '{"tags": ["caf\\u00e9", "\\ud800"]}'
    assert arguments(call) == {"tags": ["café", "\ud800"]}


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_a_non_finite_xml_number_stays_a_string(number: str) -> None:
    message = assistant(content=xml_call("configure", ratio=number, tags=f"[{number}]"))

    [call] = parse_tool_calls(message, OFFERED)

    assert call.arguments_json == json.dumps({"ratio": number, "tags": f"[{number}]"})


@pytest.mark.parametrize(
    "text",
    ['{"limit": NaN,}', '{"limit": Infinity', '{"limit": -Infinity,}', '{"limit": 1e999,}'],
)
def test_repair_gives_up_on_non_finite_numbers(text: str) -> None:
    assert repair_arguments(text) is None


def test_a_standard_call_with_a_non_finite_number_gets_no_repaired_arguments() -> None:
    message = assistant(
        calls=[ToolCall(id="call_a", name="execute_sql", arguments_json='{"limit": NaN}')]
    )

    [call] = parse_tool_calls(message, OFFERED)

    assert call.repaired_arguments_json == ""


def test_a_calls_id_is_the_sha256_of_its_index_name_and_arguments() -> None:
    message = assistant(calls=[ToolCall(name="execute_sql", arguments_json='{"sql": "SELECT 1"}')])

    [call] = parse_tool_calls(message, OFFERED)

    # The first 24 hex digits of sha256('0\nexecute_sql\n{"sql": "SELECT 1"}').
    assert call.id == "call_edd3f0731f55e9d52bf87b18"


@pytest.mark.parametrize(
    ("name", "arguments_json"),
    [("submit", '{"sql": "SELECT 1"}'), ("execute_sql", '{"sql": "SELECT 2"}')],
    ids=["only the name differs", "only the arguments differ"],
)
def test_calls_that_differ_in_name_or_arguments_get_different_ids(
    name: str, arguments_json: str
) -> None:
    first = assistant(calls=[ToolCall(name="execute_sql", arguments_json='{"sql": "SELECT 1"}')])
    second = assistant(calls=[ToolCall(name=name, arguments_json=arguments_json)])

    [one], [other] = parse_tool_calls(first, OFFERED), parse_tool_calls(second, OFFERED)

    assert one.id != other.id


async def test_a_gateway_id_repeated_in_one_reply_is_replaced_so_each_result_answers_one_call() -> (
    None
):
    calls = [
        ToolCall(id="call_0", name="execute_sql", arguments_json='{"sql": "SELECT 1"}'),
        ToolCall(id="call_0", name="execute_sql", arguments_json='{"sql": "SELECT 1"}'),
    ]
    env = FakeSqlEnvironment({"SELECT 1": SqlResult(row_count=1)})

    first, second = parse_tool_calls(assistant(calls=calls), OFFERED)
    results = [await REGISTRY.execute(call, env) for call in (first, second)]

    # The first keeps the gateway's id; the second gets the formula's id for index 1.
    assert (first.id, second.id) == ("call_0", "call_7a300c3273e4108aa15a6c53")
    assert [result.tool_call_id for result in results] == [
        "call_0",
        "call_7a300c3273e4108aa15a6c53",
    ]


def test_xml_parameters_are_typed_through_one_of_and_all_of_too() -> None:
    parameters: dict[str, JsonValue] = {
        "properties": {
            "choice": {"oneOf": [{"type": "integer"}, {"type": "boolean"}]},
            "frame": {"allOf": [{"$ref": "#/$defs/Window"}], "description": "A range of rows."},
        },
        "$defs": {"Window": {"type": "object"}},
    }
    tools: list[dict[str, JsonValue]] = [
        {"type": "function", "function": {"name": "pick", "parameters": parameters}}
    ]
    message = assistant(content=xml_call("pick", choice="7", frame='{"start": 1, "end": 2}'))

    [call] = parse_tool_calls(message, tools)

    assert arguments(call) == {"choice": 7, "frame": {"start": 1, "end": 2}}
