"""Tool calls: reading the model's calls out of its reply, and repairing their JSON (spec §5.1).

A call can be in three places. :func:`parse_tool_calls` takes the first that has any:

1. the reply's standard ``tool_calls``, which the gateway parses out of Qwen's output;
2. Qwen3-Coder XML in the content, which the gateway leaves there when it does not parse it, as
   when the request offered no tools;
3. Qwen3-Coder XML in the reasoning_content, in ``<tool_call>`` blocks only: Qwen sometimes
   writes a call before it ends its thinking, where the gateway does not look for calls, but
   thinking can also mention a call in prose, which must never run.

Only a terminated XML call counts: markup cut off before its closing tag is never a call, so a
draft at the end of a reply cannot become its answer. :func:`strip_tool_call_markup` removes the
XML markup from a text, unterminated markup included, so that a reply whose calls were parsed
from it can be replayed with them as structured calls only.
"""

import hashlib
import json
import re
import string
from collections.abc import Iterable, Mapping
from typing import NoReturn

from goaly.contracts.gen.trajectory_pb2 import (
    TOOL_CALL_SOURCE_CONTENT_XML,
    TOOL_CALL_SOURCE_REASONING_XML,
    TOOL_CALL_SOURCE_STANDARD,
    ChatMessage,
    ToolCall,
    ToolCallSource,
)
from goaly.contracts.interfaces import JsonValue

__all__ = ["parse_tool_calls", "repair_arguments", "strip_tool_call_markup"]

# Qwen3-Coder's XML form of a call: <tool_call><function=NAME><parameter=KEY>VALUE</parameter>
# ...</function></tool_call>. A block ends at its closing tag, at the next block, or at the end
# of the text; the last group is the closing tag, and empty when the block has none. A name is at
# most 256 characters, so that even a reply full of unclosed tags is read in linear time: the
# regex engine is the event loop's, and a quadratic scan would stall every task.
_TOOL_CALL = re.compile(r"<tool_call>(.*?)(</tool_call>|(?=<tool_call>)|\Z)", re.DOTALL)
_FUNCTION = re.compile(r"<function=([^>\n]{0,256})>(.*?)(</function>|(?=<function=)|\Z)", re.DOTALL)
_PARAMETER = re.compile(
    r"<parameter=([^>\n]{0,256})>(.*?)(?:</parameter>|(?=<parameter=)|\Z)", re.DOTALL
)

# The opening of a Markdown code fence, with an optional language tag.
_FENCE_OPENING = re.compile(r"```[\w+-]*[ \t]*\n?")


def parse_tool_calls(
    message: ChatMessage, tools: Iterable[Mapping[str, JsonValue]]
) -> list[ToolCall]:
    """Return the calls in the assistant ``message``, in order, ready for ToolRegistry.execute.

    An XML call counts only when it is terminated: a ``<tool_call>`` block with its
    ``</tool_call>``, or, in a content with no ``<tool_call>`` tag, a bare function block with
    its ``</function>``. Inside a terminated block, a parameter or function tag may go unclosed.

    ``tools`` is the ``tools`` list the model was offered, as ToolRegistry.json_schemas()
    gives it: a parameter written in XML takes the type its tool's JSON Schema gives it (a
    string, a number, a boolean, null, or JSON), and stays a string when the schema says
    nothing of it or the value does not parse as that type.

    Each call records the source it was read from. A call with no ID, or with one an earlier
    call in the reply already has, gets one that depends only on its place in the reply, its
    name and its arguments: ``"call_"`` and the first 24 hex digits of the sha256 of
    ``f"{index}\\n{name}\\n{arguments_json}"``. Parsing the same reply again gives the same IDs,
    and each result answers exactly one call.

    A standard call whose arguments are not a JSON object gets the repaired arguments, if
    :func:`repair_arguments` can repair them; otherwise executing it gives an error result. An
    XML call's arguments are written from its parameters, so they need none.
    """
    if message.tool_calls:
        calls = [_standard(call) for call in message.tool_calls]
    else:
        schemas = _parameter_schemas(tools)
        calls = _xml_calls(message.content, TOOL_CALL_SOURCE_CONTENT_XML, schemas)
        if not calls:
            calls = _xml_calls(message.reasoning_content, TOOL_CALL_SOURCE_REASONING_XML, schemas)
    taken: set[str] = set()
    for index, call in enumerate(calls):
        if not call.id or call.id in taken:
            call.id = _call_id(index, call)
        taken.add(call.id)
    return calls


def strip_tool_call_markup(text: str, *, reasoning: bool = False) -> str:
    """Return ``text`` without the Qwen3-Coder XML call markup that :func:`parse_tool_calls` reads.

    The markup is each ``<tool_call>`` block, to its closing tag, or to the end of the text when
    it was cut off, whether or not the block holds a call. In a content (``reasoning`` false)
    with no ``<tool_call>`` tag, it is each bare ``<function=...>`` block, as the parser reads
    those there. A bare block is kept otherwise: outside ``<tool_call>`` blocks, and anywhere in
    a reasoning (``reasoning`` true), the parser never reads one as a call. Unterminated markup,
    which the parser never takes as a call, is removed all the same, so that none reaches the
    model. Everything else stays as it was, whitespace included.

    When a reply's calls came from XML, its replay strips both its content and its reasoning,
    ``strip_tool_call_markup(content)`` and ``strip_tool_call_markup(reasoning_content,
    reasoning=True)``, and carries the calls as structured calls only. Otherwise the model sees
    each call twice.
    """
    markup = _TOOL_CALL if reasoning or _TOOL_CALL.search(text) else _FUNCTION
    return markup.sub("", text)


def repair_arguments(text: str) -> str | None:
    """Repair ``text`` into a JSON object, or return None if it cannot become one.

    The repairs: strip a Markdown code fence around the text; drop each comma before a closing
    bracket or brace; close each string, bracket and brace still open at the end, as when the
    reply was cut off, first dropping an escape the cut fell inside, such as a partial
    ``\\u00``. Strings may hold control characters, such as raw newlines, and blank text is an
    empty object. The object comes back as JSON text, as json.dumps writes it.
    """
    stripped = _strip_fence(text.strip())
    if not stripped:
        return "{}"
    try:
        value = _loads(_close(stripped), strict=False)
        return _to_json(value) if isinstance(value, dict) else None
    except (ValueError, RecursionError):  # RecursionError: nested too deep to parse or write.
        return None


def _standard(call: ToolCall) -> ToolCall:
    parsed = ToolCall(
        id=call.id,
        name=call.name,
        arguments_json=call.arguments_json,
        source=TOOL_CALL_SOURCE_STANDARD,
    )
    if not _is_json_object(call.arguments_json):
        parsed.repaired_arguments_json = repair_arguments(call.arguments_json) or ""
    return parsed


def _xml_calls(
    text: str,
    source: ToolCallSource.ValueType,
    schemas: Mapping[str, Mapping[str, JsonValue]],
) -> list[ToolCall]:
    calls = []
    for written_name, body in _function_blocks(text, source):
        name = written_name.strip()
        parameters = schemas.get(name, {})
        values: dict[str, JsonValue] = {}
        for written_key, value in _PARAMETER.findall(body):
            key = written_key.strip()
            values[key] = _coerce(_trim(value), _types(parameters, key))
        arguments_json = _to_json(values)  # Each value was checked as it was coerced.
        calls.append(ToolCall(name=name, arguments_json=arguments_json, source=source))
    return calls


def _function_blocks(text: str, source: ToolCallSource.ValueType) -> list[tuple[str, str]]:
    """The name and the body of each function block in ``text`` that is a call.

    Only a terminated block is a call: a ``<tool_call>`` block with its ``</tool_call>``, and in
    the content of a text with no ``<tool_call>`` tag, a bare function block with its
    ``</function>``. A draft cut off before its closing tag is never a call.
    """
    if _TOOL_CALL.search(text):
        blocks = [match[1] for match in _TOOL_CALL.finditer(text) if match[2]]
        return [(name, body) for block in blocks for name, body, _ in _FUNCTION.findall(block)]
    if source == TOOL_CALL_SOURCE_CONTENT_XML:
        return [(match[1], match[2]) for match in _FUNCTION.finditer(text) if match[3]]
    return []


def _parameter_schemas(
    tools: Iterable[Mapping[str, JsonValue]],
) -> dict[str, Mapping[str, JsonValue]]:
    """Map each offered tool's name to the JSON Schema of its parameters."""
    schemas: dict[str, Mapping[str, JsonValue]] = {}
    for entry in tools:
        function = entry.get("function")
        if isinstance(function, dict):
            name, parameters = function.get("name"), function.get("parameters")
            if isinstance(name, str) and isinstance(parameters, dict):
                schemas[name] = parameters
    return schemas


def _types(parameters: Mapping[str, JsonValue], key: str) -> frozenset[str]:
    """The JSON types the schema ``parameters`` allows parameter ``key``; none if it is unknown."""
    properties = parameters.get("properties")
    schema = properties.get(key) if isinstance(properties, dict) else None
    if not isinstance(schema, dict):
        return frozenset()
    return frozenset(_schema_types(schema, parameters, set()))


def _schema_types(
    schema: Mapping[str, JsonValue], root: Mapping[str, JsonValue], seen: set[str]
) -> set[str]:
    """The types of ``schema``, through anyOf, oneOf, allOf and $ref, as pydantic writes them."""
    kind = schema.get("type")
    types = {item for item in (kind if isinstance(kind, list) else [kind]) if isinstance(item, str)}
    for keyword in ("anyOf", "oneOf", "allOf"):
        options = schema.get(keyword)
        for option in options if isinstance(options, list) else []:
            if isinstance(option, dict):
                types |= _schema_types(option, root, seen)
    ref, definitions = schema.get("$ref"), root.get("$defs")
    if isinstance(ref, str) and ref not in seen and isinstance(definitions, dict):
        seen.add(ref)  # A definition that refers to itself is read once.
        target = definitions.get(ref.removeprefix("#/$defs/"))
        if isinstance(target, dict):
            types |= _schema_types(target, root, seen)
    return types


def _coerce(value: str, types: frozenset[str]) -> JsonValue:
    """Turn the XML text ``value`` into the JSON value that ``types`` calls for."""
    text = value.strip()
    if "null" in types and text == "null":
        return None
    if not types or "string" in types:
        return value
    if "boolean" in types and text.lower() in ("true", "false"):
        return text.lower() == "true"
    try:
        parsed = _loads(text)  # A number, an object or an array.
        _to_json([parsed])  # Writable inside the arguments, a level deeper, or kept as written.
    except (ValueError, RecursionError):
        return value
    return parsed


def _trim(value: str) -> str:
    """Drop the newline Qwen writes after an opening tag and the one before a closing tag."""
    return value.removeprefix("\n").removesuffix("\n")


def _call_id(index: int, call: ToolCall) -> str:
    """An ID, in the form the gateway gives, for the call at ``index`` in its reply."""
    digest = hashlib.sha256(f"{index}\n{call.name}\n{call.arguments_json}".encode()).hexdigest()
    return f"call_{digest[:24]}"


def _is_json_object(text: str) -> bool:
    try:
        return isinstance(_loads(text), dict)
    except (ValueError, RecursionError):
        return False


def _loads(text: str, *, strict: bool = True) -> JsonValue:
    """Parse ``text`` as JSON, refusing NaN and the infinities, which JSON has no token for.

    Raise ValueError if it is not JSON, and RecursionError if it nests too deep to parse.
    """
    parsed: JsonValue = json.loads(text, strict=strict, parse_constant=_refuse)
    return parsed


def _refuse(constant: str) -> NoReturn:
    raise ValueError(f"{constant} is not a JSON number")


def _to_json(value: JsonValue) -> str:
    """Write ``value`` as JSON text that a protobuf string can hold.

    Text UTF-8 cannot encode, such as the lone surrogate that a \\ud800 escape decodes to, is
    written in ASCII, with \\u escapes. Raise ValueError for a number JSON has no token for, such
    as the infinity that 1e999 parses to, and RecursionError for a value nested too deep.
    """
    text = json.dumps(value, ensure_ascii=False, allow_nan=False)
    try:
        text.encode()
    except UnicodeEncodeError:
        return json.dumps(value, allow_nan=False)
    return text


def _strip_fence(text: str) -> str:
    """Strip a Markdown code fence around ``text``, whose closing fence a cut-off reply lacks."""
    opening = _FENCE_OPENING.match(text)
    if opening is None:
        return text
    return text[opening.end() :].rstrip().removesuffix("```").strip()


def _close(text: str) -> str:
    """Drop the commas before closing brackets and braces, and close what is open at the end."""
    out: list[str] = []
    closers: list[str] = []
    in_string = escaped = False
    unicode_start = -1  # Where an unfinished \uXXXX escape starts in out, or -1.
    for char in text:
        if in_string:
            if escaped:
                escaped = False
                unicode_start = len(out) - 1 if char == "u" else -1
            else:
                if char not in string.hexdigits:
                    unicode_start = -1  # Not a \uXXXX escape: json.loads judges it.
                if char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
        elif char == '"':
            in_string = True
        elif char in "[{":
            closers.append("]" if char == "[" else "}")
        elif char in "]}":
            _drop_trailing_comma(out)
            if closers and closers[-1] == char:
                closers.pop()
        out.append(char)
        if unicode_start >= 0 and len(out) - unicode_start == 6:
            unicode_start = -1  # A complete \uXXXX escape.
    if in_string:
        if escaped:
            out.pop()  # The text was cut inside an escape: drop its backslash.
        elif unicode_start >= 0:
            del out[unicode_start:]  # It was cut inside a \uXXXX escape: drop the escape.
        out.append('"')
    _drop_trailing_comma(out)
    out.extend(reversed(closers))
    return "".join(out)


def _drop_trailing_comma(out: list[str]) -> None:
    end = len(out)
    while end and out[end - 1].isspace():
        end -= 1
    if end and out[end - 1] == ",":
        del out[end - 1]
