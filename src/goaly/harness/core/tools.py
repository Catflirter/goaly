"""Tools: binding a pack's tools, the registry that holds them, and the executor (spec §5.1).

A capability pack implements its tools against goaly.contracts.interfaces.Tool. :func:`bind`
turns each into a :class:`BoundTool`, which erases the tool's argument type and runs each call
by the executor's rules (:meth:`BoundTool.invoke`). A :class:`ToolRegistry` holds a policy's
bound tools: it lists them for the model and answers each of the model's calls.
"""

import asyncio
import copy
import json
import logging
import math
import time
from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable, Iterator
from dataclasses import dataclass
from typing import final

from pydantic import BaseModel, ValidationError

from goaly.contracts.errors import EnvironmentFailure
from goaly.contracts.gen.task_pb2 import Prediction
from goaly.contracts.gen.trajectory_pb2 import ToolCall, ToolResult
from goaly.contracts.interfaces import Environment, JsonValue, Tool

__all__ = ["DEFAULT_TIMEOUT_S", "BoundTool", "ToolRegistry", "bind"]

# How long a call may run when the tool's config sets no timeout.
DEFAULT_TIMEOUT_S = 60.0

# A result's content keeps this many characters at its start, and as many at its end.
_CONTENT_KEEP = 4_000
# The model sees this many characters at each end of an exception's message.
_MESSAGE_KEEP = 250
# The model sees this many characters at each end of arguments that are not valid JSON.
_EXCERPT_KEEP = 100

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class BoundTool[E: Environment](ABC):
    """A tool ready to run in an environment of type ``E``, with its argument type erased.

    :func:`bind` makes one. Erasing the argument type lets a registry hold tools whose
    arguments differ, as a ``Sequence[Tool[Any, E]]`` would, but with no ``Any``: a bound tool
    validates the model's JSON and runs the tool where the argument type is still known.
    """

    name: str
    description: str
    # How long a call may run, in seconds.
    timeout_s: float

    @abstractmethod
    def json_schema(self) -> dict[str, JsonValue]:
        """Return the tool's entry in the OpenAI ``tools`` list, a fresh copy each time."""

    @abstractmethod
    async def invoke(self, arguments_json: str, env: E) -> ToolResult:
        """Validate ``arguments_json`` against the tool's arguments, then run the tool in ``env``.

        These are the executor's rules (spec §5.1):

        - The tool's ``content`` and ``submission`` pass through; the executor fills every
          other field except ``tool_call_id``, which ToolRegistry.execute fills. ``elapsed_s``
          covers the whole invocation, validation included.
        - Content longer than 8,000 characters keeps its first and last 4,000, with a marker
          between them that says how many were cut, and ``truncated`` is set.
        - ``is_error`` is set when the call itself failed, and only then. Arguments that fail
          validation get pydantic's errors listed, and the tool does not run. A call that runs
          past ``timeout_s`` is stopped. A tool that raised gets its exception's type and
          message, never a traceback; the traceback goes to the log. A statement or command
          that ran and failed is a normal result.
        - EnvironmentFailure propagates, never as a result, and so does an exception group that
          holds one, as a TaskGroup raises. Cancellation propagates too.
        """


def bind[A: BaseModel, E: Environment](
    tool: Tool[A, E], *, timeout_s: float = DEFAULT_TIMEOUT_S
) -> BoundTool[E]:
    """Bind ``tool`` for a registry; each of its calls may run for ``timeout_s`` seconds.

    ``timeout_s`` is the timeout the pack's config gives the tool, and DEFAULT_TIMEOUT_S if it
    gives none; one that is not a positive, finite number raises ValueError. The tool's name,
    description and JSON Schema are read once, here: a tool whose arguments have no JSON Schema
    fails now, with pydantic's PydanticInvalidForJsonSchema, rather than at the first model
    call.
    """
    if not 0 < timeout_s < math.inf:  # Also rejects NaN, which compares false.
        raise ValueError(f"timeout_s must be a positive, finite number of seconds: {timeout_s}")
    return _Binding(
        name=tool.name,
        description=tool.description,
        timeout_s=timeout_s,
        tool=tool,
        parameters=tool.args_model.model_json_schema(),
    )


@final
@dataclass(frozen=True, eq=False)
class _Binding[A: BaseModel, E: Environment](BoundTool[E]):
    """The bound tool :func:`bind` makes, which knows the tool's argument type ``A``."""

    tool: Tool[A, E]
    # The JSON Schema of the tool's arguments.
    parameters: dict[str, JsonValue]

    def json_schema(self) -> dict[str, JsonValue]:
        function: dict[str, JsonValue] = {
            "name": self.name,
            "description": self.description,
            "parameters": copy.deepcopy(self.parameters),
        }
        return {"type": "function", "function": function}

    async def invoke(self, arguments_json: str, env: E) -> ToolResult:
        started = time.perf_counter()
        result = await self._call(arguments_json, env)
        result.elapsed_s = time.perf_counter() - started
        return result

    async def _call(self, arguments_json: str, env: E) -> ToolResult:
        try:
            args = self.tool.args_model.model_validate_json(arguments_json)
        except ValidationError as error:
            content = _invalid_arguments(self.name, arguments_json, error)
            return _result(self.name, content, is_error=True)
        timeout = asyncio.timeout(self.timeout_s)
        try:
            async with timeout:
                returned = await self.tool.run(args, env)
        except Exception as error:
            if _environment_failed(error):
                raise
            if timeout.expired():
                content = (
                    f"Error: {self.name} timed out after {self.timeout_s:g} s and was stopped."
                )
            else:
                _logger.warning(
                    "a tool raised an exception", extra={"tool": self.name}, exc_info=error
                )
                content = f"Error: {self.name} failed with {_describe(error)}"
            return _result(self.name, content, is_error=True)
        submission = returned.submission if returned.HasField("submission") else None
        return _result(self.name, returned.content, submission=submission)


class ToolRegistry[E: Environment]:
    """The tools a policy offers the model, each with a name no other tool has.

    It lists them for the LLM client in the order they were registered. A tool that works in
    any environment fits a registry for a narrower one: ``bind(EchoTool())`` fits a
    ``ToolRegistry[SqlEnvironment]``.
    """

    def __init__(self, tools: Iterable[BoundTool[E]] = ()) -> None:
        """Register each of ``tools`` in order, as :meth:`register` does."""
        self._tools: dict[str, BoundTool[E]] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: BoundTool[E]) -> None:
        """Add ``tool``. Raise ValueError if a registered tool already has its name."""
        if tool.name in self._tools:
            raise ValueError(f"a tool named {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def __iter__(self) -> Iterator[BoundTool[E]]:
        """Iterate over the tools in the order they were registered."""
        return iter(self._tools.values())

    def json_schemas(self) -> list[dict[str, JsonValue]]:
        """Return the ``tools`` list for the LLM client: each tool's entry, in order."""
        return [tool.json_schema() for tool in self._tools.values()]

    async def execute(self, call: ToolCall, env: E) -> ToolResult:
        """Run ``call`` in ``env``, as :meth:`BoundTool.invoke` does, and answer it.

        The result's ``tool_call_id`` is the call's ID. The call runs with its repaired
        arguments if it has them. A call to a tool the registry does not hold gives an error
        result that lists the tools it does hold; no tool ran, so its ``elapsed_s`` is 0.
        """
        tool = self._tools.get(call.name)
        if tool is None:
            result = _result(call.name, _unknown_tool(call.name, self._tools), is_error=True)
        else:
            result = await tool.invoke(call.repaired_arguments_json or call.arguments_json, env)
        result.tool_call_id = call.id
        return result


def _result(
    name: str, content: str, *, is_error: bool = False, submission: Prediction | None = None
) -> ToolResult:
    """Build a result of the tool ``name``, its content cut down if it is long (spec §5.1)."""
    content, truncated = _truncate(content, _CONTENT_KEEP)
    return ToolResult(
        name=name, content=content, is_error=is_error, truncated=truncated, submission=submission
    )


def _invalid_arguments(name: str, arguments_json: str, error: ValidationError) -> str:
    details = error.errors(include_url=False, include_context=False, include_input=False)
    if details[0]["type"] == "json_invalid":  # Invalid JSON is the only error pydantic reports.
        reason = details[0]["msg"].removeprefix("Invalid JSON: ")
        excerpt, _ = _truncate(arguments_json, _EXCERPT_KEEP)
        return f"Error: the arguments for {name} are not valid JSON ({reason}):\n{excerpt}"
    lines = [f"Error: invalid arguments for {name}:"]
    for detail in details:
        where = ".".join(str(part) for part in detail["loc"])
        lines.append(f"- {where}: {detail['msg']}" if where else f"- {detail['msg']}")
    return "\n".join(lines)


def _environment_failed(error: Exception) -> bool:
    """Whether ``error`` is an EnvironmentFailure, or a group holding one, as a TaskGroup raises."""
    if isinstance(error, ExceptionGroup):
        return error.subgroup(EnvironmentFailure) is not None
    return isinstance(error, EnvironmentFailure)


def _describe(error: Exception) -> str:
    """Name ``error``'s type and give its message, cut short: never a traceback."""
    message, _ = _truncate(str(error), _MESSAGE_KEEP)
    return f"{type(error).__name__}: {message}" if message else type(error).__name__


def _truncate(text: str, keep: int) -> tuple[str, bool]:
    """Keep the first and the last ``keep`` characters of ``text`` and say how many were cut.

    Return the text, and whether it was cut: text no longer than ``2 * keep`` stays whole.
    """
    if len(text) <= 2 * keep:
        return text, False
    marker = f"[... {len(text) - 2 * keep} of {len(text)} characters cut ...]"
    return f"{text[:keep]}\n{marker}\n{text[-keep:]}", True


def _unknown_tool(name: str, names: Collection[str]) -> str:
    listed = f"The tools are: {', '.join(names)}." if names else "There are no tools."
    return f"Error: there is no tool named {json.dumps(name, ensure_ascii=False)}. {listed}"
