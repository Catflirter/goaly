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

__all__ = ["DEFAULT_TIMEOUT_S", "BoundTool", "ToolRegistry", "bind", "environment_failure_in"]

# How long a call may run when the tool's config sets no timeout.
DEFAULT_TIMEOUT_S = 60.0

# A result's content keeps this many characters at its start, and as many at its end.
_CONTENT_KEEP = 4_000
# The model sees this many characters at each end of an exception's message.
_MESSAGE_KEEP = 250
# An invalid-arguments error lists this many of pydantic's errors, and counts the rest.
_LISTED_ERRORS = 10

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
        """Return the tool's entry in the OpenAI ``tools`` list, a fresh copy each time.

        The entry holds the tool's name, its description and its arguments' JSON Schema as
        pydantic writes it, and the model reads all of it as written: the args model's
        docstring, its field descriptions and its titles. Nothing is stripped, so whoever writes
        a tool, RSI's implementer included, writes that text for the model.
        """

    @abstractmethod
    async def invoke(self, arguments_json: str, env: E) -> ToolResult:
        """Validate ``arguments_json`` against the tool's arguments, then run the tool in ``env``.

        These are the executor's rules (spec §5.1):

        - The tool's ``content`` and ``submission`` pass through; the executor fills every
          other field except ``tool_call_id``, which ToolRegistry.execute fills. ``elapsed_s``
          covers the whole invocation, validation included.
        - Content longer than 8,000 characters keeps its first and last 4,000, with a marker
          between them that says how many were cut, and ``truncated`` is set.
        - ``is_error`` is set when the call itself failed, and only then:
          - arguments that fail validation get pydantic's first 10 errors listed, or the type
            and message of what a validator raised, and the tool does not run;
          - a call that runs past ``timeout_s`` is cancelled;
          - a tool that raised, or returned something other than a ToolResult, gets the
            exception's type and message, and an exception group each of its leaves', never a
            traceback; the traceback goes to the log, at WARNING.

          A statement or command that ran and failed is a normal result.
        - EnvironmentFailure propagates, from a validator too, and never becomes a result. When
          an exception group holds one, as a TaskGroup raises, the first is raised by itself
          (:func:`environment_failure_in`), so that ``except EnvironmentFailure`` catches it:
          from the group, unless it has a cause of its own, which it keeps. Cancellation
          propagates too, and wins: when the caller is being cancelled, an EnvironmentFailure
          the tool raised meanwhile becomes the CancelledError's cause.
        - The tool runs in a task of its own. It must let CancelledError propagate; one that
          does not still cannot change the outcome. A cancellation of the caller that it
          swallowed, or turned into another exception, still propagates, and a call that ran
          past its timeout gets the timeout error, whatever the tool returned.
        """


def bind[A: BaseModel, E: Environment](
    tool: Tool[A, E], *, timeout_s: float = DEFAULT_TIMEOUT_S
) -> BoundTool[E]:
    """Bind ``tool`` for a registry; each of its calls may run for ``timeout_s`` seconds.

    ``timeout_s`` bounds the whole call, including any time the environment queues the work.
    It is the timeout the pack's config gives the tool, or DEFAULT_TIMEOUT_S, which suits a
    fast tool; one that is not a positive, finite number raises ValueError.

    A tool that enforces its own timeout needs a ``timeout_s`` that is a backstop, not the
    limit. Otherwise the executor cancels the tool first, and what would have been an ordinary
    timeout result becomes an error result, which counts toward the agent loop's stop after 5
    consecutive errors (spec §5.1). Effi-SQL's execute_sql is one: it stops a statement after
    120 s, with up to 2 statements in flight per solving instance and a queue in front of them,
    so its pack binds it with the task's wall-clock budget, 600 s in v0, as the backstop.

    The tool's name, description and JSON Schema are read once, here, and the model reads them
    as written (see :meth:`BoundTool.json_schema`). A tool whose arguments have no JSON Schema
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
            return _result(self.name, _invalid_arguments(self.name, error), is_error=True)
        except Exception as error:  # A validator raised what pydantic does not wrap.
            _reraise_environment_failure(error)
            _logger.warning(
                "validating the arguments of the tool %s raised an exception",
                self.name,
                extra={"tool": self.name},
                exc_info=error,
            )
            content = f"Error: invalid arguments for {self.name}: {_describe(error)}"
            return _result(self.name, content, is_error=True)
        cancel_requests = _cancel_requests()
        timeout = asyncio.timeout(self.timeout_s)
        try:
            async with timeout:
                # The tool runs in a task of its own, which a cancellation of this one cancels,
                # so that its cancellation bookkeeping never touches this task's: Python 3.12's
                # TaskGroup leaves an extra cancel request on its parent when a child fails
                # while it exits. The result is an object: pack code can break the contract.
                returned: object = await asyncio.create_task(self.tool.run(args, env))
            if not isinstance(returned, ToolResult):
                kind = type(returned).__name__
                raise TypeError(f"{self.name} returned {kind}, not a ToolResult")
        except Exception as error:
            if _cancel_requests() > cancel_requests:
                # The tool turned an outer cancellation into an exception; it still propagates.
                raise asyncio.CancelledError from error
            _reraise_environment_failure(error)
            if timeout.expired():
                return self._timed_out()
            _logger.warning(
                "the tool %s raised an exception",
                self.name,
                extra={"tool": self.name},
                exc_info=error,
            )
            content = f"Error: {self.name} failed with {_describe(error)}"
            return _result(self.name, content, is_error=True)
        if _cancel_requests() > cancel_requests:
            raise asyncio.CancelledError  # The tool swallowed an outer cancellation.
        if timeout.expired():
            return self._timed_out()  # The tool swallowed its timeout's cancellation.
        submission = returned.submission if returned.HasField("submission") else None
        return _result(self.name, returned.content, submission=submission)

    def _timed_out(self) -> ToolResult:
        content = f"Error: {self.name} timed out after {self.timeout_s:g} s and was cancelled."
        return _result(self.name, content, is_error=True)


class ToolRegistry[E: Environment]:
    """The tools a policy offers the model, each with a name no other tool has.

    It lists them for the LLM client in the order they were registered. A tool that works in
    any environment fits a registry for a narrower one: ``bind(EchoTool())`` fits a
    ``ToolRegistry[SqlEnvironment]``. A registry or a list that mixes tools for different
    environments must state its type, because mypy joins their bound tools' types to
    ``object``: ``registry: ToolRegistry[SqlEnvironment] = ToolRegistry([...])``, or
    ``tools: list[BoundTool[SqlEnvironment]] = [...]``.
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


def environment_failure_in(error: BaseException) -> EnvironmentFailure | None:
    """Return ``error`` if it is an EnvironmentFailure, or else the first one inside it.

    The first one is looked for depth first, through the exception groups that a TaskGroup
    raises; there is none, and the result is None, when no leaf is an EnvironmentFailure. The
    executor raises what it finds, and the runner can apply it to a group that a pack's own
    TaskGroup raised, so that the task is recorded as an environment failure.
    """
    return next((leaf for leaf in _leaves(error) if isinstance(leaf, EnvironmentFailure)), None)


def _reraise_environment_failure(error: Exception) -> None:
    """Raise the EnvironmentFailure that ``error`` is or holds, if there is one.

    A failure inside a group is raised by itself, so that a caller's ``except
    EnvironmentFailure`` catches it, as the contract asks. The group stays reachable: as its
    __cause__, or as its __context__ when the failure already has a cause of its own, such as
    the connection error it was raised from, which it keeps.
    """
    failure = environment_failure_in(error)
    if failure is error:
        raise error
    if failure is not None and failure.__cause__ is not None:
        raise failure  # The group, being handled, becomes its __context__.
    if failure is not None:
        raise failure from error


def _leaves(error: BaseException) -> Iterator[BaseException]:
    """Yield ``error``, or if it is an exception group, each exception at its leaves, in order."""
    if isinstance(error, BaseExceptionGroup):
        for inner in error.exceptions:
            yield from _leaves(inner)
    else:
        yield error


def _cancel_requests() -> int:
    """How many requests to cancel the current task are pending, as Task.cancelling counts."""
    task = asyncio.current_task()
    return 0 if task is None else task.cancelling()


def _invalid_arguments(name: str, error: ValidationError) -> str:
    details = error.errors(include_url=False, include_context=False, include_input=False)
    lines = [f"Error: invalid arguments for {name}:"]
    for detail in details[:_LISTED_ERRORS]:
        where = ".".join(str(part) for part in detail["loc"])
        lines.append(f"- {where}: {detail['msg']}" if where else f"- {detail['msg']}")
    if len(details) > _LISTED_ERRORS:
        lines.append(f"- and {len(details) - _LISTED_ERRORS} more")
    return "\n".join(lines)


def _describe(error: BaseException) -> str:
    """Name ``error``'s type and give its message, never a traceback; a group lists its leaves."""
    text = _type_and_message(error)
    if isinstance(error, BaseExceptionGroup):
        text += ":" + "".join(f"\n- {_type_and_message(leaf)}" for leaf in _leaves(error))
    return text


def _type_and_message(error: BaseException) -> str:
    """``error``'s type and its message, cut short; the type alone if it has no message."""
    try:
        message = str(error)
    except Exception:  # noqa: BLE001 - A broken __str__ must not take the executor down.
        message = ""
    message, _ = _truncate(message, _MESSAGE_KEEP)
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
