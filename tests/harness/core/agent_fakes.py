"""FakeLLM: a scripted executor model, for tests of the agent loop and of the packs that drive it.

FakeLLM stands in for the exec-bound LLM client the runner hands the loop (llm.BoundLlm). It
replays a script, one item per call: an LlmReply is returned, an LlmError is raised, and HANG
waits until the call is cancelled, as a call cut short by a budget or the runner is. It records
every request, with its own copies of the messages, so a test can check what the model was sent.

:func:`reply` and :func:`tool_call` build scripted replies the way the client returns them: the
message as the model wrote it, and a call record with the call's ID, timing and usage.
:func:`qwen_reply` replays a real Qwen reply from T04's qwen_replies.json.
"""

import asyncio
import itertools
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from goaly.contracts.gen.llm_call_pb2 import (
    CallPurpose,
    LlmCallRecord,
    LlmCallStatus,
    Usage,
)
from goaly.contracts.gen.trajectory_pb2 import ChatMessage, ChatRole, ToolCall, ToolCallSource
from goaly.contracts.interfaces import JsonValue
from goaly.harness.core.llm import LlmReply
from goaly.harness.core.llm_errors import LlmError
from goaly.harness.core.llm_models import EXECUTOR_MODEL, ModelParams

# What a scripted reply costs unless the test says otherwise.
DEFAULT_USAGE: Final = Usage(input_tokens=1000, output_tokens=200, cost_usd=0.001)

# The query that the real Qwen replies in qwen_replies.json set out to rewrite.
QWEN_BASE_SQL: Final = (
    "SELECT count(*) FROM orders WHERE o_custkey IN "
    "(SELECT c_custkey FROM customer WHERE c_nationkey = 7);"
)
_QWEN_REPLIES: Final[dict[str, Any]] = json.loads(
    Path(__file__).with_name("qwen_replies.json").read_text(encoding="utf-8")
)["replies"]

_numbers = itertools.count(1)


class Hang:
    """A script item for a call that never returns: only cancelling it ends it."""


HANG: Final = Hang()


@dataclass(frozen=True)
class Request:
    """One call the loop made: the messages, the tools and the parameters it sent."""

    messages: tuple[ChatMessage, ...]
    tools: tuple[Mapping[str, JsonValue], ...]
    params: ModelParams | None


class FakeLLM:
    """A BoundLlm that answers each call with the next item of its script.

    A call past the end of the script fails the test with an AssertionError, which the loop lets
    propagate like any bug.
    """

    def __init__(self, script: Iterable[LlmReply | LlmError | Hang]) -> None:
        self._script = iter(script)
        self.requests: list[Request] = []

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[Mapping[str, JsonValue]] = (),
        params: ModelParams | None = None,
    ) -> LlmReply:
        self.requests.append(Request(tuple(map(_copy, messages)), tuple(tools), params))
        item = next(self._script, None)
        if item is None:
            raise AssertionError(f"the script has no item for call {len(self.requests)}")
        if isinstance(item, Hang):
            await asyncio.Event().wait()  # Nothing sets it: only cancelling the call ends it.
            raise AssertionError("a hanging call returned")
        if isinstance(item, LlmError):
            raise item
        return item


def reply(
    *calls: ToolCall,
    content: str = "",
    reasoning: str = "",
    finish_reason: str = "tool_calls",
    usage: Usage = DEFAULT_USAGE,
    latency_s: float = 0.25,
) -> LlmReply:
    """A reply the model sends, as the client returns it, with a fresh call ID."""
    message = ChatMessage(
        role=ChatRole.CHAT_ROLE_ASSISTANT,
        content=content,
        reasoning_content=reasoning,
        tool_calls=calls,
    )
    return LlmReply(message, finish_reason, call_record(usage, latency_s=latency_s))


def call_record(usage: Usage = DEFAULT_USAGE, *, latency_s: float = 0.25) -> LlmCallRecord:
    """The record of a successful executor call, with a fresh call ID."""
    record = LlmCallRecord(
        call_id=f"llm-{next(_numbers)}",
        purpose=CallPurpose.CALL_PURPOSE_EXEC,
        model=EXECUTOR_MODEL,
        usage=usage,
        latency_s=latency_s,
        status=LlmCallStatus.LLM_CALL_STATUS_OK,
    )
    record.started_at.GetCurrentTime()
    return record


def qwen_reply(name: str, *, usage: Usage = DEFAULT_USAGE) -> LlmReply:
    """A real Qwen reply from T04's qwen_replies.json, as the client returns it.

    Their task rewrites QWEN_BASE_SQL, with execute_sql(sql, limit=20) and submit(sql) offered.
    """
    fixture = _QWEN_REPLIES[name]
    calls = [
        tool_call(call["name"], call["arguments"], call_id=call["id"])
        for call in fixture["tool_calls"]
    ]
    return reply(
        *calls,
        content=fixture["content"],
        reasoning=fixture["reasoning_content"],
        finish_reason=fixture["finish_reason"],
        usage=usage,
    )


def tool_call(
    name: str, arguments: str | Mapping[str, JsonValue] = "{}", *, call_id: str | None = None
) -> ToolCall:
    """A standard tool call, as the gateway returns one; ``arguments`` may be malformed text."""
    text = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return ToolCall(
        id=f"call-{next(_numbers)}" if call_id is None else call_id,
        name=name,
        arguments_json=text,
        source=ToolCallSource.TOOL_CALL_SOURCE_STANDARD,
    )


def _copy(message: ChatMessage) -> ChatMessage:
    copy = ChatMessage()
    copy.CopyFrom(message)
    return copy
