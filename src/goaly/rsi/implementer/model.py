"""The implementer's model: mini-swe-agent's model protocol on top of core's LLM client (spec §8.8).

Each step is one ``LlmClient.complete_sync`` call to GLM, with purpose ``rsi-impl`` and the
experiment ID as the run ID, from the worker thread the agent runs in. The client applies the GLM
profile, its retries, ``Retry-After``, AIMD, the circuit breaker and the lane's GLM share, and
writes the call's ``LlmCallRecord``; mini-swe-agent's litellm model is not used.

The assistant message goes into the agent's history exactly as the gateway returned it, its
``reasoning_content`` and ``tool_calls`` included, and goes back unchanged in every later request:
GLM chats replay their reasoning (spec §5.1). Each step's ``extra`` records the gateway's cost,
which the agent adds up, and the call's ID, which joins the step to its record.

The adapter enforces the run's step, cost and wall-time limits before each call, each with an exit
status of its own, so the agent's own step and cost limits stay off and nothing else can end a
run as LimitsExceeded. It also stops the run, as ``Interrupted``, once ``stop`` is set.
"""

import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final, Protocol

from goaly.contracts.gen.llm_call_pb2 import CallPurpose
from goaly.contracts.gen.trajectory_pb2 import ChatMessage, ChatRole, ToolCall, ToolCallSource
from goaly.contracts.interfaces import JsonValue
from goaly.harness.core.llm import CallTags, LlmReply
from goaly.harness.core.llm_errors import LlmError
from goaly.harness.core.llm_models import IMPROVEMENT_MODEL, ModelParams
from goaly.rsi.implementer import msa
from goaly.rsi.implementer.config import Limits, ModelTemplates

__all__ = [
    "EXIT_COST_LIMIT",
    "EXIT_INTERRUPTED",
    "EXIT_STEP_LIMIT",
    "EXIT_TIME_LIMIT",
    "GlmModel",
    "SyncLlm",
    "assistant_message",
    "chat_message",
]

# The exit statuses with which the adapter ends a run; mini-swe-agent's agent adds its own, such
# as Submitted and RepeatedFormatError.
EXIT_STEP_LIMIT: Final = "StepLimitExceeded"
EXIT_COST_LIMIT: Final = "CostLimitExceeded"
EXIT_TIME_LIMIT: Final = "TimeExceeded"  # mini-swe-agent's own name for it.
EXIT_INTERRUPTED: Final = "Interrupted"

_ROLES: Final = {
    "system": ChatRole.CHAT_ROLE_SYSTEM,
    "user": ChatRole.CHAT_ROLE_USER,
    "assistant": ChatRole.CHAT_ROLE_ASSISTANT,
    "tool": ChatRole.CHAT_ROLE_TOOL,
}


class SyncLlm(Protocol):
    """What the adapter needs of core's client: LlmClient.complete_sync, for a worker thread."""

    def complete_sync(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str,
        tags: CallTags,
        tools: Sequence[Mapping[str, JsonValue]] = (),
        params: ModelParams | None = None,
    ) -> LlmReply:
        """Make one call on the client's event loop and wait for it, as LlmClient does."""


class GlmModel:
    """A mini-swe-agent model whose every query is one GLM call through ``llm``.

    ``llm`` is core's LlmClient, which must run on an event loop in another thread: the agent runs
    in a worker thread, and a query made on the loop's own thread raises RuntimeError rather than
    deadlock. ``clock`` is for tests.
    """

    def __init__(
        self,
        llm: SyncLlm,
        *,
        experiment_id: str,
        limits: Limits,
        templates: ModelTemplates,
        stop: threading.Event | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = {
            "model": IMPROVEMENT_MODEL,
            "purpose": CallPurpose.Name(CallPurpose.CALL_PURPOSE_RSI_IMPL),
            "run_id": experiment_id,
            **limits.model_dump(),
        }
        self._llm = llm
        self._tags = CallTags(CallPurpose.CALL_PURPOSE_RSI_IMPL, run_id=experiment_id)
        self._limits = limits
        self._templates = templates
        self._stop = threading.Event() if stop is None else stop
        self._clock = clock
        self._started: float | None = None
        # The calls that returned a reply, and what every call cost, failed ones included.
        self.calls = 0
        self.cost_usd = 0.0

    def query(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        """Send the agent's history to GLM and return the reply, with its actions in ``extra``.

        Raises msa.LimitsExceeded or msa.TimeExceeded, ending the run, once a limit is reached;
        msa.InterruptAgentFlow once ``stop`` is set; msa.FormatError, which the agent shows the
        model, for a reply with no valid bash call; and the client's errors, LlmError among them.
        """
        self._check_limits()
        history = [chat_message(message) for message in messages]
        try:
            reply = self._llm.complete_sync(
                history, model=IMPROVEMENT_MODEL, tags=self._tags, tools=[msa.BASH_TOOL]
            )
        except LlmError as error:
            if error.record is not None:  # The attempts the gateway billed before it failed.
                self.cost_usd += error.record.usage.cost_usd
            raise
        self.calls += 1
        cost = reply.usage.cost_usd
        self.cost_usd += cost
        message = assistant_message(reply.message)
        usage = reply.usage
        extra: dict[str, Any] = {
            "cost": cost,
            "call_id": reply.record.call_id,
            "finish_reason": reply.finish_reason,
            "usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "reasoning_tokens": usage.reasoning_tokens,
            },
            "timestamp": time.time(),
        }
        try:
            actions = msa.parse_toolcall_actions(
                message.get("tool_calls", []),
                format_error_template=self._templates.format_error_template,
                finish_reason=reply.finish_reason,
            )
        except msa.FormatError as error:
            # The agent charges this cost, and the trajectory keeps the reply it could not parse.
            error.messages[0]["extra"].update(extra, response=message)
            raise
        message["extra"] = {"actions": actions, **extra}
        return message

    def format_message(self, **kwargs: Any) -> dict[str, Any]:
        """Return a message the agent adds to its history, such as the system prompt."""
        return dict(kwargs)

    def format_observation_messages(
        self,
        message: dict[str, Any],
        outputs: list[dict[str, Any]],
        template_vars: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the tool messages that answer ``message``'s bash calls with their outputs."""
        return msa.format_toolcall_observations(
            message.get("extra", {}).get("actions", []),
            outputs,
            observation_template=self._templates.observation_template,
            template_vars=template_vars,
        )

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        """Return the variables the agent's templates may use: the model and the limits."""
        return {**self.config, **kwargs}

    def serialize(self) -> dict[str, Any]:
        """Return the trajectory's record of this model. It holds no key: the client has that."""
        return {
            "info": {
                "config": {
                    "model": dict(self.config),
                    "model_type": f"{type(self).__module__}.{type(self).__qualname__}",
                },
            }
        }

    def _check_limits(self) -> None:
        if self._started is None:
            self._started = self._clock()
        if self._stop.is_set():
            raise msa.InterruptAgentFlow(_exit(EXIT_INTERRUPTED))
        if self.calls >= self._limits.step_limit:
            raise msa.LimitsExceeded(_exit(EXIT_STEP_LIMIT))
        if self.cost_usd >= self._limits.cost_limit_usd:
            raise msa.LimitsExceeded(_exit(EXIT_COST_LIMIT))
        if self._clock() - self._started >= self._limits.wall_time_limit_s:
            raise msa.TimeExceeded(_exit(EXIT_TIME_LIMIT))


def chat_message(message: Mapping[str, Any]) -> ChatMessage:
    """Convert a message of the agent's history to the client's ChatMessage.

    Its ``extra``, which only the trajectory needs, stays behind; content, reasoning and tool calls
    go over unchanged. Raises ValueError for a role the gateway does not take, such as ``exit``,
    or content that is not text.
    """
    role = _ROLES.get(message.get("role", ""))
    if role is None:
        raise ValueError(f"cannot send a message with role {message.get('role')!r} to the model")
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    if not isinstance(content, str) or not isinstance(reasoning, str):
        raise ValueError("a message's content and reasoning must be text")
    return ChatMessage(
        role=role,
        content=content,
        reasoning_content=reasoning,
        tool_calls=[
            ToolCall(
                id=call["id"],
                name=call["function"]["name"],
                arguments_json=call["function"]["arguments"],
                source=ToolCallSource.TOOL_CALL_SOURCE_STANDARD,
            )
            for call in message.get("tool_calls") or ()
        ],
        tool_call_id=message.get("tool_call_id", ""),
    )


def assistant_message(message: ChatMessage) -> dict[str, Any]:
    """Convert the client's reply to an OpenAI-format assistant message for the agent's history.

    Content and reasoning are kept as the gateway sent them. A tool call the gateway sent without
    an ID gets one, since the client sends none back without an ID.
    """
    converted: dict[str, Any] = {"role": "assistant", "content": message.content}
    if message.reasoning_content:
        converted["reasoning_content"] = message.reasoning_content
    if message.tool_calls:
        converted["tool_calls"] = [_tool_call(call) for call in message.tool_calls]
    return converted


def _tool_call(call: ToolCall) -> dict[str, Any]:
    return {
        "id": call.id or f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {"name": call.name, "arguments": call.arguments_json},
    }


def _exit(status: str) -> dict[str, Any]:
    """The exit message with which the agent ends a run."""
    return {"role": "exit", "content": status, "extra": {"exit_status": status, "submission": ""}}
