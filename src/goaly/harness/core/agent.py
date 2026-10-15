"""The agent loop (spec §5.1): core's engine, driving the executor model through a pack's tools.

A capability pack's Policy.solve calls :func:`run_agent` with the task's environment, the task's
:class:`AgentContext` and the pack's :class:`AgentHooks`. The loop sends the chat to the model,
runs the tool calls in its reply, and repeats until a tool's result carries a submission, which
it returns. When a budget runs out first, it returns the pack's fallback, marked ``fallback``
(spec §5.4).

Bad model output never raises. These are errors, and each counts toward the error streak: a
call to an unknown tool, or whose arguments do not parse or validate, or whose tool failed; a
call that repeats the one that ran before it, which runs and says so; a reply that calls no
tool; a reply cut off at the output limit, none of whose calls run; and a reply with more calls
than the budgets allow, whose calls past the cap do not run. After 3 errors in a row (by
default) the model is reminded of the tool schemas, and after 5 the loop stops.

The prompt is kept within the model's context (context_window): past a threshold, older tool
outputs fold. A prompt the gateway still says is too long folds as far as it may and goes once
more; a second overflow, or one with nothing left to fold, ends in the fallback.

What raises is what the Policy.solve docstring lets through: an EnvironmentFailure, an LLM
failure that outlasted the client's retries, and cancellation; so does a bug in the pack's
hooks. No exception becomes a result or the fallback, an exception group included, apart from
the loop's own wall-clock timeout and a context overflow.

The context records every chat and step of the task in memory, with why the agent stopped, and
the runner reads it with :meth:`AgentContext.trajectory` once the policy returns or raises (T06).
"""

import asyncio
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, NamedTuple

from google.protobuf.message import Message

from goaly.contracts.gen.task_pb2 import Prediction
from goaly.contracts.gen.trajectory_pb2 import (
    ChatMessage,
    ChatRole,
    ContextFold,
    Step,
    StopReason,
    ToolCall,
    ToolCallSource,
    ToolResult,
    Trajectory,
)
from goaly.contracts.interfaces import Environment
from goaly.harness.core.budgets import Budgets, BudgetUse, ErrorStreak
from goaly.harness.core.context_window import ChatPrompt, ContextWindow
from goaly.harness.core.llm import BoundLlm, LlmReply
from goaly.harness.core.llm_errors import ContextOverflow, LlmError
from goaly.harness.core.llm_models import ModelParams
from goaly.harness.core.tool_calls import parse_tool_calls, strip_tool_call_markup
from goaly.harness.core.tools import ToolRegistry
from goaly.harness.core.trajectory import TrajectoryRecorder

__all__ = ["AgentContext", "AgentHooks", "run_agent"]

# The finish_reason of a reply that max_tokens cut off.
_CUT_OFF: Final = "length"
# Where the parser finds calls written as Qwen3-Coder XML.
_XML_SOURCES: Final = frozenset(
    {ToolCallSource.TOOL_CALL_SOURCE_CONTENT_XML, ToolCallSource.TOOL_CALL_SOURCE_REASONING_XML}
)

# What the loop tells the model. {tools} lists the tools' names.
_CUT_OFF_CALL: Final = (
    "Error: your reply was cut off at the output limit, so this call was not run. "
    "Send it again, more concisely: think briefly and keep the call short."
)
_CUT_OFF_NO_CALL: Final = (
    "Error: your reply was cut off at the output limit before it called a tool. "
    "Think briefly, then call one of the tools: {tools}."
)
_NO_CALL: Final = (
    "Error: your reply called no tool. Each reply must call one of the tools: {tools}."
)
_REPEATED: Final = (
    "Note: this call repeats your previous one, the same tool with the same arguments, so its "
    "result is unchanged. Repeating a call does not help: change the arguments or try another "
    "approach."
)
_OVER_CAP: Final = (
    "Error: a reply may make at most {cap} tool calls, so this call was not run. "
    "Make fewer calls in each reply."
)
_REMINDER: Final = (
    "You have made {errors} in a row. Call the tools exactly as these schemas define them:\n"
    "{schemas}"
)


class AgentContext:
    """One task's context for the agent loop: its model, budgets and trajectory (spec §5.1).

    The runner's context factory builds one for each task and replicate seed (T06), and T11
    wires in the pack's config, its ``budgets``, request ``params`` and context ``window``
    (ContextWindow's defaults if none), and ``llm``: the client
    bound to purpose exec, the executor model, and the task's run, instance and seed
    (LlmClient.bind). A pack reaches the model only through :func:`run_agent`, so it can choose
    neither the purpose nor the model, and every call it makes is held to the budgets and
    recorded (spec §8.5).

    The context is the task's: every loop a policy runs with it adds a chat to one trajectory
    and spends the same budgets, whose wall clock starts with the first loop.
    """

    def __init__(
        self,
        llm: BoundLlm,
        *,
        budgets: Budgets,
        params: ModelParams | None = None,
        window: ContextWindow | None = None,
    ) -> None:
        self._llm = llm
        self._budgets = budgets
        self._params = params
        self._window = ContextWindow() if window is None else window
        self._recorder = TrajectoryRecorder()
        self._started: float | None = None  # The event loop's time when the first loop started.

    @property
    def budgets(self) -> Budgets:
        """What the task may spend."""
        return self._budgets

    def trajectory(self) -> Trajectory:
        """Return a copy of the task's trajectory so far: its chats, steps, usage and stop reason.

        The runner fills in the rest once the policy returns or raises (T06): the run, task and
        seed, the prediction and the grade, the error and the timestamps.
        """
        return self._recorder.trajectory()

    def _clock_start(self) -> float:
        """Start the task's wall clock, unless a loop already has, and return when it started."""
        if self._started is None:
            self._started = asyncio.get_running_loop().time()
        return self._started


@dataclass(frozen=True, kw_only=True, eq=False)
class AgentHooks[E: Environment]:
    """What a capability pack gives the agent loop for one task (spec §5.2, §5.3).

    ``system_prompt`` and ``task_prompt`` open the chat, as its system message and its first
    user message. ``tools`` are the tools the model may call, run in the task's environment.
    ``fallback`` returns the pack's answer for when a budget runs out before a tool submits:
    v0 Effi-SQL's submits ``base_sql`` unchanged (spec §5.4).

    Guidance goes into tool results and between steps, not into a longer system prompt
    (spec §5.3). ``on_tool_result`` sees each call that ran, with its result, and may return
    text to append to the result the model sees. ``on_step`` sees each step that another step
    follows, with what the task has spent so far, and may return a message to send before the
    next call. Both get copies, and return None, or "", for no guidance.
    """

    system_prompt: str
    task_prompt: str
    tools: ToolRegistry[E]
    fallback: Callable[[], Prediction]
    on_tool_result: Callable[[ToolCall, ToolResult], str | None] | None = None
    on_step: Callable[[Step, BudgetUse], str | None] | None = None


async def run_agent[E: Environment](
    env: E, ctx: AgentContext, hooks: AgentHooks[E], *, label: str = "main"
) -> Prediction:
    """Drive the model through ``hooks.tools`` in ``env`` until a tool submits or a budget runs out.

    Return the submission, or the fallback, with ``fallback`` set, if a budget runs out first.
    The chat goes into ``ctx``'s trajectory under ``label``, with why it stopped. What may raise
    is as the module docstring says; the trajectory then records STOP_REASON_CANCELLED, or
    STOP_REASON_CRASHED and the error.
    """
    return await _Chat(env, ctx, hooks, label).run()


class _Outcome(NamedTuple):
    """How a chat ended: why, the submission if a tool made one, and any error."""

    stop_reason: StopReason.ValueType
    submission: Prediction | None = None
    error_message: str = ""


class _Sent(NamedTuple):
    """A model call that got a reply: the reply, and the prompt it was sent, as folded."""

    reply: LlmReply
    folds: list[ContextFold]
    estimated_tokens: int


class _Chat[E: Environment]:
    """One loop of :func:`run_agent`: a chat with the model, recorded in the task's trajectory."""

    def __init__(self, env: E, ctx: AgentContext, hooks: AgentHooks[E], label: str) -> None:
        self._env = env
        self._ctx = ctx
        self._hooks = hooks
        self._recorder = ctx._recorder
        self._tools = hooks.tools.json_schemas()
        self._tool_names = ", ".join(tool.name for tool in hooks.tools) or "none"
        opening = [
            ChatMessage(role=ChatRole.CHAT_ROLE_SYSTEM, content=hooks.system_prompt),
            ChatMessage(role=ChatRole.CHAT_ROLE_USER, content=hooks.task_prompt),
        ]
        self._chat = self._recorder.start_chat(label, opening)
        self._prompt = ChatPrompt(opening, self._tools, ctx._window)
        self._streak = ErrorStreak(ctx.budgets)
        # The last call that ran: its tool and normalized arguments, for repeat detection.
        self._previous: tuple[str, str] | None = None
        self._last: Step | None = None  # The step the next call follows.
        self._notes: list[str] = []  # The loop's own messages to send after that step.

    async def run(self) -> Prediction:
        try:
            outcome = await self._converse_in_time()
            prediction = outcome.submission
            if prediction is None:
                prediction = self._fallback()
        except asyncio.CancelledError:
            self._recorder.stop(StopReason.STOP_REASON_CANCELLED)
            raise
        except BaseException as error:
            self._recorder.stop(StopReason.STOP_REASON_CRASHED, error_message=_describe(error))
            raise
        self._recorder.stop(outcome.stop_reason, error_message=outcome.error_message)
        return prediction

    async def _converse_in_time(self) -> _Outcome:
        """Converse until the chat ends, or the task's wall clock runs out (spec §5.1)."""
        timeout = asyncio.timeout_at(self._ctx._clock_start() + self._ctx.budgets.wall_clock_s)
        try:
            async with timeout:
                return await self._converse()
        except TimeoutError:
            if not timeout.expired():
                raise
        # The call or the tool that was running was cancelled; the client records the call.
        return _Outcome(StopReason.STOP_REASON_TIME_BUDGET_EXHAUSTED)

    async def _converse(self) -> _Outcome:
        while True:
            stop = self._stop_reason()
            if stop is not None:
                return _Outcome(stop)
            self._send_guidance()
            try:
                sent = await self._send()
            except ContextOverflow as error:
                stop = StopReason.STOP_REASON_TOKEN_BUDGET_EXHAUSTED
                return _Outcome(stop, error_message=_describe(error))
            submission = await self._take_step(sent)
            if submission is not None:
                return _Outcome(StopReason.STOP_REASON_SUBMITTED, submission)

    def _stop_reason(self) -> StopReason.ValueType | None:
        """Why the chat must stop before another call, if it must."""
        if self._streak.exhausted:
            return StopReason.STOP_REASON_CONSECUTIVE_ERRORS
        return self._ctx.budgets.exhausted(self._spent())

    def _spent(self) -> BudgetUse:
        usage = self._recorder.usage
        return BudgetUse(
            steps=self._recorder.step_count,
            elapsed_s=asyncio.get_running_loop().time() - self._ctx._clock_start(),
            tokens=usage.input_tokens + usage.output_tokens,
            cost_usd=usage.cost_usd,
        )

    async def _send(self) -> _Sent:
        """Call the model with the chat's prompt, folded to fit its context (spec §5.1).

        A prompt that the gateway says does not fit folds every older output it may and goes
        once more. With nothing more to fold, or a second overflow, ContextOverflow propagates.
        """
        folds = self._prompt.fold()
        estimate = self._prompt.estimated_tokens
        try:
            reply = await self._complete()
        except ContextOverflow:
            more = self._prompt.fold(everything_older=True)
            if not more:
                raise
            folds += more
            estimate = self._prompt.estimated_tokens
            reply = await self._complete()
        self._prompt.sent(reply.usage.input_tokens)
        return _Sent(reply, folds, estimate)

    async def _complete(self) -> LlmReply:
        try:
            return await self._ctx._llm.complete(
                self._prompt.messages(), tools=self._tools, params=self._ctx._params
            )
        except LlmError as error:
            if error.record is not None:  # The gateway may have billed a call that failed.
                self._recorder.add_usage(error.record.usage)
            raise

    async def _take_step(self, sent: _Sent) -> Prediction | None:
        """Record the step of ``sent``'s reply and run its calls, in order.

        Return the first submission, if a call makes one.
        """
        reply = sent.reply
        calls = parse_tool_calls(reply.message, self._tools)
        assistant = _with_calls(reply.message, calls)
        step = self._recorder.add_step(self._chat, reply.record, assistant)
        step.estimated_prompt_tokens = sent.estimated_tokens
        step.folds.extend(sent.folds)
        self._last = step
        self._prompt.add_turn(step.index, _replayed(assistant))
        if reply.finish_reason == _CUT_OFF:
            for call in calls:  # A cut-off call may be a wrong submit: none runs.
                self._answer(step, _not_run(call, _CUT_OFF_CALL))
            self._streak.add(error=True)  # One error for the reply, however many calls it had.
            if not calls:
                self._notes.append(_CUT_OFF_NO_CALL.format(tools=self._tool_names))
            return None
        if not calls:
            self._streak.add(error=True)
            self._notes.append(_NO_CALL.format(tools=self._tool_names))
            return None
        cap = self._ctx.budgets.calls_per_reply
        for call in calls[:cap]:
            result = await self._hooks.tools.execute(call, self._env)
            repeated = self._repeats(call)
            if repeated:
                result.content = _appended(result.content, _REPEATED)
            self._streak.add(error=result.is_error or repeated)
            self._add_guidance(call, result)
            self._answer(step, result)
            if result.HasField("submission"):  # The episode ends: later calls do not run.
                return result.submission
        if len(calls) > cap:
            for call in calls[cap:]:
                self._answer(step, _not_run(call, _OVER_CAP.format(cap=cap)))
            self._streak.add(error=True)  # One error for the reply, however far past the cap.
        return None

    def _repeats(self, call: ToolCall) -> bool:
        """Whether ``call``, which ran, repeats the call that ran before it in this chat.

        Calls are compared by tool and normalized arguments, never by ID: an XML call's ID is
        only unique within its reply.
        """
        key = (call.name, _normalized_arguments(call))
        repeated = key == self._previous
        self._previous = key
        return repeated

    def _add_guidance(self, call: ToolCall, result: ToolResult) -> None:
        hook = self._hooks.on_tool_result
        guidance = hook(_copy(call), _copy(result)) if hook is not None else None
        if guidance:
            result.content = _appended(result.content, guidance)

    def _answer(self, step: Step, result: ToolResult) -> None:
        step.tool_results.append(result)
        self._prompt.add(
            ChatMessage(
                role=ChatRole.CHAT_ROLE_TOOL,
                content=result.content,
                tool_call_id=result.tool_call_id,
            )
        )

    def _send_guidance(self) -> None:
        """Add the messages due after the last step to it and to the prompt (spec §5.3)."""
        step = self._last
        if step is None:
            return
        texts, self._notes = self._notes, []
        if self._streak.take_reminder():
            errors = self._streak.count
            schemas = json.dumps(self._tools, ensure_ascii=False, indent=2)
            texts.append(
                _REMINDER.format(errors=f"{errors} error{'s' * (errors != 1)}", schemas=schemas)
            )
        hook = self._hooks.on_step
        guidance = hook(_copy(step), self._spent()) if hook is not None else None
        if guidance:
            texts.append(guidance)
        for text in texts:
            message = ChatMessage(role=ChatRole.CHAT_ROLE_USER, content=text)
            step.injected.append(message)
            self._prompt.add(message)

    def _fallback(self) -> Prediction:
        prediction = _copy(self._hooks.fallback())
        prediction.fallback = True
        return prediction


def _with_calls(message: ChatMessage, calls: Sequence[ToolCall]) -> ChatMessage:
    """``message`` as the model wrote it, with the calls parsed from it as its tool calls."""
    assistant = _copy(message)
    del assistant.tool_calls[:]
    assistant.tool_calls.extend(calls)
    return assistant


def _replayed(assistant: ChatMessage) -> ChatMessage:
    """``assistant`` as later prompts carry it: each call only once, as a structured call.

    When its calls were parsed from XML, the call markup is taken out of both the content and
    the reasoning. A call drafted in one and written again in the other then reaches the model
    once, and a drafted call that never ran cannot look as if it ran. The trajectory keeps the
    reply as it came.
    """
    message = _copy(assistant)
    if any(call.source in _XML_SOURCES for call in assistant.tool_calls):
        message.content = strip_tool_call_markup(message.content)
        message.reasoning_content = strip_tool_call_markup(message.reasoning_content)
    return message


def _not_run(call: ToolCall, content: str) -> ToolResult:
    """The error result of a call that the loop did not run, telling the model why."""
    return ToolResult(tool_call_id=call.id, name=call.name, content=content, is_error=True)


def _normalized_arguments(call: ToolCall) -> str:
    """``call``'s arguments, in one layout: sorted keys, no spaces; as written if not JSON."""
    text = call.repaired_arguments_json or call.arguments_json
    try:
        value = json.loads(text)
    except (ValueError, RecursionError):  # RecursionError: nested too deep to parse.
        return text.strip()
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _appended(content: str, note: str) -> str:
    return f"{content}\n\n{note}" if content else note


def _copy[M: Message](message: M) -> M:
    copy = type(message)()
    copy.CopyFrom(message)
    return copy


def _describe(error: BaseException) -> str:
    """Name ``error``'s type and give its message, and those of the errors a group holds."""
    text = f"{type(error).__name__}: {error}" if str(error) else type(error).__name__
    if isinstance(error, BaseExceptionGroup):
        text += f" [{'; '.join(map(_describe, error.exceptions))}]"
    return text
