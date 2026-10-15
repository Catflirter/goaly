"""The LLM client: the one way to call the model gateway (spec §5.1, §9).

``LlmClient.complete`` sends a chat to the gateway and returns the assistant's reply. Callers
pass and get contract messages: ``ChatMessage`` in, the assistant's ``ChatMessage`` out, with its
tool calls as the model wrote them, for core's parser (T04). The OpenAI wire format exists only
inside this module. For each call the client:

- checks the model allowlist and applies the model's profile (``llm_models``);
- gives every attempt a timeout, and retries 429s, 5xx responses, timeouts and connection
  errors with exponential backoff (``RetryPolicy``);
- raises a typed error when the call fails (``llm_errors``);
- hands one ``LlmCallRecord`` to a ``CallRecorder``, however the call ends.

A process uses one client, on its event loop::

    async with LlmClient(load_settings(), JsonlCallRecorder(path)) as llm:
        reply = await llm.complete(
            messages, model=EXECUTOR_MODEL, tags=CallTags(CallPurpose.CALL_PURPOSE_EXEC)
        )

A task gets a client bound to one model and one call's tags, ``LlmClient.bind``, which offers
just ``complete(messages, *, tools, params)``: whoever holds it cannot choose another purpose or
model. T11's per-task factory binds CALL_PURPOSE_EXEC with the task's run, instance and seed;
only goaly.rsi binds the rsi-* purposes.

The API key goes in the Authorization header and nowhere else: no message, log line, repr or
record holds it, and the client removes it from any text it quotes from the gateway.
"""

import asyncio
import json
import logging
import math
import random
import threading
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from types import TracebackType
from typing import Final, Protocol, Self

import httpx
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveFloat, ValidationError

from goaly.contracts.gen.llm_call_pb2 import (
    CallPurpose,
    LlmCallAttempt,
    LlmCallRecord,
    LlmCallStatus,
    Usage,
)
from goaly.contracts.gen.trajectory_pb2 import ChatMessage, ChatRole, ToolCall, ToolCallSource
from goaly.contracts.interfaces import JsonValue
from goaly.contracts.jsonl import append_jsonl
from goaly.harness.core.llm_errors import ModelNotAllowed, error_type
from goaly.harness.core.llm_models import (
    DEFAULT_PROFILES,
    ModelParams,
    ModelProfile,
    allowed_model,
)
from goaly.harness.core.settings import Settings

__all__ = [
    "BoundLlm",
    "BoundLlmClient",
    "CallRecorder",
    "CallTags",
    "Clock",
    "JsonlCallRecorder",
    "LlmClient",
    "LlmReply",
    "RetryPolicy",
]

_logger = logging.getLogger(__name__)

_S = LlmCallStatus
_RETRYABLE: Final = frozenset(
    {
        _S.LLM_CALL_STATUS_RATE_LIMITED,
        _S.LLM_CALL_STATUS_SERVER_ERROR,
        _S.LLM_CALL_STATUS_TIMEOUT,
        _S.LLM_CALL_STATUS_CONNECTION_ERROR,
    }
)
# How a 400 says the request does not fit in the context, in both forms seen on 2026-10-15:
# LiteLLM marks a prompt too long by itself, but passes SGLang's message on as a plain
# BadRequestError when the prompt fits and max_tokens does not.
_CONTEXT_OVERFLOW_MARKS: Final = ("contextwindowexceedederror", "maximum context length")
_THINKING_SWITCHES: Final = ("enable_thinking", "preserve_thinking")
_ROLES: Final = {
    ChatRole.CHAT_ROLE_SYSTEM: "system",
    ChatRole.CHAT_ROLE_USER: "user",
    ChatRole.CHAT_ROLE_ASSISTANT: "assistant",
    ChatRole.CHAT_ROLE_TOOL: "tool",
}
_JSON_HEADERS: Final = {"Content-Type": "application/json"}
_QUOTE_CHARS: Final = 500  # How much of the gateway's error text an error message quotes.


class RetryPolicy(BaseModel):
    """How long an attempt may take, and how the client retries a failed one (spec §5.1, §6.2).

    ``timeout_s`` runs from sending an attempt's request to reading all of its response. A retry
    waits for full jitter on an exponential backoff, or for the server's Retry-After when that is
    longer; either way no longer than ``backoff_cap_s``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    timeout_s: PositiveFloat = 300.0
    max_retries: NonNegativeInt = 5
    backoff_base_s: PositiveFloat = 1.0
    backoff_cap_s: PositiveFloat = 60.0

    def delay_s(self, retry: int, retry_after_s: float | None, rng: random.Random) -> float:
        """Return the wait before retry number ``retry``, counting from 0."""
        ceiling = self.backoff_base_s * 2.0 ** min(retry, 64)  # The bound keeps it a float.
        jitter = rng.uniform(0.0, min(ceiling, self.backoff_cap_s))
        if retry_after_s is None:
            return jitter
        # Even a Retry-After of 0, or a date in the past, waits for the backoff.
        return min(max(retry_after_s, jitter), self.backoff_cap_s)


@dataclass(frozen=True, slots=True)
class CallTags:
    """What a call is for: its purpose and, if it has them, its run, task and replicate seed.

    The call's record carries them, so calls join to trajectories on run_id, instance_id and seed.
    """

    purpose: CallPurpose.ValueType
    run_id: str = ""
    instance_id: str = ""
    seed: int = 0


@dataclass(frozen=True, slots=True)
class LlmReply:
    """A successful call: the assistant's message, why the model stopped, and the call's record.

    The message's content and reasoning are exactly as the gateway sent them, and its tool calls
    are the response's standard ones, raw. ``finish_reason`` is the gateway's, e.g. "stop",
    "tool_calls", or "length" when max_tokens ran out.
    """

    message: ChatMessage
    finish_reason: str
    record: LlmCallRecord

    @property
    def usage(self) -> Usage:
        """The call's tokens and cost, as its record holds them."""
        return self.record.usage


class CallRecorder(Protocol):
    """Where the client writes each call's record, exactly once, however the call ends."""

    def record(self, record: LlmCallRecord) -> None:
        """Take ``record``, which is the recorder's own copy.

        The client calls this on its event loop, so it must not block for long: the runner's
        single-writer call log (T06) only queues the record. It must not raise either. If it
        does, the client logs the lost record and goes on, so a call never loses its reply or
        its own error to its recorder.
        """


class JsonlCallRecorder:
    """A CallRecorder that appends each record to a JSONL file at once, for scripts and tests.

    Each write blocks the event loop for a moment; a run's call log goes through the runner's
    single writer instead (spec §6.2).
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()  # append_jsonl does not lock, and threads may share this.

    def record(self, record: LlmCallRecord) -> None:
        with self._lock:
            append_jsonl(self.path, record)


class Clock(Protocol):
    """The time a client measures waits and latency with, and sleeps through backoffs on."""

    def monotonic(self) -> float:
        """Return the time in seconds on a clock that never goes back."""

    async def sleep(self, seconds: float) -> None:
        """Wait ``seconds``."""


class _SystemClock:
    def monotonic(self) -> float:
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


# The parts of a chat completion the client reads. Anything else in the body is ignored.


class _WireFunction(BaseModel):
    name: str
    arguments: str | None = None


class _WireToolCall(BaseModel):
    id: str | None = None
    function: _WireFunction


class _WireMessage(BaseModel):
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[_WireToolCall] | None = None


class _WireChoice(BaseModel):
    message: _WireMessage
    finish_reason: str | None = None


class _WireTokenDetails(BaseModel):
    reasoning_tokens: NonNegativeInt | None = None


class _WireUsage(BaseModel):
    prompt_tokens: NonNegativeInt = 0
    completion_tokens: NonNegativeInt = 0
    completion_tokens_details: _WireTokenDetails | None = None


class _WireResponse(BaseModel):
    choices: list[_WireChoice] = Field(min_length=1)
    usage: _WireUsage | None = None


@dataclass(frozen=True, slots=True)
class _Outcome:
    """How one attempt ended: ``detail`` says what went wrong, and ``response`` is set if OK."""

    status: LlmCallStatus.ValueType
    http_status: int = 0
    detail: str = ""
    retry_after_s: float | None = None
    cost_usd: float | None = None
    response: _WireResponse | None = None


class BoundLlm(Protocol):
    """A model to chat with, its purpose and tags fixed: what T05's agent loop calls.

    LlmClient.bind makes one; tests may stand in their own.
    """

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[Mapping[str, JsonValue]] = (),
        params: ModelParams | None = None,
    ) -> LlmReply:
        """Send ``messages`` and return the assistant's reply, as LlmClient.complete does."""


class LlmClient:
    """An async client of the OpenAI-compatible model gateway; the module docstring has more.

    ``profiles`` replace the default profiles of their models (llm_models.DEFAULT_PROFILES).
    ``transport``, ``clock`` and ``rng`` are for tests: a mock transport, a clock that skips
    waits, and a seeded source of jitter.
    """

    def __init__(
        self,
        settings: Settings,
        recorder: CallRecorder,
        *,
        profiles: Iterable[ModelProfile] = (),
        retry: RetryPolicy | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Clock | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._profiles = {profile.model: profile for profile in DEFAULT_PROFILES}
        for profile in profiles:
            if profile.model not in self._profiles:
                raise ValueError(f"no call purpose may use {profile.model}, so it has no profile")
            self._profiles[profile.model] = profile
        self._recorder = recorder
        self._retry = RetryPolicy() if retry is None else retry
        self._clock: Clock = _SystemClock() if clock is None else clock
        self._rng = random.Random() if rng is None else rng  # noqa: S311 - jitter, not secrecy.
        self._api_key = settings.api_key
        self._http = httpx.AsyncClient(
            base_url=settings.base_url,
            headers={"Authorization": f"Bearer {settings.api_key.get_secret_value()}"},
            timeout=self._retry.timeout_s,
            transport=transport,
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the client's connections; it cannot make calls after."""
        await self._http.aclose()

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str,
        tags: CallTags,
        tools: Sequence[Mapping[str, JsonValue]] = (),
        params: ModelParams | None = None,
    ) -> LlmReply:
        """Send ``messages`` to ``model`` and return the assistant's reply.

        ``tools`` are OpenAI tool entries, as core's tool registry makes them (T04). ``params``
        override the fields of the model's profile that they set.

        A purpose that may not use ``model`` raises ModelNotAllowed, and arguments that cannot
        be sent raise ValueError, both before any I/O and with no record. Any other call gets
        exactly one record, however it ends: a failure raises an LlmError, and
        asyncio.CancelledError goes through at once, with no retry.
        """
        if allowed_model(tags.purpose) != model:
            raise self._refuse(model, tags)
        body = _request_body(self._profiles[model].with_overrides(params), messages, tools)
        record = LlmCallRecord(
            call_id=uuid.uuid4().hex,
            run_id=tags.run_id,
            instance_id=tags.instance_id,
            seed=tags.seed,
            purpose=tags.purpose,
            model=model,
        )
        record.started_at.GetCurrentTime()
        start = self._clock.monotonic()
        try:
            response = await self._send(body, record)
            self._read_usage(response, record)
        finally:
            self._finish(record, start)
        choice = response.choices[0]
        return LlmReply(_assistant_message(choice.message), choice.finish_reason or "", record)

    def bind(self, *, model: str, tags: CallTags) -> "BoundLlmClient":
        """Return a client whose calls all go to ``model``, tagged ``tags``.

        The pair is checked against the model allowlist now, before any I/O: a purpose that
        may not use ``model`` raises ModelNotAllowed. Every bound client calls through this
        one, so all of them share its limits, connections and recorder.
        """
        if allowed_model(tags.purpose) != model:
            raise self._refuse(model, tags)
        return BoundLlmClient(self, model, tags)

    def _refuse(self, model: str, tags: CallTags) -> ModelNotAllowed:
        try:
            purpose = CallPurpose.Name(tags.purpose)
        except ValueError:  # A value no CallPurpose has, as only a cast can make one.
            purpose = str(tags.purpose)
        _logger.warning(
            "refused an LLM call: its purpose may not use the model",
            extra={
                "event": "llm_model_not_allowed",
                "model": model,
                "purpose": purpose,
                "run_id": tags.run_id,
                "instance_id": tags.instance_id,
                "seed": tags.seed,
            },
        )
        allowed = allowed_model(tags.purpose)
        may = f"it may call only {allowed}" if allowed else "it may call no model"
        return ModelNotAllowed(f"{purpose} may not call {model}: {may}")

    async def _send(self, body: bytes, record: LlmCallRecord) -> _WireResponse:
        """Make attempts until one succeeds, one fails for good, or the retries run out."""
        delay = 0.0
        while True:
            outcome = await self._attempt(record, record.attempts.add(), body, delay)
            if outcome.cost_usd is not None:
                record.usage.cost_usd += outcome.cost_usd  # Every attempt the gateway billed.
            if outcome.response is not None:
                return outcome.response
            retries = len(record.attempts) - 1
            if outcome.status not in _RETRYABLE or retries >= self._retry.max_retries:
                count = f"{retries + 1} attempt{'s' if retries else ''}"
                message = f"{record.model}: {outcome.detail} ({count})"
                raise error_type(outcome.status)(message, record=record)
            delay = self._retry.delay_s(retries, outcome.retry_after_s, self._rng)
            self._log_backoff(record, outcome, delay)

    async def _attempt(
        self, record: LlmCallRecord, attempt: LlmCallAttempt, body: bytes, delay: float
    ) -> _Outcome:
        """Wait ``delay``, send the request once, and fill in ``attempt`` with how it went."""
        attempt.started_at.GetCurrentTime()
        began = self._clock.monotonic()
        sent: float | None = None
        try:
            if delay > 0:
                await self._clock.sleep(delay)
            sent = self._clock.monotonic()
            outcome = await self._request(body, record)
        except asyncio.CancelledError:
            attempt.status = _S.LLM_CALL_STATUS_CANCELLED
            raise
        finally:
            now = self._clock.monotonic()
            attempt.wait_s = (now if sent is None else sent) - began
            attempt.latency_s = 0.0 if sent is None else now - sent
        attempt.status = outcome.status
        attempt.http_status = outcome.http_status
        if outcome.retry_after_s is not None:
            attempt.retry_after_s = outcome.retry_after_s
        return outcome

    async def _request(self, body: bytes, record: LlmCallRecord) -> _Outcome:
        """Send the request and say how it went. Only cancellation and bugs raise."""
        try:
            async with asyncio.timeout(self._retry.timeout_s):
                response = await self._http.post(
                    "/chat/completions", content=body, headers=_JSON_HEADERS
                )
        except (TimeoutError, httpx.TimeoutException):
            detail = f"no response within {self._retry.timeout_s:g} s"
            return _Outcome(_S.LLM_CALL_STATUS_TIMEOUT, detail=detail)
        except httpx.LocalProtocolError as error:
            # The request itself breaks HTTP, so a retry fails the same way. The error's text
            # quotes the offending header, which may be the one holding the key: leave it out.
            detail = f"httpx refused to send the request ({type(error).__name__})"
            return _Outcome(_S.LLM_CALL_STATUS_CLIENT_ERROR, detail=detail)
        except httpx.TransportError as error:
            detail = self._quote(f"connection failed: {type(error).__name__}: {error}")
            return _Outcome(_S.LLM_CALL_STATUS_CONNECTION_ERROR, detail=detail)
        except httpx.RequestError as error:  # For example, a body that cannot be decoded.
            detail = self._quote(f"unreadable response: {type(error).__name__}: {error}")
            return _Outcome(_S.LLM_CALL_STATUS_INVALID_RESPONSE, detail=detail)
        return self._outcome(response, record)

    def _outcome(self, response: httpx.Response, record: LlmCallRecord) -> _Outcome:
        """Say how an attempt that got ``response`` ended (spec §5.1)."""
        code = response.status_code
        retry_after = _retry_after_s(response.headers)
        cost = self._cost_usd(response.headers, record)
        if response.is_success:
            try:
                parsed = _WireResponse.model_validate_json(response.content)
            except ValidationError as error:
                detail = f"HTTP {code}, but no chat completion: {_problems(error)}"
                return _Outcome(
                    _S.LLM_CALL_STATUS_INVALID_RESPONSE, code, detail, retry_after, cost
                )
            return _Outcome(_S.LLM_CALL_STATUS_OK, code, "", retry_after, cost, parsed)
        said = self._redact(_error_message(response))
        if code == httpx.codes.TOO_MANY_REQUESTS:
            status = _S.LLM_CALL_STATUS_RATE_LIMITED
        elif code == httpx.codes.REQUEST_TIMEOUT:  # LiteLLM's own upstream request timed out.
            status = _S.LLM_CALL_STATUS_TIMEOUT
        elif code >= httpx.codes.INTERNAL_SERVER_ERROR:
            status = _S.LLM_CALL_STATUS_SERVER_ERROR
        elif code == httpx.codes.BAD_REQUEST and any(
            mark in said.lower() for mark in _CONTEXT_OVERFLOW_MARKS
        ):
            status = _S.LLM_CALL_STATUS_CONTEXT_OVERFLOW
        elif code >= httpx.codes.BAD_REQUEST:
            status = _S.LLM_CALL_STATUS_CLIENT_ERROR
        else:  # 1xx or 3xx: no chat completion comes that way.
            status = _S.LLM_CALL_STATUS_INVALID_RESPONSE
        detail = f"HTTP {code}: {_cut(said)}" if said else f"HTTP {code}"
        return _Outcome(status, code, detail, retry_after, cost)

    def _cost_usd(self, headers: httpx.Headers, record: LlmCallRecord) -> float | None:
        """Return the cost the gateway reported in x-litellm-response-cost, or None for none."""
        value = headers.get("x-litellm-response-cost")
        if value is None:
            return None
        try:
            cost = float(value)
        except ValueError:
            cost = math.nan
        if math.isfinite(cost) and cost >= 0:
            return cost
        _logger.warning(
            "ignored a malformed cost header",
            extra={
                "event": "llm_bad_cost",
                "model": record.model,
                "call_id": record.call_id,
                "value": self._redact(value)[:40],  # Redacted first: a cut could split the key.
            },
        )
        return None

    def _read_usage(self, response: _WireResponse, record: LlmCallRecord) -> None:
        """Copy the tokens the gateway reported into ``record``, whose cost is already in."""
        usage = response.usage
        if usage is None:
            _logger.warning(
                "the gateway reported no usage, so the call counts no tokens",
                extra={"event": "llm_no_usage", "model": record.model, "call_id": record.call_id},
            )
            return
        record.usage.input_tokens = usage.prompt_tokens
        record.usage.output_tokens = usage.completion_tokens
        details = usage.completion_tokens_details
        if details is not None and details.reasoning_tokens is not None:
            record.usage.reasoning_tokens = details.reasoning_tokens

    def _finish(self, record: LlmCallRecord, start: float) -> None:
        """Complete ``record`` from its attempts, and hand the recorder a copy of its own."""
        record.latency_s = self._clock.monotonic() - start
        record.wait_s = sum(attempt.wait_s for attempt in record.attempts)
        last = record.attempts[-1]  # _send makes the first attempt before anything can fail.
        record.status = last.status
        record.http_status = last.http_status
        copy = LlmCallRecord()
        copy.CopyFrom(record)
        try:
            self._recorder.record(copy)
        except Exception:  # A broken recorder must not take the call's outcome with it.
            _logger.exception(
                "the call recorder failed, so this call's record is lost",
                extra={
                    "event": "llm_record_lost",
                    "model": record.model,
                    "call_id": record.call_id,
                },
            )

    def _log_backoff(self, record: LlmCallRecord, outcome: _Outcome, delay: float) -> None:
        """Log a throttling event (spec §6.2): the wait before the next attempt, and why."""
        _logger.info(
            "LLM call backing off before its next attempt",
            extra={
                "event": "llm_backoff",
                "model": record.model,
                "purpose": CallPurpose.Name(record.purpose),
                "call_id": record.call_id,
                "attempt": len(record.attempts) + 1,
                "after": LlmCallStatus.Name(outcome.status),
                "http_status": outcome.http_status,
                "detail": outcome.detail,
                "delay_s": delay,
                "retry_after_s": outcome.retry_after_s,
            },
        )

    def _redact(self, text: str) -> str:
        return text.replace(self._api_key.get_secret_value(), "[redacted]")

    def _quote(self, text: str) -> str:
        return _cut(self._redact(text))


class BoundLlmClient:
    """A BoundLlm that calls through the LlmClient that bound it; build one with bind()."""

    __slots__ = ("_client", "_model", "_tags")

    def __init__(self, client: LlmClient, model: str, tags: CallTags) -> None:
        self._client = client
        self._model = model
        self._tags = tags

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[Mapping[str, JsonValue]] = (),
        params: ModelParams | None = None,
    ) -> LlmReply:
        """Send ``messages`` to the bound model with the bound tags; see LlmClient.complete."""
        return await self._client.complete(
            messages, model=self._model, tags=self._tags, tools=tools, params=params
        )


def _request_body(
    profile: ModelProfile,
    messages: Sequence[ChatMessage],
    tools: Sequence[Mapping[str, JsonValue]],
) -> bytes:
    """Build the request as JSON. Qwen's thinking switches go in chat_template_kwargs."""
    body: dict[str, object] = {
        "model": profile.model,
        "messages": [_wire_message(message) for message in messages],
    }
    params = profile.params.model_dump(exclude_none=True)
    switches = {name: params.pop(name) for name in _THINKING_SWITCHES if name in params}
    body.update(params)  # The sampling fields, named as the wire names them.
    if switches:
        body["chat_template_kwargs"] = switches
    if tools:
        body["tools"] = [dict(tool) for tool in tools]
    try:
        text = json.dumps(body, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except TypeError as error:  # A value JSON has no form for; NaN raises ValueError itself.
        raise ValueError(f"the request cannot be sent as JSON: {error}") from None
    return text.encode()


def _wire_message(message: ChatMessage) -> dict[str, object]:
    """Convert a ChatMessage to the wire format. Reasoning goes back unchanged, as GLM needs."""
    role = _ROLES.get(message.role)
    if role is None:
        raise ValueError(f"a message needs a role, not {ChatRole.Name(message.role)}")
    wire: dict[str, object] = {"role": role, "content": message.content}
    if message.reasoning_content:
        wire["reasoning_content"] = message.reasoning_content
    if message.tool_calls:
        wire["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": _wire_arguments(call)},
            }
            for call in message.tool_calls
        ]
    if message.tool_call_id:
        wire["tool_call_id"] = message.tool_call_id
    return wire


def _wire_arguments(call: ToolCall) -> str:
    """Return the arguments to send back for ``call``: those its tool ran with, a JSON object.

    That is the repaired arguments if there are any, since the original was malformed, and
    otherwise the arguments as the model wrote them. The gateway answers 400 to a history whose
    tool arguments are not JSON (seen 2026-10-15), so arguments that are not a JSON object go
    back as {}, with a warning. The tool result that answers the call tells the model why.
    """
    arguments = call.repaired_arguments_json or call.arguments_json
    try:
        parsed = json.loads(arguments)
    except (ValueError, RecursionError):  # RecursionError: nested too deep to parse.
        parsed = None
    if isinstance(parsed, dict):
        return arguments
    _logger.warning(
        "sent a tool call's arguments back as {}: they are not a JSON object",
        extra={"event": "llm_arguments_replaced", "tool_call_id": call.id, "tool": call.name},
    )
    return "{}"


def _assistant_message(message: _WireMessage) -> ChatMessage:
    """Convert the reply to a ChatMessage: text unchanged, and tool calls raw, for T04's parser."""
    return ChatMessage(
        role=ChatRole.CHAT_ROLE_ASSISTANT,
        content=message.content or "",
        reasoning_content=message.reasoning_content or "",
        tool_calls=[
            ToolCall(
                id=call.id or "",
                name=call.function.name,
                arguments_json=call.function.arguments or "",
                source=ToolCallSource.TOOL_CALL_SOURCE_STANDARD,
            )
            for call in message.tool_calls or ()
        ],
    )


def _error_message(response: httpx.Response) -> str:
    """Return what the gateway said went wrong: LiteLLM's ``error.message``, or else the body."""
    try:
        message = response.json()["error"]["message"]
    except (ValueError, LookupError, TypeError):  # Not JSON, or not shaped as LiteLLM's errors.
        message = None
    return message if isinstance(message, str) else response.text.strip()


def _retry_after_s(headers: httpx.Headers) -> float | None:
    """Return Retry-After in seconds, from a number or an HTTP date; None if none is readable."""
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            moment = parsedate_to_datetime(value)
        except ValueError:
            return None
        if moment.tzinfo is None:  # A "-0000" zone, which RFC 5322 reads as UTC.
            moment = moment.replace(tzinfo=UTC)
        return max((moment - datetime.now(UTC)).total_seconds(), 0.0)
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def _problems(error: ValidationError) -> str:
    """Say what is wrong with a response body, without quoting the body."""
    problems = error.errors(include_input=False)[:3]
    return "; ".join(
        f"{'.'.join(map(str, problem['loc'])) or 'body'}: {problem['msg']}" for problem in problems
    )


def _cut(text: str) -> str:
    return text if len(text) <= _QUOTE_CHARS else text[:_QUOTE_CHARS] + "…"
