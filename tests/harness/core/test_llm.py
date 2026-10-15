"""Tests for goaly.harness.core.llm: the client, against a scripted gateway on httpx.MockTransport.

No test reaches the network. The replies are shaped as the real gateway (LiteLLM 1.92 on SGLang)
sent them on 2026-10-15, and FakeClock makes every backoff take no time.
"""

import asyncio
import inspect
import json
import logging
import math
import random
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from goaly.contracts.gen.llm_call_pb2 import CallPurpose, LlmCallRecord, LlmCallStatus, Usage
from goaly.contracts.gen.trajectory_pb2 import ChatMessage, ChatRole, ToolCall, ToolCallSource
from goaly.contracts.jsonl import read_jsonl, to_json_line
from goaly.harness.core.llm import (
    BoundLlm,
    BoundLlmClient,
    CallTags,
    JsonlCallRecorder,
    LlmClient,
    RetryPolicy,
)
from goaly.harness.core.llm_errors import (
    ClientError,
    ConnectionFailed,
    ContextOverflow,
    InvalidResponse,
    LlmError,
    ModelNotAllowed,
    RateLimited,
    ServerError,
    TimedOut,
)
from goaly.harness.core.llm_models import (
    EXECUTOR_MODEL,
    IMPROVEMENT_MODEL,
    ModelParams,
    ModelProfile,
)
from goaly.harness.core.logs import JsonFormatter
from goaly.harness.core.settings import Settings

pytestmark = pytest.mark.anyio

KEY = "sk-test-0123456789abcdef-never-in-any-output"
BASE_URL = "https://gateway.test/v1"
QWEN = EXECUTOR_MODEL
GLM = IMPROVEMENT_MODEL
EXEC = CallTags(CallPurpose.CALL_PURPOSE_EXEC, run_id="run-1", instance_id="task-7", seed=2)
ANALYST = CallTags(CallPurpose.CALL_PURPOSE_RSI_ANALYST)

S = LlmCallStatus
OK = S.LLM_CALL_STATUS_OK
TIMEOUT = S.LLM_CALL_STATUS_TIMEOUT
RATE_LIMITED = S.LLM_CALL_STATUS_RATE_LIMITED
SERVER_ERROR = S.LLM_CALL_STATUS_SERVER_ERROR
CONNECTION_ERROR = S.LLM_CALL_STATUS_CONNECTION_ERROR
CONTEXT_OVERFLOW = S.LLM_CALL_STATUS_CONTEXT_OVERFLOW
INVALID_RESPONSE = S.LLM_CALL_STATUS_INVALID_RESPONSE
CLIENT_ERROR = S.LLM_CALL_STATUS_CLIENT_ERROR
CANCELLED = S.LLM_CALL_STATUS_CANCELLED

SYSTEM = ChatRole.CHAT_ROLE_SYSTEM
USER = ChatRole.CHAT_ROLE_USER
ASSISTANT = ChatRole.CHAT_ROLE_ASSISTANT
TOOL = ChatRole.CHAT_ROLE_TOOL
STANDARD = ToolCallSource.TOOL_CALL_SOURCE_STANDARD

USAGE: Mapping[str, Any] = {
    "completion_tokens": 309,
    "prompt_tokens": 26,
    "total_tokens": 335,
    "completion_tokens_details": {"reasoning_tokens": 304, "text_tokens": 5},
}
SQL_TOOL: Mapping[str, Any] = {
    "type": "function",
    "function": {
        "name": "execute_sql",
        "description": "Run a SQL query and return its rows.",
        "parameters": {
            "type": "object",
            "properties": {"sql": {"type": "string"}},
            "required": ["sql"],
        },
    },
}


def user(text: str) -> ChatMessage:
    return ChatMessage(role=USER, content=text)


def completion(
    content: str = "391",
    *,
    reasoning: str = "",
    tool_calls: Sequence[Mapping[str, Any]] = (),
    finish_reason: str = "stop",
    usage: Mapping[str, Any] | None = USAGE,
    cost: str | None = "0.0003129",
) -> httpx.Response:
    """A chat completion as the gateway sends one; ``usage=None`` leaves the usage out."""
    message: dict[str, Any] = {"content": content, "role": "assistant"}
    if reasoning:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = [dict(call) for call in tool_calls]
    message["provider_specific_fields"] = {"refusal": None}
    body: dict[str, Any] = {
        "id": "chatcmpl-1",
        "created": 1760500000,
        "model": QWEN,
        "object": "chat.completion",
        "choices": [{"finish_reason": finish_reason, "index": 0, "message": message}],
    }
    if usage is not None:
        body["usage"] = dict(usage)
    headers = {} if cost is None else {"x-litellm-response-cost": cost}
    return httpx.Response(200, json=body, headers=headers)


def failure(
    status: int,
    message: str = "something went wrong",
    *,
    retry_after: str | None = None,
    cost: str = "0",
) -> httpx.Response:
    """An error response as LiteLLM sends one, cost header included."""
    headers = {"x-litellm-response-cost": cost}
    if retry_after is not None:
        headers["retry-after"] = retry_after
    body = {"error": {"message": message, "type": None, "param": None, "code": str(status)}}
    return httpx.Response(status, json=body, headers=headers)


type Handler = Callable[[httpx.Request], Awaitable[httpx.Response]]
type Reply = httpx.Response | Exception | Handler


class Gateway:
    """A model gateway that answers each request with the next of its scripted replies.

    A reply is a response, an exception for the transport to raise, or a coroutine function that
    makes the response, such as one that never answers.
    """

    def __init__(self, *replies: Reply) -> None:
        self.replies = list(replies)
        self.requests: list[httpx.Request] = []

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, httpx.Response):
            return reply
        return await reply(request)

    @property
    def bodies(self) -> list[Any]:
        return [json.loads(request.content) for request in self.requests]


async def hang(request: httpx.Request) -> httpx.Response:
    """A reply that never comes."""
    await asyncio.Event().wait()
    raise AssertionError("unreachable")


class Recorder:
    """A CallRecorder that keeps the records it gets."""

    def __init__(self) -> None:
        self.records: list[LlmCallRecord] = []

    def record(self, record: LlmCallRecord) -> None:
        self.records.append(record)


class FakeClock:
    """A Clock whose sleeps return at once and move its time on by as long."""

    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
        await asyncio.sleep(0)


class TopOfRange(random.Random):
    """Jitter that always picks the top of its range, so the backoff ceilings show."""

    def uniform(self, a: float, b: float) -> float:
        return b


def settings() -> Settings:
    return Settings(api_key=SecretStr(KEY), base_url=BASE_URL)


def connect(
    gateway: Gateway,
    *,
    recorder: Recorder | None = None,
    clock: FakeClock | None = None,
    retry: RetryPolicy | None = None,
    rng: random.Random | None = None,
    profiles: Sequence[ModelProfile] = (),
) -> LlmClient:
    return LlmClient(
        settings(),
        Recorder() if recorder is None else recorder,
        profiles=profiles,
        retry=retry,
        transport=httpx.MockTransport(gateway.handle),
        clock=FakeClock() if clock is None else clock,
        rng=rng,
    )


def attempts(record: LlmCallRecord) -> list[tuple[int, int]]:
    """Each attempt's status and HTTP status."""
    return [(attempt.status, attempt.http_status) for attempt in record.attempts]


# The request: the wire format and the model profiles.


async def test_sends_qwen_its_thinking_switch_and_recommended_sampling() -> None:
    gateway = Gateway(completion())
    async with connect(gateway) as llm:
        await llm.complete([user("What is 17 * 23?")], model=QWEN, tags=EXEC)

    (request,) = gateway.requests
    assert request.method == "POST"
    assert request.url == f"{BASE_URL}/chat/completions"
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert request.headers["content-type"] == "application/json"
    # Exactly these keys: no reasoning_effort, no top-level enable_thinking, no stream.
    assert gateway.bodies == [
        {
            "model": "qwen3.6-35b-a3b",
            "messages": [{"role": "user", "content": "What is 17 * 23?"}],
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 16384,
            "chat_template_kwargs": {"enable_thinking": True},
        }
    ]


async def test_a_call_overrides_the_profile_field_by_field() -> None:
    gateway = Gateway(completion(), completion())
    params = ModelParams(temperature=0.0, max_tokens=512, enable_thinking=False)
    async with connect(gateway) as llm:
        await llm.complete([user("hi")], model=QWEN, tags=EXEC, params=params)
        await llm.complete(
            [user("hi")], model=QWEN, tags=EXEC, params=ModelParams(preserve_thinking=True)
        )

    first, second = gateway.bodies
    assert first == {
        "model": QWEN,
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.0,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": 512,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    assert second["chat_template_kwargs"] == {"enable_thinking": True, "preserve_thinking": True}


async def test_sends_glm_no_thinking_parameter_and_no_sampling_defaults() -> None:
    gateway = Gateway(completion())
    async with connect(gateway) as llm:
        await llm.complete([user("hi")], model=GLM, tags=ANALYST)

    assert gateway.bodies == [{"model": "glm-5.3", "messages": [{"role": "user", "content": "hi"}]}]


async def test_refuses_to_switch_glm_thinking_before_any_io() -> None:
    gateway, recorder = Gateway(), Recorder()
    async with connect(gateway, recorder=recorder) as llm:
        with pytest.raises(ValidationError, match="always thinks"):
            await llm.complete(
                [user("hi")], model=GLM, tags=ANALYST, params=ModelParams(enable_thinking=False)
            )

    assert gateway.requests == []
    assert recorder.records == []


async def test_a_configured_profile_replaces_the_default_for_its_model() -> None:
    profile = ModelProfile(model=QWEN, params=ModelParams(temperature=1.0, enable_thinking=False))
    gateway = Gateway(completion())
    async with connect(gateway, profiles=[profile]) as llm:
        await llm.complete([user("hi")], model=QWEN, tags=EXEC)

    assert gateway.bodies == [
        {
            "model": QWEN,
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 1.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ]


def test_refuses_a_profile_for_a_model_no_purpose_may_use() -> None:
    with pytest.raises(ValueError, match="gpt-4o"):
        LlmClient(settings(), Recorder(), profiles=[ModelProfile(model="gpt-4o")])


async def test_sends_the_tools_as_given() -> None:
    gateway = Gateway(completion())
    async with connect(gateway) as llm:
        await llm.complete([user("Count the orders.")], model=QWEN, tags=EXEC, tools=[SQL_TOOL])

    assert gateway.bodies[0]["tools"] == [SQL_TOOL]


@pytest.mark.parametrize("value", [object(), math.nan], ids=["not-json", "nan"])
async def test_refuses_tools_that_are_not_json_before_any_io(value: object) -> None:
    gateway = Gateway()
    not_json: Any = {"x": value}  # The wrong type is the point.
    async with connect(gateway) as llm:
        with pytest.raises(ValueError, match="JSON"):
            await llm.complete([user("hi")], model=QWEN, tags=EXEC, tools=[not_json])

    assert gateway.requests == []


async def test_converts_each_kind_of_message() -> None:
    history = [
        ChatMessage(role=SYSTEM, content="You make SQL faster."),
        user("Make it faster."),
        ChatMessage(
            role=ASSISTANT,
            content="",
            reasoning_content="Let me look at the plan.",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="execute_sql",
                    arguments_json='{"sql": "EXPLAIN SELECT 1"}',
                    source=STANDARD,
                )
            ],
        ),
        ChatMessage(role=TOOL, tool_call_id="call_1", content="Seq Scan on orders"),
    ]
    gateway = Gateway(completion())
    async with connect(gateway) as llm:
        await llm.complete(history, model=QWEN, tags=EXEC)

    assert gateway.bodies[0]["messages"] == [
        {"role": "system", "content": "You make SQL faster."},
        {"role": "user", "content": "Make it faster."},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "Let me look at the plan.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "execute_sql", "arguments": '{"sql": "EXPLAIN SELECT 1"}'},
                }
            ],
        },
        {"role": "tool", "content": "Seq Scan on orders", "tool_call_id": "call_1"},
    ]


@pytest.mark.parametrize(
    ("written", "repaired", "sent"),
    [
        pytest.param('{"sql": "SELECT 1"}', "", '{"sql": "SELECT 1"}', id="valid"),
        pytest.param(
            '{"sql": "SELECT 1"', '{"sql": "SELECT 1"}', '{"sql": "SELECT 1"}', id="repaired"
        ),
        pytest.param('{"sql": "SELECT', "", "{}", id="unrepaired"),
        pytest.param("[1, 2]", "", "{}", id="not-an-object"),
    ],
)
async def test_sends_back_tool_arguments_the_gateway_accepts(
    written: str, repaired: str, sent: str
) -> None:
    # The gateway answers 400 to a history holding arguments that are not JSON (2026-10-15).
    call = ToolCall(
        id="c1", name="execute_sql", arguments_json=written, repaired_arguments_json=repaired
    )
    history = [user("Count."), ChatMessage(role=ASSISTANT, tool_calls=[call])]
    gateway = Gateway(completion())
    async with connect(gateway) as llm:
        await llm.complete(history, model=QWEN, tags=EXEC)

    assert gateway.bodies[0]["messages"][1]["tool_calls"][0]["function"]["arguments"] == sent


async def test_refuses_a_message_without_a_role_before_any_io() -> None:
    gateway = Gateway()
    async with connect(gateway) as llm:
        with pytest.raises(ValueError, match="role"):
            await llm.complete([ChatMessage(content="hi")], model=QWEN, tags=EXEC)

    assert gateway.requests == []


# The reply.


async def test_returns_content_reasoning_and_raw_tool_calls_unchanged() -> None:
    truncated = '{"sql": "SELECT COUNT(*) FROM orders;"'  # Malformed: the parser's job (T04).
    call = {
        "index": 0,
        "function": {"arguments": truncated, "name": "execute_sql"},
        "id": "call_c779e04186054799bd2545cf",
        "type": "function",
    }
    reasoning = "Thinking Process:\n1.  **Identify the request**  \n"
    gateway = Gateway(
        completion("\n\n", reasoning=reasoning, tool_calls=[call], finish_reason="tool_calls")
    )
    async with connect(gateway) as llm:
        reply = await llm.complete([user("Count the orders.")], model=QWEN, tags=EXEC)

    assert reply.message == ChatMessage(
        role=ASSISTANT,
        content="\n\n",
        reasoning_content=reasoning,
        tool_calls=[
            ToolCall(
                id="call_c779e04186054799bd2545cf",
                name="execute_sql",
                arguments_json=truncated,
                source=STANDARD,
            )
        ],
    )
    assert reply.finish_reason == "tool_calls"


async def test_a_plain_answer_has_no_reasoning_and_no_tool_calls() -> None:
    gateway = Gateway(completion("Hi", finish_reason="length"))
    async with connect(gateway) as llm:
        reply = await llm.complete([user("hi")], model=QWEN, tags=EXEC)

    assert reply.message == ChatMessage(role=ASSISTANT, content="Hi")
    assert reply.finish_reason == "length"


async def test_glm_gets_its_reasoning_back_unchanged() -> None:
    reasoning = "17 * 23 = 391\n\nLet me verify:  17 * 20 + 17 * 3 = 391 \n"
    gateway = Gateway(completion("391", reasoning=reasoning), completion("392"))
    async with connect(gateway) as llm:
        history = [user("What is 17 * 23?")]
        first = await llm.complete(history, model=GLM, tags=ANALYST)
        history += [first.message, user("Now add 1.")]
        await llm.complete(history, model=GLM, tags=ANALYST)

    assert gateway.bodies[1]["messages"] == [
        {"role": "user", "content": "What is 17 * 23?"},
        {"role": "assistant", "content": "391", "reasoning_content": reasoning},
        {"role": "user", "content": "Now add 1."},
    ]


# Usage and cost.


async def test_reads_the_tokens_reasoning_tokens_and_cost() -> None:
    gateway = Gateway(completion())
    async with connect(gateway) as llm:
        reply = await llm.complete([user("hi")], model=QWEN, tags=EXEC)

    assert reply.usage == Usage(
        input_tokens=26, output_tokens=309, reasoning_tokens=304, cost_usd=0.0003129
    )
    assert reply.record.usage == reply.usage


@pytest.mark.parametrize(
    "usage",
    [
        # As the gateway answers Qwen with thinking off: no completion_tokens_details.
        pytest.param({"completion_tokens": 2, "prompt_tokens": 18}, id="no-details"),
        pytest.param(
            {"completion_tokens": 2, "prompt_tokens": 18, "completion_tokens_details": None},
            id="null-details",
        ),
        pytest.param(
            {
                "completion_tokens": 2,
                "prompt_tokens": 18,
                "completion_tokens_details": {"reasoning_tokens": None},
            },
            id="null-count",
        ),
    ],
)
async def test_leaves_reasoning_tokens_unset_when_none_are_reported(usage: dict[str, Any]) -> None:
    gateway = Gateway(completion("Hi", usage=usage))
    async with connect(gateway) as llm:
        reply = await llm.complete([user("hi")], model=QWEN, tags=EXEC)

    assert (reply.usage.input_tokens, reply.usage.output_tokens) == (18, 2)
    assert not reply.usage.HasField("reasoning_tokens")


async def test_leaves_the_cost_unset_without_the_header() -> None:
    gateway = Gateway(completion(cost=None))
    async with connect(gateway) as llm:
        reply = await llm.complete([user("hi")], model=QWEN, tags=EXEC)

    assert not reply.usage.HasField("cost_usd")
    assert reply.usage.output_tokens == 309


async def test_reads_a_cost_in_exponent_notation() -> None:
    gateway = Gateway(completion(cost="4.7e-06"))
    async with connect(gateway) as llm:
        reply = await llm.complete([user("hi")], model=QWEN, tags=EXEC)

    assert reply.usage.cost_usd == 4.7e-06


@pytest.mark.parametrize("value", ["free", "-0.01", "nan", "inf"])
async def test_ignores_a_malformed_cost_with_a_warning(
    value: str, caplog: pytest.LogCaptureFixture
) -> None:
    gateway = Gateway(completion(cost=value))
    async with connect(gateway) as llm:
        reply = await llm.complete([user("hi")], model=QWEN, tags=EXEC)

    assert not reply.usage.HasField("cost_usd")
    (warning,) = [entry for entry in caplog.records if entry.levelno == logging.WARNING]
    assert getattr(warning, "event", None) == "llm_bad_cost"


async def test_sums_the_cost_of_every_attempt() -> None:
    gateway = Gateway(failure(503, cost="0.0001"), completion(cost="0.0002"))
    async with connect(gateway) as llm:
        reply = await llm.complete([user("hi")], model=QWEN, tags=EXEC)

    assert reply.usage.cost_usd == pytest.approx(0.0003)


async def test_a_failed_call_keeps_the_cost_its_attempts_reported() -> None:
    recorder = Recorder()
    gateway = Gateway(failure(400, "bad request", cost="0"))
    async with connect(gateway, recorder=recorder) as llm:
        with pytest.raises(ClientError):
            await llm.complete([user("hi")], model=QWEN, tags=EXEC)

    (record,) = recorder.records
    assert record.usage.HasField("cost_usd")
    assert record.usage == Usage(cost_usd=0.0)


async def test_counts_no_tokens_and_warns_when_no_usage_is_reported(
    caplog: pytest.LogCaptureFixture,
) -> None:
    gateway = Gateway(completion(usage=None))
    async with connect(gateway) as llm:
        reply = await llm.complete([user("hi")], model=QWEN, tags=EXEC)

    assert reply.usage == Usage(cost_usd=0.0003129)
    (warning,) = [entry for entry in caplog.records if entry.levelno == logging.WARNING]
    assert getattr(warning, "event", None) == "llm_no_usage"
    assert getattr(warning, "call_id", None) == reply.record.call_id


# Retries and backoff.

TRANSIENT = [
    pytest.param(lambda: failure(429), RATE_LIMITED, 429, id="429"),
    # LiteLLM answers 408 when its own upstream request timed out.
    pytest.param(lambda: failure(408, "Request Timeout"), TIMEOUT, 408, id="408"),
    pytest.param(lambda: failure(500), SERVER_ERROR, 500, id="500"),
    pytest.param(lambda: failure(502), SERVER_ERROR, 502, id="502"),
    pytest.param(lambda: failure(503), SERVER_ERROR, 503, id="503"),
    pytest.param(lambda: httpx.ReadTimeout("timed out"), TIMEOUT, 0, id="read-timeout"),
    pytest.param(lambda: httpx.ConnectError("refused"), CONNECTION_ERROR, 0, id="connect"),
    pytest.param(
        lambda: httpx.RemoteProtocolError("server disconnected"),
        CONNECTION_ERROR,
        0,
        id="disconnect",
    ),
]


@pytest.mark.parametrize(("make", "status", "http_status"), TRANSIENT)
async def test_retries_a_transient_failure(
    make: Callable[[], Reply], status: LlmCallStatus.ValueType, http_status: int
) -> None:
    recorder = Recorder()
    gateway = Gateway(make(), completion())
    async with connect(gateway, recorder=recorder) as llm:
        reply = await llm.complete([user("hi")], model=QWEN, tags=EXEC)

    assert attempts(reply.record) == [(status, http_status), (OK, 200)]
    assert (reply.record.status, reply.record.http_status) == (OK, 200)
    assert recorder.records == [reply.record]
    assert len(gateway.bodies) == 2
    assert gateway.bodies[0] == gateway.bodies[1]


GIVE_UP = [
    pytest.param(lambda: failure(429), RateLimited, RATE_LIMITED, 429, id="429"),
    pytest.param(lambda: failure(408), TimedOut, TIMEOUT, 408, id="408"),
    pytest.param(lambda: failure(502), ServerError, SERVER_ERROR, 502, id="502"),
    pytest.param(lambda: httpx.ReadTimeout("timed out"), TimedOut, TIMEOUT, 0, id="timeout"),
    pytest.param(
        lambda: httpx.ConnectError("refused"), ConnectionFailed, CONNECTION_ERROR, 0, id="connect"
    ),
]


@pytest.mark.parametrize(("make", "error", "status", "http_status"), GIVE_UP)
async def test_gives_up_after_five_retries(
    make: Callable[[], Reply],
    error: type[LlmError],
    status: LlmCallStatus.ValueType,
    http_status: int,
) -> None:
    recorder, clock = Recorder(), FakeClock()
    gateway = Gateway(*(make() for _ in range(6)))
    async with connect(gateway, recorder=recorder, clock=clock) as llm:
        with pytest.raises(error, match=r"\(6 attempts\)") as raised:
            await llm.complete([user("hi")], model=QWEN, tags=EXEC)

    (record,) = recorder.records
    assert raised.value.record == record
    assert len(gateway.requests) == 6
    assert attempts(record) == [(status, http_status)] * 6
    assert (record.status, record.http_status) == (status, http_status)
    assert len(clock.sleeps) == 5


async def test_retries_as_often_as_the_policy_says() -> None:
    gateway = Gateway(failure(503), failure(503))
    async with connect(gateway, retry=RetryPolicy(max_retries=1)) as llm:
        with pytest.raises(ServerError, match=r"\(2 attempts\)"):
            await llm.complete([user("hi")], model=QWEN, tags=EXEC)

    assert len(gateway.requests) == 2


def test_the_retry_policy_defaults_are_the_specs() -> None:
    policy = RetryPolicy()

    assert (policy.timeout_s, policy.max_retries) == (300.0, 5)
    assert (policy.backoff_base_s, policy.backoff_cap_s) == (1.0, 60.0)


class RecordingJitter(random.Random):
    """Jitter that notes each range it is asked to pick from, and picks its middle."""

    def __init__(self) -> None:
        super().__init__()
        self.ranges: list[tuple[float, float]] = []

    def uniform(self, a: float, b: float) -> float:
        self.ranges.append((a, b))
        return (a + b) / 2


async def test_backs_off_with_full_jitter_below_a_doubling_ceiling() -> None:
    clock, jitter = FakeClock(), RecordingJitter()
    gateway = Gateway(*(failure(503) for _ in range(9)))
    async with connect(gateway, clock=clock, retry=RetryPolicy(max_retries=8), rng=jitter) as llm:
        with pytest.raises(ServerError):
            await llm.complete([user("hi")], model=QWEN, tags=EXEC)

    # Anywhere from 0 to a ceiling that starts at 1 s, doubles, and stops at 60 s.
    ceilings = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0]
    assert jitter.ranges == [(0.0, ceiling) for ceiling in ceilings]
    assert clock.sleeps == [ceiling / 2 for ceiling in ceilings]


async def test_backoff_jitter_is_full() -> None:
    clock = FakeClock()
    gateway = Gateway(*(failure(503) for _ in range(6)))
    rng = random.Random(7)  # noqa: S311 - a seeded jitter source, not security.
    async with connect(gateway, clock=clock, rng=rng) as llm:
        with pytest.raises(ServerError):
            await llm.complete([user("hi")], model=QWEN, tags=EXEC)

    ceilings = [1, 2, 4, 8, 16]
    assert all(0 <= slept <= ceiling for slept, ceiling in zip(clock.sleeps, ceilings, strict=True))
    assert clock.sleeps != ceilings


async def test_waits_as_long_as_retry_after_says() -> None:
    clock = FakeClock()
    gateway = Gateway(failure(429, retry_after="7"), completion())
    async with connect(gateway, clock=clock) as llm:
        reply = await llm.complete([user("hi")], model=QWEN, tags=EXEC)

    assert clock.sleeps == [7.0]
    first, second = reply.record.attempts
    assert first.retry_after_s == 7.0
    assert not second.HasField("retry_after_s")
    assert (first.wait_s, second.wait_s) == (0.0, 7.0)


async def test_honors_retry_after_on_a_5xx_too() -> None:
    clock = FakeClock()
    gateway = Gateway(failure(503, retry_after="2.5"), completion())
    async with connect(gateway, clock=clock) as llm:
        await llm.complete([user("hi")], model=QWEN, tags=EXEC)

    assert clock.sleeps == [2.5]


@pytest.mark.parametrize(
    "zone",
    [
        pytest.param("GMT", id="gmt"),
        pytest.param("-0000", id="no-zone"),  # RFC 5322 reads it as UTC.
    ],
)
async def test_reads_retry_after_as_an_http_date(zone: str) -> None:
    clock = FakeClock()
    moment = datetime.now(UTC) + timedelta(seconds=30)
    text = format_datetime(moment, usegmt=True).replace("GMT", zone)
    gateway = Gateway(failure(429, retry_after=text), completion())
    async with connect(gateway, clock=clock) as llm:
        reply = await llm.complete([user("hi")], model=QWEN, tags=EXEC)

    (slept,) = clock.sleeps
    assert 28 < slept <= 30  # HTTP dates have whole seconds.
    assert reply.record.attempts[0].retry_after_s == slept


async def test_a_retry_after_in_the_past_means_retry_at_once() -> None:
    clock = FakeClock()
    moment = datetime.now(UTC) - timedelta(minutes=5)
    gateway = Gateway(failure(429, retry_after=format_datetime(moment, usegmt=True)), completion())
    async with connect(gateway, clock=clock) as llm:
        await llm.complete([user("hi")], model=QWEN, tags=EXEC)

    assert clock.sleeps == []


async def test_waits_no_longer_than_the_cap_whatever_retry_after_says() -> None:
    clock = FakeClock()
    gateway = Gateway(failure(429, retry_after="3600"), completion())
    async with connect(gateway, clock=clock) as llm:
        reply = await llm.complete([user("hi")], model=QWEN, tags=EXEC)

    assert clock.sleeps == [60.0]
    assert reply.record.attempts[0].retry_after_s == 3600.0


@pytest.mark.parametrize("value", ["soon", "-5", "nan", ""])
async def test_backs_off_with_jitter_when_retry_after_is_malformed(value: str) -> None:
    clock = FakeClock()
    gateway = Gateway(failure(429, retry_after=value), completion())
    async with connect(gateway, clock=clock, rng=TopOfRange()) as llm:
        reply = await llm.complete([user("hi")], model=QWEN, tags=EXEC)

    assert clock.sleeps == [1.0]
    assert not reply.record.attempts[0].HasField("retry_after_s")


async def test_the_system_clock_really_waits() -> None:
    gateway = Gateway(failure(429, retry_after="0.05"), completion())
    async with LlmClient(
        settings(), Recorder(), transport=httpx.MockTransport(gateway.handle)
    ) as llm:
        reply = await llm.complete([user("hi")], model=QWEN, tags=EXEC)

    assert reply.record.attempts[1].wait_s >= 0.05
    assert reply.record.latency_s >= reply.record.wait_s


LOG_FIELDS = [
    "model",
    "purpose",
    "call_id",
    "attempt",
    "after",
    "http_status",
    "detail",
    "delay_s",
    "retry_after_s",
]


async def test_logs_each_backoff_with_the_purpose_and_model(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="goaly.harness.core.llm")
    gateway = Gateway(failure(429, "slow down", retry_after="2"), completion())
    async with connect(gateway) as llm:
        reply = await llm.complete([user("hi")], model=QWEN, tags=EXEC)

    (entry,) = [entry for entry in caplog.records if getattr(entry, "event", None) == "llm_backoff"]
    assert entry.levelno == logging.INFO
    assert {key: getattr(entry, key) for key in LOG_FIELDS} == {
        "model": QWEN,
        "purpose": "CALL_PURPOSE_EXEC",
        "call_id": reply.record.call_id,
        "attempt": 2,
        "after": "LLM_CALL_STATUS_RATE_LIMITED",
        "http_status": 429,
        "detail": "HTTP 429: slow down",
        "delay_s": 2.0,
        "retry_after_s": 2.0,
    }


# Failures that are not retried.

CONTEXT_OVERFLOWS = [
    # The prompt alone is too long: LiteLLM marks it ContextWindowExceededError.
    "litellm.ContextWindowExceededError: litellm.BadRequestError: ContextWindowExceededError: "
    "OpenAIException - The input (2333794 tokens) is longer than the model's context length "
    "(1010000 tokens).",
    # The prompt fits, but not with max_tokens: LiteLLM passes SGLang's message on as it is.
    "litellm.BadRequestError: OpenAIException - Requested token count exceeds the model's "
    "maximum context length of 1010000 tokens. You requested a total of 1016018 tokens: 999634 "
    "tokens from the input messages and 16384 tokens for the completion.",
]


@pytest.mark.parametrize("message", CONTEXT_OVERFLOWS)
async def test_raises_context_overflow_without_retrying(message: str) -> None:
    gateway = Gateway(failure(400, message))
    async with connect(gateway) as llm:
        with pytest.raises(ContextOverflow, match="1010000 tokens") as raised:
            await llm.complete([user("hi")], model=QWEN, tags=EXEC)

    assert len(gateway.requests) == 1
    assert raised.value.record is not None
    assert attempts(raised.value.record) == [(CONTEXT_OVERFLOW, 400)]


@pytest.mark.parametrize(
    ("status", "message"),
    [
        (400, "litellm.UnsupportedParamsError: openai does not support parameters: ['thinking']"),
        (400, "Assistant tool call function.arguments must be valid JSON."),
        (401, "Authentication Error"),
        (404, "Not Found"),
        (422, "Unprocessable Entity"),
    ],
)
async def test_raises_client_error_without_retrying(status: int, message: str) -> None:
    gateway = Gateway(failure(status, message))
    async with connect(gateway) as llm:
        with pytest.raises(ClientError, match=re.escape(f"HTTP {status}: {message}")) as raised:
            await llm.complete([user("hi")], model=QWEN, tags=EXEC)

    assert len(gateway.requests) == 1
    assert raised.value.record is not None
    assert attempts(raised.value.record) == [(CLIENT_ERROR, status)]


async def test_a_request_httpx_refuses_to_send_is_not_retried_or_quoted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # h11's error for an illegal header quotes the header, which holds the key.
    refused = httpx.LocalProtocolError(f"Illegal header value b'Bearer {KEY}\\n'")
    gateway = Gateway(refused, completion())
    async with connect(gateway) as llm:
        with pytest.raises(ClientError, match="refused to send the request") as raised:
            await llm.complete([user("hi")], model=QWEN, tags=EXEC)

    assert len(gateway.requests) == 1
    assert raised.value.record is not None
    assert attempts(raised.value.record) == [(CLIENT_ERROR, 0)]
    assert KEY not in str(raised.value)
    assert all(KEY not in JsonFormatter().format(entry) for entry in caplog.records)


INVALID = [
    pytest.param(lambda: httpx.Response(200, text="<html>Bad gateway</html>"), 200, id="not-json"),
    pytest.param(lambda: httpx.Response(200, json=[1, 2]), 200, id="not-an-object"),
    pytest.param(lambda: httpx.Response(200, json={"choices": []}), 200, id="no-choices"),
    pytest.param(
        lambda: httpx.Response(200, json={"choices": [{"message": {"content": 5}}]}),
        200,
        id="wrong-type",
    ),
    pytest.param(
        lambda: httpx.Response(302, headers={"location": "https://elsewhere.test/"}),
        302,
        id="redirect",
    ),
    pytest.param(lambda: httpx.DecodingError("bad gzip"), 0, id="undecodable"),
]


@pytest.mark.parametrize(("make", "http_status"), INVALID)
async def test_raises_invalid_response_without_retrying(
    make: Callable[[], Reply], http_status: int
) -> None:
    gateway = Gateway(make())
    async with connect(gateway) as llm:
        with pytest.raises(InvalidResponse) as raised:
            await llm.complete([user("hi")], model=QWEN, tags=EXEC)

    assert len(gateway.requests) == 1
    assert raised.value.record is not None
    assert attempts(raised.value.record) == [(INVALID_RESPONSE, http_status)]


async def test_an_error_says_what_the_gateway_said() -> None:
    gateway = Gateway(failure(400, "openai does not support parameters: ['reasoning_effort']"))
    async with connect(gateway) as llm:
        with pytest.raises(ClientError) as raised:
            await llm.complete([user("hi")], model=GLM, tags=ANALYST)

    assert str(raised.value) == (
        "glm-5.3: HTTP 400: openai does not support parameters: ['reasoning_effort'] (1 attempt)"
    )


async def test_an_error_body_that_is_not_json_is_quoted_as_text() -> None:
    gateway = Gateway(httpx.Response(403, text="Forbidden by the proxy"))
    async with connect(gateway) as llm:
        with pytest.raises(ClientError, match="HTTP 403: Forbidden by the proxy"):
            await llm.complete([user("hi")], model=QWEN, tags=EXEC)


async def test_cuts_a_long_gateway_message_short() -> None:
    gateway = Gateway(failure(400, "x" * 5000))
    async with connect(gateway) as llm:
        with pytest.raises(ClientError) as raised:
            await llm.complete([user("hi")], model=QWEN, tags=EXEC)

    assert len(str(raised.value)) < 600


# Timeouts and cancellation.


async def test_times_out_an_attempt_that_takes_too_long() -> None:
    gateway = Gateway(hang)
    async with connect(gateway, retry=RetryPolicy(timeout_s=0.05, max_retries=0)) as llm:
        with pytest.raises(TimedOut, match=r"no response within 0\.05 s \(1 attempt\)") as raised:
            await llm.complete([user("hi")], model=QWEN, tags=EXEC)

    assert raised.value.record is not None
    assert attempts(raised.value.record) == [(TIMEOUT, 0)]


async def test_retries_an_attempt_that_timed_out() -> None:
    gateway = Gateway(hang, completion())
    async with connect(gateway, retry=RetryPolicy(timeout_s=0.05)) as llm:
        reply = await llm.complete([user("hi")], model=QWEN, tags=EXEC)

    assert attempts(reply.record) == [(TIMEOUT, 0), (OK, 200)]


async def test_cancellation_propagates_at_once_and_is_recorded() -> None:
    recorder, arrived = Recorder(), asyncio.Event()

    async def hold(request: httpx.Request) -> httpx.Response:
        arrived.set()
        return await hang(request)

    gateway = Gateway(hold, completion())
    async with connect(gateway, recorder=recorder) as llm:
        task = asyncio.create_task(llm.complete([user("hi")], model=QWEN, tags=EXEC))
        await arrived.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    (record,) = recorder.records
    assert attempts(record) == [(CANCELLED, 0)]
    assert (record.status, record.http_status) == (CANCELLED, 0)
    assert len(gateway.requests) == 1


async def test_a_call_cancelled_during_its_backoff_is_not_retried() -> None:
    recorder, backing_off = Recorder(), asyncio.Event()

    class StuckClock(FakeClock):
        async def sleep(self, seconds: float) -> None:
            backing_off.set()
            await asyncio.Event().wait()

    gateway = Gateway(failure(429), completion())
    async with connect(gateway, recorder=recorder, clock=StuckClock()) as llm:
        task = asyncio.create_task(llm.complete([user("hi")], model=QWEN, tags=EXEC))
        await backing_off.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    (record,) = recorder.records
    assert attempts(record) == [(RATE_LIMITED, 429), (CANCELLED, 0)]
    assert record.attempts[1].latency_s == 0.0  # Its request was never sent.
    assert len(gateway.requests) == 1


# Binding a model and tags (spec ruling V3): what a task's agent loop gets.


async def test_a_bound_client_calls_its_model_with_its_tags() -> None:
    recorder = Recorder()
    gateway = Gateway(completion())
    async with connect(gateway, recorder=recorder) as llm:
        bound: BoundLlm = llm.bind(model=QWEN, tags=EXEC)
        reply = await bound.complete(
            [user("hi")], tools=[SQL_TOOL], params=ModelParams(max_tokens=512)
        )

    record = reply.record
    assert (record.model, record.purpose) == (QWEN, CallPurpose.CALL_PURPOSE_EXEC)
    assert (record.run_id, record.instance_id, record.seed) == ("run-1", "task-7", 2)
    assert recorder.records == [record]
    (body,) = gateway.bodies
    assert (body["model"], body["max_tokens"], body["tools"]) == (QWEN, 512, [SQL_TOOL])


def test_a_bound_client_offers_no_way_to_choose_its_purpose_or_model() -> None:
    public = [name for name in dir(BoundLlmClient) if not name.startswith("_")]
    parameters = list(inspect.signature(BoundLlmClient.complete).parameters)

    assert public == ["complete"]
    assert parameters == ["self", "messages", "tools", "params"]


@pytest.mark.parametrize(
    ("model", "tags"),
    [pytest.param(GLM, EXEC, id="exec-glm"), pytest.param(QWEN, ANALYST, id="analyst-qwen")],
)
async def test_binding_a_pair_the_allowlist_refuses_raises_before_any_io(
    model: str, tags: CallTags
) -> None:
    gateway, recorder = Gateway(), Recorder()
    async with connect(gateway, recorder=recorder) as llm:
        with pytest.raises(ModelNotAllowed, match=f"may not call {model}"):
            llm.bind(model=model, tags=tags)

    assert (gateway.requests, recorder.records) == ([], [])


async def test_bound_clients_call_through_the_one_client_that_bound_them() -> None:
    recorder = Recorder()
    gateway = Gateway(completion(), completion())
    llm = connect(gateway, recorder=recorder)
    first = llm.bind(model=QWEN, tags=EXEC)
    second = llm.bind(model=QWEN, tags=CallTags(CallPurpose.CALL_PURPOSE_EXEC, instance_id="x"))
    await first.complete([user("hi")])
    await second.complete([user("hi")])

    await llm.aclose()  # Its connections are theirs too: neither has any of its own.
    with pytest.raises(RuntimeError, match="closed"):
        await first.complete([user("hi")])

    assert [record.instance_id for record in recorder.records] == ["task-7", "x", "task-7"]


# The model allowlist.


@pytest.mark.parametrize(
    ("model", "tags"),
    [
        pytest.param(GLM, EXEC, id="exec-glm"),
        pytest.param(QWEN, ANALYST, id="analyst-qwen"),
        pytest.param(QWEN, CallTags(CallPurpose.CALL_PURPOSE_RSI_IMPL), id="impl-qwen"),
        pytest.param("gpt-4o", EXEC, id="exec-other"),
        pytest.param(QWEN, CallTags(CallPurpose.CALL_PURPOSE_UNSPECIFIED), id="no-purpose"),
    ],
)
async def test_refuses_a_model_the_purpose_may_not_use_before_any_io(
    model: str, tags: CallTags, caplog: pytest.LogCaptureFixture
) -> None:
    gateway, recorder = Gateway(), Recorder()
    async with connect(gateway, recorder=recorder) as llm:
        with pytest.raises(ModelNotAllowed, match=f"may not call {model}") as raised:
            await llm.complete([user("hi")], model=model, tags=tags)

    assert gateway.requests == []
    assert recorder.records == []
    assert raised.value.record is None
    (warning,) = [entry for entry in caplog.records if entry.levelno == logging.WARNING]
    assert getattr(warning, "event", None) == "llm_model_not_allowed"
    assert (getattr(warning, "model", None), getattr(warning, "purpose", None)) == (
        model,
        CallPurpose.Name(tags.purpose),
    )


# Records.


async def test_records_what_the_call_was_for_and_when_it_started() -> None:
    gateway = Gateway(completion())
    before = datetime.now(UTC)
    async with connect(gateway) as llm:
        reply = await llm.complete([user("hi")], model=QWEN, tags=EXEC)
    after = datetime.now(UTC)

    record = reply.record
    assert len(record.call_id) == 32
    assert int(record.call_id, 16) >= 0  # Hex.
    assert (record.run_id, record.instance_id, record.seed) == ("run-1", "task-7", 2)
    assert (record.purpose, record.model) == (CallPurpose.CALL_PURPOSE_EXEC, QWEN)
    assert before <= record.started_at.ToDatetime(tzinfo=UTC) <= after
    assert record.started_at.ToDatetime(tzinfo=UTC) <= (
        record.attempts[0].started_at.ToDatetime(tzinfo=UTC)
    )


async def test_gives_every_call_its_own_id() -> None:
    gateway = Gateway(completion(), completion())
    async with connect(gateway) as llm:
        first = await llm.complete([user("hi")], model=QWEN, tags=EXEC)
        second = await llm.complete([user("hi")], model=QWEN, tags=EXEC)

    assert first.record.call_id != second.record.call_id


async def test_times_each_attempt_and_the_whole_call() -> None:
    clock = FakeClock()

    async def slow(request: httpx.Request) -> httpx.Response:
        clock.now += 1.5  # The request takes 1.5 s.
        return completion()

    gateway = Gateway(failure(429, retry_after="3"), slow)
    async with connect(gateway, clock=clock) as llm:
        reply = await llm.complete([user("hi")], model=QWEN, tags=EXEC)

    first, second = reply.record.attempts
    assert (first.wait_s, first.latency_s) == (0.0, 0.0)
    assert (second.wait_s, second.latency_s) == (3.0, 1.5)
    assert (reply.record.wait_s, reply.record.latency_s) == (3.0, 4.5)


async def test_the_recorder_gets_its_own_copy() -> None:
    recorder = Recorder()
    gateway = Gateway(completion())
    async with connect(gateway, recorder=recorder) as llm:
        reply = await llm.complete([user("hi")], model=QWEN, tags=EXEC)

    reply.record.model = "changed"
    assert recorder.records[0].model == QWEN


async def test_a_bug_still_leaves_exactly_one_record() -> None:
    recorder = Recorder()
    gateway = Gateway(RuntimeError("a bug in the transport"))
    async with connect(gateway, recorder=recorder) as llm:
        with pytest.raises(RuntimeError, match="a bug in the transport"):
            await llm.complete([user("hi")], model=QWEN, tags=EXEC)

    (record,) = recorder.records
    assert record.status == S.LLM_CALL_STATUS_UNSPECIFIED
    assert len(record.attempts) == 1


async def test_jsonl_recorder_appends_one_line_per_call(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    gateway = Gateway(completion(), failure(400, "bad request"))
    async with LlmClient(
        settings(), JsonlCallRecorder(path), transport=httpx.MockTransport(gateway.handle)
    ) as llm:
        reply = await llm.complete([user("hi")], model=QWEN, tags=EXEC)
        with pytest.raises(ClientError) as raised:
            await llm.complete([user("hi")], model=GLM, tags=ANALYST)

    assert list(read_jsonl(path, LlmCallRecord)) == [reply.record, raised.value.record]


# Keeping the key secret.


async def test_the_key_is_in_no_error_log_line_repr_or_record(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    recorder = Recorder()
    leak = f"Authentication Error, Received API Key = {KEY}"
    gateway = Gateway(
        failure(503, leak), httpx.ConnectError(f"refused for {KEY}"), failure(401, leak)
    )
    async with connect(gateway, recorder=recorder) as llm:
        with pytest.raises(ClientError) as raised:
            await llm.complete([user("hi")], model=QWEN, tags=EXEC)
        texts = [
            str(raised.value),
            repr(raised.value),
            repr(llm),
            *(JsonFormatter().format(entry) for entry in caplog.records),
            *(to_json_line(record) for record in recorder.records),
        ]

    assert len(caplog.records) >= 2  # Both backoffs were logged, with what the gateway said.
    assert all(KEY not in text for text in texts)
    assert "Received API Key = [redacted]" in str(raised.value)
