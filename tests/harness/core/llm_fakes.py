"""A scripted model gateway and the other doubles the LLM client tests share.

``Gateway`` answers each request with the next of its scripted replies, shaped as the real
gateway (LiteLLM 1.92 on SGLang) sent them on 2026-10-15. It runs behind httpx.MockTransport, so
no test reaches the network. ``FakeClock`` makes every wait take no time, and ``FakeLLM``
injects faults at random, for the stress test.

pytest puts the repository root on sys.path, so tests import these as
``tests.harness.core.llm_fakes``.
"""

import asyncio
import json
import random
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

import httpx
from pydantic import SecretStr

from goaly.contracts.gen.llm_call_pb2 import CallPurpose, LlmCallRecord
from goaly.contracts.gen.trajectory_pb2 import ChatMessage, ChatRole
from goaly.harness.core.llm import CallRecorder, CallTags, LlmClient, RetryPolicy
from goaly.harness.core.llm_models import EXECUTOR_MODEL, IMPROVEMENT_MODEL, ModelProfile
from goaly.harness.core.settings import Settings

KEY = "sk-test-0123456789abcdef-never-in-any-output"
BASE_URL = "https://gateway.test/v1"
QWEN = EXECUTOR_MODEL
GLM = IMPROVEMENT_MODEL
EXEC = CallTags(CallPurpose.CALL_PURPOSE_EXEC, run_id="run-1", instance_id="task-7", seed=2)
ANALYST = CallTags(CallPurpose.CALL_PURPOSE_RSI_ANALYST)

USAGE: Mapping[str, Any] = {
    "completion_tokens": 309,
    "prompt_tokens": 26,
    "total_tokens": 335,
    "completion_tokens_details": {"reasoning_tokens": 304, "text_tokens": 5},
}


def user(text: str) -> ChatMessage:
    return ChatMessage(role=ChatRole.CHAT_ROLE_USER, content=text)


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
    recorder: CallRecorder | None = None,
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


class FakeLLM:
    """A gateway that answers after a random latency and injects faults: 429s and timeouts.

    Over its first ``storm_requests`` requests, a request gets a 429 with probability
    ``rate_limited``, or else, with probability ``timeouts``, hangs for ``hang_s``: longer than
    the client's timeout. After the storm it only answers, still after a random latency. It counts
    each model's requests in flight, so a test can check the client never went over its share.
    """

    def __init__(
        self,
        *,
        seed: int,
        latency_s: float,
        hang_s: float,
        rate_limited: float,
        timeouts: float,
        storm_requests: int,
    ) -> None:
        self._rng = random.Random(seed)  # noqa: S311 - simulated faults, not security.
        self._latency_s = latency_s
        self._hang_s = hang_s
        self._rate_limited = rate_limited
        self._timeouts = timeouts
        self._storm_requests = storm_requests
        self.requests = 0
        self.in_flight: Counter[str] = Counter()
        self.max_in_flight: dict[str, int] = {}

    async def handle(self, request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        self.requests += 1
        roll = self._rng.random() if self.requests <= self._storm_requests else 1.0
        latency = self._rng.uniform(0.0, self._latency_s)
        self.in_flight[model] += 1
        self.max_in_flight[model] = max(self.max_in_flight.get(model, 0), self.in_flight[model])
        try:
            if roll < self._rate_limited:
                await asyncio.sleep(latency)
                return failure(429)
            if roll < self._rate_limited + self._timeouts:
                await asyncio.sleep(self._hang_s)
            await asyncio.sleep(latency)
            return completion()
        finally:
            self.in_flight[model] -= 1
