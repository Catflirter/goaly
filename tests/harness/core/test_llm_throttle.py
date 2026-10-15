"""Tests for goaly.harness.core.llm_throttle: slots, AIMD and the circuit breaker.

The unit tests drive a ModelThrottle directly, on a FakeClock. The tests at the end run the
client: waits counted into the record, one throttle per model, and a stress test of 500 calls
against a FakeLLM that injects 429s, latency and timeouts.
"""

import asyncio
import logging
from contextlib import AsyncExitStack
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from goaly.contracts.gen.llm_call_pb2 import CallPurpose, LlmCallRecord, LlmCallStatus
from goaly.harness.core.llm import CallTags, LlmClient, LlmReply, RetryPolicy
from goaly.harness.core.llm_errors import LlmError, ServerError
from goaly.harness.core.llm_throttle import DEFAULT_SHARES, ModelThrottle, ThrottlePolicy
from tests.harness.core.llm_fakes import (
    ANALYST,
    EXEC,
    GLM,
    QWEN,
    FakeClock,
    FakeLLM,
    Gateway,
    Recorder,
    TopOfRange,
    completion,
    failure,
    settings,
    user,
)

pytestmark = pytest.mark.anyio

S = LlmCallStatus
OK = S.LLM_CALL_STATUS_OK
RATE_LIMITED = S.LLM_CALL_STATUS_RATE_LIMITED
SERVER_ERROR = S.LLM_CALL_STATUS_SERVER_ERROR
TIMEOUT = S.LLM_CALL_STATUS_TIMEOUT
CONNECTION_ERROR = S.LLM_CALL_STATUS_CONNECTION_ERROR
CLIENT_ERROR = S.LLM_CALL_STATUS_CLIENT_ERROR
CONTEXT_OVERFLOW = S.LLM_CALL_STATUS_CONTEXT_OVERFLOW
INVALID_RESPONSE = S.LLM_CALL_STATUS_INVALID_RESPONSE

RECORD = LlmCallRecord(call_id="c-1", purpose=CallPurpose.CALL_PURPOSE_EXEC, model=QWEN)


def new_throttle(share: int = 8, clock: FakeClock | None = None, **policy: Any) -> ModelThrottle:
    clock = FakeClock() if clock is None else clock
    return ModelThrottle(QWEN, share, ThrottlePolicy.model_validate(policy), clock)


async def settle() -> None:
    """Let every task that can run do so, until each one waits."""
    for _ in range(10):
        await asyncio.sleep(0)


async def hold(limiter: ModelThrottle, done: asyncio.Event, entered: list[int], n: int) -> None:
    """Take a slot, note it in ``entered``, and keep it until ``done`` is set."""
    async with limiter.slot(RECORD):
        entered.append(n)
        await done.wait()


def observe(
    limiter: ModelThrottle, status: LlmCallStatus.ValueType, http_status: int, sent_at: float
) -> None:
    limiter.observe(status, http_status, sent_at, RECORD)


def succeed(limiter: ModelThrottle, clock: FakeClock, times: int) -> None:
    for _ in range(times):
        observe(limiter, OK, 200, clock.now)


def fail(
    limiter: ModelThrottle,
    clock: FakeClock,
    times: int,
    status: LlmCallStatus.ValueType = SERVER_ERROR,
) -> None:
    for _ in range(times):
        observe(limiter, status, 500 if status == SERVER_ERROR else 0, clock.now)


def events(caplog: pytest.LogCaptureFixture, name: str) -> list[logging.LogRecord]:
    return [entry for entry in caplog.records if getattr(entry, "event", None) == name]


def fields(entry: logging.LogRecord) -> tuple[str, int]:
    """The model and the new limit of a log entry about a change of the limit."""
    return getattr(entry, "model", ""), getattr(entry, "limit", 0)


# The policy and the shares.


def test_the_default_policy_is_the_specs() -> None:
    policy = ThrottlePolicy()

    assert policy.successes_to_grow == 20
    assert (policy.breaker_window_s, policy.breaker_open_s) == (120.0, 60.0)
    assert policy.breaker_failure_ratio == 0.5


def test_a_process_alone_may_use_each_models_whole_budget() -> None:
    assert dict(DEFAULT_SHARES) == {QWEN: 48, GLM: 16}


@pytest.mark.parametrize(
    "policy",
    [
        {"successes_to_grow": 0},
        {"breaker_window_s": 0.0},
        {"breaker_open_s": -1.0},
        {"breaker_min_attempts": 0},
        {"breaker_failure_ratio": 1.0},
        {"breaker_failure_ratio": 0.0},
        {"breaker_open_s": float("nan")},
    ],
)
def test_the_policy_rejects_values_out_of_range(policy: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ThrottlePolicy.model_validate(policy)


def test_a_throttle_needs_a_share_of_at_least_one() -> None:
    with pytest.raises(ValueError, match="share"):
        new_throttle(share=0)


# Slots.


async def test_no_more_calls_than_the_share_are_in_flight() -> None:
    limiter = new_throttle(share=2)
    done, entered = asyncio.Event(), list[int]()
    tasks = [asyncio.create_task(hold(limiter, done, entered, n)) for n in range(3)]
    await settle()

    assert (entered, limiter.in_flight) == ([0, 1], 2)

    done.set()
    await asyncio.gather(*tasks)
    assert (entered, limiter.in_flight) == ([0, 1, 2], 0)


async def test_slots_go_to_waiting_calls_in_the_order_they_came() -> None:
    limiter = new_throttle(share=1)
    dones = [asyncio.Event() for _ in range(4)]
    entered: list[int] = []
    tasks = [asyncio.create_task(hold(limiter, dones[n], entered, n)) for n in range(4)]
    for n in range(4):
        await settle()
        assert entered == list(range(n + 1))
        dones[n].set()
    await asyncio.gather(*tasks)


async def test_a_call_cancelled_while_it_waits_takes_no_slot() -> None:
    limiter = new_throttle(share=1)
    async with AsyncExitStack() as stack:
        await stack.enter_async_context(limiter.slot(RECORD))
        waiter = asyncio.create_task(hold(limiter, asyncio.Event(), [], 1))
        await settle()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

    assert limiter.in_flight == 0
    async with limiter.slot(RECORD):  # The slot is free at once.
        assert limiter.in_flight == 1


async def test_a_call_cancelled_just_after_its_turn_came_gives_its_slot_back() -> None:
    limiter = new_throttle(share=1)
    stack = AsyncExitStack()
    await stack.enter_async_context(limiter.slot(RECORD))
    waiter = asyncio.create_task(hold(limiter, asyncio.Event(), [], 1))
    await settle()

    await stack.aclose()  # Hands the slot to the waiter, which has not resumed yet.
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    assert limiter.in_flight == 0


async def test_a_call_that_raises_gives_its_slot_back() -> None:
    limiter = new_throttle(share=1)
    with pytest.raises(RuntimeError, match="boom"):
        async with limiter.slot(RECORD):
            raise RuntimeError("boom")

    assert limiter.in_flight == 0


# AIMD.


@pytest.mark.parametrize(
    ("status", "http_status"),
    [
        pytest.param(RATE_LIMITED, 429, id="429"),
        pytest.param(SERVER_ERROR, 502, id="502"),
        pytest.param(SERVER_ERROR, 503, id="503"),
        pytest.param(SERVER_ERROR, 504, id="504"),
        pytest.param(TIMEOUT, 0, id="timeout"),
        pytest.param(TIMEOUT, 408, id="408"),
    ],
)
def test_halves_the_limit_on_congestion(status: LlmCallStatus.ValueType, http_status: int) -> None:
    clock = FakeClock()
    limiter = new_throttle(share=8, clock=clock)

    observe(limiter, status, http_status, clock.now)

    assert limiter.limit == 4


@pytest.mark.parametrize(
    ("status", "http_status"),
    [
        pytest.param(OK, 200, id="ok"),
        pytest.param(SERVER_ERROR, 500, id="500"),
        pytest.param(CONNECTION_ERROR, 0, id="connection-error"),
        pytest.param(CLIENT_ERROR, 400, id="400"),
        pytest.param(CONTEXT_OVERFLOW, 400, id="context-overflow"),
        pytest.param(INVALID_RESPONSE, 200, id="invalid-response"),
    ],
)
def test_other_outcomes_do_not_lower_the_limit(
    status: LlmCallStatus.ValueType, http_status: int
) -> None:
    clock = FakeClock()
    limiter = new_throttle(share=8, clock=clock)

    observe(limiter, status, http_status, clock.now)

    assert limiter.limit == 8


def test_halves_once_per_burst() -> None:
    clock = FakeClock()
    limiter = new_throttle(share=8, clock=clock)
    sent = clock.now
    clock.now += 1.0

    observe(limiter, RATE_LIMITED, 429, sent)
    # The rest of the burst: requests sent before the halving, under the old limit.
    observe(limiter, RATE_LIMITED, 429, sent)
    observe(limiter, TIMEOUT, 0, sent)
    assert limiter.limit == 4

    # A request sent after the halving shows the lower limit is still too high.
    clock.now += 1.0
    observe(limiter, RATE_LIMITED, 429, clock.now - 0.5)
    assert limiter.limit == 2


def test_a_burst_spread_over_time_still_halves_once() -> None:
    clock = FakeClock()
    limiter = new_throttle(share=8, clock=clock)
    first_sent, second_sent = clock.now, clock.now + 0.5
    clock.now += 1.0  # The first congested reply comes back, after both were sent.

    observe(limiter, RATE_LIMITED, 429, first_sent)
    clock.now += 0.2
    observe(limiter, SERVER_ERROR, 503, second_sent)  # Sent before the halving: same burst.

    assert limiter.limit == 4


def test_never_halves_below_one() -> None:
    clock = FakeClock()
    limiter = new_throttle(share=3, clock=clock)
    for _ in range(4):
        clock.now += 1.0
        observe(limiter, RATE_LIMITED, 429, clock.now)

    assert limiter.limit == 1


def test_grows_by_one_after_every_20_consecutive_successes() -> None:
    clock = FakeClock()
    limiter = new_throttle(share=8, clock=clock)
    observe(limiter, RATE_LIMITED, 429, clock.now)
    clock.now += 1.0  # The successes below are of requests sent after the halving.

    succeed(limiter, clock, 19)
    assert limiter.limit == 4
    succeed(limiter, clock, 1)
    assert limiter.limit == 5
    succeed(limiter, clock, 20)
    assert limiter.limit == 6


@pytest.mark.parametrize(
    ("status", "http_status"),
    [
        pytest.param(RATE_LIMITED, 429, id="old-burst-429"),
        pytest.param(SERVER_ERROR, 500, id="500"),
        pytest.param(CONNECTION_ERROR, 0, id="connection-error"),
    ],
)
def test_any_failure_restarts_the_run_of_successes(
    status: LlmCallStatus.ValueType, http_status: int
) -> None:
    clock = FakeClock()
    limiter = new_throttle(share=8, clock=clock)
    sent = clock.now
    clock.now += 1.0
    observe(limiter, RATE_LIMITED, 429, sent)
    clock.now += 1.0

    succeed(limiter, clock, 19)
    observe(limiter, status, http_status, sent)  # No halving: a 500, or the old burst's 429.
    succeed(limiter, clock, 19)
    assert limiter.limit == 4
    succeed(limiter, clock, 1)
    assert limiter.limit == 5


@pytest.mark.parametrize(
    ("status", "http_status"),
    [
        pytest.param(CLIENT_ERROR, 400, id="400"),
        pytest.param(CONTEXT_OVERFLOW, 400, id="context-overflow"),
        pytest.param(INVALID_RESPONSE, 200, id="invalid-response"),
    ],
)
def test_a_refused_request_leaves_the_run_of_successes_alone(
    status: LlmCallStatus.ValueType, http_status: int
) -> None:
    clock = FakeClock()
    limiter = new_throttle(share=8, clock=clock)
    observe(limiter, RATE_LIMITED, 429, clock.now)
    clock.now += 1.0

    succeed(limiter, clock, 19)
    observe(limiter, status, http_status, clock.now)
    succeed(limiter, clock, 1)

    assert limiter.limit == 5


def test_successes_sent_before_the_last_halving_do_not_count() -> None:
    clock = FakeClock()
    limiter = new_throttle(share=8, clock=clock)
    sent = clock.now
    clock.now += 1.0
    observe(limiter, RATE_LIMITED, 429, sent)

    for _ in range(40):  # They ran under the old limit, so they say nothing of the new one.
        observe(limiter, OK, 200, sent)

    assert limiter.limit == 4


def test_never_grows_above_the_share() -> None:
    clock = FakeClock()
    limiter = new_throttle(share=2, clock=clock)

    succeed(limiter, clock, 100)

    assert limiter.limit == 2


async def test_a_lower_limit_lets_calls_in_flight_finish_but_starts_no_new_one() -> None:
    clock = FakeClock()
    limiter = new_throttle(share=4, clock=clock)
    dones = [asyncio.Event() for _ in range(5)]
    entered: list[int] = []
    tasks = [asyncio.create_task(hold(limiter, dones[n], entered, n)) for n in range(4)]
    await settle()
    observe(limiter, RATE_LIMITED, 429, clock.now)
    assert (limiter.limit, limiter.in_flight) == (2, 4)

    tasks.append(asyncio.create_task(hold(limiter, dones[4], entered, 4)))
    for n in range(3):  # Two must finish before the new call starts, and a third.
        dones[n].set()
        await settle()
        assert entered == [0, 1, 2, 3] + ([4] if n == 2 else [])
    for done in dones:
        done.set()
    await asyncio.gather(*tasks)


async def test_a_higher_limit_lets_a_waiting_call_start() -> None:
    clock = FakeClock()
    limiter = new_throttle(share=2, clock=clock)
    observe(limiter, RATE_LIMITED, 429, clock.now)
    clock.now += 1.0
    done, entered = asyncio.Event(), list[int]()
    tasks = [asyncio.create_task(hold(limiter, done, entered, n)) for n in range(2)]
    await settle()
    assert entered == [0]

    succeed(limiter, clock, 20)
    await settle()

    assert (limiter.limit, entered) == (2, [0, 1])
    done.set()
    await asyncio.gather(*tasks)


def test_logs_each_change_of_the_limit_with_the_purpose_and_model(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="goaly.harness.core.llm_throttle")
    clock = FakeClock()
    limiter = new_throttle(share=8, clock=clock)

    observe(limiter, RATE_LIMITED, 429, clock.now)
    clock.now += 1.0
    succeed(limiter, clock, 20)

    (down,) = events(caplog, "llm_limit_down")
    (up,) = events(caplog, "llm_limit_up")
    assert down.levelno == logging.WARNING
    assert up.levelno == logging.INFO
    keys = ["model", "purpose", "call_id", "limit", "previous_limit", "share"]
    assert {key: getattr(down, key) for key in [*keys, "after", "http_status"]} == {
        "model": QWEN,
        "purpose": "CALL_PURPOSE_EXEC",
        "call_id": "c-1",
        "limit": 4,
        "previous_limit": 8,
        "share": 8,
        "after": "LLM_CALL_STATUS_RATE_LIMITED",
        "http_status": 429,
    }
    assert {key: getattr(up, key) for key in keys} == {
        "model": QWEN,
        "purpose": "CALL_PURPOSE_EXEC",
        "call_id": "c-1",
        "limit": 5,
        "previous_limit": 4,
        "share": 8,
    }


# The circuit breaker.


async def test_opens_when_more_than_half_of_the_attempts_failed() -> None:
    clock = FakeClock()
    limiter = new_throttle(clock=clock)
    succeed(limiter, clock, 4)
    fail(limiter, clock, 6)

    async with limiter.slot(RECORD):
        assert clock.sleeps == [60.0]  # No new attempt starts for 60 s.


async def test_stays_closed_when_exactly_half_failed() -> None:
    clock = FakeClock()
    limiter = new_throttle(clock=clock)
    succeed(limiter, clock, 5)
    fail(limiter, clock, 5)

    async with limiter.slot(RECORD):
        assert clock.sleeps == []


async def test_needs_enough_attempts_before_it_opens() -> None:
    clock = FakeClock()
    limiter = new_throttle(clock=clock)
    fail(limiter, clock, 9)
    async with limiter.slot(RECORD):
        assert clock.sleeps == []

    fail(limiter, clock, 1)
    async with limiter.slot(RECORD):
        assert clock.sleeps == [60.0]


@pytest.mark.parametrize("status", [RATE_LIMITED, SERVER_ERROR, TIMEOUT, CONNECTION_ERROR])
async def test_counts_each_transient_failure(status: LlmCallStatus.ValueType) -> None:
    clock = FakeClock()
    limiter = new_throttle(clock=clock)
    fail(limiter, clock, 10, status=status)

    async with limiter.slot(RECORD):
        assert clock.sleeps == [60.0]


@pytest.mark.parametrize("status", [CLIENT_ERROR, CONTEXT_OVERFLOW, INVALID_RESPONSE])
async def test_a_refused_request_is_no_failure_of_the_gateway(
    status: LlmCallStatus.ValueType,
) -> None:
    clock = FakeClock()
    limiter = new_throttle(clock=clock)
    fail(limiter, clock, 10, status=status)

    async with limiter.slot(RECORD):
        assert clock.sleeps == []


async def test_forgets_attempts_older_than_two_minutes() -> None:
    clock = FakeClock()
    limiter = new_throttle(clock=clock)
    fail(limiter, clock, 9)
    clock.now += 120.0
    fail(limiter, clock, 1)  # Only one failure in the last two minutes.

    async with limiter.slot(RECORD):
        assert clock.sleeps == []


async def test_closes_after_its_open_period_and_judges_afresh() -> None:
    clock = FakeClock()
    limiter = new_throttle(clock=clock)
    fail(limiter, clock, 10)
    async with limiter.slot(RECORD):
        pass
    async with limiter.slot(RECORD):
        assert clock.sleeps == [60.0]  # Closed again: the second call did not wait.

    fail(limiter, clock, 9)  # The failures that opened it no longer count.
    async with limiter.slot(RECORD):
        assert clock.sleeps == [60.0]
    fail(limiter, clock, 1)
    async with limiter.slot(RECORD):
        assert clock.sleeps == [60.0, 60.0]


async def test_attempts_sent_before_it_opened_do_not_count_toward_its_next_judgement(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = FakeClock()
    limiter = new_throttle(share=20, clock=clock)
    done, entered = asyncio.Event(), list[int]()
    held = [asyncio.create_task(hold(limiter, done, entered, n)) for n in range(10)]
    await settle()
    held_sent = clock.now  # Ten attempts in flight, sent at t=100.
    clock.now += 1.0
    fail(limiter, clock, 10)  # Ten others, sent at t=101, open the breaker.

    for _ in range(10):  # The held ones fail too, while it is open.
        observe(limiter, SERVER_ERROR, 500, held_sent)
    done.set()
    await asyncio.gather(*held)
    async with limiter.slot(RECORD):  # Waits out the 60 s.
        pass
    succeed(limiter, clock, 1)  # A fresh success, judged on its own.
    async with limiter.slot(RECORD):
        pass

    assert clock.sleeps == [60.0]
    assert len(events(caplog, "llm_breaker_open")) == 1


async def test_failures_reported_while_it_is_open_extend_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = FakeClock()
    limiter = new_throttle(clock=clock)
    fail(limiter, clock, 10)  # Opens at t=100, until t=160.
    clock.now += 0.5
    fail(limiter, clock, 10)  # Enough to open it again, but it is open already.

    async with limiter.slot(RECORD):
        pass

    assert clock.sleeps == [59.5]  # Until t=160, not t=160.5.
    assert len(events(caplog, "llm_breaker_open")) == 1


async def test_calls_in_flight_finish_while_it_is_open() -> None:
    clock = FakeClock()
    limiter = new_throttle(share=2, clock=clock)
    done, entered = asyncio.Event(), list[int]()
    task = asyncio.create_task(hold(limiter, done, entered, 0))
    await settle()

    fail(limiter, clock, 10)
    done.set()
    await task

    assert (entered, clock.sleeps, limiter.in_flight) == ([0], [], 0)


async def test_a_call_waiting_for_a_slot_also_waits_for_it_to_close() -> None:
    clock = FakeClock()
    limiter = new_throttle(share=1, clock=clock)
    done, entered = asyncio.Event(), list[int]()
    tasks = [asyncio.create_task(hold(limiter, done, entered, n)) for n in range(2)]
    await settle()
    assert entered == [0]

    fail(limiter, clock, 10)
    done.set()
    await asyncio.gather(*tasks)

    assert (entered, clock.sleeps) == ([0, 1], [60.0])


async def test_logs_opening_and_closing_with_the_purpose_and_model(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="goaly.harness.core.llm_throttle")
    clock = FakeClock()
    limiter = new_throttle(clock=clock)
    succeed(limiter, clock, 3)
    fail(limiter, clock, 7)

    async with limiter.slot(RECORD):
        pass

    (opened,) = events(caplog, "llm_breaker_open")
    (closed,) = events(caplog, "llm_breaker_closed")
    assert opened.levelno == logging.WARNING
    assert closed.levelno == logging.INFO
    base = {"model": QWEN, "purpose": "CALL_PURPOSE_EXEC", "call_id": "c-1"}
    assert {key: getattr(opened, key) for key in [*base, "failures", "attempts", "open_s"]} == {
        **base,
        "failures": 7,
        "attempts": 10,
        "open_s": 60.0,
    }
    assert {key: getattr(closed, key) for key in [*base, "open_s"]} == {**base, "open_s": 60.0}


class PausingClock(FakeClock):
    """A FakeClock whose sleeps hold until ``resume`` is set, so a test sees the pause."""

    def __init__(self) -> None:
        super().__init__()
        self.resume = asyncio.Event()

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        await self.resume.wait()
        self.now += seconds


async def test_calls_keep_their_order_across_a_pause_and_wake_once_it_ends() -> None:
    clock = PausingClock()
    limiter = new_throttle(share=2, clock=clock)
    dones = {name: asyncio.Event() for name in "ABCDE"}
    entered: list[int] = []
    tasks = [
        asyncio.create_task(hold(limiter, dones[name], entered, n))
        for n, name in enumerate("ABCDE")
    ]
    await settle()
    assert entered == [0, 1]  # C, D and E wait, in that order.

    fail(limiter, clock, 10)  # The breaker opens.
    dones["A"].set()
    dones["B"].set()
    await settle()
    assert (entered, limiter.in_flight) == ([0, 1], 0)  # No slot goes out while it is open.
    assert clock.sleeps == [60.0]  # One wait for the whole queue.

    clock.resume.set()
    await settle()
    assert entered == [0, 1, 2, 3]  # The queue wakes once, in order.
    dones["C"].set()
    await settle()
    assert entered == [0, 1, 2, 3, 4]
    for done in dones.values():
        done.set()
    await asyncio.gather(*tasks)
    assert clock.sleeps == [60.0]


async def test_a_call_whose_slot_came_just_before_a_halving_waits_again_first_in_line() -> None:
    clock = FakeClock()
    limiter = new_throttle(share=2, clock=clock)
    first = AsyncExitStack()
    await first.enter_async_context(limiter.slot(RECORD))
    dones = [asyncio.Event() for _ in range(3)]
    entered: list[int] = []
    tasks = [asyncio.create_task(hold(limiter, dones[n], entered, n)) for n in range(3)]
    await settle()
    assert entered == [0]  # 1 and 2 wait, in that order.

    await first.aclose()  # Its slot goes to 1, which has not resumed yet...
    observe(limiter, RATE_LIMITED, 429, clock.now)  # ...when the limit halves to 1.
    await settle()
    assert (entered, limiter.in_flight, limiter.limit) == ([0], 1, 1)  # 1 gave the slot back.

    dones[0].set()
    await settle()
    assert entered == [0, 1]  # 1 kept its turn, ahead of 2.
    for done in dones:
        done.set()
    await asyncio.gather(*tasks)


async def test_a_call_whose_slot_came_just_before_the_breaker_opened_waits_it_out() -> None:
    clock = FakeClock()
    limiter = new_throttle(share=1, clock=clock)
    first = AsyncExitStack()
    await first.enter_async_context(limiter.slot(RECORD))
    done, entered = asyncio.Event(), list[int]()
    task = asyncio.create_task(hold(limiter, done, entered, 1))
    await settle()

    await first.aclose()  # Its slot goes to 1, which has not resumed yet...
    fail(limiter, clock, 10)  # ...when the breaker opens.
    await settle()
    assert (entered, clock.sleeps) == ([1], [60.0])  # 1 started only after the pause.
    done.set()
    await task


# Through the client.


async def test_slot_waits_count_into_the_attempts_wait() -> None:
    clock = FakeClock()

    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0)
        clock.now += 2.0  # The first call holds the only slot for 2 s.
        return completion()

    gateway = Gateway(slow, completion())
    async with LlmClient(
        settings(),
        Recorder(),
        shares={QWEN: 1, GLM: 1},
        transport=httpx.MockTransport(gateway.handle),
        clock=clock,
    ) as llm:
        first, second = await asyncio.gather(
            llm.complete([user("one")], model=QWEN, tags=EXEC),
            llm.complete([user("two")], model=QWEN, tags=EXEC),
        )

    assert first.record.attempts[0].wait_s == 0.0
    assert second.record.attempts[0].wait_s == 2.0
    assert second.record.wait_s == 2.0


async def test_an_open_breaker_holds_back_new_calls_and_counts_as_their_wait() -> None:
    clock = FakeClock()
    gateway = Gateway(failure(500), failure(500), completion(), completion())
    async with LlmClient(
        settings(),
        Recorder(),
        retry=RetryPolicy(max_retries=1),
        throttle=ThrottlePolicy(breaker_min_attempts=2),
        transport=httpx.MockTransport(gateway.handle),
        clock=clock,
        rng=TopOfRange(),  # A fixed backoff keeps the clock's sums exact.
    ) as llm:
        with pytest.raises(ServerError):
            await llm.complete([user("hi")], model=QWEN, tags=EXEC)
        clock.sleeps.clear()
        held_back = await llm.complete([user("hi")], model=QWEN, tags=EXEC)
        free = await llm.complete([user("hi")], model=GLM, tags=ANALYST)

    assert clock.sleeps == [60.0]
    assert held_back.record.attempts[0].wait_s == 60.0
    assert free.record.wait_s == 0.0  # Each model has its own breaker.


async def test_bound_clients_share_the_limit_of_the_client_that_bound_them() -> None:
    in_flight, peak = 0, 0

    async def answer(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return completion()

    gateway = Gateway(*(answer for _ in range(6)))
    async with LlmClient(
        settings(),
        Recorder(),
        shares={QWEN: 2, GLM: 1},
        transport=httpx.MockTransport(gateway.handle),
    ) as llm:
        tasks = [
            llm.bind(model=QWEN, tags=CallTags(CallPurpose.CALL_PURPOSE_EXEC, instance_id=f"t{n}"))
            for n in range(3)
        ]
        replies = await asyncio.gather(
            *(task.complete([user("hi")]) for task in tasks for _ in range(2))
        )

    assert len(replies) == 6
    assert peak == 2  # Three tasks under one limit of 2: not 2 each.


async def test_a_client_serves_only_the_loop_it_was_first_used_on() -> None:
    # Its limits wait on futures of that loop: a call from another loop must not join them.
    gateway = Gateway(completion())
    async with LlmClient(
        settings(), Recorder(), transport=httpx.MockTransport(gateway.handle)
    ) as llm:

        def call_on_a_second_loop() -> None:
            asyncio.run(llm.complete([user("hi")], model=QWEN, tags=EXEC))

        with pytest.raises(RuntimeError, match="one event loop"):
            await asyncio.to_thread(call_on_a_second_loop)

    assert gateway.requests == []


async def test_each_model_has_its_own_limit(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="goaly.harness.core.llm_throttle")
    gateway = Gateway(failure(429), completion(), completion())
    async with LlmClient(
        settings(), Recorder(), transport=httpx.MockTransport(gateway.handle), clock=FakeClock()
    ) as llm:
        await llm.complete([user("hi")], model=QWEN, tags=EXEC)
        await llm.complete([user("hi")], model=GLM, tags=ANALYST)

    (down,) = events(caplog, "llm_limit_down")
    assert {key: getattr(down, key) for key in ("model", "limit", "share")} == {
        "model": QWEN,
        "limit": 24,
        "share": 48,
    }


@pytest.mark.parametrize(
    ("shares", "message"),
    [
        pytest.param({QWEN: 4}, "every model", id="a-model-missing"),
        pytest.param({QWEN: 4, GLM: 2, "gpt-4o": 1}, "every model", id="an-unknown-model"),
        pytest.param({QWEN: 0, GLM: 2}, "positive int", id="zero"),
        pytest.param({QWEN: True, GLM: 2}, "positive int", id="a-bool"),
        pytest.param({QWEN: 2.0, GLM: 2}, "positive int", id="a-float"),
    ],
)
def test_the_client_refuses_shares_it_cannot_use(shares: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        LlmClient(settings(), Recorder(), shares=shares)


@pytest.mark.parametrize(
    ("shares", "logged"),
    [
        pytest.param(None, {QWEN: 48, GLM: 16}, id="the-whole-budgets"),
        pytest.param({QWEN: 4, GLM: 2}, {QWEN: 4, GLM: 2}, id="given"),
    ],
)
async def test_logs_the_shares_it_runs_with(
    shares: dict[str, int] | None, logged: dict[str, int], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="goaly.harness.core.llm")
    async with LlmClient(settings(), Recorder(), shares=shares):
        pass

    (entry,) = events(caplog, "llm_shares")
    assert entry.levelno == logging.INFO
    assert getattr(entry, "shares", None) == logged


# The stress test (spec §6.2): faults, latency and timeouts, 500 calls at once.

STRESS_SHARES = {QWEN: 8, GLM: 3}


async def test_500_faulty_calls_stay_within_the_share_and_are_each_recorded_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="goaly.harness.core.llm_throttle")
    # A storm of 429s and timeouts over the first 100 requests drives both limits down to 1;
    # the calm after it is long enough for them to grow back, with room to spare.
    gateway = FakeLLM(
        seed=20261015,
        latency_s=0.005,
        hang_s=1.0,
        rate_limited=0.25,
        timeouts=0.05,
        storm_requests=100,
    )
    recorder = Recorder()
    retry = RetryPolicy(timeout_s=0.2, backoff_base_s=0.001, backoff_cap_s=0.01)
    policy = ThrottlePolicy(breaker_window_s=1.0, breaker_open_s=0.05)
    async with LlmClient(
        settings(),
        recorder,
        shares=STRESS_SHARES,
        retry=retry,
        throttle=policy,
        transport=httpx.MockTransport(gateway.handle),
    ) as llm:

        async def one(n: int) -> LlmReply | LlmError:
            model, tags = (GLM, ANALYST) if n % 4 == 0 else (QWEN, EXEC)
            try:
                return await llm.complete([user(f"task {n}")], model=model, tags=tags)
            except LlmError as error:  # Retries can run out during the storm.
                return error

        async with asyncio.timeout(60):  # Nothing deadlocks.
            results = await asyncio.gather(*(one(n) for n in range(500)))

    assert gateway.max_in_flight == STRESS_SHARES
    records = [result.record for result in results]
    assert all(record is not None for record in records)
    call_ids = [record.call_id for record in records if record is not None]
    assert len(set(call_ids)) == 500
    assert sorted(record.call_id for record in recorder.records) == sorted(call_ids)
    assert sum(len(record.attempts) for record in recorder.records) == gateway.requests
    # The limits backed off in the storm and grew back to the share in the calm.
    for model, share in STRESS_SHARES.items():
        downs = [entry for entry in events(caplog, "llm_limit_down") if fields(entry)[0] == model]
        changes = [
            fields(entry)[1]
            for entry in caplog.records
            if getattr(entry, "event", "").startswith("llm_limit") and fields(entry)[0] == model
        ]
        assert downs, model
        assert changes[-1] == share, model
