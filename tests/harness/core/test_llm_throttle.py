"""Tests for goaly.harness.core.llm_throttle: slots, AIMD and the circuit breaker.

The tests drive a ModelThrottle directly, on a FakeClock.
"""

import asyncio
import logging
from contextlib import AsyncExitStack
from typing import Any

import pytest
from pydantic import ValidationError

from goaly.contracts.gen.llm_call_pb2 import CallPurpose, LlmCallRecord, LlmCallStatus
from goaly.harness.core.llm_throttle import DEFAULT_SHARES, ModelThrottle, ThrottlePolicy
from tests.harness.core.llm_fakes import (
    GLM,
    QWEN,
    FakeClock,
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
    [(RATE_LIMITED, 429), (SERVER_ERROR, 503), (TIMEOUT, 0)],
    ids=["429", "503", "timeout"],
)
def test_halves_the_limit_after_a_429_a_503_or_a_timeout(
    status: LlmCallStatus.ValueType, http_status: int
) -> None:
    clock = FakeClock()
    limiter = new_throttle(share=8, clock=clock)

    observe(limiter, status, http_status, clock.now)

    assert limiter.limit == 4


@pytest.mark.parametrize(
    ("status", "http_status"),
    [
        (OK, 200),
        (SERVER_ERROR, 500),
        (SERVER_ERROR, 502),
        (CONNECTION_ERROR, 0),
        (CLIENT_ERROR, 400),
        (CONTEXT_OVERFLOW, 400),
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

    succeed(limiter, clock, 19)
    assert limiter.limit == 4
    succeed(limiter, clock, 1)
    assert limiter.limit == 5
    succeed(limiter, clock, 20)
    assert limiter.limit == 6


def test_congestion_restarts_the_count_of_successes() -> None:
    clock = FakeClock()
    limiter = new_throttle(share=8, clock=clock)
    sent = clock.now
    clock.now += 1.0
    observe(limiter, RATE_LIMITED, 429, sent)

    succeed(limiter, clock, 19)
    observe(limiter, RATE_LIMITED, 429, sent)  # Part of the old burst: no halving, but a reset.
    succeed(limiter, clock, 19)
    assert limiter.limit == 4
    succeed(limiter, clock, 1)
    assert limiter.limit == 5


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
    succeed(limiter, clock, 20)

    (down,) = events(caplog, "llm_limit_down")
    (up,) = events(caplog, "llm_limit_up")
    assert down.levelno == logging.WARNING
    assert up.levelno == logging.INFO
    fields = ["model", "purpose", "call_id", "limit", "previous_limit", "share"]
    assert {key: getattr(down, key) for key in [*fields, "after", "http_status"]} == {
        "model": QWEN,
        "purpose": "CALL_PURPOSE_EXEC",
        "call_id": "c-1",
        "limit": 4,
        "previous_limit": 8,
        "share": 8,
        "after": "LLM_CALL_STATUS_RATE_LIMITED",
        "http_status": 429,
    }
    assert {key: getattr(up, key) for key in fields} == {
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
    assert {key: getattr(closed, key) for key in base} == base
