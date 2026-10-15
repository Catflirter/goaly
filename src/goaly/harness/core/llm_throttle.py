"""Concurrency limits for the LLM client (spec §6.2): slots, AIMD and a circuit breaker per model.

A process may have at most its share of each model's global budget in flight: the share its
config sets, or the whole budget for a process that runs alone. Other teams share the gateway,
so its capacity can drop at any moment, and each model's limit adapts below the share:

- AIMD. A burst of congestion halves the limit once, never below 1: 429s, 502s, 503s, 504s
  or timeouts, 408s included. A burst is the congestion reported by requests sent before the
  last halving: they ran under the old limit, so only a request sent after it can show that the
  new limit is still too high. After every 20 successes in a row it grows by 1, never above the
  share. Any failure ends the run, and only requests sent after the last halving count toward
  it. A 500 or a connection error is a failure, not congestion: it lowers nothing.
- Circuit breaker. When more than half of a model's attempts in the last 2 minutes failed (429,
  5xx, timeout or connection error), no new attempt starts for 60 s; the breaker then judges
  afresh, on attempts sent after it opened only. A retry is a new attempt, so it waits too,
  while requests already sent run on.

Calls wait for a slot in the order they came, across a pause of the breaker too. Each change of
a limit, and each opening and closing of a breaker, is logged with the model and the purpose of
the call that caused it. A breaker's closing is logged lazily, by the first call to go through
after it, or the first outcome judged after it.
"""

import asyncio
import logging
import math
import time
from collections import deque
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from types import MappingProxyType
from typing import Final, Protocol

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt

from goaly.contracts.gen.llm_call_pb2 import CallPurpose, LlmCallRecord, LlmCallStatus
from goaly.harness.core.llm_models import EXECUTOR_MODEL, IMPROVEMENT_MODEL

__all__ = [
    "CONGESTION_HTTP_STATUSES",
    "DEFAULT_SHARES",
    "TRANSIENT_FAILURES",
    "Clock",
    "ModelThrottle",
    "SystemClock",
    "ThrottlePolicy",
]

_logger = logging.getLogger(__name__)

_S = LlmCallStatus
# The gateway failing, or too busy, which the client retries: 429, 5xx, timeouts (a 408
# included) and connection errors. The breaker counts them, and each ends AIMD's run of
# successes.
TRANSIENT_FAILURES: Final = frozenset(
    {
        _S.LLM_CALL_STATUS_RATE_LIMITED,
        _S.LLM_CALL_STATUS_SERVER_ERROR,
        _S.LLM_CALL_STATUS_TIMEOUT,
        _S.LLM_CALL_STATUS_CONNECTION_ERROR,
    }
)
# AIMD's congestion (spec §6.2): these responses, and any timeout, a 408 included.
CONGESTION_HTTP_STATUSES: Final = frozenset({429, 502, 503, 504})

# Each model's global budget (spec §6.2): all that a process running alone may use.
DEFAULT_SHARES: Final[Mapping[str, int]] = MappingProxyType(
    {EXECUTOR_MODEL: 48, IMPROVEMENT_MODEL: 16}
)


class Clock(Protocol):
    """The time the client measures waits and latency with, and sleeps through waits on."""

    def monotonic(self) -> float:
        """Return the time in seconds on a clock that never goes back."""

    async def sleep(self, seconds: float) -> None:
        """Wait ``seconds``."""


class SystemClock:
    """The real time."""

    def monotonic(self) -> float:
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class ThrottlePolicy(BaseModel):
    """How each model's limit adapts below its share. The defaults are the spec's (§6.2)."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    successes_to_grow: PositiveInt = 20
    breaker_window_s: PositiveFloat = 120.0
    breaker_failure_ratio: float = Field(default=0.5, gt=0, lt=1)
    breaker_min_attempts: PositiveInt = 10  # Fewer attempts say too little about the gateway.
    breaker_open_s: PositiveFloat = 60.0


class ModelThrottle:
    """One model's limit on calls in flight in this process, with AIMD and a circuit breaker.

    Each attempt of a call holds a ``slot``, and before it gives the slot back reports how it
    went to ``observe``. ``record`` is the call's record, for the log.
    """

    def __init__(self, model: str, share: int, policy: ThrottlePolicy, clock: Clock) -> None:
        if share < 1:
            raise ValueError(f"{model}: a share must be at least 1, not {share}")
        self.model = model
        self.share = share
        self._policy = policy
        self._clock = clock
        self._limit = share
        self._in_flight = 0
        self._waiters: deque[tuple[asyncio.Future[None], LlmCallRecord]] = deque()
        self._closer: asyncio.Task[None] | None = None  # Wakes the queue when the pause ends.
        self._successes = 0  # In a row.
        self._halved_at = -math.inf
        self._window: deque[tuple[float, bool]] = deque()  # When recent attempts ended; failed?
        self._failures = 0  # In the window.
        self._opened_at = -math.inf
        self._open_until = -math.inf
        self._opened = False  # Open, or past its open period but not yet logged as closed.

    @property
    def limit(self) -> int:
        """How many of the model's calls may be in flight now: never more than the share."""
        return self._limit

    @property
    def in_flight(self) -> int:
        """How many slots are held. After a halving this can be more than the limit for a while."""
        return self._in_flight

    @asynccontextmanager
    async def slot(self, record: LlmCallRecord) -> AsyncIterator[None]:
        """Hold a slot for one attempt, waiting while the breaker is open or no slot is free.

        Slots go to waiting calls in the order they came, and no slot goes out while the
        breaker is open. However the attempt ends, cancellation included, its slot is given back.
        """
        await self._acquire(record)
        try:
            yield
        finally:
            self._release()

    def observe(
        self,
        status: LlmCallStatus.ValueType,
        http_status: int,
        sent_at: float,
        record: LlmCallRecord,
    ) -> None:
        """Take in how an attempt sent at ``sent_at`` ended: feed AIMD and the breaker."""
        now = self._clock.monotonic()
        failed = status in TRANSIENT_FAILURES
        self._judge(failed, sent_at, now, record)
        if failed:
            self._successes = 0  # Any failure ends the run of successes.
            congested = status == _S.LLM_CALL_STATUS_TIMEOUT or (
                http_status in CONGESTION_HTTP_STATUSES
            )
            if congested and sent_at > self._halved_at:  # The first sign of a new burst.
                self._halved_at = now
                self._halve(status, http_status, record)
        elif status == _S.LLM_CALL_STATUS_OK and sent_at > self._halved_at:
            self._successes += 1  # Sent under the current limit, so it speaks for that limit.
            if self._successes >= self._policy.successes_to_grow:
                self._successes = 0
                if self._limit < self.share:
                    self._grow(record)

    async def _acquire(self, record: LlmCallRecord) -> None:
        first_in_line = False
        while True:
            waiter = asyncio.get_running_loop().create_future()
            if first_in_line:
                self._waiters.appendleft((waiter, record))
            else:
                self._waiters.append((waiter, record))
            self._wake()
            try:
                await waiter
            except asyncio.CancelledError:
                if not waiter.cancelled():  # The slot came just before the cancellation did.
                    self._release()
                raise
            # The slot came, but the limit may have halved, or the breaker opened, since.
            if self._in_flight <= self._limit and not self._is_open(self._clock.monotonic()):
                return
            self._in_flight -= 1  # Give it back, and wait again, keeping this call's turn.
            first_in_line = True

    def _release(self) -> None:
        self._in_flight -= 1
        self._wake()

    def _wake(self) -> None:
        """Hand the free slots to the calls waiting longest; skip any cancelled meanwhile.

        While the breaker is open no slot goes out: one task wakes the queue when it closes.
        """
        now = self._clock.monotonic()
        if self._is_open(now):
            if self._waiters and self._closer is None:
                self._closer = asyncio.get_running_loop().create_task(self._wake_when_closed())
            return
        while self._waiters and self._in_flight < self._limit:
            waiter, record = self._waiters.popleft()
            if not waiter.done():
                self._close_if_due(now, record)
                self._in_flight += 1
                waiter.set_result(None)

    async def _wake_when_closed(self) -> None:
        try:
            await self._clock.sleep(self._open_until - self._clock.monotonic())
        finally:
            self._closer = None
        self._wake()

    def _is_open(self, now: float) -> bool:
        return now < self._open_until

    def _halve(
        self, status: LlmCallStatus.ValueType, http_status: int, record: LlmCallRecord
    ) -> None:
        previous, self._limit = self._limit, max(1, self._limit // 2)
        if self._limit < previous:
            _logger.warning(
                "halved the LLM concurrency limit",
                extra={
                    **self._context(record),
                    "event": "llm_limit_down",
                    "limit": self._limit,
                    "previous_limit": previous,
                    "share": self.share,
                    "after": LlmCallStatus.Name(status),
                    "http_status": http_status,
                },
            )

    def _grow(self, record: LlmCallRecord) -> None:
        previous, self._limit = self._limit, self._limit + 1
        _logger.info(
            "raised the LLM concurrency limit",
            extra={
                **self._context(record),
                "event": "llm_limit_up",
                "limit": self._limit,
                "previous_limit": previous,
                "share": self.share,
            },
        )
        self._wake()

    def _judge(self, failed: bool, sent_at: float, now: float, record: LlmCallRecord) -> None:
        """Count an attempt that just ended, and open the breaker if too many failed."""
        self._close_if_due(now, record)
        if sent_at <= self._opened_at:  # It went out before the breaker last opened.
            return
        self._window.append((now, failed))
        self._failures += failed
        horizon = now - self._policy.breaker_window_s
        while self._window[0][0] <= horizon:  # Stops at the attempt just added, at the latest.
            _, old = self._window.popleft()
            self._failures -= old
        attempts = len(self._window)
        if (
            self._is_open(now)
            or attempts < self._policy.breaker_min_attempts
            or self._failures / attempts <= self._policy.breaker_failure_ratio
        ):
            return
        self._opened_at = now
        self._open_until = now + self._policy.breaker_open_s
        self._opened = True
        _logger.warning(
            "opened the LLM circuit breaker: no new requests to the model for a while",
            extra={
                **self._context(record),
                "event": "llm_breaker_open",
                "failures": self._failures,
                "attempts": attempts,
                "open_s": self._policy.breaker_open_s,
            },
        )
        self._window.clear()  # Once it closes, the breaker judges afresh.
        self._failures = 0

    def _close_if_due(self, now: float, record: LlmCallRecord) -> None:
        """Log the breaker's closing, if it is past its open period and not logged yet."""
        if self._opened and not self._is_open(now):
            self._opened = False
            _logger.info(
                "closed the LLM circuit breaker",
                extra={
                    **self._context(record),
                    "event": "llm_breaker_closed",
                    "open_s": self._open_until - self._opened_at,
                },
            )

    def _context(self, record: LlmCallRecord) -> dict[str, object]:
        return {
            "model": self.model,
            "purpose": CallPurpose.Name(record.purpose),
            "call_id": record.call_id,
        }
