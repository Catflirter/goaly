"""The grading instances, and which grading gets one next (spec §6.2).

Grading capacity binds the whole day, so each instance runs one evaluation at a time and serves
waiting work by phase priority: ``goaly-smoke`` gradings first, then setup, v0's runs, RSI and
the final runs. Within a priority, work is served in arrival order. :class:`GradingPool` lends
out any kind of resource; the grader lends grading slots (goaly.bench.effisql.grading).

This is the scheduler's side of T09's grading-instance pool. Until T09's pool lands, it holds
the instances itself.
"""

import asyncio
import enum
import heapq
import itertools
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass

__all__ = [
    "GRADING_INSTANCES",
    "SETUP_INSTANCES",
    "GradingInstance",
    "GradingPool",
    "Priority",
]


class Priority(enum.IntEnum):
    """A grading's phase priority: a lower value is served first (spec §6.2)."""

    SMOKE = 0  # goaly-smoke: at most 2 per hypothesis, ahead of every phase.
    SETUP = 1  # The one-time setup and exclusion run (T10).
    BASELINE = 2  # v0's dev runs on train and val (T12).
    BASELINE_HELDOUT = 3  # v0's official held-out run, ahead of the RSI lane.
    RSI = 4  # Screening, validation and confirmation in the RSI rounds (T22).
    FINAL = 5  # The final runs (T23), the only phase then.


@dataclass(frozen=True, slots=True)
class GradingInstance:
    """A PostgreSQL instance that grades: its name and its port on 127.0.0.1."""

    name: str
    port: int


# The grading instances (T09). Each holds tpch and tpch_alt.
GRADING_INSTANCES = (
    GradingInstance("grade-1", 55434),
    GradingInstance("grade-2", 55435),
    GradingInstance("grade-3", 55436),
)
# The setup grades on all 5, since nothing solves before T12 (spec §6.2).
SETUP_INSTANCES = (
    *GRADING_INSTANCES,
    GradingInstance("solve-1", 55432),
    GradingInstance("solve-2", 55433),
)


class GradingPool[R]:
    """Lends each resource to one holder at a time, by priority, then in arrival order.

    Use it from one event loop.
    """

    def __init__(self, resources: Sequence[R]) -> None:
        if not resources:
            raise ValueError("a grading pool needs at least one resource")
        self._free = list(resources)
        self._waiting: list[tuple[int, int, asyncio.Future[R]]] = []
        self._arrivals = itertools.count()

    @property
    def free_count(self) -> int:
        return len(self._free)

    @property
    def waiting_count(self) -> int:
        return sum(not future.done() for _, _, future in self._waiting)

    @asynccontextmanager
    async def lease(self, priority: Priority) -> AsyncIterator[R]:
        """Wait for a resource, hold it for the body, then hand it to the next waiter."""
        resource = await self._acquire(priority)
        try:
            yield resource
        finally:
            self._release(resource)

    async def _acquire(self, priority: Priority) -> R:
        if self._free and not self.waiting_count:
            return self._free.pop(0)
        future: asyncio.Future[R] = asyncio.get_running_loop().create_future()
        heapq.heappush(self._waiting, (int(priority), next(self._arrivals), future))
        try:
            return await future
        except asyncio.CancelledError:
            if future.done() and not future.cancelled():
                self._release(future.result())  # Handed over just as the wait was cancelled.
            raise

    def _release(self, resource: R) -> None:
        while self._waiting:
            _, _, future = heapq.heappop(self._waiting)
            if not future.done():
                future.set_result(resource)
                return
        self._free.append(resource)
