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
from collections.abc import AsyncIterator, Callable, Sequence
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
        self._resources = tuple(resources)
        self._free = list(resources)
        self._waiting: list[tuple[int, int, asyncio.Future[R], Callable[[R], bool]]] = []
        self._arrivals = itertools.count()

    @property
    def resources(self) -> tuple[R, ...]:
        """Every resource, lent out or not."""
        return self._resources

    @property
    def free_count(self) -> int:
        return len(self._free)

    @property
    def waiting_count(self) -> int:
        return sum(not future.done() for _, _, future, _ in self._waiting)

    @asynccontextmanager
    async def lease(
        self, priority: Priority, *, which: Callable[[R], bool] | None = None
    ) -> AsyncIterator[R]:
        """Wait for a resource, hold it for the body, then hand it to the next waiter.

        With ``which``, wait for a resource it accepts, such as one particular instance.
        """
        resource = await self._acquire(priority, which)
        try:
            yield resource
        finally:
            self._release(resource)

    async def _acquire(self, priority: Priority, which: Callable[[R], bool] | None = None) -> R:
        accepts = which or _any
        for index, free in enumerate(self._free):
            # A free resource that a waiter accepts is that waiter's, by priority and arrival.
            if accepts(free) and not self._wanted(free):
                return self._free.pop(index)
        future: asyncio.Future[R] = asyncio.get_running_loop().create_future()
        heapq.heappush(self._waiting, (int(priority), next(self._arrivals), future, accepts))
        try:
            return await future
        except asyncio.CancelledError:
            if future.done() and not future.cancelled():
                self._release(future.result())  # Handed over just as the wait was cancelled.
            raise

    def _wanted(self, resource: R) -> bool:
        return any(not f.done() and accepts(resource) for _, _, f, accepts in self._waiting)

    def _release(self, resource: R) -> None:
        skipped = []
        try:
            while self._waiting:
                waiter = heapq.heappop(self._waiting)
                _, _, future, accepts = waiter
                if future.done():
                    continue
                if accepts(resource):
                    future.set_result(resource)
                    return
                skipped.append(waiter)
            self._free.append(resource)
        finally:
            for waiter in skipped:
                heapq.heappush(self._waiting, waiter)


def _any(_resource: object) -> bool:
    return True
