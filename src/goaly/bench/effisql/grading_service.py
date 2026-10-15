"""The async face of the Effi-SQL grader: one worker per grading instance, and the priority pool.

Each grading instance gets a :class:`SlotRunner` that runs its evaluations one at a time:

- :class:`ProcessRunner`, for real grading, runs them in a worker process of its own. Fetching
  and hashing a result is Python work. Measured on 2M lineitem rows, three threads in one
  process each took 4 times as long as one thread alone, and five threads 7 times, since they
  share one interpreter lock. Results reach 18-20M rows, so threads would push the largest
  tasks past the fetch's time limit.
- :class:`ThreadRunner` runs them in a thread of this process: for tests, and for one instance.

:class:`EffiSqlGrader` leases a runner from the pool by priority (goaly.bench.effisql.
grading_pool), runs the evaluation on it, and never raises for grading trouble: the Grade's
status says what went wrong. A cancelled grading stops its statements before it frees the
instance, so no orphaned statement can disturb the next evaluation's timing.
"""

import asyncio
import logging
import multiprocessing
import threading
import time
from collections.abc import Awaitable, Callable, Sequence
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Self

from goaly.bench.effisql.data import EffiSqlInstance
from goaly.bench.effisql.grading import (
    AbortFlag,
    BaseCache,
    GradingSlot,
    InstanceCheck,
    Mode,
    grader_error,
    prediction_sql,
)
from goaly.bench.effisql.grading_pool import GradingInstance, GradingPool, Priority
from goaly.bench.effisql.pg_build import Connect, connector
from goaly.bench.effisql.pg_settings import GRADER_ROLE, load_credentials
from goaly.contracts.gen.task_pb2 import Grade, Prediction

__all__ = [
    "EffiSqlGrader",
    "ProcessRunner",
    "SlotRunner",
    "ThreadRunner",
    "WorkerSpec",
    "grade_on_slot",
]

_log = logging.getLogger(__name__)

_ABORT_POLL_S = 0.2  # How often a worker's abort watcher checks that an abort is over.


class SlotRunner(Protocol):
    """Runs work on one grading instance's slot, one call at a time."""

    @property
    def instance(self) -> GradingInstance: ...

    async def call[T](self, work: Callable[..., T], *args: Any) -> T:
        """Return ``work(slot, *args)``, run on the slot. ``work`` is a module-level function.

        If the caller is cancelled, stop the work's statements and wait for it to end first.
        """
        ...

    def close(self) -> None: ...


class ThreadRunner:
    """Runs a slot's work in one thread of this process."""

    def __init__(self, slot: GradingSlot) -> None:
        self.slot = slot
        self._executor = ThreadPoolExecutor(1, thread_name_prefix=f"grade-{slot.instance.name}")

    @property
    def instance(self) -> GradingInstance:
        return self.slot.instance

    async def call[T](self, work: Callable[..., T], *args: Any) -> T:
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self._executor, work, self.slot, *args)
        return await _shielded(future, self._abort)

    def _abort(self) -> None:
        sessions = self.slot.open_sessions
        if sessions is not None:
            sessions.abort()

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        self.slot.reset()


@dataclass(frozen=True, slots=True)
class WorkerSpec:
    """What a worker process needs to open its slot; it holds no secret."""

    instance: GradingInstance
    host: str = "127.0.0.1"
    check: InstanceCheck | None = None  # A module-level function, so it can be pickled.

    def connect(self) -> Connect:
        """Connect as the grader's role, with the password from the credentials file."""
        password = load_credentials().grader_password
        return connector(self.host, self.instance.port, password, user=GRADER_ROLE)


_worker_slot: GradingSlot | None = None


def _start_worker(spec: WorkerSpec, aborted: AbortFlag) -> None:
    global _worker_slot  # The worker process's one slot.
    connect = spec.connect()
    slot = GradingSlot(spec.instance, lambda _: connect, check=spec.check, aborted=aborted)
    _worker_slot = slot
    threading.Thread(target=_watch_aborts, args=(slot, aborted), daemon=True).start()


def _watch_aborts(slot: GradingSlot, aborted: AbortFlag) -> None:
    """Cancel the slot's statements whenever the parent sets ``aborted``.

    Only the connection that runs a statement holds its cancel key, and the grader's role may
    not cancel other sessions, so the cancel comes from inside the worker.
    """
    while True:
        aborted.wait()
        sessions = slot.open_sessions
        if sessions is not None:
            sessions.abort()
        while aborted.is_set():  # The next evaluation clears it.
            time.sleep(_ABORT_POLL_S)


def _run_in_worker[T](work: Callable[..., T], args: tuple[Any, ...]) -> T:
    if _worker_slot is None:
        raise RuntimeError("the grading worker was not started")
    return work(_worker_slot, *args)


class ProcessRunner:
    """Runs a slot's work in a worker process of its own, started with spawn."""

    def __init__(self, spec: WorkerSpec) -> None:
        self.spec = spec
        context = multiprocessing.get_context("spawn")  # Never fork a process with threads.
        self._aborted = context.Event()
        self._executor: Executor = ProcessPoolExecutor(
            1, mp_context=context, initializer=_start_worker, initargs=(spec, self._aborted)
        )

    @property
    def instance(self) -> GradingInstance:
        return self.spec.instance

    async def call[T](self, work: Callable[..., T], *args: Any) -> T:
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self._executor, _run_in_worker, work, args)
        return await _shielded(future, self._abort)

    def _abort(self) -> None:
        self._aborted.set()  # The worker's watcher cancels the statements.

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)


async def _shielded[T](future: asyncio.Future[T], abort: Callable[[], None]) -> T:
    """Await ``future``; if cancelled, run ``abort`` and wait for ``future`` before re-raising."""
    try:
        return await asyncio.shield(future)
    except asyncio.CancelledError:
        await _finish(asyncio.get_running_loop().run_in_executor(None, abort))
        await _finish(future)
        raise


async def _finish[T](future: asyncio.Future[T]) -> None:
    """Wait for ``future`` however often the waiting task is cancelled meanwhile."""
    while not future.done():
        try:
            await asyncio.shield(future)
        except asyncio.CancelledError:
            continue  # Cancelled again: keep waiting.
        except Exception:  # noqa: BLE001  # The cancelled work's outcome is discarded.
            break


def grade_on_slot(
    slot: GradingSlot, inst: EffiSqlInstance, sql: str | None, mode: Mode, cache: BaseCache | None
) -> Grade:
    """Grade on ``slot``: the work a runner runs for :meth:`EffiSqlGrader.grade`."""
    return slot.evaluate(inst, sql, mode, cache)


class EffiSqlGrader:
    """Grades predictions on a pool of grading instances, in dev or official mode.

    Each mode has its own directory under ``cache_root``: dev mode's holds the base cache it
    reads, and official mode reads and writes none.
    """

    def __init__(self, pool: GradingPool[SlotRunner], *, cache_root: Path) -> None:
        self._pool = pool
        self._caches = {mode: BaseCache(cache_root / mode.value) for mode in Mode}

    @classmethod
    def on_instances(
        cls,
        instances: Sequence[GradingInstance],
        *,
        cache_root: Path,
        check: InstanceCheck | None = None,
    ) -> Self:
        """A grader with a worker process per instance; :meth:`close` stops them."""
        runners: list[SlotRunner] = [ProcessRunner(WorkerSpec(i, check=check)) for i in instances]
        return cls(GradingPool(runners), cache_root=cache_root)

    def cache(self, mode: Mode) -> BaseCache:
        return self._caches[mode]

    @property
    def pool(self) -> GradingPool[SlotRunner]:
        return self._pool

    async def grade(
        self,
        inst: EffiSqlInstance,
        prediction: Prediction,
        mode: Mode,
        *,
        priority: Priority = Priority.RSI,
    ) -> Grade:
        """Grade ``prediction`` for ``inst``. Grading trouble sets the status; it never raises."""
        sql = prediction_sql(inst, prediction)
        cache = self._caches[Mode.DEV] if mode is Mode.DEV else None
        try:
            async with self._pool.lease(priority) as runner:
                return await runner.call(grade_on_slot, inst, sql, mode, cache)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001  # Never raise for grading trouble.
            _log.exception("grading failed", extra={"instance_id": inst.instance_id})
            return grader_error(inst, f"grader: {type(error).__name__}: {error}")

    def benchmark_grade(
        self, mode: Mode, *, priority: Priority = Priority.RSI
    ) -> Callable[[EffiSqlInstance, Prediction], Awaitable[Grade]]:
        """Return ``Benchmark.grade`` for the adapter: this grader in one mode and priority."""

        async def grade(inst: EffiSqlInstance, prediction: Prediction) -> Grade:
            return await self.grade(inst, prediction, mode, priority=priority)

        return grade

    def close(self) -> None:
        """Stop every runner; call it once no grading is running."""
        for runner in self._pool.resources:
            runner.close()
