"""The async face of the Effi-SQL grader: one worker per grading instance, and the priority pool.

Each grading instance gets a :class:`SlotRunner` that runs its evaluations one at a time:

- :class:`ProcessRunner`, for real grading, runs them in a worker process of its own. Fetching
  and hashing a result is Python work. Measured on 2M lineitem rows, three threads in one
  process each took 4 times as long as one thread alone, and five threads 7 times, since they
  share one interpreter lock. Results reach 18-20M rows, so threads would push the largest
  tasks past the fetch's time limit. A worker that dies is replaced, and the instance is held
  back with a growing pause after repeated deaths, so a broken worker cannot drain the queue.
  A file lock keeps a second grader process off the same instance.
- :class:`ThreadRunner` runs them in a thread of this process: for tests, and for one instance.

Every call gets a token, and a cancel stops only the call with that token
(goaly.bench.effisql.grading.GradingSlot), so a late cancel never reaches the next evaluation.
:class:`EffiSqlGrader` grades what needs no database at once, leases a runner from the pool by
priority (goaly.bench.effisql.grading_pool, the one scheduler of the grading instances, with
``goaly-smoke`` first) for the rest, and never raises for grading trouble: the Grade's status
says what went wrong. A cancelled grading stops its statements before it frees the instance.
"""

import asyncio
import contextlib
import fcntl
import itertools
import logging
import multiprocessing
import multiprocessing.synchronize
import os
import threading
from collections.abc import Awaitable, Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Protocol, Self

from goaly.bench.effisql.data import EffiSqlInstance
from goaly.bench.effisql.grader import GraderError, TimingError, time_interleaved
from goaly.bench.effisql.grading import (
    BaseCache,
    GradingSlot,
    InstanceCheck,
    Mode,
    grader_error,
    prediction_sql,
    quick_grade,
)
from goaly.bench.effisql.grading_pool import GradingInstance, GradingPool, Priority
from goaly.bench.effisql.pg_build import Connect, connector
from goaly.bench.effisql.pg_settings import GRADER_ROLE, load_credentials
from goaly.contracts.gen.task_pb2 import GRADE_STATUS_GRADER_ERROR, Grade, Prediction

__all__ = [
    "EffiSqlGrader",
    "ProcessRunner",
    "SlotRunner",
    "ThreadRunner",
    "WorkerSpec",
    "default_lock_dir",
    "grade_on_slot",
    "probe_identity",
]

if TYPE_CHECKING:  # Generic only to type checkers; a plain class at run time.
    from multiprocessing.sharedctypes import Synchronized

_log = logging.getLogger(__name__)

_BACKOFF_AFTER = 2  # A worker's deaths in a row before its instance is held back.
_BACKOFF_STEP_S = 30.0  # Each further death in a row holds the instance back this much more.
_BACKOFF_MAX_S = 600.0
_CLOSE_JOIN_S = 30.0  # How long close() waits for a worker before it terminates it.
_GRADER_ERROR_ATTEMPTS = 2  # A grader failure is retried once (the controller's ruling).


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
        self._tokens = itertools.count()

    @property
    def instance(self) -> GradingInstance:
        return self.slot.instance

    async def call[T](self, work: Callable[..., T], *args: Any) -> T:
        token = next(self._tokens)
        future = asyncio.get_running_loop().run_in_executor(
            self._executor, self.slot.run, token, lambda slot: work(slot, *args)
        )
        return await _shielded(future, lambda: self.slot.abort(token), inline=False)

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        self.slot.reset()


def default_lock_dir() -> Path:
    """Where grader processes take their per-instance locks: ``~/.cache/goaly/grading``."""
    return Path.home() / ".cache" / "goaly" / "grading"


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


def _start_worker(
    spec: WorkerSpec, cancel: "Synchronized[int]", poke: multiprocessing.synchronize.Event
) -> None:
    global _worker_slot  # The worker process's one slot.
    slot = GradingSlot(
        spec.instance, lambda _: spec.connect(), check=spec.check, cancelled=lambda: cancel.value
    )
    _worker_slot = slot
    threading.Thread(target=_watch_aborts, args=(slot, cancel, poke), daemon=True).start()


def _watch_aborts(
    slot: GradingSlot,
    cancel: "Synchronized[int]",
    poke: multiprocessing.synchronize.Event,
    stop: threading.Event | None = None,
) -> None:
    """Abort the call whose token the parent wrote to ``cancel``, each time it pokes.

    Only the connection that runs a statement holds its cancel key, and the grader's role may
    not cancel other sessions, so the cancel comes from inside the worker.
    """
    while stop is None or not stop.is_set():
        if not poke.wait(0.5):
            continue
        poke.clear()
        slot.abort(cancel.value)


def _run_in_worker[T](token: int, work: Callable[..., T], args: tuple[Any, ...]) -> T:
    if _worker_slot is None:
        raise RuntimeError("the grading worker was not started")
    return _worker_slot.run(token, lambda slot: work(slot, *args))


class ProcessRunner:
    """Runs a slot's work in a worker process of its own, started with spawn.

    It takes the instance's lock under ``lock_dir`` for its lifetime, so no other grader process
    can grade on the same instance, and raises GraderError if another holds it.
    """

    def __init__(self, spec: WorkerSpec, *, lock_dir: Path | None = None) -> None:
        self.spec = spec
        self._lock_file = _lock_instance(lock_dir or default_lock_dir(), spec.instance)
        self._context = multiprocessing.get_context("spawn")  # Never fork with threads around.
        self._tokens = itertools.count()
        self._deaths = 0
        self._start()

    def _start(self) -> None:
        self._cancel: Synchronized[int] = self._context.Value("q", -1)
        self._poke = self._context.Event()
        self._executor = ProcessPoolExecutor(
            1,
            mp_context=self._context,
            initializer=_start_worker,
            initargs=(self.spec, self._cancel, self._poke),
        )

    @property
    def instance(self) -> GradingInstance:
        return self.spec.instance

    async def call[T](self, work: Callable[..., T], *args: Any) -> T:
        token = next(self._tokens)
        future = asyncio.get_running_loop().run_in_executor(
            self._executor, _run_in_worker, token, work, args
        )
        try:
            result = await _shielded(future, lambda: self._abort(token), inline=True)
        except BrokenProcessPool:
            await self._replace_dead_worker()
            raise GraderError(
                f"the grading worker for {self.instance.name} died; it was replaced"
            ) from None
        self._deaths = 0
        return result

    def _abort(self, token: int) -> None:
        self._cancel.value = token
        self._poke.set()  # The worker's watcher cancels the statements.

    async def _replace_dead_worker(self) -> None:
        self._deaths += 1
        _log.error("grading worker died", extra={"instance": self.instance.name})
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._start()
        # The dead worker's statement ends by the server's statement_timeout. Hold the instance
        # back after repeated deaths, so this runner does not win every lease and fail it.
        if self._deaths >= _BACKOFF_AFTER:
            pause = min(_BACKOFF_STEP_S * (self._deaths - _BACKOFF_AFTER + 1), _BACKOFF_MAX_S)
            await asyncio.sleep(pause)

    def close(self) -> None:
        """Stop the worker, waiting a bounded time for it, then release the instance's lock.

        It blocks: call it once no grading runs, outside the event loop or at its end.
        """
        processes = list(getattr(self._executor, "_processes", {}).values())
        self._executor.shutdown(wait=False, cancel_futures=False)
        for process in processes:
            process.join(_CLOSE_JOIN_S)
            if process.is_alive():
                _log.warning("terminating a hung grading worker", extra={"pid": process.pid})
                process.terminate()
        self._lock_file.close()


def _lock_instance(lock_dir: Path, instance: GradingInstance) -> IO[str]:
    lock_dir.mkdir(parents=True, exist_ok=True)
    handle = (lock_dir / f"{instance.name}.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise GraderError(f"another grader process is grading on {instance.name}") from None
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    return handle


async def _shielded[T](future: asyncio.Future[T], abort: Callable[[], None], *, inline: bool) -> T:
    """Await ``future``; if cancelled, run ``abort`` and wait for ``future`` before re-raising.

    ``abort`` runs inline when it cannot block, else on the loop's default executor.
    """
    try:
        return await asyncio.shield(future)
    except asyncio.CancelledError:
        if inline:
            abort()
        else:
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


def probe_identity(slot: GradingSlot, inst: EffiSqlInstance) -> float:
    """Time ``inst``'s base against itself on ``slot``; the work of an identity probe."""
    try:
        sessions = slot.sessions()

        def time_base() -> float:
            return sessions.main.execution_ms(inst.base_sql)

        return time_interleaved(time_base, time_base, rng=slot.rng).speedup
    except TimingError as error:
        slot.reset()
        raise GraderError(f"the identity probe failed: {error}") from None
    except BaseException:
        slot.reset()
        raise


def grade_on_slot(
    slot: GradingSlot, inst: EffiSqlInstance, sql: str | None, mode: Mode, cache: BaseCache | None
) -> Grade:
    """Grade on ``slot``: the work a runner runs for :meth:`EffiSqlGrader.grade`."""
    return slot.evaluate(inst, sql, mode, cache)


class EffiSqlGrader:
    """Grades predictions on a pool of grading instances, in dev or official mode.

    Dev mode reads the base cache under ``cache_root / "dev"``; official mode reads and writes
    no cache.
    """

    def __init__(self, pool: GradingPool[SlotRunner], *, cache_root: Path) -> None:
        self._pool = pool
        self._dev_cache = BaseCache(cache_root / Mode.DEV.value)

    @classmethod
    def on_instances(
        cls,
        instances: Sequence[GradingInstance],
        *,
        cache_root: Path,
        check: InstanceCheck | None = None,
        lock_dir: Path | None = None,
    ) -> Self:
        """A grader with a worker process per instance; :meth:`close` stops them."""
        runners: list[SlotRunner] = []
        try:
            for instance in instances:
                runners.append(ProcessRunner(WorkerSpec(instance, check=check), lock_dir=lock_dir))
        except GraderError:
            for runner in runners:
                runner.close()
            raise
        return cls(GradingPool(runners), cache_root=cache_root)

    @property
    def dev_cache(self) -> BaseCache:
        return self._dev_cache

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
        cache = self._dev_cache if mode is Mode.DEV else None
        try:
            quick = quick_grade(inst, sql, mode, cache)
            if quick is not None:
                return quick
            for attempt in range(1, _GRADER_ERROR_ATTEMPTS + 1):
                async with self._pool.lease(priority) as runner:
                    grade = await runner.call(grade_on_slot, inst, sql, mode, cache)
                if grade.status != GRADE_STATUS_GRADER_ERROR or attempt == _GRADER_ERROR_ATTEMPTS:
                    return grade
                _log.warning(
                    "retrying a grader failure",
                    extra={"instance_id": inst.instance_id, "feedback": grade.feedback},
                )
        except Exception as error:  # noqa: BLE001  # Never raise for grading trouble.
            _log.exception("grading failed", extra={"instance_id": inst.instance_id})
            return grader_error(inst, f"grader: {type(error).__name__}: {error}")
        raise AssertionError("unreachable")  # pragma: no cover

    async def identity_probe(
        self, inst: EffiSqlInstance, *, instance: GradingInstance, priority: Priority = Priority.RSI
    ) -> float:
        """Measure ``inst``'s identity τ on ``instance`` now: its base against itself, with the
        interleaved protocol, as the setup measured it (§6.2's periodic identity probes).

        It holds that grading instance for about 6 base runs (4 for a base of 60 s or more).
        Compare the result with :meth:`identity_reference`. Raises GraderError if the probe
        could not run.
        """
        async with self._pool.lease(priority, which=lambda r: r.instance == instance) as runner:
            return await runner.call(probe_identity, inst)

    def identity_reference(self, inst: EffiSqlInstance) -> float | None:
        """Return the identity τ the setup measured for ``inst``, if the base cache has it."""
        entry = self._dev_cache.get(inst)
        return None if entry is None else entry.identity_speedup

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
            with contextlib.suppress(Exception):
                runner.close()
