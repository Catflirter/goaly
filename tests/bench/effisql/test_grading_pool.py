"""Unit tests of the grading pool, the grading slots and the async grader."""

import asyncio
import functools
import multiprocessing
import random
import threading
import time
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

import psycopg
import pytest

from goaly.bench.effisql import grading, grading_service
from goaly.bench.effisql.data import EffiSqlInstance
from goaly.bench.effisql.grader import GraderError, PgSession, QueryError
from goaly.bench.effisql.grading import BaseCache, GradingSlot, Mode
from goaly.bench.effisql.grading_pool import (
    GRADING_INSTANCES,
    SETUP_INSTANCES,
    GradingInstance,
    GradingPool,
    Priority,
)
from goaly.bench.effisql.grading_service import EffiSqlGrader, ThreadRunner, WorkerSpec
from goaly.contracts.gen.task_pb2 import (
    GRADE_STATUS_GRADED,
    GRADE_STATUS_GRADER_ERROR,
    GRADE_STATUS_PREDICTION_FAILED,
    EffiSqlPrediction,
    Grade,
    Prediction,
)
from tests.bench.effisql import grading_fakes, worker_helpers
from tests.bench.effisql.grading_fakes import BASE_SQL, INST, FakeSession, entry


def run_async[**P](test: Callable[P, Coroutine[Any, Any, None]]) -> Callable[P, None]:
    """Run an async test on a fresh event loop; the project has no async pytest plugin."""

    @functools.wraps(test)
    def run(*args: P.args, **kwargs: P.kwargs) -> None:
        asyncio.run(test(*args, **kwargs))

    return run


def test_the_instances() -> None:
    assert [i.port for i in GRADING_INSTANCES] == [55434, 55435, 55436]
    assert len(SETUP_INSTANCES) == 5


class TestGradingPool:
    @run_async
    async def test_lends_each_resource_to_one_holder_at_a_time(self) -> None:
        pool = GradingPool(["a", "b"])
        async with pool.lease(Priority.RSI) as first, pool.lease(Priority.RSI) as second:
            assert {first, second} == {"a", "b"}
            assert pool.free_count == 0
        assert pool.free_count == 2

    @run_async
    async def test_smoke_is_served_ahead_of_queued_phase_work(self) -> None:
        pool = GradingPool(["only"])
        order: list[str] = []

        async def wait(name: str, priority: Priority) -> None:
            async with pool.lease(priority):
                order.append(name)

        async with pool.lease(Priority.RSI):
            tasks = [
                asyncio.create_task(wait("rsi-1", Priority.RSI)),
                asyncio.create_task(wait("final", Priority.FINAL)),
                asyncio.create_task(wait("rsi-2", Priority.RSI)),
                asyncio.create_task(wait("setup", Priority.SETUP)),
            ]
            await asyncio.sleep(0)
            tasks.append(asyncio.create_task(wait("smoke", Priority.SMOKE)))
            await asyncio.sleep(0)
            assert pool.waiting_count == 5
        await asyncio.gather(*tasks)
        assert order == ["smoke", "setup", "rsi-1", "rsi-2", "final"]

    @run_async
    async def test_a_cancelled_waiter_is_skipped(self) -> None:
        pool = GradingPool(["only"])
        async with pool.lease(Priority.RSI):
            waiter = asyncio.create_task(pool._acquire(Priority.SMOKE))
            await asyncio.sleep(0)
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
        assert pool.free_count == 1

    @run_async
    async def test_a_resource_handed_to_a_cancelled_waiter_goes_back(self) -> None:
        pool = GradingPool(["only"])
        async with pool.lease(Priority.RSI):
            waiter = asyncio.create_task(pool._acquire(Priority.SMOKE))
            await asyncio.sleep(0)
        waiter.cancel()  # Handed over, but cancelled before it ran.
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert pool.free_count == 1

    def test_needs_a_resource(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            GradingPool([])


class FakeSlot:
    """A slot whose evaluate is the test's; run and abort follow GradingSlot's contract."""

    instance = GradingInstance("fake", 1)

    def __init__(self, work: Any = None) -> None:
        self.resets = 0
        self.calls: list[tuple[str, str | None, Mode, BaseCache | None]] = []
        self.work = work
        self.aborts: list[int] = []
        self.running: int | None = None

    def run(self, token: int, work: Callable[["FakeSlot"], Any]) -> Any:
        self.running = token
        try:
            return work(self)
        finally:
            self.running = None

    def abort(self, token: int) -> None:
        self.aborts.append(token)

    def evaluate(
        self, inst: EffiSqlInstance, sql: str | None, mode: Mode, cache: BaseCache | None
    ) -> Grade:
        self.calls.append((inst.instance_id, sql, mode, cache))
        if self.work is not None:
            return self.work()  # type: ignore[no-any-return]  # The test's own work.
        return Grade(instance_id=inst.instance_id, status=GRADE_STATUS_GRADED)

    def reset(self) -> None:
        self.resets += 1


def grader_on(slot: FakeSlot, tmp_path: Path) -> EffiSqlGrader:
    runner = ThreadRunner(slot)  # type: ignore[arg-type]  # A fake slot.
    return EffiSqlGrader(GradingPool([runner]), cache_root=tmp_path)


def answer(sql: str) -> Prediction:
    return Prediction(effisql=EffiSqlPrediction(sql=sql))


class TestEffiSqlGrader:
    @run_async
    async def test_dev_mode_reads_its_own_cache_and_official_none(self, tmp_path: Path) -> None:
        slot = FakeSlot()
        grader = grader_on(slot, tmp_path)
        await grader.grade(INST, answer("SELECT 1"), Mode.DEV)
        await grader.grade(INST, answer("SELECT 1"), Mode.OFFICIAL)
        dev_cache = slot.calls[0][3]
        assert dev_cache is grader.dev_cache
        assert dev_cache.directory == tmp_path / "dev"
        assert slot.calls[1][3] is None

    @run_async
    async def test_no_branch_is_graded_as_base_sql(self, tmp_path: Path) -> None:
        slot = FakeSlot()
        await grader_on(slot, tmp_path).grade(INST, Prediction(), Mode.DEV)
        assert slot.calls[0][1] == BASE_SQL

    @run_async
    async def test_what_needs_no_database_is_graded_without_a_lease(self, tmp_path: Path) -> None:
        slot = FakeSlot()
        grader = grader_on(slot, tmp_path)
        grader.dev_cache.put(entry())
        empty = await grader.grade(INST, answer("  "), Mode.DEV)
        identity = await grader.grade(INST, Prediction(), Mode.DEV)
        assert empty.status == GRADE_STATUS_PREDICTION_FAILED
        assert identity.effisql.identity_cached
        assert slot.calls == []
        official = await grader.grade(INST, Prediction(), Mode.OFFICIAL)
        assert official.status == GRADE_STATUS_GRADED
        assert len(slot.calls) == 1, "official mode always measures"

    @run_async
    async def test_an_unexpected_failure_is_a_grader_error(self, tmp_path: Path) -> None:
        def broken() -> Grade:
            raise RuntimeError("bug")

        grade = await grader_on(FakeSlot(broken), tmp_path).grade(INST, Prediction(), Mode.DEV)
        assert grade.status == GRADE_STATUS_GRADER_ERROR
        assert grade.instance_id == INST.instance_id
        assert "RuntimeError: bug" in grade.feedback

    @run_async
    async def test_benchmark_grade_binds_the_mode(self, tmp_path: Path) -> None:
        slot = FakeSlot()
        grade = grader_on(slot, tmp_path).benchmark_grade(Mode.OFFICIAL, priority=Priority.SMOKE)
        result = await grade(INST, Prediction())
        assert result.status == GRADE_STATUS_GRADED
        assert slot.calls[0][2] is Mode.OFFICIAL

    @run_async
    async def test_cancelling_aborts_the_call_before_freeing_the_slot(self, tmp_path: Path) -> None:
        started, release = threading.Event(), threading.Event()

        def blocking() -> Grade:
            started.set()
            release.wait(5)
            return Grade()

        slot = FakeSlot(blocking)

        def abort(token: int) -> None:
            slot.aborts.append(token)
            release.set()

        slot.abort = abort  # type: ignore[method-assign]  # Releases the blocked call.
        grader = grader_on(slot, tmp_path)
        await grader.grade(INST, answer("SELECT 1"), Mode.OFFICIAL)  # Call 0 runs to its end.
        slot.work = blocking
        task = asyncio.create_task(grader.grade(INST, answer("SELECT 2"), Mode.OFFICIAL))
        await asyncio.to_thread(started.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert slot.aborts == [1], "only the running call's token is aborted"
        assert grader.pool.free_count == 1
        grader.close()
        assert slot.resets == 1


class TestGradingSlot:
    @pytest.fixture
    def opened(self, monkeypatch: pytest.MonkeyPatch) -> list[FakeSession]:
        opened: list[FakeSession] = []

        def open_session(_connect: Any, db: str, **_: Any) -> FakeSession:
            session = FakeSession()
            session.db = db  # type: ignore[attr-defined]  # Tagged for the test.
            opened.append(session)
            return session

        monkeypatch.setattr(PgSession, "open", open_session)
        monkeypatch.setattr(grading, "read_only_problems", lambda _conn: [])
        return opened

    def slot(self, **kwargs: Any) -> GradingSlot:
        return GradingSlot(GradingInstance("grade-1", 55434), lambda _i: lambda _db: None, **kwargs)  # type: ignore[arg-type,return-value]  # Never called.

    def test_opens_both_databases_once_and_runs_the_check_each_time(
        self, opened: list[FakeSession]
    ) -> None:
        checked: list[str] = []
        slot = self.slot(check=lambda sessions: checked.append(sessions.instance))
        first = slot.sessions()
        assert slot.sessions() is first
        assert [s.db for s in opened] == ["tpch", "tpch_alt"]  # type: ignore[attr-defined]  # Tagged above.
        assert checked == ["grade-1", "grade-1"]
        slot.reset()
        assert all(session.closed for session in opened)
        assert slot.open_sessions is None

    def test_refuses_an_instance_that_is_not_read_only(
        self, opened: list[FakeSession], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(grading, "read_only_problems", lambda _conn: ["may write to region"])
        grade = self.slot().evaluate(INST, "SELECT 1", Mode.DEV, None)
        assert grade.status == GRADE_STATUS_GRADER_ERROR
        assert "unsafe to grade on" in grade.feedback
        assert all(session.closed for session in opened)

    def test_a_failing_role_check_is_a_grader_error(
        self, opened: list[FakeSession], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken(_conn: Any) -> list[str]:
            raise psycopg.OperationalError("lost")

        monkeypatch.setattr(grading, "read_only_problems", broken)
        grade = self.slot().evaluate(INST, "SELECT 1", Mode.DEV, None)
        assert "cannot check the grading role" in grade.feedback
        assert all(session.closed for session in opened)

    def test_closes_the_main_session_if_the_alternate_fails(
        self, opened: list[FakeSession], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def open_session(_connect: Any, db: str, **_: Any) -> FakeSession:
            if db == "tpch_alt":
                raise GraderError("cannot connect to tpch_alt")
            session = FakeSession()
            opened.append(session)
            return session

        monkeypatch.setattr(PgSession, "open", open_session)
        grade = self.slot().evaluate(INST, "SELECT 1", Mode.DEV, None)
        assert grade.status == GRADE_STATUS_GRADER_ERROR
        assert opened[0].closed

    @pytest.mark.parametrize("failure", ["grade", "exception"])
    def test_a_failure_reopens_the_sessions_next_time(
        self, opened: list[FakeSession], monkeypatch: pytest.MonkeyPatch, failure: str
    ) -> None:
        def failing(*_args: Any, **_kwargs: Any) -> Grade:
            if failure == "exception":
                raise KeyError("unexpected")
            return Grade(status=GRADE_STATUS_GRADER_ERROR)

        monkeypatch.setattr(grading, "evaluate", failing)
        slot = self.slot()
        grade = slot.evaluate(INST, "SELECT 1", Mode.DEV, None)
        assert grade.status == GRADE_STATUS_GRADER_ERROR
        assert slot.open_sessions is None
        assert all(session.closed for session in opened)

    def test_abort_stops_only_the_running_call(self, opened: list[FakeSession]) -> None:
        slot = self.slot()
        slot.run(0, lambda s: s.sessions())
        slot.abort(0)  # Late: call 0 has ended.
        sessions = slot.run(1, lambda s: s.sessions())
        assert not sessions.aborted.is_set()
        assert [s.connection.cancels for s in opened] == [0, 0]

        def aborted_while_running(running: GradingSlot) -> None:
            running.abort(2)
            running.sessions().check_abort()

        with pytest.raises(grading.AbortedError):
            slot.run(2, aborted_while_running)
        assert [s.connection.cancels for s in opened] == [1, 1]
        after = slot.run(3, lambda s: s.sessions())
        assert not after.aborted.is_set(), "the next call gets a flag of its own"

    def test_a_cancel_before_the_call_starts_stops_it(self, opened: list[FakeSession]) -> None:
        slot = self.slot()
        slot.abort(5)
        with pytest.raises(grading.AbortedError):
            slot.run(5, lambda s: s.sessions())
        assert opened == [], "it never opened a session"
        slot.run(6, lambda s: s.sessions())

    def test_a_cancel_from_another_process_is_read_at_the_start(
        self, opened: list[FakeSession]
    ) -> None:
        slot = self.slot(cancelled=lambda: 7)
        with pytest.raises(grading.AbortedError):
            slot.run(7, lambda s: s.instance)
        assert slot.run(8, lambda s: s.instance.name) == "grade-1"


class TestWorkerPlumbing:
    def test_the_abort_watcher_aborts_the_token_it_is_given(self) -> None:
        cancel = multiprocessing.get_context("spawn").Value("q", -1)
        poke, stop = threading.Event(), threading.Event()
        aborted: list[int] = []

        class Slot:
            def abort(self, token: int) -> None:
                aborted.append(token)

        watcher = threading.Thread(
            target=grading_service._watch_aborts, args=(Slot(), cancel, poke, stop)
        )
        watcher.start()
        try:
            for token in (3, 4):
                cancel.value = token
                poke.set()
                deadline = time.monotonic() + 5
                while token not in aborted and time.monotonic() < deadline:
                    time.sleep(0.01)
            assert aborted == [3, 4]
        finally:
            stop.set()
            watcher.join(5)
        assert not watcher.is_alive()

    def test_running_work_needs_a_started_worker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(grading_service, "_worker_slot", None)
        with pytest.raises(RuntimeError, match="not started"):
            grading_service._run_in_worker(0, lambda slot: slot, ())

    def test_a_started_worker_runs_work_on_its_slot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        started: list[Any] = []
        monkeypatch.setattr(grading_service, "_worker_slot", None)
        monkeypatch.setattr(
            threading,
            "Thread",
            lambda **kwargs: type("T", (), {"start": lambda _self: started.append(kwargs)})(),
        )
        cancel = multiprocessing.get_context("spawn").Value("q", -1)
        poke = multiprocessing.get_context("spawn").Event()
        grading_service._start_worker(WorkerSpec(GRADING_INSTANCES[0]), cancel, poke)
        assert grading_service._run_in_worker(0, worker_helpers.instance_name, ()) == "grade-1"
        assert started, "the abort watcher was started"

    def test_grade_on_slot_evaluates(self) -> None:
        slot = FakeSlot()
        grade = grading_service.grade_on_slot(slot, INST, "SELECT 1", Mode.DEV, None)  # type: ignore[arg-type]  # A fake.
        assert grade.status == GRADE_STATUS_GRADED


class TestProcessRunner:
    """A real spawned worker; its work never touches a database."""

    @pytest.fixture
    def runner(self, tmp_path: Path) -> Any:
        runner = grading_service.ProcessRunner(
            WorkerSpec(GradingInstance("fake-1", 1)), lock_dir=tmp_path
        )
        yield runner
        runner.close()

    @run_async
    async def test_runs_work_in_its_worker(self, runner: Any) -> None:
        assert await runner.call(worker_helpers.instance_name) == "fake-1"

    def test_holds_the_instances_lock(self, runner: Any, tmp_path: Path) -> None:
        with pytest.raises(GraderError, match="another grader process is grading on fake-1"):
            grading_service.ProcessRunner(
                WorkerSpec(GradingInstance("fake-1", 1)), lock_dir=tmp_path
            )

    def test_the_lock_is_released_on_close(self, tmp_path: Path) -> None:
        spec = WorkerSpec(GradingInstance("fake-2", 1))
        grading_service.ProcessRunner(spec, lock_dir=tmp_path).close()
        grading_service.ProcessRunner(spec, lock_dir=tmp_path).close()

    @run_async
    async def test_a_dead_worker_is_replaced(self, runner: Any) -> None:
        with pytest.raises(GraderError, match="died; it was replaced"):
            await runner.call(worker_helpers.die)
        assert await runner.call(worker_helpers.instance_name) == "fake-1"

    @run_async
    async def test_repeated_deaths_hold_the_instance_back(
        self, runner: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pauses: list[float] = []

        async def sleep(seconds: float) -> None:
            pauses.append(seconds)

        monkeypatch.setattr(asyncio, "sleep", sleep)
        for _ in range(3):
            with pytest.raises(GraderError):
                await runner.call(worker_helpers.die)
        assert pauses == [30.0, 60.0]
        assert await runner.call(worker_helpers.instance_name) == "fake-1"
        assert runner._deaths == 0

    @run_async
    async def test_a_cancel_aborts_only_that_call(self, runner: Any) -> None:
        assert await runner.call(worker_helpers.instance_name) == "fake-1"  # Warm the worker.
        task = asyncio.create_task(runner.call(worker_helpers.wait_for_abort))
        await asyncio.sleep(1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await runner.call(worker_helpers.was_aborted) is False


class TestRulingAdditions:
    @run_async
    async def test_a_lease_can_wait_for_one_particular_resource(self) -> None:
        pool = GradingPool(["a", "b"])
        order: list[str] = []

        async def want(name: str, which: str) -> None:
            async with pool.lease(Priority.RSI, which=lambda r: r == which) as got:
                order.append(f"{name}:{got}")

        async with pool.lease(Priority.RSI, which=lambda r: r == "a"):
            waiting_for_a = asyncio.create_task(want("probe", "a"))
            await asyncio.sleep(0)
            async with pool.lease(Priority.RSI) as other:
                assert other == "b", "a waiter for a does not hold up b"
        await waiting_for_a
        assert order == ["probe:a"]
        assert pool.free_count == 2

    @run_async
    async def test_a_grader_failure_is_retried_once(self, tmp_path: Path) -> None:
        outcomes = [Grade(status=GRADE_STATUS_GRADER_ERROR), Grade(status=GRADE_STATUS_GRADED)]
        slot = FakeSlot(lambda: outcomes.pop(0))
        grade = await grader_on(slot, tmp_path).grade(INST, answer("SELECT 1"), Mode.OFFICIAL)
        assert grade.status == GRADE_STATUS_GRADED
        assert len(slot.calls) == 2

    @run_async
    async def test_a_second_grader_failure_is_returned(self, tmp_path: Path) -> None:
        slot = FakeSlot(lambda: Grade(status=GRADE_STATUS_GRADER_ERROR))
        grade = await grader_on(slot, tmp_path).grade(INST, answer("SELECT 1"), Mode.OFFICIAL)
        assert grade.status == GRADE_STATUS_GRADER_ERROR
        assert len(slot.calls) == 2

    @run_async
    async def test_identity_probe_times_the_base_against_itself_on_the_instance(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        main = FakeSession(times={BASE_SQL: [100.0, 100.0, 120.0, 100.0, 100.0, 80.0]})
        slot = FakeSlot()
        slot.sessions = lambda: grading_fakes.sessions(main)  # type: ignore[attr-defined]  # The fake's.
        slot.rng = random.Random(0)  # type: ignore[attr-defined]  # noqa: S311  # Reproducible.
        grader = grader_on(slot, tmp_path)
        tau = await grader.identity_probe(INST, instance=slot.instance)
        assert 0.8 <= tau <= 1.25
        assert main.kinds(BASE_SQL) == ["time"] * 6
        assert grader.identity_reference(INST) is None
        grader.dev_cache.put(entry())
        assert grader.identity_reference(INST) == 1.02

    def test_a_failed_probe_is_a_grader_error_and_resets_the_slot(self) -> None:
        main = FakeSession(times={BASE_SQL: [QueryError("boom")]})
        slot = FakeSlot()
        slot.sessions = lambda: grading_fakes.sessions(main)  # type: ignore[attr-defined]  # The fake's.
        slot.rng = random.Random(0)  # type: ignore[attr-defined]  # noqa: S311  # Reproducible.
        with pytest.raises(GraderError, match="identity probe failed"):
            grading_service.probe_identity(slot, INST)  # type: ignore[arg-type]  # A fake.
        assert slot.resets == 1
