"""Unit tests of the grading pool, the grading slots and the async grader."""

import asyncio
import functools
import threading
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

import pytest

from goaly.bench.effisql import grading
from goaly.bench.effisql.data import EffiSqlInstance
from goaly.bench.effisql.grader import GraderError, PgSession
from goaly.bench.effisql.grading import BaseCache, EffiSqlGrader, GradingSlot, Mode
from goaly.bench.effisql.grading_pool import (
    GRADING_INSTANCES,
    SETUP_INSTANCES,
    GradingInstance,
    GradingPool,
    Priority,
)
from goaly.contracts.gen.task_pb2 import (
    GRADE_STATUS_GRADED,
    GRADE_STATUS_GRADER_ERROR,
    EffiSqlPrediction,
    Grade,
    Prediction,
)
from tests.bench.effisql.grading_fakes import BASE_SQL, INST, FakeSession


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
    def __init__(self, work: Any = None) -> None:
        self.calls: list[tuple[str, str | None, Mode, BaseCache | None]] = []
        self.work = work
        self.open_sessions: Any = None

    def evaluate(
        self, inst: EffiSqlInstance, sql: str | None, mode: Mode, cache: BaseCache | None
    ) -> Grade:
        self.calls.append((inst.instance_id, sql, mode, cache))
        if self.work is not None:
            return self.work()  # type: ignore[no-any-return]  # The test's own work.
        return Grade(instance_id=inst.instance_id, status=GRADE_STATUS_GRADED)


def grader_on(slot: FakeSlot, tmp_path: Path) -> EffiSqlGrader:
    return EffiSqlGrader(GradingPool([slot]), cache_root=tmp_path)  # type: ignore[list-item]  # A fake slot.


class TestEffiSqlGrader:
    @run_async
    async def test_dev_mode_reads_its_own_cache_and_official_none(self, tmp_path: Path) -> None:
        slot = FakeSlot()
        grader = grader_on(slot, tmp_path)
        prediction = Prediction(effisql=EffiSqlPrediction(sql="SELECT 1"))
        await grader.grade(INST, prediction, Mode.DEV)
        await grader.grade(INST, prediction, Mode.OFFICIAL)
        dev_cache = slot.calls[0][3]
        assert dev_cache is not None
        assert dev_cache.directory == tmp_path / "dev"
        assert grader.cache(Mode.OFFICIAL).directory == tmp_path / "official"
        assert slot.calls[1][3] is None

    @run_async
    async def test_no_branch_is_graded_as_base_sql(self, tmp_path: Path) -> None:
        slot = FakeSlot()
        await grader_on(slot, tmp_path).grade(INST, Prediction(), Mode.DEV)
        assert slot.calls[0][1] == BASE_SQL

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
    async def test_cancelling_aborts_the_evaluation_before_freeing_the_slot(
        self, tmp_path: Path
    ) -> None:
        started, release = threading.Event(), threading.Event()

        class Sessions:
            aborted = False

            def abort(self) -> None:
                self.aborted = True
                release.set()

        def blocking() -> Grade:
            started.set()
            release.wait(5)
            return Grade()

        slot = FakeSlot(blocking)
        slot.open_sessions = Sessions()
        grader = grader_on(slot, tmp_path)
        task = asyncio.create_task(grader.grade(INST, Prediction(), Mode.DEV))
        await asyncio.to_thread(started.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert slot.open_sessions.aborted
        assert grader._pool.free_count == 1


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

    def test_a_grader_error_reopens_the_sessions_next_time(
        self, opened: list[FakeSession], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def failing(*_args: Any, **_kwargs: Any) -> Grade:
            return Grade(status=GRADE_STATUS_GRADER_ERROR)

        monkeypatch.setattr(grading, "evaluate", failing)
        slot = self.slot()
        slot.evaluate(INST, "SELECT 1", Mode.DEV, None)
        assert slot.open_sessions is None

    def test_abort_flags_and_cancels_both_sessions(self, opened: list[FakeSession]) -> None:
        sessions = self.slot().sessions()
        sessions.abort()
        assert sessions.aborted.is_set()
        assert [s.connection.cancels for s in opened] == [1, 1]
        with pytest.raises(grading._AbortedError):
            sessions.check_abort()
