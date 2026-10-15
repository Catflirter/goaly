"""Unit tests of the grader's measurements: digests, row strings, timing, speedup, the session."""

import hashlib
import random
import threading
import time
from collections.abc import Callable, Generator, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, cast

import psycopg
import pytest

from goaly.bench.effisql import grader
from goaly.bench.effisql.grader import (
    LONG_RUN_MS,
    SPEEDUP_CAP,
    DigestBuilder,
    GraderError,
    PgSession,
    QueryError,
    QueryTimeoutError,
    ResultDigest,
    TimingError,
    Watchdog,
    row_hash,
    speedup,
    statement_body,
    time_interleaved,
    time_prediction,
)

Row = tuple[Any, ...]


def digest_of(rows: Sequence[Row]) -> ResultDigest:
    builder = DigestBuilder()
    builder.add(rows)
    return builder.digest()


ROWS: list[Row] = [(1, "a", Decimal("1.50")), (2, "b", Decimal("2.00")), (3, "c", None)]


class TestDigest:
    def test_row_hash_is_sha256_of_the_rows_string(self) -> None:
        expected = hashlib.sha256(b"(1, 'a', Decimal('1.50'))").digest()
        assert row_hash(ROWS[0]) == int.from_bytes(expected, "big")

    def test_the_digest_is_the_sum_of_row_hashes_mod_2_256_and_the_count(self) -> None:
        digest = digest_of(ROWS)
        assert digest.row_count == 3
        assert digest.row_hash_sum == sum(row_hash(row) for row in ROWS) % (1 << 256)

    def test_order_does_not_matter(self) -> None:
        assert digest_of(ROWS).matches(digest_of(list(reversed(ROWS))))

    def test_batches_add_up_to_the_whole(self) -> None:
        builder = DigestBuilder()
        builder.add(ROWS[:1])
        builder.add(ROWS[1:])
        assert builder.digest() == digest_of(ROWS)

    @pytest.mark.parametrize(
        "variant",
        [
            pytest.param(ROWS[:-1], id="dropped row"),
            pytest.param([*ROWS, ROWS[0]], id="duplicated row"),
            pytest.param([*ROWS[:-1], ROWS[0]], id="one row swapped for a duplicate"),
            pytest.param([(r[1], r[0], r[2]) for r in ROWS], id="columns reordered"),
            pytest.param([*ROWS[:-1], (3, "c", Decimal(0))], id="value perturbed"),
            pytest.param([*ROWS[:-1], (3, "c", None, None)], id="column added"),
            pytest.param([(1, "a", Decimal("1.5")), *ROWS[1:]], id="numeric scale differs"),
        ],
    )
    def test_a_changed_result_does_not_match(self, variant: list[Row]) -> None:
        assert not digest_of(variant).matches(digest_of(ROWS))

    def test_two_empty_results_match(self) -> None:
        assert digest_of([]).matches(digest_of([]))
        assert not digest_of([]).matches(digest_of([(None,)]))

    def test_an_incomplete_digest_matches_nothing(self) -> None:
        digest = digest_of(ROWS)
        partial = ResultDigest(digest.row_count, digest.row_hash_sum, complete=False)
        assert not partial.matches(digest)
        assert not digest.matches(partial)

    def test_json_round_trip(self) -> None:
        digest = digest_of(ROWS)
        assert ResultDigest.from_json(digest.to_json()) == digest
        assert len(str(digest.to_json()["row_hash_sum"])) == 64

    def test_from_json_rejects_a_malformed_value(self) -> None:
        with pytest.raises(ValueError, match="not a result digest"):
            ResultDigest.from_json({"row_count": "3", "row_hash_sum": "00", "complete": True})


class TestPsycopg2RowStrings:
    """The strings psycopg2 2.9 gives str(row) for each type a TPC-H query returns.

    psycopg 3 builds the same Python values (tests/bench/effisql/test_grader_integration.py
    loads each from the server), so these pin the row strings the official evaluator hashes.
    """

    @pytest.mark.parametrize(
        ("row", "expected"),
        [
            ((Decimal("35098.38"),), "(Decimal('35098.38'),)"),
            ((Decimal("0.0700"), Decimal("NaN")), "(Decimal('0.0700'), Decimal('NaN'))"),
            ((date(1997, 2, 9),), "(datetime.date(1997, 2, 9),)"),
            ((1.5, 0.1, float("nan")), "(1.5, 0.1, nan)"),
            ((None,), "(None,)"),
            (("COLLECT COD   ", "it's"), "('COLLECT COD   ', \"it's\")"),
            ((999939, True), "(999939, True)"),
            ((timedelta(days=32),), "(datetime.timedelta(days=32),)"),
            (
                (datetime(2026, 10, 15, 5, 28, tzinfo=UTC),),
                "(datetime.datetime(2026, 10, 15, 5, 28, tzinfo=datetime.timezone.utc),)",
            ),
            (
                (datetime(1995, 1, 1, 0, 0, tzinfo=timezone(timedelta(hours=5, minutes=30))),),
                "(datetime.datetime(1995, 1, 1, 0, 0, "
                "tzinfo=datetime.timezone(datetime.timedelta(seconds=19800))),)",
            ),
        ],
    )
    def test_row_string(self, row: Row, expected: str) -> None:
        assert str(row) == expected


class TestSpeedup:
    def test_base_over_pred(self) -> None:
        assert speedup(300.0, 100.0) == 3.0

    def test_capped_at_1000(self) -> None:
        assert speedup(1e9, 1.0) == SPEEDUP_CAP
        assert speedup(1000.0, 1.0) == SPEEDUP_CAP

    def test_a_zero_pred_time_counts_as_a_thousandth_of_a_ms(self) -> None:
        assert speedup(0.0005, 0.0) == pytest.approx(0.5)
        assert speedup(10.0, 0.0) == SPEEDUP_CAP

    def test_a_zero_base_time_gives_zero(self) -> None:
        assert speedup(0.0, 5.0) == 0.0


class TestStatementBody:
    def test_strips_whitespace_and_trailing_semicolons(self) -> None:
        assert statement_body("  SELECT 1 ;; \n") == "SELECT 1"


class RecordingTimer:
    def __init__(self, label: str, log: list[str], times: Sequence[float | Exception]) -> None:
        self._label = label
        self._log = log
        self._times = list(times)

    def __call__(self) -> float:
        self._log.append(self._label)
        value = self._times.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


class TestTimeInterleaved:
    def test_three_runs_in_random_order_and_the_medians(self) -> None:
        log: list[str] = []
        timing = time_interleaved(
            RecordingTimer("pred", log, [10.0, 30.0, 20.0]),
            RecordingTimer("base", log, [100.0, 300.0, 200.0]),
            rng=random.Random(1),  # noqa: S311  # A reproducible order, not a secret.
        )
        assert (timing.pred_ms, timing.base_ms, timing.runs) == (20.0, 200.0, 3)
        assert timing.speedup == 10.0
        runs = [tuple(log[i : i + 2]) for i in range(0, 6, 2)]
        assert all(sorted(run) == ["base", "pred"] for run in runs)
        assert len(set(runs)) == 2, "the order is shuffled per run"

    def test_the_order_follows_the_rng(self) -> None:
        orders = set()
        for seed in range(20):
            log: list[str] = []
            pred = RecordingTimer("pred", log, [1.0] * 3)
            rng = random.Random(seed)  # noqa: S311  # A reproducible order, not a secret.
            time_interleaved(pred, RecordingTimer("base", log, [1.0] * 3), rng=rng)
            orders.add(log[0])
        assert orders == {"pred", "base"}

    @pytest.mark.parametrize("long_side", ["pred", "base"])
    def test_a_long_second_run_stops_after_two_runs(self, long_side: str) -> None:
        log: list[str] = []
        long = [1.0, LONG_RUN_MS, 1.0]
        short = [1.0, 1.0, 1.0]
        timing = time_interleaved(
            RecordingTimer("pred", log, long if long_side == "pred" else short),
            RecordingTimer("base", log, long if long_side == "base" else short),
            rng=random.Random(0),  # noqa: S311  # A reproducible order, not a secret.
        )
        assert timing.runs == 2
        assert len(log) == 4
        median = (1.0 + LONG_RUN_MS) / 2
        assert (timing.pred_ms if long_side == "pred" else timing.base_ms) == median

    def test_a_long_first_run_alone_does_not_stop_the_timing(self) -> None:
        log: list[str] = []
        timing = time_interleaved(
            RecordingTimer("pred", log, [LONG_RUN_MS, 1.0, 1.0]),
            RecordingTimer("base", log, [1.0, 1.0, 1.0]),
            rng=random.Random(0),  # noqa: S311  # A reproducible order, not a secret.
        )
        assert timing.runs == 3

    def test_just_under_a_minute_runs_three_times(self) -> None:
        log: list[str] = []
        timing = time_interleaved(
            RecordingTimer("pred", log, [LONG_RUN_MS - 1] * 3),
            RecordingTimer("base", log, [1.0] * 3),
            rng=random.Random(0),  # noqa: S311  # A reproducible order, not a secret.
        )
        assert timing.runs == 3

    @pytest.mark.parametrize("side", ["pred", "base"])
    @pytest.mark.parametrize("error", [QueryTimeoutError("late"), QueryError("boom")])
    def test_a_failed_run_names_its_side(self, side: str, error: Exception) -> None:
        log: list[str] = []
        failing: list[float | Exception] = [1.0, error, 1.0]
        with pytest.raises(TimingError) as caught:
            time_interleaved(
                RecordingTimer("pred", log, failing if side == "pred" else [1.0] * 3),
                RecordingTimer("base", log, failing if side == "base" else [1.0] * 3),
                rng=random.Random(0),  # noqa: S311  # A reproducible order, not a secret.
            )
        assert caught.value.side == side
        assert caught.value.timed_out is isinstance(error, QueryTimeoutError)
        assert caught.value.__cause__ is error

    def test_a_grader_error_propagates_unchanged(self) -> None:
        log: list[str] = []
        with pytest.raises(GraderError):
            time_interleaved(
                RecordingTimer("pred", log, [GraderError("lost")]),
                RecordingTimer("base", log, [GraderError("lost")]),
                rng=random.Random(0),  # noqa: S311  # A reproducible order, not a secret.
            )


class TestTimePrediction:
    def test_only_the_prediction_runs_against_the_cached_base(self) -> None:
        log: list[str] = []
        timing = time_prediction(RecordingTimer("pred", log, [5.0, 3.0, 4.0]), base_ms=40.0)
        assert log == ["pred"] * 3
        assert (timing.pred_ms, timing.base_ms, timing.runs, timing.speedup) == (4.0, 40.0, 3, 10)

    def test_a_long_cached_base_stops_after_two_runs(self) -> None:
        log: list[str] = []
        timing = time_prediction(RecordingTimer("pred", log, [5.0, 3.0, 4.0]), base_ms=LONG_RUN_MS)
        assert timing.runs == 2
        assert timing.pred_ms == 4.0

    def test_a_long_prediction_stops_after_two_runs(self) -> None:
        log: list[str] = []
        timing = time_prediction(RecordingTimer("pred", log, [1.0, LONG_RUN_MS, 1.0]), base_ms=1.0)
        assert timing.runs == 2

    def test_a_failed_run_is_the_predictions(self) -> None:
        log: list[str] = []
        with pytest.raises(TimingError) as caught:
            time_prediction(RecordingTimer("pred", log, [QueryTimeoutError("late")]), base_ms=1.0)
        assert caught.value.side == "pred"
        assert caught.value.timed_out


class TestWatchdog:
    def test_fires_once_the_deadline_passes(self) -> None:
        fired = threading.Event()
        watchdog = Watchdog(fired.set)
        try:
            watchdog.arm(0.05)
            assert fired.wait(2.0)
            assert watchdog.disarm()
        finally:
            watchdog.close()

    def test_disarming_in_time_never_fires(self) -> None:
        fired = threading.Event()
        watchdog = Watchdog(fired.set)
        try:
            watchdog.arm(0.2)
            assert not watchdog.disarm()
            assert not fired.wait(0.4)
        finally:
            watchdog.close()

    def test_extend_moves_the_deadline_up_to_the_hard_limit(self) -> None:
        fired = threading.Event()
        watchdog = Watchdog(fired.set)
        try:
            started = time.monotonic()
            watchdog.arm(0.1, hard_s=0.3)
            watchdog.extend(10.0)
            assert fired.wait(2.0)
            assert 0.25 <= time.monotonic() - started < 1.5
        finally:
            watchdog.close()

    def test_extend_without_a_deadline_does_nothing(self) -> None:
        fired = threading.Event()
        watchdog = Watchdog(fired.set)
        try:
            watchdog.extend(1.0)
            assert not fired.wait(0.1)
        finally:
            watchdog.close()

    def test_a_failing_cancel_is_ignored(self) -> None:
        calls: list[int] = []

        def cancel() -> None:
            calls.append(1)
            raise psycopg.OperationalError("already idle")

        watchdog = Watchdog(cancel)
        try:
            watchdog.arm(0.01)
            deadline = time.monotonic() + 2
            while not calls and time.monotonic() < deadline:
                time.sleep(0.01)
            watchdog.arm(0.01)  # Still running: it fires again.
            while len(calls) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            assert len(calls) == 2
        finally:
            watchdog.close()


class FakeCursor:
    def __init__(self, conn: "FakeConn") -> None:
        self._conn = conn
        self._row: Row | None = None

    def stream(self, query: str, *, size: int) -> Iterator[Row]:
        self._conn.statements.append(query)
        assert size == grader.FETCH_CHUNK_ROWS
        return self._conn.stream_rows(query)

    def execute(self, query: str, *, binary: bool) -> None:
        assert binary
        self._conn.statements.append(query)
        outcome = self._conn.explain(query)
        if isinstance(outcome, BaseException):
            raise outcome
        self._row = outcome

    def fetchone(self) -> Row | None:
        return self._row


class FakeConn:
    """Enough of psycopg.Connection for PgSession: canned results and errors per query."""

    def __init__(
        self,
        *,
        rows: Callable[[], Iterator[Row]] = lambda: iter(()),
        explain: Callable[[str], Row | BaseException | None] = lambda _: None,
    ) -> None:
        self.statements: list[str] = []
        self.transactions = 0
        self.closed = False
        self.broken = False
        self.autocommit = False
        self.read_only: bool | None = None
        self.cancels = 0
        self.stream_closed = False
        self.registered: list[str] = []
        self.adapters = self
        self._rows = rows
        self.explain = explain
        self.fail_setup = False

    def register_loader(self, name: str, _loader: object) -> None:
        self.registered.append(name)

    def execute(self, query: str) -> None:
        if self.fail_setup:
            raise psycopg.OperationalError("")
        self.statements.append(query)

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    @contextmanager
    def transaction(self, *, force_rollback: bool) -> Iterator[None]:
        assert force_rollback
        assert self.read_only is True
        self.transactions += 1
        yield

    def stream_rows(self, _query: str) -> Generator[Row]:
        try:
            yield from self._rows()
        finally:
            self.stream_closed = True

    def cancel_safe(self) -> None:
        self.cancels += 1

    def close(self) -> None:
        self.closed = True


def session_on(conn: FakeConn, **kwargs: Any) -> PgSession:
    return PgSession(conn, **kwargs)  # type: ignore[arg-type]  # A structural fake.


class TestPgSession:
    def test_setup_makes_the_session_read_only_with_psycopg2_types(self) -> None:
        conn = FakeConn()
        session_on(conn).close()
        assert conn.autocommit
        assert conn.read_only is True
        assert conn.registered == ["uuid", "inet", "cidr", "record", "timestamptz"]
        assert conn.statements == ["SET statement_timeout = 660000"]
        assert conn.closed

    def test_digest_streams_and_hashes_every_row(self) -> None:
        rows = [(i, f"r{i}") for i in range(12_001)]
        conn = FakeConn(rows=lambda: iter(rows))
        session = session_on(conn)
        try:
            digest = session.digest(" SELECT * FROM t ; ")
        finally:
            session.close()
        assert digest == digest_of(rows)
        assert conn.statements[-1] == "SELECT * FROM t"
        assert conn.transactions == 1

    def test_digest_stops_past_the_row_limit_and_closes_the_stream(self) -> None:
        def endless() -> Iterator[Row]:
            i = 0
            while True:
                i += 1
                yield (i,)

        conn = FakeConn(rows=endless)
        session = session_on(conn)
        try:
            digest = session.digest("SELECT x", row_limit=7)
        finally:
            session.close()
        assert not digest.complete
        assert digest.row_count > 7
        assert conn.stream_closed

    @pytest.mark.parametrize(
        ("error", "expected", "message"),
        [
            (psycopg.errors.SyntaxError("syntax error at or near x"), QueryError, "syntax"),
            (psycopg.errors.QueryCanceled("canceling statement"), QueryTimeoutError, "after 300"),
        ],
    )
    def test_errors_map_to_the_graders(
        self, error: Exception, expected: type[Exception], message: str
    ) -> None:
        def failing() -> Iterator[Row]:
            raise error

        conn = FakeConn(rows=failing)
        session = session_on(conn)
        try:
            with pytest.raises(expected, match=message):
                session.digest("SELECT 1")
        finally:
            session.close()

    def test_a_broken_connection_is_a_grader_error(self) -> None:
        conn = FakeConn()

        def lost() -> Iterator[Row]:
            conn.broken = True
            raise psycopg.OperationalError("server closed the connection unexpectedly")

        conn._rows = lost
        session = session_on(conn)
        try:
            with pytest.raises(GraderError, match="server closed"):
                session.digest("SELECT 1")
        finally:
            session.close()

    def test_the_watchdog_cancels_a_slow_statement(self) -> None:
        conn = FakeConn()

        def slow() -> Iterator[Row]:
            deadline = time.monotonic() + 5
            while not conn.cancels and time.monotonic() < deadline:
                time.sleep(0.005)
            raise psycopg.errors.QueryCanceled("canceling statement due to user request")

        conn._rows = slow
        session = session_on(conn, timeout_s=0.05)
        try:
            with pytest.raises(QueryTimeoutError):
                session.digest("SELECT pg_sleep(10)")
        finally:
            session.close()
        assert conn.cancels == 1

    def test_execution_ms_reads_explain_analyze(self) -> None:
        conn = FakeConn(explain=lambda _: ([{"Plan": {}, "Execution Time": 12.5}],))
        session = session_on(conn)
        try:
            assert session.execution_ms("SELECT 1;") == 12.5
        finally:
            session.close()
        assert conn.statements[-1] == "EXPLAIN (ANALYZE, FORMAT JSON) SELECT 1"

    def test_plan_parses_text_and_only_plans(self) -> None:
        conn = FakeConn(explain=lambda _: ('[{"Plan": {"Node Type": "Result"}}]',))
        session = session_on(conn)
        try:
            assert session.plan("SELECT 1") == {"Plan": {"Node Type": "Result"}}
        finally:
            session.close()
        assert conn.statements[-1] == "EXPLAIN (FORMAT JSON) SELECT 1"

    @pytest.mark.parametrize(
        ("row", "message"),
        [
            (None, "no plan document"),
            (([],), "no plan document"),
            (([{"Plan": {}}],), "no Execution Time"),
        ],
    )
    def test_a_malformed_explain_is_a_grader_error(self, row: Row | None, message: str) -> None:
        conn = FakeConn(explain=lambda _: row)
        session = session_on(conn)
        try:
            with pytest.raises(GraderError, match=message):
                session.execution_ms("SELECT 1")
        finally:
            session.close()

    def test_an_explain_error_is_the_querys(self) -> None:
        conn = FakeConn(
            explain=lambda _: psycopg.errors.UndefinedTable("relation x does not exist")
        )
        session = session_on(conn)
        try:
            with pytest.raises(QueryError, match="relation x does not exist"):
                session.plan("SELECT * FROM x")
        finally:
            session.close()

    def test_open_reports_a_failed_connection(self) -> None:
        def connect(_db: str) -> psycopg.Connection[Any]:
            raise psycopg.OperationalError("connection refused")

        with pytest.raises(GraderError, match="cannot connect to tpch: connection refused"):
            PgSession.open(connect, "tpch")

    def test_open_closes_a_connection_it_cannot_set_up(self) -> None:
        conn = FakeConn()

        conn.fail_setup = True
        with pytest.raises(GraderError, match="cannot set up the session on tpch"):
            PgSession.open(lambda _db: conn, "tpch")  # type: ignore[arg-type,return-value]  # A structural fake.
        assert conn.closed

    def test_open_returns_a_session(self) -> None:
        conn = FakeConn()
        session = PgSession.open(lambda _db: conn, "tpch")  # type: ignore[arg-type,return-value]  # A structural fake.
        assert session.connection is cast(Any, conn)
        session.close()
