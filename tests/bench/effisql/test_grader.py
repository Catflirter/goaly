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
from psycopg.rows import tuple_row

from goaly.bench.effisql import grader
from goaly.bench.effisql.grader import (
    HARD_CAP_MARGIN_S,
    LONG_RUN_MS,
    SPEEDUP_CAP,
    DigestBuilder,
    GraderError,
    GradingFailure,
    PgSession,
    Psycopg2Timestamptz,
    QueryError,
    QueryTimeoutError,
    ResourceError,
    ResultDigest,
    TimingError,
    Watchdog,
    row_hash,
    speedup,
    statement_body,
    time_base,
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

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(
                {"row_count": "3", "row_hash_sum": "0" * 64, "complete": True}, id="text count"
            ),
            pytest.param(
                {"row_count": True, "row_hash_sum": "0" * 64, "complete": True}, id="bool count"
            ),
            pytest.param(
                {"row_count": -1, "row_hash_sum": "0" * 64, "complete": True}, id="negative"
            ),
            pytest.param(
                {"row_count": 3, "row_hash_sum": "0" * 63, "complete": True}, id="short sum"
            ),
            pytest.param(
                {"row_count": 3, "row_hash_sum": "-" + "0" * 63, "complete": True}, id="sign"
            ),
            pytest.param(
                {"row_count": 3, "row_hash_sum": "f" * 65, "complete": True}, id="long sum"
            ),
            pytest.param({"row_count": 3, "row_hash_sum": "0" * 64, "complete": 1}, id="int flag"),
            pytest.param({"row_count": 3, "complete": True}, id="missing sum"),
            pytest.param([3, "0" * 64, True], id="not an object"),
        ],
    )
    def test_from_json_rejects_a_malformed_value(self, value: object) -> None:
        with pytest.raises(ValueError, match="result digest"):
            ResultDigest.from_json(value)


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
            watchdog.arm(0.01)
            assert fired.wait(10)
            assert watchdog.disarm()
        finally:
            watchdog.close()

    def test_disarming_in_time_never_fires(self) -> None:
        fired = threading.Event()
        watchdog = Watchdog(fired.set)
        try:
            watchdog.arm(60)
            assert not watchdog.disarm()
            watchdog.arm(60)  # Re-arming resets the deadline and the fired flag.
            assert not watchdog.disarm()
        finally:
            watchdog.close()
        assert not fired.is_set()

    def test_disarm_waits_for_a_cancel_in_flight(self) -> None:
        """A cancel fired as a statement ends must finish before the next statement starts."""
        entered, release = threading.Event(), threading.Event()
        log: list[str] = []

        def slow_cancel() -> None:
            entered.set()
            release.wait(10)
            log.append("cancel done")

        watchdog = Watchdog(slow_cancel)
        try:
            watchdog.arm(0.0)
            assert entered.wait(10)
            disarmed = threading.Thread(target=lambda: log.append(f"disarm {watchdog.disarm()}"))
            disarmed.start()
            disarmed.join(0.2)
            assert disarmed.is_alive(), "disarm returned while the cancel was in flight"
            release.set()
            disarmed.join(10)
            assert log == ["cancel done", "disarm True"]
        finally:
            release.set()
            watchdog.close()

    def test_a_failing_cancel_is_ignored_and_the_watchdog_still_works(self) -> None:
        calls = threading.Semaphore(0)

        def cancel() -> None:
            calls.release()
            raise psycopg.OperationalError("already idle")

        watchdog = Watchdog(cancel)
        try:
            for _ in range(2):
                watchdog.arm(0.0)
                assert calls.acquire(timeout=10)
                assert watchdog.disarm()
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

    @property
    def description(self) -> list[Any] | None:
        if self._row is None:
            return None
        oids = [psycopg.adapters.types[name].oid for name in self._row]
        return [type("Column", (), {"type_code": oid})() for oid in [*oids, 1]]


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
        self.row_factory: object = None
        self.prepare_threshold: int | None = 5
        self.cancels = 0
        self.stream_closed = False
        self.registered: list[str | int] = []
        self.adapters = self
        self.types = psycopg.adapters.types
        self._rows = rows
        self.explain = explain
        self.fail_on: str | None = None  # A session statement that fails, such as DISCARD ALL.

    def register_loader(self, name: str | int, _loader: object) -> None:
        self.registered.append(name)

    def execute(self, query: str) -> None:
        if self.fail_on is not None and query.startswith(self.fail_on):
            raise psycopg.OperationalError("")
        self.statements.append(query)

    def cursor(self) -> FakeCursor:
        if self.closed:
            raise psycopg.OperationalError("the connection is closed")
        return FakeCursor(self)

    @contextmanager
    def transaction(self, *, force_rollback: bool) -> Iterator[None]:
        assert force_rollback
        assert self.read_only is True
        self.transactions += 1
        yield
        if self.closed:
            raise psycopg.OperationalError("the connection is closed")

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


SETTINGS = "SET statement_timeout = 300000"
RESET = ["DISCARD ALL", SETTINGS]


def raising(error: Exception) -> Callable[[], Iterator[Row]]:
    def rows() -> Iterator[Row]:
        raise error

    return rows


class TestPgSession:
    def test_setup_makes_the_session_read_only_with_psycopg2_types(self) -> None:
        conn = FakeConn()
        with session_on(conn):
            pass
        assert conn.autocommit
        assert conn.read_only is True
        assert conn.row_factory is tuple_row
        assert conn.prepare_threshold is None
        types = psycopg.adapters.types
        arrays = [types[name].array_oid for name in ("uuid", "inet", "cidr", "record", "money")]
        assert conn.registered[:4] == ["uuid", "inet", "cidr", "record"]
        assert set(arrays) <= set(conn.registered)
        assert conn.registered[-1] == "timestamptz"
        assert conn.statements == [SETTINGS]
        assert conn.closed

    def test_the_server_timeout_follows_the_session_timeout(self) -> None:
        conn = FakeConn()
        session_on(conn, timeout_s=1.5).close()
        assert conn.statements == ["SET statement_timeout = 1500"]

    def test_digest_streams_and_hashes_every_row_then_resets(self) -> None:
        rows = [(i, f"r{i}") for i in range(12_001)]
        conn = FakeConn(rows=lambda: iter(rows))
        with session_on(conn) as session:
            digest = session.digest(" SELECT * FROM t ; ")
        assert digest == digest_of(rows)
        assert conn.statements == [SETTINGS, "SELECT * FROM t", *RESET]
        assert conn.transactions == 1

    def test_digest_stops_past_the_row_limit_and_closes_the_stream(self) -> None:
        def endless() -> Iterator[Row]:
            i = 0
            while True:
                i += 1
                yield (i,)

        conn = FakeConn(rows=endless)
        with session_on(conn) as session:
            digest = session.digest("SELECT x", row_limit=7)
        assert not digest.complete
        assert digest.row_count > 7
        assert conn.stream_closed

    @pytest.mark.parametrize(
        ("error", "expected", "message"),
        [
            (psycopg.errors.SyntaxError("syntax error at or near x"), QueryError, "syntax"),
            (
                psycopg.errors.QueryCanceled("canceling statement due to statement timeout"),
                QueryTimeoutError,
                r"statement timeout \(300 s\)",
            ),
        ],
    )
    def test_errors_map_to_the_graders_and_the_session_resets(
        self, error: Exception, expected: type[Exception], message: str
    ) -> None:
        conn = FakeConn(rows=raising(error))
        with session_on(conn) as session:
            with pytest.raises(expected, match=message):
                session.digest("SELECT 1")
            assert conn.statements[-2:] == RESET
            assert isinstance(expected("x"), GradingFailure)

    def test_running_out_of_resources_is_the_graders(self) -> None:
        error = psycopg.errors.OutOfMemory("could not resize shared memory segment")
        conn = FakeConn(rows=raising(error))
        with (
            session_on(conn) as session,
            pytest.raises(ResourceError, match="ran out of resources"),
        ):
            session.digest("SELECT 1")
        assert issubclass(ResourceError, GraderError)

    def test_a_dead_connection_is_a_grader_error_every_time(self) -> None:
        conn = FakeConn()

        def lost() -> Iterator[Row]:
            conn.broken = conn.closed = True
            raise psycopg.OperationalError("server closed the connection unexpectedly")

        conn._rows = lost
        session = session_on(conn)
        with pytest.raises(GraderError, match="server closed"):
            session.digest("SELECT 1")
        for _ in range(2):  # The connection stays dead: never a raw psycopg error.
            with pytest.raises(GraderError, match="the connection is closed"):
                session.digest("SELECT 1")
            with pytest.raises(GraderError):
                session.plan("SELECT 1")
        assert "DISCARD ALL" not in conn.statements

    def test_a_failed_reset_is_a_grader_error(self) -> None:
        conn = FakeConn(rows=lambda: iter([(1,)]))
        conn.fail_on = "DISCARD ALL"
        with session_on(conn) as session, pytest.raises(GraderError, match="cannot reset"):
            session.digest("SELECT 1")

    def test_the_hard_cap_cancels_a_hung_statement(self) -> None:
        conn = FakeConn()

        def hung() -> Iterator[Row]:
            deadline = time.monotonic() + 10
            while not conn.cancels and time.monotonic() < deadline:
                time.sleep(0.005)
            raise psycopg.errors.QueryCanceled("canceling statement due to user request")

        conn._rows = hung
        with (
            session_on(conn, hard_cap_s=0.01) as session,
            pytest.raises(QueryTimeoutError, match="hard cap"),
        ):
            session.digest("SELECT pg_sleep(1000)")
        assert conn.cancels == 1

    def test_the_default_hard_cap_is_twice_the_timeout_plus_a_margin(self) -> None:
        with session_on(FakeConn(), timeout_s=10) as session:
            assert session._hard_cap_s == 20 + HARD_CAP_MARGIN_S

    def test_execution_ms_reads_explain_analyze(self) -> None:
        conn = FakeConn(explain=lambda _: ([{"Plan": {}, "Execution Time": 12.5}],))
        with session_on(conn) as session:
            assert session.execution_ms("SELECT 1;") == 12.5
        assert conn.statements == [SETTINGS, "EXPLAIN (ANALYZE, FORMAT JSON) SELECT 1", *RESET]

    def test_plan_parses_text_and_only_plans(self) -> None:
        conn = FakeConn(explain=lambda _: ('[{"Plan": {"Node Type": "Result"}}]',))
        with session_on(conn) as session:
            assert session.plan("SELECT 1") == {"Plan": {"Node Type": "Result"}}
        assert conn.statements[1] == "EXPLAIN (FORMAT JSON) SELECT 1"

    @pytest.mark.parametrize(
        ("row", "message"),
        [
            (None, "no plan document"),
            (([],), "no plan document"),
            (("not json",), "not JSON"),
            (([{"Plan": {}}],), "no Execution Time"),
        ],
    )
    def test_a_malformed_explain_is_a_grader_error(self, row: Row | None, message: str) -> None:
        conn = FakeConn(explain=lambda _: row)
        with session_on(conn) as session, pytest.raises(GraderError, match=message):
            session.execution_ms("SELECT 1")

    def test_an_explain_error_is_the_querys(self) -> None:
        error = psycopg.errors.UndefinedTable("relation x does not exist")
        conn = FakeConn(explain=lambda _: error)
        with session_on(conn) as session, pytest.raises(QueryError, match="x does not exist"):
            session.plan("SELECT * FROM x")

    def test_open_reports_a_failed_connection(self) -> None:
        def connect(_db: str) -> psycopg.Connection[Any]:
            raise psycopg.OperationalError("connection refused")

        with pytest.raises(GraderError, match="cannot connect to tpch: connection refused"):
            PgSession.open(connect, "tpch")

    def test_open_closes_a_connection_it_cannot_set_up(self) -> None:
        conn = FakeConn()
        conn.fail_on = "SET"
        with pytest.raises(GraderError, match="cannot set up the session on tpch"):
            PgSession.open(lambda _db: conn, "tpch")  # type: ignore[arg-type,return-value]  # A structural fake.
        assert conn.closed

    def test_an_old_libpq_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(psycopg.pq, "version", lambda: 160004)
        conn = FakeConn()
        with pytest.raises(GraderError, match="libpq 160004 is too old"):
            PgSession.open(lambda _db: conn, "tpch")  # type: ignore[arg-type,return-value]  # A structural fake.
        assert conn.closed

    def test_open_returns_a_session(self) -> None:
        conn = FakeConn()
        session = PgSession.open(lambda _db: conn, "tpch")  # type: ignore[arg-type,return-value]  # A structural fake.
        assert session.connection is cast(Any, conn)
        session.close()


class TestTimeBase:
    def test_the_median_of_three_base_runs(self) -> None:
        log: list[str] = []
        assert time_base(RecordingTimer("base", log, [9.0, 7.0, 8.0])) == (8.0, 3)

    def test_a_long_second_run_stops_after_two(self) -> None:
        log: list[str] = []
        assert time_base(RecordingTimer("base", log, [1.0, LONG_RUN_MS, 1.0])) == (
            (1.0 + LONG_RUN_MS) / 2,
            2,
        )

    def test_a_failed_run_is_the_bases(self) -> None:
        log: list[str] = []
        with pytest.raises(TimingError) as caught:
            time_base(RecordingTimer("base", log, [QueryError("boom")]))
        assert caught.value.side == "base"


class TestPsycopg2Timestamptz:
    """The loader on its own, without a server: the session's time zone is UTC then."""

    def load(self, text: bytes) -> datetime:
        loader = Psycopg2Timestamptz(psycopg.adapters.types["timestamptz"].oid, None)
        return loader.load(text)

    def test_a_fixed_offset_utc_as_psycopg2_builds_it(self) -> None:
        value = self.load(b"2026-10-15 05:28:00+00")
        assert value.tzinfo is UTC
        assert str((value,)) == (
            "(datetime.datetime(2026, 10, 15, 5, 28, tzinfo=datetime.timezone.utc),)"
        )

    def test_another_offset_is_the_same_instant_with_a_fixed_offset(self) -> None:
        value = self.load(b"1995-01-01 00:00:00+05:30")
        assert value == datetime(1994, 12, 31, 18, 30, tzinfo=UTC)
        assert type(value.tzinfo) is timezone


def test_the_evaluators_constants() -> None:
    assert grader.STATEMENT_TIMEOUT_S == 300.0
    assert grader.LONG_RUN_MS == 60_000.0
    assert grader.TIMING_RUNS == 3
    assert grader.SPEEDUP_CAP == 1000.0
    assert grader.ZERO_TIME_MS == 0.001
    assert grader.FETCH_CHUNK_ROWS == 5000
    assert grader.MIN_LIBPQ_VERSION == 170000


@pytest.mark.parametrize("runs", [0, -1])
def test_timing_needs_a_run(runs: int) -> None:
    rng = random.Random(0)  # noqa: S311  # A reproducible order, not a secret.
    with pytest.raises(ValueError, match="at least 1 run"):
        time_interleaved(lambda: 1.0, lambda: 1.0, rng=rng, runs=runs)
    with pytest.raises(ValueError, match="at least 1 run"):
        time_prediction(lambda: 1.0, base_ms=1.0, runs=runs)
    with pytest.raises(ValueError, match="at least 1 run"):
        time_base(lambda: 1.0, runs=runs)


def test_column_types_names_each_column_without_fetching() -> None:
    conn = FakeConn(explain=lambda _: ("numeric", "date", "bpchar"))
    with session_on(conn) as session:
        types = session.column_types("SELECT 1;")
    assert types == ["numeric", "date", "bpchar", "unknown"]
    assert conn.statements[1] == "SELECT * FROM (SELECT 1) AS q LIMIT 0"
