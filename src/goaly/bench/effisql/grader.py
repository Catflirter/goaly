"""The Effi-SQL grader's measurements (spec §4.1): result digests, timing and speedup.

A clean reimplementation of the official evaluator's algorithm, from its description:

- **Equivalence.** A result's signature is the multiset of ``sha256(str(row))`` plus the row
  count. :class:`ResultDigest` keeps it compactly, as the row hashes' sum mod 2^256 plus the
  count, which is order-independent and never holds a hash per row. Rows load as the official
  evaluator's psycopg2 would build them (:func:`use_psycopg2_types`), so ``str(row)`` is the same.
- **Timing.** ``EXPLAIN (ANALYZE, FORMAT JSON)``'s ``"Execution Time"``, in ms. Up to 3 runs,
  each timing both sides in random order (:func:`time_interleaved`); timing stops after the
  second run if either side of it took 60 s or more. Each side's time is its runs' median.
- **Speedup.** τ = base / pred, capped at 1000, a pred time of 0 counting as 0.001 ms.

:class:`PgSession` runs grading statements on one database. Each runs alone, in a read-only
transaction that is always rolled back, and the session is reset after it (``DISCARD ALL``), so
nothing a prediction does outlives its statement. The timeout is the evaluator's: the server's
``statement_timeout`` of 300 s on every statement. A client-side hard cap, well above it, only
catches a hang, such as a query that lifted its own server timeout. Use a session from one worker
thread or process, never on the event loop: it fetches and hashes the whole result.

Rows are streamed in chunks, not read through a DECLAREd cursor: PostgreSQL never runs a cursor's
query in parallel, while the evaluator's client-side execution can get a parallel plan. One
difference remains: the evaluator's client receives the whole result as fast as the server sends
it, while here the server waits whenever the client is still hashing the previous chunk, and
that wait counts toward the server's statement time. At about 5 µs per row it matters only for
results of tens of millions of rows from queries near the 300 s limit.
"""

import contextlib
import hashlib
import json
import random
import re
import statistics
import threading
import time
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import batched
from types import TracebackType
from typing import Any, Literal, Self, cast

import psycopg
from psycopg.abc import AdaptContext
from psycopg.rows import tuple_row
from psycopg.types.datetime import TimestamptzLoader
from psycopg.types.string import TextLoader

from goaly.bench.effisql.pg_build import Connect

__all__ = [
    "FETCH_CHUNK_ROWS",
    "HARD_CAP_MARGIN_S",
    "LONG_RUN_MS",
    "MIN_LIBPQ_VERSION",
    "SPEEDUP_CAP",
    "STATEMENT_TIMEOUT_S",
    "TIMING_RUNS",
    "ZERO_TIME_MS",
    "DigestBuilder",
    "GraderError",
    "GradingFailure",
    "PgSession",
    "Psycopg2Timestamptz",
    "QueryError",
    "QueryTimeoutError",
    "ResourceError",
    "ResultDigest",
    "Timing",
    "TimingError",
    "Watchdog",
    "row_hash",
    "speedup",
    "statement_body",
    "time_base",
    "time_interleaved",
    "time_prediction",
    "use_psycopg2_types",
]

SPEEDUP_CAP = 1000.0
ZERO_TIME_MS = 0.001  # A pred time of 0 counts as this.
STATEMENT_TIMEOUT_S = 300.0  # The evaluator's statement_timeout, set on the server.
TIMING_RUNS = 3
LONG_RUN_MS = 60_000.0  # A second run at least this long, on either side, ends the timing.
FETCH_CHUNK_ROWS = 5000  # Rows per streamed chunk, as the official evaluator fetches them.
# The client-side hard cap is twice the statement timeout plus this: it only catches hangs.
HARD_CAP_MARGIN_S = 60.0
MIN_LIBPQ_VERSION = 170000  # Chunked streaming needs libpq 17.

_HASH_MODULUS = 1 << 256
_HEX_256 = re.compile(r"[0-9a-f]{64}")
# Types psycopg 3 loads differently from psycopg2 2.9, which leaves them, and arrays of them, as
# the server's text. (psycopg2 has no typecaster for them by default.)
_TEXT_TYPES = ("uuid", "inet", "cidr", "record")
_TEXT_ARRAY_TYPES = (*_TEXT_TYPES, "money", "bit", "varbit")


def row_hash(row: tuple[Any, ...]) -> int:
    """Return the official row fingerprint, ``sha256(str(row))``, as an integer."""
    return int.from_bytes(hashlib.sha256(str(row).encode("utf-8")).digest(), "big")


@dataclass(frozen=True, slots=True)
class ResultDigest:
    """A result's signature: its row count and the sum of its row hashes mod 2^256.

    Equal multisets of rows give equal digests, in any order. ``complete`` is false if fetching
    stopped once the result had more rows than a limit: ``row_count`` is then a lower bound,
    and the digest matches nothing.
    """

    row_count: int
    row_hash_sum: int
    complete: bool = True

    def matches(self, other: "ResultDigest") -> bool:
        """Say whether both results hold the same multiset of rows."""
        return (
            self.complete
            and other.complete
            and self.row_count == other.row_count
            and self.row_hash_sum == other.row_hash_sum
        )

    def to_json(self) -> dict[str, int | str | bool]:
        return {
            "row_count": self.row_count,
            "row_hash_sum": f"{self.row_hash_sum:064x}",
            "complete": self.complete,
        }

    @classmethod
    def from_json(cls, value: object) -> Self:
        """Read :meth:`to_json`'s form back; raise ValueError for anything else."""
        if not isinstance(value, Mapping):
            raise ValueError("a result digest is a JSON object")
        count, total, complete = (value.get(k) for k in ("row_count", "row_hash_sum", "complete"))
        if type(count) is not int or count < 0:
            raise ValueError("a result digest's row_count is a non-negative integer")
        if not isinstance(total, str) or not _HEX_256.fullmatch(total):
            raise ValueError("a result digest's row_hash_sum is 64 lowercase hex digits")
        if type(complete) is not bool:
            raise ValueError("a result digest's complete is true or false")
        return cls(count, int(total, 16), complete)


class DigestBuilder:
    """Accumulates rows into a :class:`ResultDigest`, one batch at a time."""

    def __init__(self) -> None:
        self._count = 0
        self._sum = 0

    @property
    def row_count(self) -> int:
        return self._count

    def add(self, rows: Iterable[tuple[Any, ...]]) -> None:
        # row_hash, inlined: this loop runs once per row of every result.
        total, count, sha256, from_bytes = self._sum, 0, hashlib.sha256, int.from_bytes
        for row in rows:
            total += from_bytes(sha256(str(row).encode("utf-8")).digest(), "big")
            count += 1
        self._sum = total % _HASH_MODULUS
        self._count += count

    def digest(self, *, complete: bool = True) -> ResultDigest:
        return ResultDigest(self._count, self._sum, complete)


class Psycopg2Timestamptz(TimestamptzLoader):
    """A timestamptz with a fixed-offset tzinfo, as psycopg2 2.9 builds it, not a ZoneInfo."""

    def load(self, data: Any) -> datetime:
        value = super().load(data)
        offset = value.utcoffset()
        return value if offset is None else value.replace(tzinfo=timezone(offset))


def use_psycopg2_types(context: AdaptContext) -> None:
    """Load values as the official evaluator's psycopg2 2.9 does, so ``str(row)`` is the same.

    psycopg 3 already builds what psycopg2 builds for every type a TPC-H query returns: numeric
    (Decimal), date, time, timestamp, interval, integers, floats, bool, text and char(n) (str,
    padded), json, NULL (None), and arrays of these. It differs for uuid, inet, cidr and
    anonymous records, which psycopg2 leaves as text; for arrays of those and of money and bit,
    which psycopg2 leaves as one string, such as ``'{...}'``; and for timestamptz, whose tzinfo
    psycopg2 gives as a fixed offset. Still different, and never in a TPC-H result: infinite
    dates and timestamps, which psycopg2 loads as the maximum and psycopg 3 refuses; bytea; and
    ranges.
    """
    types = context.adapters.types
    for name in _TEXT_TYPES:
        context.adapters.register_loader(name, TextLoader)
    for name in _TEXT_ARRAY_TYPES:
        info = types.get(name)
        if info is not None and info.array_oid:
            context.adapters.register_loader(info.array_oid, TextLoader)
    context.adapters.register_loader("timestamptz", Psycopg2Timestamptz)


def speedup(base_ms: float, pred_ms: float) -> float:
    """Return τ = base / pred, capped at 1000; a pred time of 0 counts as 0.001 ms."""
    return min(base_ms / (pred_ms if pred_ms > 0 else ZERO_TIME_MS), SPEEDUP_CAP)


def statement_body(sql: str) -> str:
    """Return ``sql`` without surrounding whitespace and trailing semicolons, for wrapping."""
    return sql.strip().rstrip(";").rstrip()


class GradingFailure(Exception):  # noqa: N818  # The base of the grader's failures, not an error.
    """Something kept a grading statement from giving a result; the subclasses say what."""


class QueryError(GradingFailure):
    """The statement failed. The message is the server's primary message."""


class QueryTimeoutError(GradingFailure):
    """The statement ran past its time limit and was cancelled."""


class GraderError(GradingFailure):
    """The grader's own infrastructure failed, such as a lost connection: nothing was measured."""


class ResourceError(GraderError):
    """The server ran out of a resource (SQLSTATE class 53), such as shared memory or disk.

    It says nothing about the query, so it is the grader's failure, retried once.
    """


@dataclass(frozen=True, slots=True)
class Timing:
    """Both sides' median execution times, in ms, and how many timed runs they come from."""

    base_ms: float
    pred_ms: float
    runs: int

    @property
    def speedup(self) -> float:
        return speedup(self.base_ms, self.pred_ms)


class TimingError(GradingFailure):
    """A timed run failed; ``side`` says which query, and ``__cause__`` holds the error."""

    def __init__(
        self, side: Literal["pred", "base"], error: QueryError | QueryTimeoutError
    ) -> None:
        super().__init__(f"timing the {side} query failed: {error}")
        self.side = side
        self.timed_out = isinstance(error, QueryTimeoutError)


# Runs one EXPLAIN ANALYZE and returns its execution time in ms; raises QueryError or
# QueryTimeoutError for the query, GraderError for the infrastructure.
type Timer = Callable[[], float]


def _timed(side: Literal["pred", "base"], timer: Timer) -> float:
    try:
        return timer()
    except (QueryError, QueryTimeoutError) as error:
        raise TimingError(side, error) from error


def _check_runs(runs: int) -> None:
    if runs < 1:
        raise ValueError(f"timing needs at least 1 run, not {runs}")


def time_interleaved(
    time_pred: Timer, time_base: Timer, *, rng: random.Random, runs: int = TIMING_RUNS
) -> Timing:
    """Time both sides, in random order in each run, as the official evaluator does.

    Raise :class:`TimingError` as soon as a run fails.
    """
    _check_runs(runs)
    pred_ms: list[float] = []
    base_ms: list[float] = []
    for run in range(runs):
        sides: list[Literal["pred", "base"]] = ["pred", "base"]
        rng.shuffle(sides)
        times = {side: _timed(side, time_pred if side == "pred" else time_base) for side in sides}
        pred_ms.append(times["pred"])
        base_ms.append(times["base"])
        if _long_second_run(run, runs, times["pred"], times["base"]):
            break
    return Timing(statistics.median(base_ms), statistics.median(pred_ms), len(pred_ms))


def time_prediction(time_pred: Timer, *, base_ms: float, runs: int = TIMING_RUNS) -> Timing:
    """Time only the prediction, against the base's cached median (dev mode, spec §4.1).

    The stop rule reads the cached base median in place of the base's run, so a task gets as
    many runs as the interleaved protocol would give it.
    """
    _check_runs(runs)
    pred_ms: list[float] = []
    for run in range(runs):
        pred_ms.append(_timed("pred", time_pred))
        if _long_second_run(run, runs, pred_ms[-1], base_ms):
            break
    return Timing(base_ms, statistics.median(pred_ms), len(pred_ms))


def time_base(time_base: Timer, *, runs: int = TIMING_RUNS) -> tuple[float, int]:
    """Time only the base query (the setup's base cache); return its median ms and run count.

    The stop rule reads the base's own runs. Raise :class:`TimingError` if a run fails.
    """
    _check_runs(runs)
    base_ms: list[float] = []
    for run in range(runs):
        base_ms.append(_timed("base", time_base))
        if _long_second_run(run, runs, 0.0, base_ms[-1]):
            break
    return statistics.median(base_ms), len(base_ms)


def _long_second_run(run: int, runs: int, pred_ms: float, base_ms: float) -> bool:
    # The evaluator's rule: with more than 2 runs planned, stop after any run from the second
    # on in which either side took 60 s or more.
    return runs > 2 and run >= 1 and max(pred_ms, base_ms) >= LONG_RUN_MS


class Watchdog:
    """Calls ``cancel`` from its own thread once an armed deadline passes.

    ``arm`` sets the deadline and ``disarm`` clears it and says whether it fired. ``disarm``
    waits for a cancel already in flight, so a cancel fired at the end of one statement can
    never reach the next.
    """

    def __init__(self, cancel: Callable[[], None]) -> None:
        self._cancel = cancel
        self._condition = threading.Condition()
        self._deadline: float | None = None
        self._fired = False
        self._cancelling = False
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="grader-watchdog", daemon=True)
        self._thread.start()

    def arm(self, timeout_s: float) -> None:
        with self._condition:
            self._deadline = time.monotonic() + timeout_s
            self._fired = False
            self._condition.notify_all()

    def disarm(self) -> bool:
        with self._condition:
            self._deadline = None
            while self._cancelling:
                self._condition.wait()
            return self._fired

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._thread.join()

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._closed:
                    if self._deadline is None:
                        self._condition.wait()
                        continue
                    remaining = self._deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._condition.wait(remaining)
                if self._closed:
                    return
                self._deadline = None
                self._fired = True
                self._cancelling = True
            try:
                # A failed cancel means the statement ended meanwhile.
                with contextlib.suppress(psycopg.Error):
                    self._cancel()
            finally:
                with self._condition:
                    self._cancelling = False
                    self._condition.notify_all()


class PgSession:
    """Grading statements on one database, over one connection, from one worker.

    Every statement runs alone in a ``BEGIN READ ONLY`` transaction that is always rolled back,
    through the extended protocol, which rejects a second statement, and the session is reset
    after it. Errors map to :class:`QueryError` (the query failed), :class:`QueryTimeoutError`
    (it timed out) and :class:`GraderError` (the connection or the grader failed). Use it as a
    context manager, or call :meth:`close`.
    """

    def __init__(
        self,
        conn: psycopg.Connection[Any],
        *,
        timeout_s: float = STATEMENT_TIMEOUT_S,
        hard_cap_s: float | None = None,
    ) -> None:
        if psycopg.pq.version() < MIN_LIBPQ_VERSION:
            raise GraderError(
                f"libpq {psycopg.pq.version()} is too old: chunked streaming needs libpq 17"
            )
        conn.autocommit = True  # Each statement gets its own transaction, below.
        conn.read_only = True
        conn.row_factory = tuple_row  # str(row) must be str(tuple), as psycopg2's rows are.
        conn.prepare_threshold = None  # Never prepare: DISCARD ALL drops prepared statements.
        use_psycopg2_types(conn)
        self._conn = conn
        self._timeout_s = timeout_s
        # Only for a hang: by default twice the server's timeout, plus a margin.
        self._hard_cap_s = 2 * timeout_s + HARD_CAP_MARGIN_S if hard_cap_s is None else hard_cap_s
        self._settings = f"SET statement_timeout = {round(timeout_s * 1000)}"
        conn.execute(self._settings)
        self._watchdog = Watchdog(conn.cancel_safe)

    @classmethod
    def open(cls, connect: Connect, db: str, *, timeout_s: float = STATEMENT_TIMEOUT_S) -> Self:
        try:
            conn = connect(db)
        except psycopg.Error as error:
            raise GraderError(f"cannot connect to {db}: {_message(error)}") from None
        try:
            return cls(conn, timeout_s=timeout_s)
        except psycopg.Error as error:
            conn.close()
            raise GraderError(f"cannot set up the session on {db}: {_message(error)}") from None
        except GraderError:
            conn.close()
            raise

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def connection(self) -> psycopg.Connection[Any]:
        return self._conn

    def digest(self, sql: str, *, row_limit: int | None = None) -> ResultDigest:
        """Run ``sql`` fully, streaming its rows, and return its result's digest.

        With ``row_limit``, stop once the result has more rows than that: the digest is then
        incomplete.
        """
        builder = DigestBuilder()
        complete = True
        with self._statement():
            # stream() is a generator, though typed as an Iterator. Closing it early cancels the
            # rest of the statement and releases the connection before the rollback.
            stream = cast(
                Generator[tuple[Any, ...]],
                self._conn.cursor().stream(statement_body(sql), size=FETCH_CHUNK_ROWS),
            )
            with contextlib.closing(stream) as rows:
                for chunk in batched(rows, FETCH_CHUNK_ROWS):
                    builder.add(chunk)
                    if row_limit is not None and builder.row_count > row_limit:
                        complete = False
                        break
        return builder.digest(complete=complete)

    def execution_ms(self, sql: str) -> float:
        """Run ``EXPLAIN (ANALYZE, FORMAT JSON)`` on ``sql``; return its execution time in ms."""
        document = self._explain("ANALYZE, FORMAT JSON", sql)
        value = document.get("Execution Time")
        if not isinstance(value, int | float):
            raise GraderError("EXPLAIN ANALYZE gave no Execution Time")
        return float(value)

    def plan(self, sql: str) -> dict[str, Any]:
        """Return ``EXPLAIN (FORMAT JSON)``'s document for ``sql``: planned, never run."""
        return self._explain("FORMAT JSON", sql)

    def close(self) -> None:
        self._watchdog.close()
        self._conn.close()

    def _explain(self, options: str, sql: str) -> dict[str, Any]:
        with self._statement():
            cursor = self._conn.cursor()
            # Binary results force the extended protocol, so a second statement is an error.
            # Only the fixed options are added to the query, as EXPLAIN takes no parameter.
            cursor.execute(f"EXPLAIN ({options}) {statement_body(sql)}", binary=True)
            row = cursor.fetchone()
        value = row[0] if row else None
        if isinstance(value, str | bytes):
            try:
                value = json.loads(value)
            except ValueError:
                raise GraderError("EXPLAIN gave a plan that is not JSON") from None
        if not isinstance(value, list) or not value or not isinstance(value[0], dict):
            raise GraderError("EXPLAIN gave no plan document")
        document: dict[str, Any] = value[0]
        return document

    @contextlib.contextmanager
    def _statement(self) -> Iterator[None]:
        """Run the body's statement in a read-only transaction, then reset the session."""
        fired = False
        try:
            try:
                with self._conn.transaction(force_rollback=True):
                    self._watchdog.arm(self._hard_cap_s)
                    try:
                        yield
                    finally:
                        fired = self._watchdog.disarm()
            finally:
                self._reset()
        except psycopg.Error as error:
            raise self._failure(error, fired=fired) from None

    def _reset(self) -> None:
        """Drop what the statement left in the session, and apply the grader's settings again.

        DISCARD ALL releases advisory locks, deallocates prepared statements, drops temporary
        tables, stops listening and resets every setting.
        """
        if self._lost():
            return  # The statement's own error reports the lost connection.
        try:
            self._conn.execute("DISCARD ALL")
            self._conn.execute(self._settings)
        except psycopg.Error as error:
            if self._lost():
                return
            raise GraderError(f"cannot reset the session: {_message(error)}") from None

    def _lost(self) -> bool:
        return self._conn.closed or self._conn.broken

    def _failure(self, error: psycopg.Error, *, fired: bool) -> GradingFailure:
        if self._lost():
            return GraderError(f"the connection failed: {_message(error)}")
        if fired:
            return QueryTimeoutError(f"stopped by the {self._hard_cap_s:g} s hard cap")
        if isinstance(error, psycopg.errors.QueryCanceled):
            return QueryTimeoutError(f"{_message(error)} ({self._timeout_s:g} s)")
        if (error.sqlstate or "").startswith("53"):
            return ResourceError(f"the server ran out of resources: {_message(error)}")
        return QueryError(_message(error))


def _message(error: psycopg.Error) -> str:
    diag = getattr(error, "diag", None)
    primary = diag.message_primary if diag is not None else None
    if primary:
        return str(primary)
    lines = str(error).splitlines()
    return lines[0] if lines else type(error).__name__
