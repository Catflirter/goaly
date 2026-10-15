"""Grading an Effi-SQL prediction in dev or official mode (spec §4.1), on the grading instances.

One evaluation (:func:`evaluate`, run in a worker process) goes:

1. **Plan.** ``EXPLAIN (FORMAT JSON)`` on the prediction. A plan that fails is a failed
   prediction, and a plan that modifies data is refused before it runs, before anything is
   measured: not matched, with the PREDICTION_FAILED status.
2. **Comparison.** The prediction runs fully, and its result's digest is compared with the
   base query's: cached in dev mode, measured in official mode, after the prediction and up to
   its row count, as the official evaluator does.
3. **Checks**, only on a matched prediction: the plan check and the robustness check on the
   alternate database (goaly.bench.effisql.checks). An unmatched prediction leaves both unset.
4. **Timing**, even when a check flags it: dev mode times only the prediction against the
   cached base median, official mode runs the interleaved protocol. A timing that fails
   leaves the prediction matched, with τ = 0 and credit 0, as the official evaluator scores it;
   the status and feedback say why.

``credit`` is the checked credit: g(τ) for a matched prediction no check flagged, else 0.

**Dev mode** reads the base cache that the setup run writes (:class:`BaseCache`) and measures
any task missing from it. A prediction identical to ``base_sql`` after :func:`normalize_sql`
gets the setup's identity τ and its credit, with no timing at all. **Official mode** measures
everything every time and never reads or writes a cache, so no dev measurement can reach an
official grade. Each mode has its own directory under the grader's cache root.

goaly.bench.effisql.grading_service is the async face that the Benchmark adapter's ``grade``
calls.
"""

import contextlib
import enum
import hashlib
import logging
import random
import re
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg
from pydantic import BaseModel, ConfigDict, ValidationError

from goaly.bench.effisql.checks import (
    alt_matches,
    inspect_plan,
    plan_nodes,
    read_only_problems,
    scanned_relations,
)
from goaly.bench.effisql.data import EffiSqlInstance
from goaly.bench.effisql.grader import (
    GraderError,
    PgSession,
    QueryError,
    QueryTimeoutError,
    ResourceError,
    ResultDigest,
    TimingError,
    time_base,
    time_interleaved,
    time_prediction,
)
from goaly.bench.effisql.grading_pool import GradingInstance
from goaly.bench.effisql.metrics import tier_credit
from goaly.bench.effisql.pg_build import Connect
from goaly.bench.effisql.pg_settings import ALT_DB, MAIN_DB, write_private
from goaly.contracts.gen.task_pb2 import (
    GRADE_STATUS_GRADED,
    GRADE_STATUS_GRADER_ERROR,
    GRADE_STATUS_PREDICTION_FAILED,
    GRADE_STATUS_TIMEOUT,
    EffiSqlGrade,
    Grade,
    GradeStatus,
    Prediction,
)

__all__ = [
    "AbortedError",
    "BaseCache",
    "BaseEntry",
    "GradingSessions",
    "GradingSlot",
    "InstanceCheck",
    "Mode",
    "evaluate",
    "grader_error",
    "measure_base",
    "normalize_sql",
    "prediction_sql",
    "quick_grade",
]

_log = logging.getLogger(__name__)


class Mode(enum.StrEnum):
    DEV = "dev"  # RSI screening and validation: the base side cached, identity τ reused.
    OFFICIAL = "official"  # The reported held-out numbers: everything measured every time.


_DOLLAR_TAG = re.compile(r"\$(?:[A-Za-z_][A-Za-z0-9_]*)?\$")
_TIGHT = frozenset("(),;")


def normalize_sql(sql: str) -> str:
    """Return ``sql`` with comments dropped, whitespace collapsed and unquoted text lowercased.

    Quoted strings, quoted identifiers and dollar-quoted bodies stay exactly as written, so two
    queries with the same normal form are the same query to PostgreSQL. Trailing semicolons go.
    """
    out: list[str] = []
    space = False
    i, n = 0, len(sql)
    while i < n:
        char = sql[i]
        if char.isspace():
            space, i = True, i + 1
            continue
        if sql.startswith("--", i):
            end = sql.find("\n", i)
            space, i = True, n if end < 0 else end
            continue
        if sql.startswith("/*", i):
            space, i = True, _block_comment_end(sql, i)
            continue
        end = _quoted_end(sql, i)
        token = sql[i:end] if end > i else char.lower()
        end = max(end, i + 1)
        if space and out and out[-1][-1] not in _TIGHT and token[0] not in _TIGHT:
            out.append(" ")
        out.append(token)
        space, i = False, end
    text = "".join(out)
    while text.endswith(";"):
        text = text[:-1].rstrip()
    return text


def _block_comment_end(sql: str, start: int) -> int:
    depth, i = 0, start
    while i < len(sql):
        if sql.startswith("/*", i):
            depth, i = depth + 1, i + 2
        elif sql.startswith("*/", i):
            depth, i = depth - 1, i + 2
            if depth == 0:
                return i
        else:
            i += 1
    return len(sql)


def _quoted_end(sql: str, i: int) -> int:
    """Return where the quoted token starting at ``i`` ends, or ``i`` if none starts there."""
    char = sql[i]
    word_before = i > 0 and (sql[i - 1].isalnum() or sql[i - 1] in "_$")
    if char in "eE" and sql.startswith("'", i + 1) and not word_before:
        j = i + 2
        while j < len(sql):
            if sql[j] == "\\" or sql.startswith("''", j):
                j += 2
            elif sql[j] == "'":
                return j + 1
            else:
                j += 1
        return len(sql)
    if char in "'\"":
        j = i + 1
        while j < len(sql):
            if sql.startswith(char * 2, j):
                j += 2
            elif sql[j] == char:
                return j + 1
            else:
                j += 1
        return len(sql)
    if char == "$" and not word_before and (tag := _DOLLAR_TAG.match(sql, i)):
        close = sql.find(tag.group(), tag.end())
        return len(sql) if close < 0 else close + len(tag.group())
    return i


def prediction_sql(inst: EffiSqlInstance, prediction: Prediction) -> str | None:
    """Return the prediction's query: ``base_sql`` unchanged if it has no benchmark branch, as
    after a crash; None if it is not an Effi-SQL answer."""
    branch = prediction.WhichOneof("benchmark")
    if branch is None:
        return inst.base_sql
    return prediction.effisql.sql if branch == "effisql" else None


class BaseEntry(BaseModel):
    """What the setup measured of one task's base query; dev mode grades against it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    instance_id: str
    mode: Mode = Mode.DEV  # The grading mode the entry serves; part of the cache key.
    base_sql_sha256: str  # The entry belongs to this base_sql only.
    row_count: int
    row_hash_sum: str
    alt_row_count: int
    alt_row_hash_sum: str
    base_ms: float  # The median of the base's timed runs.
    timing_runs: int
    relations: tuple[str, ...]  # The tables the base query's plan scans.
    identity_speedup: float | None = None  # The setup's identity τ; None until measured.
    measured_on: str
    measured_at: datetime

    @property
    def digest(self) -> ResultDigest:
        return ResultDigest(self.row_count, int(self.row_hash_sum, 16))

    @property
    def alt_digest(self) -> ResultDigest:
        return ResultDigest(self.alt_row_count, int(self.alt_row_hash_sum, 16))


def _sql_sha256(sql: str) -> str:
    return hashlib.sha256(sql.encode("utf-8")).hexdigest()


class BaseCache:
    """One JSON file per task under ``directory``, each written atomically.

    An entry is keyed by the grading mode, the task and its ``base_sql``: a lookup in one mode
    never returns an entry for another, nor for another base query.
    """

    def __init__(self, directory: Path, *, mode: Mode = Mode.DEV) -> None:
        self.directory = directory
        self.mode = mode

    def path(self, instance_id: str) -> Path:
        return self.directory / f"{instance_id}.json"

    def get(self, inst: EffiSqlInstance) -> BaseEntry | None:
        """Return the task's entry, or None if it has none for this ``base_sql``."""
        path = self.path(inst.instance_id)
        try:
            entry = BaseEntry.model_validate_json(path.read_bytes())
        except FileNotFoundError:
            return None
        except (OSError, ValidationError) as error:
            _log.warning("ignoring an unreadable base cache entry", extra={"path": str(path)})
            _log.debug("base cache error", exc_info=error)
            return None
        key = (entry.mode, entry.base_sql_sha256)
        return entry if key == (self.mode, _sql_sha256(inst.base_sql)) else None

    def put(self, entry: BaseEntry) -> None:
        if entry.mode is not self.mode:
            raise ValueError(f"a {entry.mode} entry does not belong in the {self.mode} cache")
        write_private(self.path(entry.instance_id), entry.model_dump_json(indent=2) + "\n")

    def put_measured(self, inst: EffiSqlInstance, entry: BaseEntry) -> None:
        """Store a dev-mode measurement, unless the task has an entry with an identity τ.

        Only the setup measures the identity τ, and an on-the-fly measurement must not lose it.
        """
        existing = self.get(inst)
        if existing is None or existing.identity_speedup is None:
            self.put(entry)


class _BaseQueryError(Exception):
    """The base query failed or timed out, so the task cannot be graded."""

    def __init__(self, error: QueryError | QueryTimeoutError) -> None:
        super().__init__(str(error))
        self.timed_out = isinstance(error, QueryTimeoutError)


def _base[T](step: Callable[[], T]) -> T:
    try:
        return step()
    except (QueryError, QueryTimeoutError) as error:
        raise _BaseQueryError(error) from error


def _query_cause(error: TimingError) -> QueryError | QueryTimeoutError:
    cause = error.__cause__
    if not isinstance(cause, QueryError | QueryTimeoutError):
        raise TypeError(f"a TimingError's cause is a query error, not {cause!r}")
    return cause


class _PredictionResourceError(Exception):
    """The server ran out of resources on one of the prediction's statements."""


def _prediction[T](step: Callable[[], T]) -> T:
    try:
        return step()
    except ResourceError as error:
        raise _PredictionResourceError from error


class AbortedError(Exception):
    """The grading was cancelled while its worker ran; its outcome is discarded."""


class GradingSessions:
    """A grading instance's two sessions, on the main and the alternate database.

    ``aborted`` is the running evaluation's own flag: the slot gives each evaluation a new one.
    """

    def __init__(self, main: PgSession, alt: PgSession, *, instance: str) -> None:
        self.main = main
        self.alt = alt
        self.instance = instance
        self.aborted = threading.Event()

    def check_abort(self) -> None:
        if self.aborted.is_set():
            raise AbortedError

    def cancel_statements(self) -> None:
        """Cancel any statement in flight on either session."""
        for session in (self.main, self.alt):
            with contextlib.suppress(Exception):
                session.connection.cancel_safe()

    def close(self) -> None:
        for session in (self.main, self.alt):
            with contextlib.suppress(Exception):
                session.close()


def _base_relations(document: dict[str, Any]) -> frozenset[str]:
    """The tables the base query's plan scans; none means the plan check does not apply."""
    return scanned_relations(document)


def measure_base(
    sessions: GradingSessions, inst: EffiSqlInstance, *, identity_rng: random.Random | None = None
) -> BaseEntry:
    """Measure what dev mode caches of the task's base query.

    With ``identity_rng``, as the setup runs it, the base is timed against itself with the
    interleaved protocol: the entry's time is one side's median and its identity τ the ratio.
    Without, the base is timed alone and the entry has no identity τ. Raise QueryError or
    QueryTimeoutError if the base query fails, GraderError if the grader does.
    """
    base_sql = inst.base_sql
    relations = _base_relations(sessions.main.plan(base_sql))
    sessions.check_abort()
    digest = sessions.main.digest(base_sql)
    sessions.check_abort()
    alt = sessions.alt.digest(base_sql)
    sessions.check_abort()

    def time_it() -> float:
        sessions.check_abort()
        return sessions.main.execution_ms(base_sql)

    identity: float | None = None
    try:
        if identity_rng is None:
            base_ms, runs = time_base(time_it)
        else:
            timing = time_interleaved(time_it, time_it, rng=identity_rng)
            base_ms, runs, identity = timing.base_ms, timing.runs, timing.speedup
    except TimingError as error:
        raise _query_cause(error) from None
    return BaseEntry(
        instance_id=inst.instance_id,
        base_sql_sha256=_sql_sha256(base_sql),
        row_count=digest.row_count,
        row_hash_sum=f"{digest.row_hash_sum:064x}",
        alt_row_count=alt.row_count,
        alt_row_hash_sum=f"{alt.row_hash_sum:064x}",
        base_ms=base_ms,
        timing_runs=runs,
        relations=tuple(sorted(relations)),
        identity_speedup=identity,
        measured_on=sessions.instance,
        measured_at=datetime.now(UTC),
    )


type _Result = tuple[EffiSqlGrade, GradeStatus.ValueType, str]


def quick_grade(
    inst: EffiSqlInstance, sql: str | None, mode: Mode, cache: BaseCache | None
) -> Grade | None:
    """Grade what needs no database: a missing or empty answer, and, in dev mode, an identity
    submission whose task has the setup's identity τ. Return None for anything else."""
    if sql is None:
        result = (EffiSqlGrade(), GRADE_STATUS_PREDICTION_FAILED, "not an Effi-SQL prediction")
    elif not sql.strip():
        result = (EffiSqlGrade(), GRADE_STATUS_PREDICTION_FAILED, "the prediction is empty")
    else:
        entry = cache.get(inst) if mode is Mode.DEV and cache is not None else None
        if entry is None or entry.identity_speedup is None:
            return None
        if normalize_sql(sql) != normalize_sql(inst.base_sql):
            return None
        result = _identity(entry, entry.identity_speedup)
    return _grade(inst, *result)


def evaluate(
    sessions: GradingSessions,
    inst: EffiSqlInstance,
    sql: str | None,
    mode: Mode,
    *,
    cache: BaseCache | None,
    rng: random.Random,
    plans: list[dict[str, Any]] | None = None,
    final_attempt: bool = True,
) -> Grade:
    """Grade ``sql`` for ``inst`` in ``mode``; blocking, so run it in a worker.

    When the server runs out of resources (SQLSTATE class 53) on the prediction's own
    statements, the grade is a GRADER_ERROR to retry, unless this is the ``final_attempt``: it
    then fails the prediction with credit 0, as the official evaluator, with the same server
    limits, would score it. Out of resources on the base's statements is always the grader's.

    With ``plans``, the prediction's ``EXPLAIN (FORMAT JSON)`` document is appended to it, once
    the plan was taken, for the run's audit trail.

    ``cache`` is dev mode's base cache; official mode ignores it. A base query that times out
    is graded TIMEOUT and scored 0, as the official evaluator scores it; the setup excludes such
    tasks. Raises AbortedError if the evaluation is cancelled, and nothing else.
    """
    quick = quick_grade(inst, sql, mode, cache)
    if quick is not None or sql is None:
        return quick or grader_error(inst, "grader: no SQL")
    started = time.monotonic()
    try:
        details, status, feedback = _evaluate(sessions, inst, sql, mode, cache, rng, plans)
    except _BaseQueryError as error:
        status = GRADE_STATUS_TIMEOUT if error.timed_out else GRADE_STATUS_GRADER_ERROR
        details, feedback = EffiSqlGrade(), f"the base query failed, so nothing was graded: {error}"
    except _PredictionResourceError as error:
        status = GRADE_STATUS_PREDICTION_FAILED if final_attempt else GRADE_STATUS_GRADER_ERROR
        details, feedback = EffiSqlGrade(), f"the prediction failed: {error.__cause__}"
    except GraderError as error:
        details, status, feedback = EffiSqlGrade(), GRADE_STATUS_GRADER_ERROR, f"grader: {error}"
    details.eval_wall_s = time.monotonic() - started
    return _grade(inst, details, status, feedback)


def _grade(
    inst: EffiSqlInstance, details: EffiSqlGrade, status: GradeStatus.ValueType, feedback: str
) -> Grade:
    flagged = details.HasField("alt_matched") and not details.alt_matched
    correct = status == GRADE_STATUS_GRADED and details.matched
    return Grade(
        instance_id=inst.instance_id,
        score=details.credit,
        correct=correct and not (flagged or details.literal_result),
        status=status,
        feedback=feedback,
        effisql=details,
    )


def _evaluate(
    sessions: GradingSessions,
    inst: EffiSqlInstance,
    sql: str,
    mode: Mode,
    cache: BaseCache | None,
    rng: random.Random,
    plans: list[dict[str, Any]] | None = None,
) -> _Result:
    main, base_sql = sessions.main, inst.base_sql
    entry = cache.get(inst) if mode is Mode.DEV and cache is not None else None
    try:
        plan = _prediction(lambda: main.plan(sql))
        if plans is not None:
            plans.append(plan)
    except QueryError as error:
        return EffiSqlGrade(), GRADE_STATUS_PREDICTION_FAILED, f"the prediction failed: {error}"
    except QueryTimeoutError as error:
        return EffiSqlGrade(), GRADE_STATUS_TIMEOUT, f"planning the prediction timed out: {error}"
    if any(node.get("Node Type") == "ModifyTable" for node in plan_nodes(plan.get("Plan", {}))):
        feedback = "the prediction's plan modifies data, so it was refused before it ran"
        return EffiSqlGrade(), GRADE_STATUS_PREDICTION_FAILED, feedback
    cached = entry is not None
    if mode is Mode.DEV and entry is None:
        entry = _base(lambda: measure_base(sessions, inst))
        if cache is not None:
            cache.put_measured(inst, entry)
    if entry is not None:
        relations = frozenset(entry.relations)
    else:
        relations = _base_relations(_base(lambda: main.plan(base_sql)))
    verdict = inspect_plan(plan, relations, sql=sql)
    sessions.check_abort()

    details = EffiSqlGrade(base_cached=cached)
    try:
        pred = _prediction(
            lambda: main.digest(sql, row_limit=None if entry is None else entry.row_count)
        )
    except QueryError as error:
        return details, GRADE_STATUS_PREDICTION_FAILED, f"the prediction failed: {error}"
    except QueryTimeoutError as error:
        return details, GRADE_STATUS_TIMEOUT, f"the prediction timed out: {error}"
    sessions.check_abort()
    if entry is not None:
        base = entry.digest
    else:  # As the official evaluator does: the base is fetched up to the prediction's count.
        base = _base(lambda: main.digest(base_sql, row_limit=pred.row_count))
    details.pred_row_count, details.base_row_count = pred.row_count, base.row_count
    if not pred.matches(base):
        return details, GRADE_STATUS_GRADED, _mismatch(pred, base)

    details.matched = True
    details.literal_result = verdict.literal_result
    base_alt = (
        entry.alt_digest if entry is not None else _base(lambda: sessions.alt.digest(base_sql))
    )
    sessions.check_abort()
    alt = _prediction(lambda: alt_matches(sessions.alt, sql, base_alt))
    details.alt_matched = alt.matched
    sessions.check_abort()
    flags = []
    if not alt.matched:
        flags.append(f"hard_coded: {alt.reason}")
    flags.extend(f"literal_result: {reason}" for reason in verdict.reasons)
    return _timed(sessions, details, sql, base_sql, entry, rng, flags)


def _mismatch(pred: ResultDigest, base: ResultDigest) -> str:
    def count(digest: ResultDigest) -> str:
        return f"{'' if digest.complete else 'more than '}{digest.row_count}"

    same_count = pred.complete and base.complete and pred.row_count == base.row_count
    detail = ", with different values" if same_count else ""
    return (
        f"not matched: the prediction returned {count(pred)} rows, "
        f"the base {count(base)} rows{detail}"
    )


def _timed(
    sessions: GradingSessions,
    details: EffiSqlGrade,
    sql: str,
    base_sql: str,
    entry: BaseEntry | None,
    rng: random.Random,
    flags: list[str],
) -> _Result:
    main = sessions.main

    def time_pred() -> float:
        sessions.check_abort()
        return _prediction(lambda: main.execution_ms(sql))

    def time_the_base() -> float:
        sessions.check_abort()
        return main.execution_ms(base_sql)

    try:
        if entry is not None:
            timing = time_prediction(time_pred, base_ms=entry.base_ms)
        else:
            timing = time_interleaved(time_pred, time_the_base, rng=rng)
    except TimingError as error:
        cause = _query_cause(error)
        if error.side == "base":
            raise _BaseQueryError(cause) from error
        # The evaluator's rule: the prediction stays matched, with τ = 0 and credit g(0) = 0.
        details.speedup = 0.0
        details.credit = tier_credit(0.0)
        status = GRADE_STATUS_TIMEOUT if error.timed_out else GRADE_STATUS_PREDICTION_FAILED
        return details, status, f"matched, but timing the prediction failed, so τ = 0: {cause}"
    details.base_time_s = timing.base_ms / 1000
    details.pred_time_s = timing.pred_ms / 1000
    details.timing_runs = timing.runs
    details.speedup = timing.speedup
    details.credit = 0.0 if flags else tier_credit(details.speedup)
    feedback = (
        f"matched; speedup {details.speedup:.3g}x (base {timing.base_ms:.1f} ms, prediction "
        f"{timing.pred_ms:.1f} ms, {timing.runs} runs); credit {details.credit:g}"
    )
    return details, GRADE_STATUS_GRADED, feedback + "".join(f"; {flag}" for flag in flags)


def _identity(entry: BaseEntry, speedup: float) -> _Result:
    details = EffiSqlGrade(
        matched=True,
        base_time_s=entry.base_ms / 1000,
        speedup=speedup,
        credit=tier_credit(speedup),
        timing_runs=0,
        pred_row_count=entry.row_count,
        base_row_count=entry.row_count,
        base_cached=True,
        identity_cached=True,
    )
    feedback = (
        f"identical to base_sql: the setup's identity speedup {speedup:.3g}x, "
        f"credit {details.credit:g}, without timing it again"
    )
    return details, GRADE_STATUS_GRADED, feedback


# Checks a grading instance before it grades: T09's catalog fingerprint, once it lands.
type InstanceCheck = Callable[[GradingSessions], None]


class GradingSlot:
    """One grading instance's sessions, opened on first use and reopened after a failure.

    Its work runs one call at a time (:meth:`run`), in one worker thread or one worker process:
    each grading instance has its own (goaly.bench.effisql.grading_service). Each call has a
    token, and :meth:`abort` stops only the call with that token, whether it is running or not
    started yet, so a late cancel can never reach the next call.
    """

    def __init__(
        self,
        instance: GradingInstance,
        connect: Callable[[GradingInstance], Connect],
        *,
        check: InstanceCheck | None = None,
        cancelled: Callable[[], int] | None = None,
    ) -> None:
        self.instance = instance
        self._connect = connect
        self._check = check
        self._cancelled = cancelled  # The token another process asked to cancel, if any.
        self._lock = threading.Lock()  # Held while a call starts, ends, aborts or resets.
        self._running: int | None = None
        self._cancel: int | None = None
        self._aborted = threading.Event()
        self._sessions: GradingSessions | None = None
        self.rng = random.Random()  # noqa: S311  # Shuffles timing order; not a secret.

    def run[T](self, token: int, work: Callable[["GradingSlot"], T]) -> T:
        """Run ``work`` on this slot as the call ``token``; raise AbortedError if cancelled."""
        with self._lock:
            if token in (self._cancel, self._cancelled() if self._cancelled else None):
                raise AbortedError
            self._running = token
            self._aborted = threading.Event()
            if self._sessions is not None:
                self._sessions.aborted = self._aborted
        try:
            return work(self)
        finally:
            with self._lock:  # Waits for an abort in progress, so it ends with this call.
                self._running = None

    def abort(self, token: int) -> None:
        """Stop the call ``token``: now if it runs, at its start if it has not started."""
        with self._lock:
            self._cancel = token
            if self._running == token:
                self._aborted.set()
                if self._sessions is not None:
                    self._sessions.cancel_statements()

    def sessions(self) -> GradingSessions:
        """Return the slot's sessions, opening them if needed, and check the instance."""
        if self._sessions is None:
            connect = self._connect(self.instance)
            main = PgSession.open(connect, MAIN_DB)
            try:
                alt = PgSession.open(connect, ALT_DB)
            except GraderError:
                main.close()
                raise
            sessions = GradingSessions(main, alt, instance=self.instance.name)
            try:
                problems = [
                    f"{db}: {problem}"
                    for db, session in ((MAIN_DB, main), (ALT_DB, alt))
                    for problem in read_only_problems(session.connection)
                ]
            except psycopg.Error as error:
                sessions.close()
                raise GraderError(f"cannot check the grading role: {error}") from None
            if problems:
                sessions.close()
                raise GraderError(f"{self.instance.name} is unsafe to grade on: {problems}")
            with self._lock:
                sessions.aborted = self._aborted
                self._sessions = sessions
        if self._check is not None:
            self._check(self._sessions)
        return self._sessions

    @property
    def open_sessions(self) -> GradingSessions | None:
        return self._sessions

    def reset(self) -> None:
        """Close the sessions, so the next evaluation reconnects."""
        with self._lock:
            sessions, self._sessions = self._sessions, None
        if sessions is not None:
            sessions.close()

    def evaluate(
        self,
        inst: EffiSqlInstance,
        sql: str | None,
        mode: Mode,
        cache: BaseCache | None,
        plans: list[dict[str, Any]] | None = None,
        final_attempt: bool = True,
    ) -> Grade:
        """Grade on this slot. Any failure but a cancel is a GRADER_ERROR grade and a reset.

        With ``plans``, the prediction's plan is appended to it once it was taken.
        """
        try:
            sessions = self.sessions()
            grade = evaluate(
                sessions,
                inst,
                sql,
                mode,
                cache=cache,
                rng=self.rng,
                plans=plans,
                final_attempt=final_attempt,
            )
        except AbortedError:
            raise
        except Exception as error:  # noqa: BLE001  # Never raise for grading trouble.
            _log.exception("grading failed on %s", self.instance.name)
            self.reset()
            return grader_error(inst, f"grader: {type(error).__name__}: {error}")
        if grade.status == GRADE_STATUS_GRADER_ERROR:
            self.reset()  # A failed connection is reopened for the next evaluation.
        return grade


def grader_error(inst: EffiSqlInstance, feedback: str) -> Grade:
    """Return the grade of an evaluation the grader itself could not carry out."""
    return Grade(
        instance_id=inst.instance_id,
        status=GRADE_STATUS_GRADER_ERROR,
        feedback=feedback,
        effisql=EffiSqlGrade(),
    )
