"""The grader's one-time setup run (spec §4.1, §6.2): the base cache, the self-checks and the
exclusions, measured on this machine before any solving starts.

Three phases, each resumable from the per-task records it writes (gold-free JSON, mode 0600,
outside the repository), then a summary:

1. **measure**, on all 5 instances, so the identity τ is measured under load. For each TPC-H
   task: the base cache entry (signatures on both databases, the base plan's relations, its
   median time, and the identity τ from base against base, interleaved), the data check, and
   the gold check: the gold graded in dev mode against that entry.
2. **official**, on the 3 grading instances with the solving instances idle: the kept val golds
   and their identity submissions graded in official mode, for the dev-versus-official
   agreement.
3. **negatives**, on a fixed sample of 10 kept tasks with small, non-empty results: variants of
   base_sql that drop, duplicate, reorder or perturb rows must not match; a renamed copy must.

A grader failure is never recorded: the task stays unrecorded, and the next run retries it.
:func:`summarize` decides the exclusions (:func:`exclusion_reason`, plus the reviewed ones in
the committed manual file), applies the 10% stop rule, and reports the self-checks and the
measured grading times.
"""

import asyncio
import json
import logging
import math
import random
import statistics
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from goaly.bench.effisql.data import EffiSqlInstance
from goaly.bench.effisql.grader import (
    STATEMENT_TIMEOUT_S,
    GraderError,
    QueryError,
    QueryTimeoutError,
    ResourceError,
    statement_body,
    time_interleaved,
)
from goaly.bench.effisql.grading import (
    BaseCache,
    BaseEntry,
    GradingSlot,
    Mode,
    evaluate,
    measure_base,
)
from goaly.bench.effisql.grading_pool import GradingPool, Priority
from goaly.bench.effisql.grading_service import SlotRunner
from goaly.bench.effisql.pg_settings import write_private
from goaly.bench.effisql.pg_snapshot import foreign_scale_plan
from goaly.bench.effisql.splits import EXCLUSIONS_PATH, SPLIT_SEED, Exclusion
from goaly.contracts.gen.task_pb2 import (
    GRADE_STATUS_GRADED,
    GRADE_STATUS_GRADER_ERROR,
    Grade,
    GradeStatus,
)

__all__ = [
    "AGREEMENT_BOUND",
    "EXCLUSION_BOUND",
    "GOLD_MIN_SPEEDUP",
    "IDENTITY_TOLERANCE",
    "MANUAL_EXCLUSIONS_PATH",
    "NEGATIVE_SAMPLE",
    "SMALL_RESULT_ROWS",
    "DataCheck",
    "GradeSummary",
    "Phase",
    "RecordStore",
    "SetupRecord",
    "SetupSummary",
    "TimeStats",
    "data_check",
    "dump_exclusions",
    "exclusion_reason",
    "load_manual_exclusions",
    "measure_task",
    "negative_sample",
    "negative_task",
    "negative_variants",
    "official_task",
    "run_phase",
    "summarize",
    "to_markdown",
]

IDENTITY_TOLERANCE = (
    0.10  # Identity τ must be within 1 ± this: the default of --identity-tolerance.
)
GOLD_MIN_SPEEDUP = 1.2  # A gold below this speedup is excluded.
SMALL_RESULT_ROWS = 10_000  # Negative checks run on tasks with fewer base rows than this.
NEGATIVE_SAMPLE = 10
EXCLUSION_BOUND = 0.10  # More than this share of tasks excluded stops the setup.
AGREEMENT_BOUND = 0.90  # Dev and official mode give the same tier on at least this share.
# A gold whose dev-mode speedup falls in this range is graded again in official mode, and its
# exclusion decided on that (the controller's ruling).
MARGINAL_GOLD = (0.8, 1.5)
# How a base failure from before ResourceError existed says the server ran out of resources.
_RESOURCE_SIGNS = ("could not resize shared memory", "no space left on device", "out of memory")
# Exclusions a review decided, such as a gold a hard-coding check flags with the gold at fault.
MANUAL_EXCLUSIONS_PATH = EXCLUSIONS_PATH.with_name("effisql_exclusions_manual.json")
_TOLERANCE_EPSILON = 1e-9  # So 1.1 counts as within 1 ± 0.1 despite floating point.
_BASE_TIMEOUT_REASON = "base query exceeds the 300 s statement timeout on this machine"
_GRADER_ERROR = GradeStatus.Name(GRADE_STATUS_GRADER_ERROR)
_GRADED = GradeStatus.Name(GRADE_STATUS_GRADED)

_log = logging.getLogger(__name__)

type Phase = Literal["measure", "recheck", "official", "negatives"]


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class GradeSummary(_Model):
    """The parts of a Grade the setup keeps."""

    status: str
    matched: bool
    alt_matched: bool | None = None
    literal_result: bool | None = None
    speedup: float | None = None
    credit: float = 0.0
    timing_runs: int = 0
    eval_wall_s: float | None = None
    feedback: str = ""

    @classmethod
    def of(cls, grade: Grade, *, keep_feedback: bool = True) -> "GradeSummary":
        """Summarize ``grade``. A gold's feedback is dropped: an error message could quote it."""
        details = grade.effisql
        return cls(
            status=GradeStatus.Name(grade.status),
            matched=details.matched,
            alt_matched=details.alt_matched if details.HasField("alt_matched") else None,
            literal_result=details.literal_result if details.HasField("literal_result") else None,
            speedup=details.speedup if details.HasField("speedup") else None,
            credit=details.credit,
            timing_runs=details.timing_runs,
            eval_wall_s=details.eval_wall_s if details.HasField("eval_wall_s") else None,
            feedback=grade.feedback if keep_feedback else "",
        )

    @property
    def flagged(self) -> bool:
        return self.alt_matched is False or self.literal_result is True


class DataCheck(_Model):
    """The base's row count against the task file's plan (spec §4.1)."""

    outcome: Literal["pass", "fail", "foreign scale", "no actual rows"]
    plan_rows: int | None = None


class SetupRecord(_Model):
    """What the setup found for one task. Gold-free: grades and measurements, never gold SQL."""

    instance_id: str
    base: BaseEntry | None = None
    base_failure: str | None = None
    base_timed_out: bool = False
    base_wall_s: float | None = None
    identity_speedups: tuple[float, ...] = ()  # Every identity τ measured; the entry has the mean.
    data: DataCheck | None = None
    gold: GradeSummary | None = None
    official_gold: GradeSummary | None = None
    official_identity: GradeSummary | None = None
    negatives: dict[str, bool] | None = None  # Variant name: whether it matched the base.
    # Measurements again, on the idle grading instances (the controller's rulings): the server
    # ran out of resources, the base was measured under the old timeout, the gold's dev-mode
    # speedup was marginal (graded again in official mode), or the identity τ was off.
    resource_failures: int = 0
    timeout_rechecked: bool = False
    gold_recheck: GradeSummary | None = None
    identity_recheck: GradeSummary | None = None


def data_check(inst: EffiSqlInstance, row_count: int) -> DataCheck:
    """Compare ``row_count`` with the root Actual Rows of the task file's base plan.

    A plan recorded on another TPC-H scale (T09's foreign_scale_plan) is a dataset anomaly, not
    a data mismatch, and a plan without actual rows has nothing to compare.
    """
    plan = inst.base_explain_analyze
    found = plan.get("Plan", plan) if plan is not None else None
    actual = found.get("Actual Rows") if isinstance(found, dict) else None
    if not isinstance(actual, int | float):
        return DataCheck(outcome="no actual rows")
    if foreign_scale_plan(plan):
        return DataCheck(outcome="foreign scale", plan_rows=int(actual))
    outcome: Literal["pass", "fail"] = "pass" if int(actual) == row_count else "fail"
    return DataCheck(outcome=outcome, plan_rows=int(actual))


def exclusion_reason(record: SetupRecord) -> str | None:
    """Return why the task is excluded (spec §4.1), or None if it is kept.

    The reason never quotes an error message, which could quote the gold. A grader failure
    never excludes a task: it is retried. A gold that a hard-coding check flags is not excluded
    here either: review decides, through the manual exclusion file.
    """
    if record.base_failure is not None:
        return _BASE_TIMEOUT_REASON if record.base_timed_out else "base query fails on this machine"
    if record.base is not None and record.base.base_ms > STATEMENT_TIMEOUT_S * 1000:
        return _BASE_TIMEOUT_REASON
    gold = record.gold
    recheck = record.gold_recheck
    if recheck is not None and recheck.status != _GRADER_ERROR:
        gold = recheck  # The official re-measure of a marginal gold decides (ruling).
    if gold is None or gold.status == _GRADER_ERROR:
        return None
    if not gold.matched:
        if gold.status == _GRADED:
            return "gold optimized_sql does not match base_sql on this machine"
        return f"gold optimized_sql does not match base_sql: {gold.status}"
    if gold.status != _GRADED or gold.speedup is None:
        return f"gold optimized_sql could not be timed: {gold.status}"
    if gold.speedup < GOLD_MIN_SPEEDUP:
        return f"gold optimized_sql is only {gold.speedup:.2f}x faster than base_sql (under 1.2x)"
    return None


def _within(speedup: float | None, tolerance: float) -> bool:
    return speedup is not None and abs(speedup - 1.0) <= tolerance + _TOLERANCE_EPSILON


def _no_grader_error(slot: GradingSlot, grade: Grade) -> Grade:
    """Return ``grade``, or reset the slot and raise if the grader failed: never record it."""
    if grade.status == GRADE_STATUS_GRADER_ERROR:
        slot.reset()
        raise GraderError(grade.feedback)
    return grade


def _resetting[T](slot: GradingSlot, work: Callable[[], T]) -> T:
    """Run ``work``; on any failure, reset the slot, so its next task reconnects, and re-raise."""
    try:
        return work()
    except BaseException:
        slot.reset()
        raise


# The work each phase runs on a slot, in the slot's worker: module-level, so a worker process
# can run it.


def measure_task(
    slot: GradingSlot,
    inst: EffiSqlInstance,
    gold_sql: str | None,
    cache: BaseCache,
    resource_failures: int = 0,
) -> SetupRecord:
    """Phase 1 for one task: the base cache entry, the data check and the gold check.

    ``resource_failures`` counts the earlier attempts the server ran out of resources on.
    """
    return _resetting(slot, lambda: _measure(slot, inst, gold_sql, cache, resource_failures))


def _measure(
    slot: GradingSlot,
    inst: EffiSqlInstance,
    gold_sql: str | None,
    cache: BaseCache,
    resource_failures: int,
) -> SetupRecord:
    sessions = slot.sessions()
    rng = random.Random()  # noqa: S311  # Shuffles timing order; not a secret.
    began = time.monotonic()
    try:
        entry = measure_base(sessions, inst, identity_rng=rng)
        taus = [entry.identity_speedup or 0.0]
        if not _within(taus[0], IDENTITY_TOLERANCE):  # Once more, timing only: short queries
            # are noisy. The entry keeps the mean of both, so the retry does not bias it.
            again = time_interleaved(
                *[lambda: sessions.main.execution_ms(inst.base_sql)] * 2, rng=rng
            )
            taus.append(again.speedup)
            entry = entry.model_copy(update={"identity_speedup": statistics.fmean(taus)})
    except ResourceError as error:  # Recorded, and retried once by the next measure phase.
        return SetupRecord(
            instance_id=inst.instance_id,
            base_failure=str(error),
            resource_failures=resource_failures + 1,
            timeout_rechecked=True,
            base_wall_s=time.monotonic() - began,
        )
    except (QueryError, QueryTimeoutError) as error:
        return SetupRecord(
            instance_id=inst.instance_id,
            base_failure=str(error),
            base_timed_out=isinstance(error, QueryTimeoutError),
            resource_failures=resource_failures,
            timeout_rechecked=True,
            base_wall_s=time.monotonic() - began,
        )
    cache.put(entry)
    gold = None
    if gold_sql is not None:
        graded = evaluate(sessions, inst, gold_sql, Mode.DEV, cache=cache, rng=rng)
        gold = GradeSummary.of(_no_grader_error(slot, graded), keep_feedback=False)
    return SetupRecord(
        instance_id=inst.instance_id,
        base=entry,
        base_wall_s=time.monotonic() - began,
        identity_speedups=tuple(taus),
        data=data_check(inst, entry.row_count),
        gold=gold,
    )


def official_task(
    slot: GradingSlot, inst: EffiSqlInstance, gold_sql: str
) -> tuple[GradeSummary, GradeSummary]:
    """Phase 2 for one val task: the gold and the identity submission in official mode."""

    def work() -> tuple[GradeSummary, GradeSummary]:
        sessions = slot.sessions()
        rng = random.Random()  # noqa: S311  # Shuffles timing order; not a secret.
        gold = evaluate(sessions, inst, gold_sql, Mode.OFFICIAL, cache=None, rng=rng)
        gold_summary = GradeSummary.of(_no_grader_error(slot, gold), keep_feedback=False)
        identity = evaluate(sessions, inst, inst.base_sql, Mode.OFFICIAL, cache=None, rng=rng)
        return gold_summary, GradeSummary.of(_no_grader_error(slot, identity))

    return _resetting(slot, work)


def recheck_task(
    slot: GradingSlot, inst: EffiSqlInstance, gold_sql: str | None, identity: bool
) -> tuple[GradeSummary | None, GradeSummary | None]:
    """The recheck phase for one task, in official mode on an idle grading instance: the gold,
    when its dev-mode speedup was marginal, and the identity submission, when its τ was off."""

    def work() -> tuple[GradeSummary | None, GradeSummary | None]:
        sessions = slot.sessions()
        rng = random.Random()  # noqa: S311  # Shuffles timing order; not a secret.
        gold = None
        if gold_sql is not None:
            graded = evaluate(sessions, inst, gold_sql, Mode.OFFICIAL, cache=None, rng=rng)
            gold = GradeSummary.of(_no_grader_error(slot, graded), keep_feedback=False)
        same = None
        if identity:
            graded = evaluate(sessions, inst, inst.base_sql, Mode.OFFICIAL, cache=None, rng=rng)
            same = GradeSummary.of(_no_grader_error(slot, graded))
        return gold, same

    return _resetting(slot, work)


def negative_variants(base_sql: str, column_types: Sequence[str], row_count: int) -> dict[str, str]:
    """Return SQL variants of ``base_sql`` for the negative checks, plus a ``control``.

    The control renames the base's columns and must match; each other variant must not:
    ``drop`` leaves out a row, ``duplicate`` repeats one, ``reorder`` rotates the columns, and
    ``perturb`` changes one value in one row, a NULL included. A variant that cannot apply is
    left out: reorder needs two columns, drop, duplicate and perturb a row, and perturb a
    number, a date or an unpadded text column.
    """
    names = [f"c{i}" for i in range(1, len(column_types) + 1)]
    columns = ", ".join(names)
    source = f"(\n{statement_body(base_sql)}\n) AS base ({columns})"
    # Only the task's own base_sql and generated names are composed here.
    variants = {"control": f"SELECT {columns} FROM {source}"}  # noqa: S608
    if row_count >= 1:
        variants["drop"] = f"SELECT {columns} FROM {source} LIMIT {row_count - 1}"  # noqa: S608
        variants["duplicate"] = (
            f"SELECT {columns} FROM {source} UNION ALL "  # noqa: S608
            f"(SELECT {columns} FROM {source} LIMIT 1)"
        )
    if len(names) >= 2:
        rotated = ", ".join([*names[1:], names[0]])
        variants["reorder"] = f"SELECT {rotated} FROM {source}"  # noqa: S608
    perturbed = _perturbation(names, column_types)
    if perturbed is not None and row_count >= 1:
        variants["perturb"] = (
            f"SELECT {perturbed} FROM (SELECT *, row_number() OVER () AS rn "  # noqa: S608
            f"FROM {source}) AS numbered"
        )
    return variants


# Each type, with a change that leaves its type and every other row unchanged. A bpchar column
# is never changed: a text result in the CASE would strip every row's padding.
_PERTURBATIONS = (
    (frozenset({"int2", "int4", "int8", "numeric", "float4", "float8"}), "COALESCE({c} + 1, 0)"),
    (frozenset({"date"}), "COALESCE({c} + 1, DATE '1970-01-01')"),
    (frozenset({"text", "varchar"}), "COALESCE({c} || '~', '~')"),
)


def _perturbation(names: Sequence[str], types: Sequence[str]) -> str | None:
    for wanted, change in _PERTURBATIONS:
        for index, kind in enumerate(types):
            if kind in wanted:
                column = names[index]
                edited = f"CASE WHEN rn = 1 THEN {change.format(c=column)} ELSE {column} END"
                return ", ".join(edited if i == index else n for i, n in enumerate(names))
    return None


def negative_task(slot: GradingSlot, inst: EffiSqlInstance, entry: BaseEntry) -> dict[str, bool]:
    """Phase 3 for one task: run each variant and say whether it matched the base."""

    def work() -> dict[str, bool]:
        main = slot.sessions().main
        variants = negative_variants(
            inst.base_sql, main.column_types(inst.base_sql), entry.row_count
        )
        return {
            name: main.digest(query, row_limit=entry.row_count).matches(entry.digest)
            for name, query in variants.items()
        }

    return _resetting(slot, work)


def negative_sample(records: Mapping[str, SetupRecord]) -> list[str]:
    """Return the fixed sample for the negative checks: 10 kept tasks whose results have between
    1 and 10K rows, drawn with the split's seed from every such task."""
    small = sorted(
        (
            task
            for task, record in records.items()
            if record.base is not None
            and 1 <= record.base.row_count < SMALL_RESULT_ROWS
            and exclusion_reason(record) is None
        ),
        key=_numeric,
    )
    rng = random.Random(SPLIT_SEED)  # noqa: S311  # A fixed sample, not a secret.
    return sorted(rng.sample(small, min(NEGATIVE_SAMPLE, len(small))), key=_numeric)


def _numeric(task: str) -> tuple[int, str]:
    return (int(task), task) if task.isdigit() else (1 << 62, task)


class RecordStore:
    """The setup's per-task records, one JSON file each, under ``directory``, mode 0600."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def load(self) -> dict[str, SetupRecord]:
        records = {}
        for path in sorted(self.directory.glob("*.json")):
            try:
                record = SetupRecord.model_validate_json(path.read_bytes())
            except ValidationError:
                _log.warning("ignoring an unreadable setup record", extra={"path": str(path)})
                continue
            records[record.instance_id] = record
        return records

    def save(self, record: SetupRecord) -> None:
        text = json.dumps(record.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        write_private(self.directory / f"{record.instance_id}.json", text)


type _Job = Callable[[SlotRunner, SetupRecord | None], Awaitable[SetupRecord]]


async def run_phase(
    phase: Phase,
    pool: GradingPool[SlotRunner],
    tasks: Sequence[EffiSqlInstance],
    store: RecordStore,
    cache: BaseCache,
    *,
    val: Iterable[str] = (),
    manual: Iterable[str] = (),
) -> int:
    """Run ``phase`` on every task that still needs it, on all the pool's runners at once.

    ``manual`` names the reviewed exclusions, which the official phase and the negative
    sample leave out, as the summary does.

    ``tasks`` carry their gold. Each record is saved as its task finishes, so a rerun resumes;
    a task whose work raises, a grader failure included, is logged and left unrecorded for the
    rerun. The negative sample is drawn only once every task has a record. Return how many
    tasks were recorded.
    """
    records = store.load()
    reviewed = set(manual)
    jobs: list[tuple[EffiSqlInstance, _Job]] = []
    if phase == "measure":
        pending = [t for t in tasks if _needs_measuring(records.get(t.instance_id))]
        pending.sort(key=lambda t: -(t.base_time or 0.0))  # Longest first, to finish together.
        jobs = [(task, _measure_job(task, cache)) for task in pending]
    elif phase == "recheck":
        jobs = [
            (task, _recheck_job(task))
            for task in tasks
            if (record := records.get(task.instance_id)) is not None and _needs_recheck(record)
        ]
    elif phase == "official":
        wanted = set(val)
        jobs = [
            (task, _official_job(task))
            for task in tasks
            if task.instance_id in wanted
            and task.optimized_sql is not None
            and (record := records.get(task.instance_id)) is not None
            and exclusion_reason(record) is None
            and task.instance_id not in reviewed
            and _needs_official(record)
        ]
    else:
        missing = [t.instance_id for t in tasks if t.instance_id not in records]
        if missing:
            _log.warning(
                "the negative sample waits for every task's record: %d missing", len(missing)
            )
            return 0
        sample = set(negative_sample(_without(records, reviewed)))
        jobs = [
            (task, _negative_job(task))
            for task in tasks
            if task.instance_id in sample and records[task.instance_id].negatives is None
        ]
    _log.info("setup phase starts", extra={"phase": phase, "tasks": len(jobs)})

    async def run(task: EffiSqlInstance, job: _Job) -> bool:
        try:
            async with pool.lease(Priority.SETUP) as runner:
                record = await job(runner, records.get(task.instance_id))
        except Exception:  # noqa: BLE001  # The task stays unrecorded; a rerun retries it.
            _log.exception(
                "setup task failed", extra={"phase": phase, "instance_id": task.instance_id}
            )
            return False
        records[task.instance_id] = record
        store.save(record)
        _log.info("setup task done", extra={"phase": phase, "instance_id": task.instance_id})
        return True

    done = await asyncio.gather(*(run(task, job) for task, job in jobs))
    return sum(done)


def _without(records: Mapping[str, SetupRecord], ids: set[str]) -> dict[str, SetupRecord]:
    return {task: record for task, record in records.items() if task not in ids}


def _needs_measuring(record: SetupRecord | None) -> bool:
    """Say whether a task needs (another) measure: never measured, a grader failure, a first
    failure for lack of server resources (retried once), or a base that timed out under the
    old client-side clock (measured again under the server's statement_timeout)."""
    if record is None:
        return True
    if record.base is None:
        if record.base_failure is None:
            return True
        if _resource_failures(record) == 1:
            return True
        return record.base_timed_out and not record.timeout_rechecked
    return record.gold is not None and record.gold.status == _GRADER_ERROR


def _resource_failures(record: SetupRecord) -> int:
    """The attempts the server ran out of resources on; old records only say so in words."""
    if record.resource_failures:
        return record.resource_failures
    failure = (record.base_failure or "").lower()
    return int(any(sign in failure for sign in _RESOURCE_SIGNS))


def _marginal_gold(record: SetupRecord) -> bool:
    gold = record.gold
    return (
        gold is not None
        and gold.matched
        and gold.speedup is not None
        and MARGINAL_GOLD[0] <= gold.speedup < MARGINAL_GOLD[1]
    )


def _off_identity(record: SetupRecord) -> bool:
    base = record.base
    return base is not None and not _within(base.identity_speedup, IDENTITY_TOLERANCE)


def _needs_recheck(record: SetupRecord) -> bool:
    gold = _marginal_gold(record) and record.gold_recheck is None
    identity = _off_identity(record) and record.identity_recheck is None
    return gold or identity


def _needs_official(record: SetupRecord) -> bool:
    grades = (record.official_gold, record.official_identity)
    return any(grade is None or grade.status == _GRADER_ERROR for grade in grades)


def _measure_job(task: EffiSqlInstance, cache: BaseCache) -> _Job:
    async def job(runner: SlotRunner, record: SetupRecord | None) -> SetupRecord:
        failures = 0 if record is None else _resource_failures(record)
        return await runner.call(measure_task, task, task.optimized_sql, cache, failures)

    return job


def _recheck_job(task: EffiSqlInstance) -> _Job:
    async def job(runner: SlotRunner, record: SetupRecord | None) -> SetupRecord:
        if record is None:
            raise ValueError(f"task {task.instance_id} was not measured")
        gold_sql = task.optimized_sql if _marginal_gold(record) else None
        gold, identity = await runner.call(recheck_task, task, gold_sql, _off_identity(record))
        return record.model_copy(
            update={
                "gold_recheck": gold or record.gold_recheck,
                "identity_recheck": identity or record.identity_recheck,
            }
        )

    return job


def _official_job(task: EffiSqlInstance) -> _Job:
    async def job(runner: SlotRunner, record: SetupRecord | None) -> SetupRecord:
        if record is None:
            raise ValueError(f"task {task.instance_id} was not measured")
        gold, identity = await runner.call(official_task, task, task.optimized_sql)
        return record.model_copy(update={"official_gold": gold, "official_identity": identity})

    return job


def _negative_job(task: EffiSqlInstance) -> _Job:
    async def job(runner: SlotRunner, record: SetupRecord | None) -> SetupRecord:
        if record is None or record.base is None:
            raise ValueError(f"task {task.instance_id} has no base cache entry")
        matched = await runner.call(negative_task, task, record.base)
        return record.model_copy(update={"negatives": matched})

    return job


class TimeStats(_Model):
    """How long a kind of work held its instance, over the tasks that did it, in seconds."""

    count: int
    total_s: float
    median_s: float
    p90_s: float  # The nearest-rank 90th percentile.
    max_s: float

    @classmethod
    def of(cls, values: Sequence[float]) -> "TimeStats | None":
        if not values:
            return None
        ordered = sorted(values)
        p90 = ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)]
        return cls(
            count=len(ordered),
            total_s=sum(ordered),
            median_s=statistics.median(ordered),
            p90_s=p90,
            max_s=ordered[-1],
        )


_EXCLUSIONS: TypeAdapter[tuple[Exclusion, ...]] = TypeAdapter(tuple[Exclusion, ...])


def load_manual_exclusions(path: Path = MANUAL_EXCLUSIONS_PATH) -> tuple[Exclusion, ...]:
    """Read the reviewed exclusions, in the exclusion list's format; none if there is no file."""
    return _EXCLUSIONS.validate_json(path.read_bytes()) if path.exists() else ()


class SetupSummary(_Model):
    """The setup's outcome: exclusions, the stop rule, the self-checks and the grading times."""

    task_count: int
    missing: list[str]  # Tasks with no record yet.
    grader_errors: list[str]  # Tasks whose last grade was a grader failure: rerun them.
    exclusions: list[Exclusion]  # Measured ones, then the reviewed ones from the manual file.
    stop: bool  # More than 10% of tasks excluded: stop and investigate before RSI.
    review: list[str]  # Kept golds that a hard-coding check flagged: review decides.
    data: dict[str, list[str]]  # Data check outcome: tasks.
    identity_failures: dict[str, float | None]  # Kept tasks whose identity τ is off.
    identity_rechecks: dict[str, list[float | None]]  # Task: [τ under load, τ idle, official].
    gold_rechecks: dict[str, list[float | None]]  # Task: [dev-mode speedup, official speedup].
    exclusions_by_reason: dict[str, int]
    exclusions_by_split: dict[str, int]
    gold_failures: list[str]  # Kept tasks whose gold check fails (should be none).
    negative_sample: list[str]  # The sample the negative checks must cover.
    negative_tasks: list[str]  # The tasks they did cover.
    negative_failures: dict[str, list[str]]  # Task: variants that matched, or "control".
    agreement_tasks: list[str]  # Kept val tasks graded in both modes.
    agreement_missing: list[str]  # Kept val tasks without both official grades.
    match_disagreements: list[str]
    same_tier_share: float | None
    identity_disagreements: dict[str, list[float | None]]  # Task: [dev τ, official τ].
    scale_invariant_bases: list[str]  # Kept tasks whose base result is the same on both DBs.
    times: dict[str, TimeStats | None]

    @property
    def passed(self) -> bool:
        """Whether every self-check passed, every task was graded, and the stop rule held."""
        return not (
            self.stop
            or self.missing
            or self.grader_errors
            or self.review
            or self.data["fail"]
            or self.identity_failures
            or self.gold_failures
            or self.negative_tasks != self.negative_sample
            or len(self.negative_sample) < NEGATIVE_SAMPLE
            or self.negative_failures
            or self.agreement_missing
            or not self.agreement_tasks
            or self.match_disagreements
            or (self.same_tier_share or 0.0) < AGREEMENT_BOUND
            or self.identity_disagreements
        )


def summarize(
    task_ids: Sequence[str],
    records: Mapping[str, SetupRecord],
    *,
    val: Iterable[str],
    tasks: Mapping[str, EffiSqlInstance] | None = None,
    manual: Sequence[Exclusion] = (),
    identity_tolerance: float = IDENTITY_TOLERANCE,
    splits: Mapping[str, str] | None = None,
) -> SetupSummary:
    """Decide the exclusions and check the setup's results (spec §4.1).

    With ``tasks``, the data check is decided again from each task's plan and measured row
    count, so a record keeps up with the current rule. ``manual`` holds the reviewed exclusions.
    """
    ordered = sorted(task_ids, key=_numeric)
    if tasks is not None:
        records = {
            task: _with_data_check(record, tasks[task]) if task in tasks else record
            for task, record in records.items()
        }
    present = [task for task in ordered if task in records]
    manual_ids = {exclusion.instance_id for exclusion in manual}
    reasons = {task: exclusion_reason(records[task]) for task in present}
    exclusions = [
        Exclusion(instance_id=task, reason=reason)
        for task, reason in reasons.items()
        if reason is not None
    ]
    measured_ids = {exclusion.instance_id for exclusion in exclusions}
    exclusions += [
        e for e in manual if e.instance_id not in measured_ids and e.instance_id in task_ids
    ]
    excluded = {exclusion.instance_id for exclusion in exclusions}
    kept = [task for task in present if task not in excluded]
    data: dict[str, list[str]] = {"pass": [], "fail": [], "foreign scale": [], "no actual rows": []}
    for task in present:
        check = records[task].data
        if check is not None:
            data[check.outcome].append(task)
    sample = negative_sample(_without(records, manual_ids))
    negatives = [task for task in present if records[task].negatives is not None]
    negative_failures = {
        task: [
            name
            for name, matched in (records[task].negatives or {}).items()
            if matched != (name == "control")
        ]
        for task in negatives
    }
    kept_val = [task for task in sorted(set(val), key=_numeric) if task in kept]
    agreement = [task for task in kept_val if not _needs_official(records[task])]
    same_tier = [
        task
        for task in agreement
        if _tier(records[task].gold) == _tier(records[task].official_gold)
    ]
    identity_disagreements: dict[str, list[float | None]] = {}
    for task in agreement:
        base, official = records[task].base, records[task].official_identity
        dev_tau = base.identity_speedup if base is not None else None
        official_tau = official.speedup if official is not None else None
        agree = (
            dev_tau is not None
            and official_tau is not None
            and _within(official_tau / dev_tau if dev_tau else None, identity_tolerance)
        )
        if not (
            agree
            and _within(dev_tau, identity_tolerance)
            and _within(official_tau, identity_tolerance)
        ):
            identity_disagreements[task] = [dev_tau, official_tau]
    return SetupSummary(
        task_count=len(ordered),
        missing=[task for task in ordered if task not in records],
        grader_errors=[task for task in present if _has_grader_error(records[task])],
        exclusions=exclusions,
        stop=len(exclusions) > EXCLUSION_BOUND * len(ordered),
        review=[task for task in kept if (gold := records[task].gold) is not None and gold.flagged],
        data=data,
        identity_failures={
            task: tau
            for task in kept
            if records[task].base is not None
            and not _within(tau := _identity_tau(records[task]), identity_tolerance)
        },
        identity_rechecks={
            task: [
                record.base.identity_speedup if record.base else None,
                record.identity_recheck.speedup,
            ]
            for task in present
            if (record := records[task]).identity_recheck is not None
        },
        gold_rechecks={
            task: [record.gold.speedup if record.gold else None, record.gold_recheck.speedup]
            for task in present
            if (record := records[task]).gold_recheck is not None
        },
        exclusions_by_reason=_count(_reason_kind(e.reason) for e in exclusions),
        exclusions_by_split=_count(
            (splits or {}).get(e.instance_id, "unsplit") for e in exclusions
        ),
        gold_failures=[
            task
            for task in kept
            if (gold := records[task].gold) is None or not gold.matched or gold.flagged
        ],
        negative_sample=sample,
        negative_tasks=negatives,
        negative_failures={task: names for task, names in negative_failures.items() if names},
        agreement_tasks=agreement,
        agreement_missing=[task for task in kept_val if task not in agreement],
        match_disagreements=[
            task
            for task in agreement
            if _matched(records[task].gold) != _matched(records[task].official_gold)
        ],
        same_tier_share=len(same_tier) / len(agreement) if agreement else None,
        identity_disagreements=identity_disagreements,
        scale_invariant_bases=[
            task
            for task in kept
            if (base := records[task].base) is not None
            and (base.row_count, base.row_hash_sum) == (base.alt_row_count, base.alt_row_hash_sum)
        ],
        times={
            "setup measure per task": TimeStats.of(
                [r.base_wall_s for r in records.values() if r.base_wall_s is not None]
            ),
            "dev grading (gold)": TimeStats.of(_walls(r.gold for r in records.values())),
            "official grading (gold)": TimeStats.of(
                _walls(r.official_gold for r in records.values())
            ),
            "official grading (identity)": TimeStats.of(
                _walls(r.official_identity for r in records.values())
            ),
        },
    )


def _identity_tau(record: SetupRecord) -> float | None:
    """The identity τ the self-check reads: the idle re-measure if there is one (ruling)."""
    recheck = record.identity_recheck
    if recheck is not None and recheck.status == _GRADED and recheck.speedup is not None:
        return recheck.speedup
    return record.base.identity_speedup if record.base is not None else None


def _reason_kind(reason: str) -> str:
    if "faster than base_sql" in reason:
        return "gold under 1.2x"
    return reason.split(":")[0]


def _count(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _has_grader_error(record: SetupRecord) -> bool:
    grades = (record.gold, record.official_gold, record.official_identity)
    return any(grade is not None and grade.status == _GRADER_ERROR for grade in grades)


def _with_data_check(record: SetupRecord, inst: EffiSqlInstance) -> SetupRecord:
    if record.base is None:
        return record
    return record.model_copy(update={"data": data_check(inst, record.base.row_count)})


def _tier(grade: GradeSummary | None) -> float:
    return grade.credit if grade is not None else -1.0


def _matched(grade: GradeSummary | None) -> bool:
    return grade is not None and grade.matched


def _walls(grades: Iterable[GradeSummary | None]) -> list[float]:
    return [g.eval_wall_s for g in grades if g is not None and g.eval_wall_s is not None]


def to_markdown(summary: SetupSummary) -> str:
    """Render the summary for the report."""

    def listed(values: Sequence[str] | Mapping[str, object]) -> str:
        return str(values) if values else "none"

    excluded = len(summary.exclusions)
    lines = [
        f"- Tasks: {summary.task_count}; missing records: {listed(summary.missing)}; "
        f"grader errors to rerun: {listed(summary.grader_errors)}",
        f"- Excluded: {excluded} ({excluded / max(summary.task_count, 1):.1%}); "
        f"stop rule {'TRIGGERED' if summary.stop else 'not triggered'}",
        f"- Gold flagged by a check (review): {listed(summary.review)}",
        "- Data check: "
        + ", ".join(f"{outcome} {len(tasks)}" for outcome, tasks in summary.data.items())
        + (f"; failures {summary.data['fail']}" if summary.data["fail"] else ""),
        f"- Exclusions by reason {summary.exclusions_by_reason}, by split "
        f"{summary.exclusions_by_split}",
        f"- Marginal golds graded again in official mode [dev, official]: "
        f"{listed(summary.gold_rechecks)}",
        f"- Identity τ outside the tolerance (kept tasks, the idle re-measure when there is "
        f"one): {listed(summary.identity_failures)}; re-measured idle [under load, idle]: "
        f"{listed(summary.identity_rechecks)}",
        f"- Gold check failures (kept tasks): {listed(summary.gold_failures)}",
        f"- Negative checks: sample {summary.negative_sample}, run on {summary.negative_tasks}; "
        f"failures {listed(summary.negative_failures)}",
        f"- Dev vs official on {len(summary.agreement_tasks)} kept val golds (missing "
        f"{listed(summary.agreement_missing)}): match disagreements "
        f"{listed(summary.match_disagreements)}; same tier "
        + (f"{summary.same_tier_share:.0%}" if summary.same_tier_share is not None else "n/a")
        + "; identity τ off in either mode or between them "
        + listed(summary.identity_disagreements),
        f"- Kept tasks whose base result is the same on both databases (the robustness check "
        f"cannot catch constants there): {listed(summary.scale_invariant_bases)}",
    ]
    for name, stats in summary.times.items():
        if stats is not None:
            lines.append(
                f"- {name}: {stats.count} tasks, total {stats.total_s / 3600:.2f} h, median "
                f"{stats.median_s:.1f} s, p90 {stats.p90_s:.1f} s, max {stats.max_s:.1f} s"
            )
    lines.append(f"- All self-checks passed: {summary.passed}")
    return "\n".join(lines) + "\n"


def dump_exclusions(exclusions: Sequence[Exclusion]) -> str:
    """Return the exclusion list as the committed file holds it."""
    return json.dumps([e.model_dump() for e in exclusions], indent=2) + "\n"
