"""The grader's one-time setup run (spec §4.1, §6.2): the base cache, the self-checks and the
exclusions, measured on this machine before any solving starts.

Three phases, each resumable from the per-task records it writes (gold-free JSON, outside the
repository), then a summary:

1. **measure**, on all 5 instances, so the identity τ is measured under load. For each TPC-H
   task: the base cache entry (signatures on both databases, the base plan's relations, its
   median time, and the identity τ from base against base, interleaved), the data check, and
   the gold check: the gold graded in dev mode against that entry.
2. **official**, on the 3 grading instances with the solving instances idle: the val golds and
   the identity submissions graded in official mode, for the dev-versus-official agreement.
3. **negatives**, on a fixed sample of 10 kept tasks with small results: variants of base_sql
   that drop, duplicate, reorder or perturb rows must not match; a renamed copy must.

:func:`summarize` decides the exclusions (:func:`exclusion_reason`), applies the 10% stop rule,
and reports the self-checks and the measured grading times.
"""

import asyncio
import json
import logging
import random
import statistics
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from psycopg import sql as pgsql
from pydantic import BaseModel, ConfigDict, ValidationError

from goaly.bench.effisql.data import EffiSqlInstance
from goaly.bench.effisql.grader import QueryError, QueryTimeoutError, statement_body
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
from goaly.bench.effisql.metrics import tier_credit
from goaly.bench.effisql.pg_snapshot import foreign_scale_plan, write_json
from goaly.bench.effisql.splits import SPLIT_SEED, Exclusion
from goaly.contracts.gen.task_pb2 import GRADE_STATUS_GRADED, Grade, GradeStatus

__all__ = [
    "AGREEMENT_BOUND",
    "EXCLUSION_BOUND",
    "GOLD_MIN_SPEEDUP",
    "IDENTITY_TOLERANCE",
    "NEGATIVE_SAMPLE",
    "SMALL_RESULT_ROWS",
    "DataCheck",
    "GradeSummary",
    "Phase",
    "RecordStore",
    "SetupRecord",
    "SetupSummary",
    "data_check",
    "dump_exclusions",
    "exclusion_reason",
    "measure_task",
    "negative_sample",
    "negative_task",
    "negative_variants",
    "official_credit",
    "official_task",
    "run_phase",
    "summarize",
    "to_markdown",
]

IDENTITY_TOLERANCE = 0.10  # Identity τ must be within 1 ± this (spec §4.1's config value).
GOLD_MIN_SPEEDUP = 1.2  # A gold below this speedup is excluded.
SMALL_RESULT_ROWS = 10_000  # Negative checks run on tasks with fewer base rows than this.
NEGATIVE_SAMPLE = 10
EXCLUSION_BOUND = 0.10  # More than this share of tasks excluded stops the setup.
AGREEMENT_BOUND = 0.90  # Dev and official mode give the same tier on at least this share.
_STATEMENT_TIMEOUT_MS = 300_000.0

_log = logging.getLogger(__name__)

type Phase = Literal["measure", "official", "negatives"]


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
    def of(cls, grade: Grade) -> "GradeSummary":
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
            feedback=grade.feedback,
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
    data: DataCheck | None = None
    gold: GradeSummary | None = None
    official_gold: GradeSummary | None = None
    official_identity: GradeSummary | None = None
    negatives: dict[str, bool] | None = None  # Variant name: whether it matched the base.


def data_check(inst: EffiSqlInstance, row_count: int) -> DataCheck:
    """Compare ``row_count`` with the root Actual Rows of the task file's base plan.

    A plan recorded on another TPC-H scale is a dataset anomaly, not a data mismatch, and a plan
    without actual rows has nothing to compare.
    """
    plan = inst.base_explain_analyze
    root = plan.get("Plan", plan) if plan is not None else None
    actual = root.get("Actual Rows") if isinstance(root, dict) else None
    if not isinstance(actual, int | float):
        return DataCheck(outcome="no actual rows")
    if foreign_scale_plan(plan):
        return DataCheck(outcome="foreign scale", plan_rows=int(actual))
    outcome: Literal["pass", "fail"] = "pass" if int(actual) == row_count else "fail"
    return DataCheck(outcome=outcome, plan_rows=int(actual))


def exclusion_reason(record: SetupRecord) -> str | None:
    """Return why the task is excluded (spec §4.1), or None if it is kept.

    The reason never quotes an error message, which could quote the gold. A gold that a
    hard-coding check flags is not excluded here: review decides whether the gold or the check
    is at fault (SetupSummary.review).
    """
    if record.base_failure is not None:
        if record.base_timed_out:
            return "base query exceeds the 300 s statement timeout on this machine"
        return "base query fails on this machine"
    if record.base is not None and record.base.base_ms > _STATEMENT_TIMEOUT_MS:
        return "base query exceeds the 300 s statement timeout on this machine"
    gold = record.gold
    if gold is None:
        return None
    if not gold.matched:
        if gold.status == GradeStatus.Name(GRADE_STATUS_GRADED):
            return "gold optimized_sql does not match base_sql on this machine"
        return f"gold optimized_sql does not match base_sql: {gold.status}"
    if gold.speedup is None:
        return f"gold optimized_sql could not be timed: {gold.status}"
    if gold.speedup < GOLD_MIN_SPEEDUP:
        return f"gold optimized_sql is only {gold.speedup:.2f}x faster than base_sql (under 1.2x)"
    return None


def _identity_ok(speedup: float | None) -> bool:
    return speedup is not None and abs(speedup - 1.0) <= IDENTITY_TOLERANCE


# The work each phase runs on a slot, in the slot's worker: module-level, so a worker process
# can run it.


def measure_task(
    slot: GradingSlot, inst: EffiSqlInstance, gold_sql: str | None, cache: BaseCache
) -> SetupRecord:
    """Phase 1 for one task: the base cache entry, the data check and the gold check."""
    sessions = slot.sessions()
    rng = random.Random()  # noqa: S311  # Shuffles timing order; not a secret.
    began = time.monotonic()
    try:
        entry = measure_base(sessions, inst, identity_rng=rng)
        if not _identity_ok(entry.identity_speedup):  # Once more: short queries are noisy.
            entry = measure_base(sessions, inst, identity_rng=rng)
    except (QueryError, QueryTimeoutError) as error:
        return SetupRecord(
            instance_id=inst.instance_id,
            base_failure=str(error),
            base_timed_out=isinstance(error, QueryTimeoutError),
            base_wall_s=time.monotonic() - began,
        )
    cache.put(entry)
    gold = None
    if gold_sql is not None:
        graded = evaluate(sessions, inst, gold_sql, Mode.DEV, cache=cache, rng=rng)
        gold = GradeSummary.of(graded)
    return SetupRecord(
        instance_id=inst.instance_id,
        base=entry,
        base_wall_s=time.monotonic() - began,
        data=data_check(inst, entry.row_count),
        gold=gold,
    )


def official_task(
    slot: GradingSlot, inst: EffiSqlInstance, gold_sql: str
) -> tuple[GradeSummary, GradeSummary]:
    """Phase 2 for one val task: the gold and the identity submission in official mode."""
    sessions = slot.sessions()
    rng = random.Random()  # noqa: S311  # Shuffles timing order; not a secret.
    gold = evaluate(sessions, inst, gold_sql, Mode.OFFICIAL, cache=None, rng=rng)
    identity = evaluate(sessions, inst, inst.base_sql, Mode.OFFICIAL, cache=None, rng=rng)
    return GradeSummary.of(gold), GradeSummary.of(identity)


def negative_variants(base_sql: str, column_types: Sequence[str], row_count: int) -> dict[str, str]:
    """Return SQL variants of ``base_sql`` for the negative checks, plus a ``control``.

    The control renames the base's columns and must match; each other variant must not:
    ``drop`` leaves out a row, ``duplicate`` repeats one, ``reorder`` rotates the columns, and
    ``perturb`` changes one value in one row. A variant that cannot apply is left out: reorder
    needs two columns, drop, duplicate and perturb a row, and perturb a number, date or text.
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


_NUMERIC_TYPES = frozenset({"int2", "int4", "int8", "numeric", "float4", "float8", "date"})
_TEXT_TYPES = frozenset({"text", "varchar", "bpchar", "name"})


def _perturbation(names: Sequence[str], types: Sequence[str]) -> str | None:
    for wanted, change in ((_NUMERIC_TYPES, "{c} + 1"), (_TEXT_TYPES, "{c} || '~'")):
        for index, kind in enumerate(types):
            if kind in wanted:
                column = names[index]
                edited = f"CASE WHEN rn = 1 THEN {change.format(c=column)} ELSE {column} END"
                return ", ".join(edited if i == index else n for i, n in enumerate(names))
    return None


def negative_task(slot: GradingSlot, inst: EffiSqlInstance, entry: BaseEntry) -> dict[str, bool]:
    """Phase 3 for one task: run each variant and say whether it matched the base."""
    sessions = slot.sessions()
    conn = sessions.main.connection
    probe = pgsql.SQL("SELECT * FROM ({}) AS base LIMIT 0").format(
        pgsql.SQL(statement_body(inst.base_sql))  # The task's own SQL.
    )
    with conn.transaction(force_rollback=True):
        description = conn.execute(probe).description or []
    types = []
    for column in description:
        info = conn.adapters.types.get(column.type_code)
        types.append(info.name if info is not None else "unknown")
    variants = negative_variants(inst.base_sql, types, entry.row_count)
    return {
        name: sessions.main.digest(query, row_limit=entry.row_count).matches(entry.digest)
        for name, query in variants.items()
    }


def negative_sample(records: Mapping[str, SetupRecord]) -> list[str]:
    """Return the fixed sample for the negative checks: 10 kept tasks with small results."""
    small = sorted(
        (
            task
            for task, record in records.items()
            if record.base is not None
            and record.base.row_count < SMALL_RESULT_ROWS
            and exclusion_reason(record) is None
        ),
        key=_numeric,
    )
    rng = random.Random(SPLIT_SEED)  # noqa: S311  # A fixed sample, not a secret.
    return sorted(rng.sample(small, min(NEGATIVE_SAMPLE, len(small))), key=_numeric)


def _numeric(task: str) -> tuple[int, str]:
    return (int(task), task) if task.isdigit() else (1 << 62, task)


class RecordStore:
    """The setup's per-task records, one JSON file each, under ``directory``."""

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
        path = self.directory / f"{record.instance_id}.json"
        write_json(path, record.model_dump(mode="json"))


type _Job = Callable[[SlotRunner, SetupRecord | None], Awaitable[SetupRecord]]


async def run_phase(
    phase: Phase,
    pool: GradingPool[SlotRunner],
    tasks: Sequence[EffiSqlInstance],
    store: RecordStore,
    cache: BaseCache,
    *,
    val: Iterable[str] = (),
) -> int:
    """Run ``phase`` on every task that still needs it, on all the pool's runners at once.

    ``tasks`` carry their gold. Each record is saved as its task finishes, so a rerun resumes;
    a task whose work raises is logged and left unrecorded for the rerun. Return how many tasks
    were recorded.
    """
    records = store.load()
    jobs: list[tuple[EffiSqlInstance, _Job]] = []
    if phase == "measure":
        pending = [t for t in tasks if _needs_measuring(records.get(t.instance_id))]
        pending.sort(key=lambda t: -(t.base_time or 0.0))  # Longest first, to finish together.
        jobs = [(task, _measure_job(task, cache)) for task in pending]
    elif phase == "official":
        wanted = set(val)
        jobs = [
            (task, _official_job(task))
            for task in tasks
            if task.instance_id in wanted
            and task.optimized_sql is not None
            and (record := records.get(task.instance_id)) is not None
            and record.official_gold is None
        ]
    else:
        sample = set(negative_sample(records))
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
            _log.exception("setup task failed", extra={"phase": phase, "id": task.instance_id})
            return False
        records[task.instance_id] = record
        store.save(record)
        _log.info("setup task done", extra={"phase": phase, "instance_id": task.instance_id})
        return True

    done = await asyncio.gather(*(run(task, job) for task, job in jobs))
    return sum(done)


def _needs_measuring(record: SetupRecord | None) -> bool:
    return record is None or (record.base is None and record.base_failure is None)


def _measure_job(task: EffiSqlInstance, cache: BaseCache) -> _Job:
    async def job(runner: SlotRunner, _: SetupRecord | None) -> SetupRecord:
        return await runner.call(measure_task, task, task.optimized_sql, cache)

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
    count: int
    total_s: float
    median_s: float
    p90_s: float
    max_s: float

    @classmethod
    def of(cls, values: Sequence[float]) -> "TimeStats | None":
        if not values:
            return None
        ordered = sorted(values)
        p90 = ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]
        return cls(
            count=len(ordered),
            total_s=sum(ordered),
            median_s=statistics.median(ordered),
            p90_s=p90,
            max_s=ordered[-1],
        )


class SetupSummary(_Model):
    """The setup's outcome: exclusions, the stop rule, the self-checks and the grading times."""

    task_count: int
    missing: list[str]  # Tasks with no record yet.
    exclusions: list[Exclusion]
    stop: bool  # More than 10% of tasks excluded: stop and investigate before RSI.
    review: list[str]  # Golds that a hard-coding check flagged: review decides.
    data: dict[str, list[str]]  # Data check outcome: tasks.
    identity_failures: dict[str, float | None]  # Kept tasks whose identity τ is off.
    gold_failures: list[str]  # Kept tasks whose gold check fails (should be none).
    negative_tasks: list[str]
    negative_failures: dict[str, list[str]]  # Task: variants that matched, or "control".
    agreement_tasks: list[str]
    match_disagreements: list[str]
    same_tier_share: float | None
    identity_disagreements: dict[str, list[float | None]]  # Task: [dev τ, official τ].
    times: dict[str, TimeStats | None]

    @property
    def passed(self) -> bool:
        """Whether every self-check passed and the stop rule did not trigger."""
        return not (
            self.stop
            or self.missing
            or self.data["fail"]
            or self.identity_failures
            or self.gold_failures
            or len(self.negative_tasks) < NEGATIVE_SAMPLE
            or self.negative_failures
            or self.match_disagreements
            or (self.same_tier_share is not None and self.same_tier_share < AGREEMENT_BOUND)
            or self.identity_disagreements
        )


def summarize(
    task_ids: Sequence[str], records: Mapping[str, SetupRecord], *, val: Iterable[str]
) -> SetupSummary:
    """Decide the exclusions and check the setup's results (spec §4.1)."""
    ordered = sorted(task_ids, key=_numeric)
    present = [task for task in ordered if task in records]
    reasons = {task: exclusion_reason(records[task]) for task in present}
    exclusions = [
        Exclusion(instance_id=task, reason=reason)
        for task, reason in reasons.items()
        if reason is not None
    ]
    kept = [task for task in present if reasons[task] is None]
    data: dict[str, list[str]] = {"pass": [], "fail": [], "foreign scale": [], "no actual rows": []}
    for task in present:
        check = records[task].data
        if check is not None:
            data[check.outcome].append(task)
    negatives = [task for task in present if records[task].negatives is not None]
    negative_failures = {
        task: [
            name
            for name, matched in (records[task].negatives or {}).items()
            if matched != (name == "control")
        ]
        for task in negatives
    }
    agreement = [
        task
        for task in sorted(set(val), key=_numeric)
        if task in records and task in kept and records[task].official_gold is not None
    ]
    same_tier = [
        task
        for task in agreement
        if _tier(records[task].gold) == _tier(records[task].official_gold)
    ]
    identity_disagreements: dict[str, list[float | None]] = {}
    for task in agreement:
        base = records[task].base
        official = records[task].official_identity
        dev_tau = base.identity_speedup if base is not None else None
        official_tau = official.speedup if official is not None else None
        if not (_identity_ok(dev_tau) and _identity_ok(official_tau)):
            identity_disagreements[task] = [dev_tau, official_tau]
    return SetupSummary(
        task_count=len(ordered),
        missing=[task for task in ordered if task not in records],
        exclusions=exclusions,
        stop=len(exclusions) > EXCLUSION_BOUND * len(ordered),
        review=[task for task in kept if (gold := records[task].gold) is not None and gold.flagged],
        data=data,
        identity_failures={
            task: base.identity_speedup
            for task in kept
            if (base := records[task].base) is not None and not _identity_ok(base.identity_speedup)
        },
        gold_failures=[
            task
            for task in kept
            if (gold := records[task].gold) is None or not gold.matched or gold.flagged
        ],
        negative_tasks=negatives,
        negative_failures={task: names for task, names in negative_failures.items() if names},
        agreement_tasks=agreement,
        match_disagreements=[
            task
            for task in agreement
            if _matched(records[task].gold) != _matched(records[task].official_gold)
        ],
        same_tier_share=len(same_tier) / len(agreement) if agreement else None,
        identity_disagreements=identity_disagreements,
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


def _tier(grade: GradeSummary | None) -> float:
    return grade.credit if grade is not None else -1.0


def _matched(grade: GradeSummary | None) -> bool:
    return grade is not None and grade.matched


def _walls(grades: Iterable[GradeSummary | None]) -> list[float]:
    return [g.eval_wall_s for g in grades if g is not None and g.eval_wall_s is not None]


def official_credit(grade: GradeSummary) -> float:
    """g(τ) for a matched grade, ignoring the flags, as the official evaluator scores it."""
    return tier_credit(grade.speedup or 0.0) if grade.matched else 0.0


def to_markdown(summary: SetupSummary) -> str:
    """Render the summary for the report."""
    lines = [
        f"- Tasks: {summary.task_count}; missing records: {len(summary.missing)}",
        f"- Excluded: {len(summary.exclusions)} "
        f"({len(summary.exclusions) / max(summary.task_count, 1):.1%}); "
        f"stop rule {'TRIGGERED' if summary.stop else 'not triggered'}",
        f"- Gold flagged by a check (review): {summary.review or 'none'}",
        "- Data check: "
        + ", ".join(f"{outcome} {len(tasks)}" for outcome, tasks in summary.data.items())
        + (f"; failures {summary.data['fail']}" if summary.data["fail"] else ""),
        f"- Identity τ outside ±{IDENTITY_TOLERANCE:.0%} (kept tasks): "
        f"{summary.identity_failures or 'none'}",
        f"- Gold check failures (kept tasks): {summary.gold_failures or 'none'}",
        f"- Negative checks on {summary.negative_tasks}: failures "
        f"{summary.negative_failures or 'none'}",
        f"- Dev vs official on {len(summary.agreement_tasks)} val golds: match disagreements "
        f"{summary.match_disagreements or 'none'}; same tier "
        + (f"{summary.same_tier_share:.0%}" if summary.same_tier_share is not None else "n/a")
        + f"; identity τ outside ±10% in either mode {summary.identity_disagreements or 'none'}",
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
    entries: list[dict[str, Any]] = [e.model_dump() for e in exclusions]
    return json.dumps(entries, indent=2) + "\n"
