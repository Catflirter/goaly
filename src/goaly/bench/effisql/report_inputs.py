"""What the final report reads (spec §6.1, §8.6, §10): run directories and the RSI ledger.

A run directory, as ``goaly run`` writes it on the host, holds:

- ``manifest.json``, the RunManifest: the run's tasks, seeds, code and data;
- ``results.jsonl``, one TaskResult per line, where the last line for a task and seed wins, so the
  grade phase's graded copy wins over the solve phase's ungraded line;
- ``official/seed-<k>.jsonl``, the host-written official records, and ``plans/seed-<k>.jsonl``,
  the plans the grade phase stored, which the constant-branch audit reads
  (goaly.bench.effisql.audit);
- ``llm_calls.jsonl``, the host-side proxy's LlmCallRecords: the only tokens and cost the report
  trusts, since files the solve container wrote are informational (ruling V27).

:func:`load_run` checks that every result belongs to the manifest's run and that every task and
seed the manifest names is graded. With ``allow_partial``, for drafting the report while a run
is still grading, it keeps only the tasks graded in every seed and says how many it left out.

The ledger directory holds ``experiments.jsonl`` and ``lessons.jsonl``; the last line for an ID
wins in each, and a deleted lesson is dropped. The ``stages.jsonl`` that
scripts/effisql/final_runs.py writes next to the run directories gives the host's clock around
each solve segment, which the efficiency cap uses (spec §8.3).
"""

import json
import math
import statistics
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from google.protobuf import json_format
from google.protobuf.message import Message

from goaly.bench.effisql.audit import prediction_sql
from goaly.bench.effisql.data import load_instances
from goaly.bench.effisql.metrics import GradeTable
from goaly.contracts.gen.experiment_pb2 import ExperimentRecord, Lesson
from goaly.contracts.gen.llm_call_pb2 import LlmCallRecord
from goaly.contracts.gen.manifest_pb2 import RunManifest
from goaly.contracts.gen.result_pb2 import TaskResult
from goaly.contracts.gen.task_pb2 import (
    GRADE_STATUS_GRADED,
    GRADE_STATUS_PREDICTION_FAILED,
    GRADE_STATUS_TIMEOUT,
)
from goaly.contracts.jsonl import read_jsonl

__all__ = [
    "DEFAULT_MAX_TASKS",
    "Official",
    "RunDirError",
    "RunRecord",
    "RunUsage",
    "Verdict",
    "VerdictKey",
    "load_difficulty",
    "load_experiments",
    "load_lessons",
    "load_run",
    "load_verdicts",
]

# goaly run's default task concurrency, when the command line sets none: spec §8.3 scales a
# run's wall time by it. It copies goaly.harness.core.runner.DEFAULT_MAX_TASKS, which is not on
# main yet; import it from there once T06 lands.
DEFAULT_MAX_TASKS = 16
# The statuses metrics can score; any other, such as GRADER_ERROR, is a grade still to come.
_SCORED = frozenset({GRADE_STATUS_GRADED, GRADE_STATUS_PREDICTION_FAILED, GRADE_STATUS_TIMEOUT})


class RunDirError(ValueError):
    """A run directory cannot be reported: it is missing, foreign or not fully graded."""


@dataclass(frozen=True, slots=True)
class RunUsage:
    """A run's LLM calls, from the host's call log."""

    calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float  # Over the calls the gateway priced.
    unpriced_calls: int  # Calls with no cost, such as a refusal or a failed call.


@dataclass(frozen=True, slots=True)
class Official:
    """One prediction's official record, as the host wrote it: its query and its task's base."""

    sql: str
    base_sql: str
    base_plan: Any = None  # The record's base_explain_analyze, if it has one.


@dataclass(frozen=True, slots=True)
class Verdict:
    """A person's verdict on a prediction the constant-branch audit listed."""

    constant_branch: bool  # True: hard-coded, so every checked number gives it no credit.
    note: str = ""


# A verdict's key: the run's role, the task and the seed.
type VerdictKey = tuple[str, str, int]
_VERDICTS = {"constant branch": True, "clean": False}


@dataclass(frozen=True, slots=True)
class RunRecord:
    """One run, as the report uses it."""

    path: Path
    manifest: RunManifest
    table: GradeTable | None  # None only if allow_partial found no task graded in every seed.
    # The (task, seed) pairs not graded yet; the table leaves out every seed of their tasks.
    missing: tuple[tuple[str, int], ...]
    usage: RunUsage | None  # None if the run has no host call log.
    # Each (task, seed)'s official record, and the seeds whose official file is missing.
    queries: Mapping[tuple[str, int], Official] = field(default_factory=dict)
    unread_seeds: tuple[int, ...] = ()
    # Each (task, seed)'s plan, from the host-written plans/seed-<k>.jsonl, where present.
    plans: Mapping[tuple[str, int], Any] = field(default_factory=dict)
    # The host clock around the run's solve phase, summed over its segments (stages.jsonl).
    solve_wall_s: float | None = None
    # The mean host-attributed wall_time_s of the graded results, the fallback for the cap.
    result_wall_s: float | None = None

    @property
    def run_id(self) -> str:
        return self.manifest.run_id

    @property
    def grading_mode(self) -> str:
        return self.manifest.environment_settings.get("grading_mode", "unknown")

    @property
    def task_seeds(self) -> int:
        """How many (task, seed) pairs the manifest names."""
        return len(self.manifest.instance_ids) * len(self.manifest.seeds)

    @property
    def run_span_s(self) -> float | None:
        """From the manifest's start to its finish: solving and grading."""
        manifest = self.manifest
        if not (manifest.HasField("started_at") and manifest.HasField("finished_at")):
            return None
        span = manifest.finished_at.ToDatetime() - manifest.started_at.ToDatetime()
        return span.total_seconds()

    @property
    def max_tasks(self) -> int:
        """The run's task concurrency, from its command line."""
        args = list(self.manifest.command_line)
        for index, arg in enumerate(args):
            if arg == "--max-tasks" and index + 1 < len(args):
                return int(args[index + 1])
            if arg.startswith("--max-tasks="):
                return int(arg.split("=", 1)[1])
        return DEFAULT_MAX_TASKS

    def wall_per_task_s(self) -> float | None:
        """Spec §8.3's wall time per task, from :attr:`wall_source`.

        The host clock around the solve phase, summed over its resumed segments so the downtime
        between them is left out, spread over the tasks and times the run's task concurrency,
        capped at its number of tasks. Without stages.jsonl, as for runs archived before it
        existed, the mean host-attributed wall_time_s of the results, as T21's cap check takes.
        """
        if self.solve_wall_s is not None and self.task_seeds:
            width = min(self.max_tasks, self.task_seeds)
            return self.solve_wall_s * width / self.task_seeds
        return self.result_wall_s

    @property
    def wall_source(self) -> str:
        if self.solve_wall_s is not None:
            return "solve phase"
        return "results" if self.result_wall_s is not None else "none"

    def cost_per_task_usd(self) -> float | None:
        if self.usage is None or not self.task_seeds:
            return None
        return self.usage.cost_usd / self.task_seeds


def load_run(path: Path, *, allow_partial: bool = False) -> RunRecord:
    """Read the run directory at ``path``; raise RunDirError if it cannot be reported."""
    manifest_file = path / "manifest.json"
    if not manifest_file.is_file():
        raise RunDirError(f"{path} has no manifest.json: it is not a run directory")
    try:
        manifest = json_format.Parse(
            manifest_file.read_text(encoding="utf-8"), RunManifest(), ignore_unknown_fields=True
        )
    except json_format.ParseError as error:
        raise RunDirError(f"{manifest_file} is not a valid manifest: {error}") from error
    expected = {(task, seed) for task in manifest.instance_ids for seed in manifest.seeds}
    if not expected:
        raise RunDirError(f"{manifest_file} names no tasks or no seeds")
    latest = _latest_results(path / "results.jsonl", manifest.run_id, expected)
    graded = {key: result for key, result in latest.items() if _is_graded(result)}
    missing = tuple(sorted(expected - graded.keys(), key=lambda key: (_id_order(key[0]), key[1])))
    if missing and not allow_partial:
        raise RunDirError(
            f"{path}: {len(missing)} of {len(expected)} tasks and seeds are not graded yet,"
            f" such as task {missing[0][0]!r} seed {missing[0][1]}"
        )
    incomplete = {task for task, _ in missing}
    kept = [result for (task, _), result in graded.items() if task not in incomplete]
    table = GradeTable.from_results(kept) if kept else None
    queries, unread = _queries(path, manifest.seeds)
    walls = [result.wall_time_s for result in graded.values()]
    return RunRecord(
        path,
        manifest,
        table,
        missing,
        _usage(path / "llm_calls.jsonl"),
        queries=queries,
        unread_seeds=unread,
        plans=_plans(path, manifest.seeds),
        solve_wall_s=_solve_wall(path),
        result_wall_s=statistics.fmean(walls) if walls else None,
    )


def load_difficulty(task_file: Path) -> dict[str, str]:
    """Each task's difficulty label, from a task file in the dataset's format."""
    return {inst.instance_id: inst.difficulty or "unlabelled" for inst in load_instances(task_file)}


def load_verdicts(path: Path) -> dict[VerdictKey, Verdict]:
    """The audit's verdicts: a JSON list of objects with role, task, seed, verdict and note.

    A verdict is "constant branch" or "clean". Raises RunDirError for anything else.
    """
    try:
        entries = [_verdict(entry) for entry in json.loads(path.read_text(encoding="utf-8"))]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        choices = " or ".join(map(repr, _VERDICTS))
        raise RunDirError(
            f"{path}: not a list of {{role, task, seed, verdict: {choices}, note}}: {error!r}"
        ) from error
    verdicts: dict[VerdictKey, Verdict] = {}
    for key, verdict in entries:
        if verdicts.setdefault(key, verdict) is not verdict:
            raise RunDirError(f"{path}: two verdicts on {key}")
    return verdicts


def _verdict(entry: Any) -> tuple[VerdictKey, Verdict]:
    key = (str(entry["role"]), str(entry["task"]), int(entry["seed"]))
    return key, Verdict(_VERDICTS[entry["verdict"]], str(entry.get("note", "")))


def load_experiments(ledger_dir: Path) -> list[ExperimentRecord]:
    """The ledger's experiments, the last line for each ID, in order of first appearance."""
    return _last_by_id(ledger_dir / "experiments.jsonl", ExperimentRecord, "experiment_id")


def load_lessons(ledger_dir: Path) -> list[Lesson]:
    """The ledger's standing lessons: the last line for each ID, leaving out deleted ones."""
    lessons = _last_by_id(ledger_dir / "lessons.jsonl", Lesson, "lesson_id")
    return [lesson for lesson in lessons if not lesson.HasField("deleted_at")]


def _latest_results(
    path: Path, run_id: str, expected: set[tuple[str, int]]
) -> dict[tuple[str, int], TaskResult]:
    latest: dict[tuple[str, int], TaskResult] = {}
    if not path.exists():
        return latest
    for result in read_jsonl(path, TaskResult):
        key = (result.instance_id, result.seed)
        if result.run_id != run_id:
            raise RunDirError(f"{path} holds a result of run {result.run_id!r}, not {run_id!r}")
        if key not in expected:
            raise RunDirError(f"{path} holds task {key[0]!r} seed {key[1]}, which the run lacks")
        latest[key] = result
    return latest


def _is_graded(result: TaskResult) -> bool:
    return result.HasField("grade") and result.grade.status in _SCORED


def _queries(
    path: Path, seeds: Iterable[int]
) -> tuple[dict[tuple[str, int], Official], tuple[int, ...]]:
    """Each prediction's official record, from ``official/seed-<k>.jsonl``, and the seeds that
    have no such file."""
    queries, unread = {}, []
    for seed in seeds:
        records = _json_lines(path / "official" / f"seed-{seed}.jsonl")
        if records is None:
            unread.append(seed)
            continue
        for record in records:
            sql = prediction_sql(str(record.get("prediction", "")))
            base = record.get("base_explain_analyze")
            official = Official(sql, str(record.get("base_sql", "")), base)
            queries[str(record["id"]), seed] = official
    return queries, tuple(unread)


def _plans(path: Path, seeds: Iterable[int]) -> dict[tuple[str, int], Any]:
    """Each prediction's plan, from the host-written ``plans/seed-<k>.jsonl``, where present."""
    plans = {}
    for seed in seeds:
        for record in _json_lines(path / "plans" / f"seed-{seed}.jsonl") or []:
            plans[str(record["instance_id"]), seed] = record["plan"]
    return plans


def _solve_wall(path: Path) -> float | None:
    """The host clock around the run's solve segments, from its root's stages.jsonl."""
    records = _json_lines(path.parent / "stages.jsonl")
    durations = [
        float(record["finished_at"]) - float(record["started_at"])
        for record in records or []
        if record.get("phase") == "solve" and _out(record.get("argv", [])) == path.resolve()
    ]
    return math.fsum(durations) if durations else None


def _out(argv: list[str]) -> Path | None:
    return Path(argv[argv.index("--out") + 1]).resolve() if "--out" in argv[:-1] else None


def _json_lines(path: Path) -> list[dict[str, Any]] | None:
    """The JSON objects in a JSONL file, or None if it does not exist."""
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as file:
        try:
            return [json.loads(line) for line in file if line.strip()]
        except json.JSONDecodeError as error:
            raise RunDirError(f"{path}:{error.lineno}: not JSON: {error.msg}") from error


def _usage(path: Path) -> RunUsage | None:
    if not path.exists():
        return None
    calls = list(read_jsonl(path, LlmCallRecord))
    return RunUsage(
        calls=len(calls),
        input_tokens=sum(call.usage.input_tokens for call in calls),
        output_tokens=sum(call.usage.output_tokens for call in calls),
        cost_usd=math.fsum(call.usage.cost_usd for call in calls),
        unpriced_calls=sum(not call.usage.HasField("cost_usd") for call in calls),
    )


def _last_by_id[M: Message](path: Path, message_type: type[M], field: str) -> list[M]:
    if not path.exists():
        return []
    latest: dict[str, M] = {}
    for record in read_jsonl(path, message_type):
        latest[getattr(record, field)] = record  # A dict keeps the first appearance's place.
    return list(latest.values())


def _id_order(task: str) -> tuple[int, str]:
    """Numeric task IDs in numeric order, then any others."""
    return (int(task), "") if task.isdigit() else (sys.maxsize, task)
