"""The analyst's input (spec §8.2 step 1, §7, §8.5): everything GLM sees, assembled in code.

:func:`build_analyst_input` reads only typed sources: T08's split, the ledger and its lessons, the
best's results and trajectories, and the dataset. What it keeps, it keeps by task ID:

- A result, a trajectory or a gold answer is kept only if T08's split puts its task in train.
  A trajectory is also kept only if its run, task and seed match a kept result: results come from
  the host's grading, while a trajectory is written in the solve container, so its own fields are
  not trusted alone. A (task, seed) graded in two runs, and a trajectory listed twice, are
  ambiguous: they are left out and logged.
- From the ledger, an experiment keeps its hypothesis, the implementer's exit and summary, the
  names of the guards that failed, its decision and reasons, and the aggregates of the phases
  whose split is the one the phase runs on: train for the screen, val for the validation and the
  confirmation. Only the screen's flagged tasks are named, and only when all are in train; other
  flags are counted. Run IDs, guard details and the decision's free-text reasoning are left out.
- Free text, which GLM wrote in earlier rounds from train data alone (hypotheses, implementer
  summaries, lessons), is trusted by where it comes from, not filtered. As a second layer, a text
  that refers to a val or held-out task by ID, as in "task 57" or "tasks 12, 57", is withheld and
  logged: a hypothesis's text is replaced by a note, and a lesson is left out.

So the input holds no per-task val result, val trajectory, val flagged ID, gold outside train or
held-out data, provided T21 passes trajectories read only from the best's train runs, on the
host: the cross-check above is a second layer, since a trajectory's run ID is container-written.

Failed train tasks are summarized deterministically, with no model call. The whole prompt, schema
included, is kept under MAX_INPUT_TOKENS, estimated as characters / CHARS_PER_TOKEN: whole
summaries are dropped, lowest priority first (flagged tasks come first, then the lowest credits);
if the other sections alone are too long, the oldest experiments, the last queued hypotheses and
the oldest lessons go first, and an input that still does not fit raises ValueError.

The prompt template is prompts/analyst.toml, and the idea bank idea_bank.md, next to this module.
"""

import logging
import re
import string
import tomllib
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from goaly.bench.effisql.data import EffiSqlInstance
from goaly.bench.effisql.metrics import SCORED_STATUSES, SOLVED_CREDIT, TIERS, is_flagged
from goaly.bench.effisql.splits import EffiSqlSplits
from goaly.contracts.gen.experiment_pb2 import (
    LANE_EFFISQL,
    ChangeKind,
    Decision,
    ExperimentRecord,
    Guard,
    Hypothesis,
    HypothesisSource,
    ImplementerExit,
    Lesson,
    PhaseResult,
    RejectionReason,
)
from goaly.contracts.gen.result_pb2 import TaskResult
from goaly.contracts.gen.task_pb2 import (
    SPLIT_HELDOUT,
    SPLIT_TRAIN,
    SPLIT_VAL,
    GradeStatus,
    Split,
)
from goaly.contracts.gen.trajectory_pb2 import StopReason, Trajectory
from goaly.rsi.analyst import AnalystAnswer, AnalystInput
from goaly.rsi.ledger import Ledger
from goaly.rsi.lessons import LessonStore
from goaly.rsi.structured import format_instructions

__all__ = ["CHARS_PER_TOKEN", "LESSONS_SHOWN", "MAX_INPUT_TOKENS", "build_analyst_input"]

MAX_INPUT_TOKENS: Final = 60_000  # The analyst's whole prompt, schema included.
CHARS_PER_TOKEN: Final = 3.5  # How the budget estimates tokens from characters.
LESSONS_SHOWN: Final = 10  # The lane's latest standing lessons.

_HERE = Path(__file__).parent
_PROMPT_PATH = _HERE / "prompts" / "analyst.toml"
_IDEA_BANK_PATH = _HERE / "idea_bank.md"
# How much of each part a failure's summary keeps, in characters.
_SQL_CHARS: Final = 1500
_FEEDBACK_CHARS: Final = 400
_ERROR_CHARS: Final = 300
_LAST_ERRORS: Final = 3
# How much of the ledger, the queue and the lessons the input shows at most.
_LEDGER_EXPERIMENTS: Final = 40
_QUEUED_SHOWN: Final = 20
_CHANGE_CHARS: Final = 400
_PATTERN_CHARS: Final = 200
_LESSON_CHARS: Final = 600
_TARGETS_SHOWN: Final = 10
_ERROR_LINE = re.compile(r"\berror\b", re.IGNORECASE)
# A reference to tasks by ID: "task 57", "tasks 12, 57 and 80", "instance #57", "IDs 12/57".
_TASK_REFERENCE = re.compile(
    r"\b(?:tasks?|instances?|instance_ids?|ids?)\b[\s#:=]*"
    r"([A-Za-z0-9][\w-]*(?:\s*(?:,|/|&|\band\b|\bor\b)\s*#?[A-Za-z0-9][\w-]*)*)",
    re.IGNORECASE,
)
_REFERENCE_SEPARATOR = re.compile(r"\s*(?:,|/|&|\band\b|\bor\b)\s*#?", re.IGNORECASE)
_WITHHELD: Final = "[withheld: it names a task outside train]"

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _Failure:
    task_id: str
    credit: float  # The task's mean over seeds.
    flagged: bool
    text: str


@dataclass(frozen=True, slots=True)
class _Scope:
    """The task IDs that may be named, train, and those that never may, val and held-out."""

    train: frozenset[str]
    secret: frozenset[str]

    def withhold(self, text: str, where: str) -> str:
        """Return ``text``, or a note in its place if it names a val or held-out task."""
        if not self.names_secret(text):
            return text
        _logger.warning(
            "withheld a text from the analyst's input: it names a task outside train",
            extra={"event": "analyst_input_withheld", "where": where},
        )
        return _WITHHELD

    def names_secret(self, text: str) -> bool:
        """Return whether ``text`` refers to a val or held-out task by its ID."""
        return any(
            task in self.secret
            for reference in _TASK_REFERENCE.finditer(text)
            for task in _REFERENCE_SEPARATOR.split(reference[1])
        )


def build_analyst_input(
    *,
    splits: EffiSqlSplits,
    ledger: Ledger,
    best_results: Iterable[TaskResult],
    trajectories: Iterable[Trajectory],
    instances: Iterable[EffiSqlInstance],
    queued: Sequence[Hypothesis],
    max_input_tokens: int = MAX_INPUT_TOKENS,
) -> AnalystInput:
    """Assemble the analyst's input for the Effi-SQL lane (spec §8.2); the module says how.

    ``best_results`` are the current best's host-graded results, one per (task, seed), and
    ``trajectories`` its trajectories, read only from its train runs on the host. ``instances``
    are the full dataset, gold included. The lessons are the lane's latest LESSONS_SHOWN standing
    ones in the ledger, read through T18's LessonStore. Raises ValueError if the sections other
    than the failure summaries cannot be cut down to fit ``max_input_tokens``.
    """
    scope = _Scope(
        frozenset(splits.ids(SPLIT_TRAIN)),
        frozenset(splits.ids(SPLIT_VAL)) | frozenset(splits.ids(SPLIT_HELDOUT)),
    )
    by_id = {inst.instance_id: inst for inst in instances if inst.instance_id in scope.train}
    results = _train_results(best_results, scope.train)
    paired = _paired_trajectories(trajectories, results)
    experiments = list(ledger.latest_experiments(LANE_EFFISQL).values())
    taken = frozenset(
        [record.hypothesis.hypothesis_id for record in experiments]
        + [hypothesis.hypothesis_id for hypothesis in queued]
    )
    failures = _failures(results, paired, by_id)
    lessons = LessonStore(ledger).lessons(LANE_EFFISQL)[-LESSONS_SHOWN:]
    blocks = {
        "ledger": [
            _experiment_text(record, scope) for record in experiments[-_LEDGER_EXPERIMENTS:]
        ],
        "queue": [f"- {_hypothesis_text(h, scope, 'queued')}" for h in queued[:_QUEUED_SHOWN]],
        "lessons": [text for lesson in lessons if (text := _lesson_text(lesson, scope))],
    }
    totals = {"ledger": len(experiments), "queue": len(queued)}
    template = _load_template()
    system = string.Template(template["system"]).substitute(
        tiers=_tier_lines(), solved_credit=f"{SOLVED_CREDIT:g}"
    )
    user = string.Template(template["user"])
    rest = {
        "train_results": _results_table(results, by_id),
        "idea_bank": _IDEA_BANK_PATH.read_text(encoding="utf-8").strip(),
    }
    limit = int(max_input_tokens * CHARS_PER_TOKEN)
    overhead = len(system) + len(format_instructions(AnalystAnswer)) + _dropped_note_size(failures)

    def fixed_size() -> int:
        return overhead + len(user.substitute(_sections(blocks, totals), **rest, failures=""))

    _fit_sections(blocks, fixed_size, limit)
    budget = limit - fixed_size()
    shown: list[_Failure] = []
    for failure in failures:
        if len(failure.text) + 2 > budget:
            break
        shown.append(failure)
        budget -= len(failure.text) + 2
    dropped = tuple(failure.task_id for failure in failures[len(shown) :])
    if dropped:
        _logger.warning(
            "dropped failed train tasks from the analyst's input to fit its budget",
            extra={"event": "analyst_input_truncated", "dropped": list(dropped)},
        )
    text = user.substitute(
        _sections(blocks, totals), **rest, failures=_failures_section(shown, dropped)
    )
    return AnalystInput(
        system=system,
        user=text,
        failed_task_ids=frozenset(failure.task_id for failure in failures),
        shown_task_ids=tuple(failure.task_id for failure in shown),
        dropped_task_ids=dropped,
        taken_ids=taken,
    )


def _train_results(
    best_results: Iterable[TaskResult], train: frozenset[str]
) -> dict[str, list[TaskResult]]:
    """Return the train results by task. A (task, seed) graded in two runs is left out.

    Several results of one run, task and seed are one result graded again: the last one wins, as
    in results.jsonl.
    """
    latest: dict[tuple[str, int], dict[str, TaskResult]] = defaultdict(dict)
    for result in best_results:
        if result.instance_id in train:
            latest[(result.instance_id, result.seed)][result.run_id] = result
    results: dict[str, list[TaskResult]] = defaultdict(list)
    ambiguous = []
    for (task_id, seed), runs in latest.items():
        if len(runs) > 1:
            ambiguous.append(f"{task_id} seed {seed}")
            continue
        results[task_id].extend(runs.values())
    if ambiguous:
        _logger.warning(
            "left train results out of the analyst's input: graded in more than one run",
            extra={"event": "analyst_input_ambiguous_results", "tasks": ambiguous},
        )
    return results


def _paired_trajectories(
    trajectories: Iterable[Trajectory], results: Mapping[str, Sequence[TaskResult]]
) -> dict[tuple[str, str, int], Trajectory]:
    """Return the trajectories of the kept results, by (run, task, seed).

    A trajectory that matches no kept result is ignored, and a key with more than one trajectory
    is ambiguous, so none of them is kept.
    """
    wanted = {
        (result.run_id, result.instance_id, result.seed)
        for task_results in results.values()
        for result in task_results
    }
    matching = [
        trajectory
        for trajectory in trajectories
        if (trajectory.run_id, trajectory.instance_id, trajectory.seed) in wanted
    ]
    counts = Counter((t.run_id, t.instance_id, t.seed) for t in matching)
    duplicated = sorted(key for key, count in counts.items() if count > 1)
    if duplicated:
        _logger.warning(
            "left trajectories out of the analyst's input: listed more than once",
            extra={
                "event": "analyst_input_duplicate_trajectories",
                "tasks": [f"{task} seed {seed} of {run}" for run, task, seed in duplicated],
            },
        )
    return {
        (t.run_id, t.instance_id, t.seed): t
        for t in matching
        if counts[(t.run_id, t.instance_id, t.seed)] == 1
    }


def _fit_sections(blocks: dict[str, list[str]], fixed_size: Callable[[], int], limit: int) -> None:
    """Drop the oldest experiments, then the last queued, then the oldest lessons, until the
    sections other than the failures fit ``limit``. Raises ValueError if they never do.
    """
    cut = Counter[str]()
    for name, end in (("ledger", 0), ("queue", -1), ("lessons", 0)):
        while blocks[name] and fixed_size() > limit:
            blocks[name].pop(end)
            cut[name] += 1
    if cut:
        _logger.warning(
            "cut the analyst's ledger, queue or lessons to fit its budget",
            extra={"event": "analyst_input_sections_cut", "cut": dict(cut)},
        )
    if fixed_size() > limit:
        raise ValueError(
            f"the analyst's input needs {fixed_size()} characters before any failure, over its "
            f"budget of {limit}"
        )


def _sections(blocks: Mapping[str, Sequence[str]], totals: Mapping[str, int]) -> dict[str, str]:
    ledger, queue = blocks["ledger"], blocks["queue"]
    ledger_text = "\n".join(ledger) or "(no experiments yet)"
    if ledger and len(ledger) < totals["ledger"]:
        hidden = totals["ledger"] - len(ledger)
        ledger_text = f"({hidden} earlier experiments are not shown.)\n{ledger_text}"
    queue_text = "\n".join(queue) or "(none)"
    if queue and len(queue) < totals["queue"]:
        queue_text += f"\n({totals['queue'] - len(queue)} more are queued.)"
    lessons_text = "\n".join(blocks["lessons"]) or "(none)"
    return {"ledger": ledger_text, "queue": queue_text, "lessons": lessons_text}


def _load_template() -> dict[str, str]:
    with _PROMPT_PATH.open("rb") as file:
        loaded = tomllib.load(file)
    return {name: str(loaded[name]).strip() for name in ("system", "user")}


def _tier_lines() -> str:
    lines = [f"  - tau >= {floor:g}: {credit:g}" for floor, credit in TIERS]
    return "\n".join([*lines, "  - otherwise: 0"])


def _order(task_id: str) -> tuple[int, int, str]:
    """Sort numeric IDs by number, before any other ID."""
    return (0, int(task_id), "") if task_id.isdigit() else (1, 0, task_id)


def _credit(result: TaskResult) -> float:
    return result.grade.effisql.credit if result.grade.WhichOneof("benchmark") == "effisql" else 0.0


def _failures(
    results: Mapping[str, Sequence[TaskResult]],
    trajectories: Mapping[tuple[str, str, int], Trajectory],
    instances: Mapping[str, EffiSqlInstance],
) -> list[_Failure]:
    """Summarize each failed train task, highest priority first.

    A task fails if its credit, the mean over its scored seeds, is below SOLVED_CREDIT (spec
    §6.3); a grade with no score, as after a grader error, says nothing and is left out. The
    summary is of the task's worst seed, with that result's own trajectory. Flagged tasks come
    first, then the lowest credits.
    """
    failures = []
    for task_id, task_results in results.items():
        scored = [result for result in task_results if result.grade.status in SCORED_STATUSES]
        if not scored:
            continue
        credit = sum(map(_credit, scored)) / len(scored)
        if credit >= SOLVED_CREDIT:
            continue
        worst = min(scored, key=lambda result: (_credit(result), result.seed))
        flagged = any(is_flagged(result.grade.effisql) for result in scored)
        trajectory = trajectories.get((worst.run_id, task_id, worst.seed))
        text = _summary(worst, trajectory, instances.get(task_id), credit)
        failures.append(_Failure(task_id, credit, flagged, text))
    failures.sort(
        key=lambda failure: (not failure.flagged, failure.credit, _order(failure.task_id))
    )
    return failures


def _summary(
    result: TaskResult,
    trajectory: Trajectory | None,
    instance: EffiSqlInstance | None,
    credit: float,
) -> str:
    grade = result.grade
    difficulty = instance.difficulty if instance and instance.difficulty else "unknown"
    lines = [
        f"### Task {result.instance_id} ({difficulty}): credit {credit:g}",
        f"Grade: {_grade_text(result)}",
        f"Grader feedback: {_cut(grade.feedback, _FEEDBACK_CHARS) or '(none)'}",
    ]
    steps = len(trajectory.steps) if trajectory else result.step_count
    lines.append(f"Stopped: {StopReason.Name(result.stop_reason)} after {steps} steps")
    base_sql = instance.base_sql if instance else ""
    if not base_sql and trajectory and trajectory.task.WhichOneof("benchmark") == "effisql":
        base_sql = trajectory.task.effisql.base_sql
    lines += _sql_block("Base SQL", base_sql)
    if trajectory is None:
        lines.append("Final SQL: (no trajectory)")
    elif not trajectory.HasField("prediction"):
        lines.append("Final SQL: (none: the policy returned no prediction)")
    else:
        label = "Final SQL (the fallback)" if trajectory.prediction.fallback else "Final SQL"
        lines += _sql_block(label, trajectory.prediction.effisql.sql)
    if instance and instance.optimized_sql:
        lines += _sql_block("Gold SQL (train only, never to be copied)", instance.optimized_sql)
    errors = _last_errors(result, trajectory)
    if errors:
        lines.append("Last errors:")
        lines += [f"- {error}" for error in errors]
    return "\n".join(lines)


def _grade_text(result: TaskResult) -> str:
    grade = result.grade
    parts = [GradeStatus.Name(grade.status).removeprefix("GRADE_STATUS_")]
    if grade.WhichOneof("benchmark") == "effisql":
        details = grade.effisql
        parts.append("matched" if details.matched else "unmatched")
        if details.HasField("speedup"):
            parts.append(f"tau={details.speedup:.3g}")
        parts.append(f"credit={details.credit:g}")
        if is_flagged(details):
            parts.append(f"flags={_flag_names(result)}")
    return ", ".join(parts)


def _flag_names(result: TaskResult) -> str:
    """Name the flags of a grade that is_flagged flags."""
    details = result.grade.effisql
    names = ["literal_result"] if details.literal_result else []
    if details.HasField("alt_matched") and not details.alt_matched:
        names.insert(0, "hard_coded")
    return "+".join(names)


def _sql_block(label: str, sql: str) -> list[str]:
    if not sql.strip():
        return [f"{label}: (empty)"]
    return [f"{label}:", "```sql", _cut(sql.strip(), _SQL_CHARS), "```"]


def _last_errors(result: TaskResult, trajectory: Trajectory | None) -> list[str]:
    errors = []
    for step in trajectory.steps if trajectory else ():
        for tool in step.tool_results:
            first_line = tool.content.strip().partition("\n")[0]
            if tool.is_error or _ERROR_LINE.search(first_line):
                errors.append(f"step {step.index} {tool.name}: {_cut(tool.content, _ERROR_CHARS)}")
    stopped = (trajectory.error_message if trajectory else "") or result.error_message
    if stopped:
        errors.append(f"the run: {_cut(stopped, _ERROR_CHARS)}")
    return errors[-_LAST_ERRORS:]


def _failures_section(shown: Sequence[_Failure], dropped: Sequence[str]) -> str:
    if not shown and not dropped:
        return "(none)"
    parts = [failure.text for failure in shown]
    if dropped:
        parts.append(
            f"{len(dropped)} more failed tasks are not summarized, to keep the input short: "
            f"{', '.join(dropped)}. Their results are in the table above."
        )
    return "\n\n".join(parts)


def _dropped_note_size(failures: Sequence[_Failure]) -> int:
    return 200 + sum(len(failure.task_id) + 2 for failure in failures)


def _results_table(
    results: Mapping[str, Sequence[TaskResult]], instances: Mapping[str, EffiSqlInstance]
) -> str:
    seeds = {result.seed for task_results in results.values() for result in task_results}
    lines = []
    for task_id in sorted(results, key=_order):
        instance = instances.get(task_id)
        difficulty = instance.difficulty if instance and instance.difficulty else "unknown"
        for result in sorted(results[task_id], key=lambda result: result.seed):
            seed = f" seed {result.seed}" if len(seeds) > 1 else ""
            lines.append(f"{task_id}{seed}, {difficulty}, {_grade_text(result)}")
    return "\n".join(lines) or "(no results)"


def _experiment_text(record: ExperimentRecord, scope: _Scope) -> str:
    name = record.experiment_id
    lines = [f"- {name}, round {record.round}: {_hypothesis_text(record.hypothesis, scope, name)}"]
    outcome = Decision.Name(record.decision).removeprefix("DECISION_")
    if record.rejection_reasons:
        reasons = (
            RejectionReason.Name(reason).removeprefix("REJECTION_REASON_")
            for reason in record.rejection_reasons
        )
        outcome += f" ({', '.join(reasons)})"
    lines.append(f"  Decision: {outcome}")
    if record.HasField("implementer"):
        run = record.implementer
        exit_name = ImplementerExit.Name(run.exit).removeprefix("IMPLEMENTER_EXIT_")
        summary = scope.withhold(run.summary, f"experiment {name}: implementer summary")
        summary = f": {_cut(summary, _CHANGE_CHARS)}" if summary else ""
        lines.append(f"  Implementer: {exit_name}{summary}")
    failed = [Guard.Name(guard.guard) for guard in record.guards if not guard.passed]
    if failed:
        lines.append(f"  Failed guards: {', '.join(failed)}")
    # Each phase with the split it runs on (spec §8.2): a phase on any other is not shown.
    phases = (
        ("screening", SPLIT_TRAIN, record.screening if record.HasField("screening") else None),
        ("validation", SPLIT_VAL, record.validation if record.HasField("validation") else None),
        (
            "confirmation",
            SPLIT_VAL,
            record.confirmation if record.HasField("confirmation") else None,
        ),
    )
    for phase_name, split, phase in phases:
        if phase is None:
            continue
        if phase.split != split:
            _logger.warning(
                "left a phase out of the analyst's input: it ran on the wrong split",
                extra={
                    "event": "analyst_input_wrong_split",
                    "experiment_id": name,
                    "phase": phase_name,
                    "split": Split.Name(phase.split),
                },
            )
            continue
        lines.append(f"  {phase_name.capitalize()}: {_phase_text(phase, phase_name, scope)}")
    return "\n".join(lines)


def _phase_text(phase: PhaseResult, name: str, scope: _Scope) -> str:
    """Return the phase's aggregates; only the screen, on train, names its flagged tasks."""
    parts = [
        f"{phase.task_count} tasks x {phase.seed_count} seeds",
        f"R-VES {100 * phase.candidate_score:.1f} vs best {100 * phase.best_score:.1f} "
        f"(delta {100 * phase.delta:+.1f})",
    ]
    if phase.HasField("wins") and phase.HasField("losses"):
        parts.append(f"wins/losses {phase.wins}/{phase.losses}")
    if phase.HasField("sign_test_p"):
        parts.append(f"sign test p={phase.sign_test_p:.3g}")
    if phase.HasField("regressions"):
        parts.append(f"regressions {phase.regressions}")
    if phase.HasField("token_ratio_vs_best"):
        parts.append(f"tokens x{phase.token_ratio_vs_best:.2f} of best")
    if phase.flagged_instance_ids:
        flagged = [task for task in phase.flagged_instance_ids if task in scope.train]
        if name == "screening" and len(flagged) == len(phase.flagged_instance_ids):
            parts.append(f"flagged tasks: {', '.join(sorted(flagged, key=_order))}")
        else:
            parts.append(f"{len(phase.flagged_instance_ids)} flagged tasks")
    return "; ".join(parts)


def _hypothesis_text(hypothesis: Hypothesis, scope: _Scope, where: str) -> str:
    kind = ChangeKind.Name(hypothesis.change_kind).removeprefix("CHANGE_KIND_").lower()
    source = HypothesisSource.Name(hypothesis.source).removeprefix("HYPOTHESIS_SOURCE_").lower()
    label = f"{where}: hypothesis {hypothesis.hypothesis_id}"
    pattern = scope.withhold(hypothesis.failure_pattern, f"{label}, failure_pattern")
    change = scope.withhold(hypothesis.change, f"{label}, change")
    text = (
        f"hypothesis {hypothesis.hypothesis_id} [{kind}, {source}] targeting "
        f"{_cut(pattern, _PATTERN_CHARS)!r}"
    )
    targets = [task for task in hypothesis.target_instance_ids if task in scope.train]
    if targets:
        more = len(targets) - _TARGETS_SHOWN
        listed = ", ".join(targets[:_TARGETS_SHOWN]) + (f" and {more} more" if more > 0 else "")
        text += f" (train tasks {listed})"
    return f"{text}: {_cut(change, _CHANGE_CHARS)}"


def _lesson_text(lesson: Lesson, scope: _Scope) -> str:
    """Return the lesson as a line of the input, or "" if it names a task outside train."""
    if scope.names_secret(lesson.text):
        _logger.warning(
            "left a lesson out of the analyst's input: it names a task outside train",
            extra={"event": "analyst_input_withheld", "where": f"lesson {lesson.lesson_id}"},
        )
        return ""
    evidence = ", ".join(lesson.experiment_ids) or "none"
    counts = f"helped {len(lesson.helped_by)}, hurt {len(lesson.hurt_by)}"
    text = _cut(lesson.text, _LESSON_CHARS)
    return f"- {lesson.lesson_id} ({counts}; evidence: {evidence}): {text}"


def _cut(text: str, limit: int) -> str:
    """Return ``text`` with its middle cut out if it is longer than ``limit`` characters."""
    text = text.strip()
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n[... {len(text) - 2 * half} characters cut ...]\n{text[-half:]}"
