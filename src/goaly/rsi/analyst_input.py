"""The analyst's input (spec §8.2 step 1, §7, §8.5): everything GLM sees, assembled in code.

:func:`build_analyst_input` reads only typed sources: T08's split, the ledger, the best's results
and trajectories, and the dataset. It keeps a task's details only if the split puts the task in
train, so the input cannot hold a per-task val result, a val trajectory, a val flagged ID, a gold
answer outside train, or anything held-out, whatever the sources hold. The ledger summary keeps
each experiment's hypothesis, decision and aggregate results.

Failed train tasks are summarized deterministically, with no model call. The whole prompt is kept
under MAX_INPUT_TOKENS, estimated as characters / CHARS_PER_TOKEN, by dropping whole summaries,
lowest priority first: flagged tasks come first, then the lowest credits.

The prompt template is prompts/analyst.toml, and the idea bank idea_bank.md, next to this module.
"""

import logging
import re
import string
import tomllib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from goaly.bench.effisql.data import EffiSqlInstance
from goaly.bench.effisql.metrics import SOLVED_CREDIT, is_flagged, tier_credit
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
    PhaseResult,
    RejectionReason,
)
from goaly.contracts.gen.result_pb2 import TaskResult
from goaly.contracts.gen.task_pb2 import (
    GRADE_STATUS_GRADED,
    GRADE_STATUS_PREDICTION_FAILED,
    GRADE_STATUS_TIMEOUT,
    SPLIT_TRAIN,
    GradeStatus,
)
from goaly.contracts.gen.trajectory_pb2 import StopReason, Trajectory
from goaly.rsi.analyst import AnalystAnswer, AnalystInput
from goaly.rsi.ledger import Ledger
from goaly.rsi.structured import format_instructions

__all__ = ["CHARS_PER_TOKEN", "MAX_INPUT_TOKENS", "build_analyst_input"]

MAX_INPUT_TOKENS: Final = 60_000  # The analyst's whole prompt, schema included.
CHARS_PER_TOKEN: Final = 3.5  # How the budget estimates tokens from characters.

_HERE = Path(__file__).parent
_PROMPT_PATH = _HERE / "prompts" / "analyst.toml"
_IDEA_BANK_PATH = _HERE / "idea_bank.md"
_TIER_FLOORS: Final = (2.0, 1.2, 1.0, 0.5)  # Where g(τ) steps up, spec §4.1; credits from T07.
_SCORED: Final = frozenset(
    {GRADE_STATUS_GRADED, GRADE_STATUS_PREDICTION_FAILED, GRADE_STATUS_TIMEOUT}
)
# How much of each part a failure's summary keeps, in characters.
_SQL_CHARS: Final = 1500
_FEEDBACK_CHARS: Final = 400
_ERROR_CHARS: Final = 300
_LAST_ERRORS: Final = 3
# How much of the ledger and the queue the input shows: the latest experiments, the first queued.
_LEDGER_EXPERIMENTS: Final = 40
_QUEUED_SHOWN: Final = 20
_CHANGE_CHARS: Final = 400
_PATTERN_CHARS: Final = 200
_ERROR_LINE = re.compile(r"\berror\b", re.IGNORECASE)

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _Failure:
    task_id: str
    credit: float  # The task's mean over seeds.
    flagged: bool
    text: str


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
    """Assemble the analyst's input for the Effi-SQL lane (spec §8.2).

    ``best_results`` and ``trajectories`` are the current best's, and ``instances`` the full
    dataset, gold included. Only tasks that ``splits`` puts in train contribute results,
    trajectories and gold; everything else is dropped here, however it was loaded. The ledger
    contributes each experiment's hypothesis, decision and aggregate results, never a per-task
    val result, a val flagged ID, a run ID or the decision's free-text reasoning.
    """
    train = frozenset(splits.ids(SPLIT_TRAIN))
    by_id = {inst.instance_id: inst for inst in instances if inst.instance_id in train}
    results: dict[str, list[TaskResult]] = defaultdict(list)
    for result in best_results:
        if result.instance_id in train:
            results[result.instance_id].append(result)
    paths = {
        (traj.instance_id, traj.seed): traj for traj in trajectories if traj.instance_id in train
    }
    experiments = list(ledger.latest_experiments(LANE_EFFISQL).values())
    taken = frozenset(
        [record.hypothesis.hypothesis_id for record in experiments]
        + [hypothesis.hypothesis_id for hypothesis in queued]
    )
    failures = _failures(results, paths, by_id)
    template = _load_template()
    sections = {
        "ledger": _ledger_summary(experiments, train),
        "queue": _queue_summary(queued, train),
        "train_results": _results_table(results, by_id),
        "idea_bank": _IDEA_BANK_PATH.read_text(encoding="utf-8").strip(),
    }
    system = string.Template(template["system"]).substitute(
        tiers=_tier_lines(), solved_credit=f"{SOLVED_CREDIT:g}"
    )
    fixed = len(system) + len(format_instructions(AnalystAnswer))
    fixed += len(string.Template(template["user"]).substitute(sections, failures=""))
    budget = int(max_input_tokens * CHARS_PER_TOKEN) - fixed - _dropped_note_size(failures)
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
    user = string.Template(template["user"]).substitute(
        sections, failures=_failures_section(shown, dropped)
    )
    return AnalystInput(
        system=system,
        user=user,
        failed_task_ids=frozenset(failure.task_id for failure in failures),
        shown_task_ids=tuple(failure.task_id for failure in shown),
        dropped_task_ids=dropped,
        taken_ids=taken,
    )


def _load_template() -> dict[str, str]:
    with _PROMPT_PATH.open("rb") as file:
        loaded = tomllib.load(file)
    return {name: str(loaded[name]).strip() for name in ("system", "user")}


def _tier_lines() -> str:
    return (
        "\n".join(f"  - tau >= {floor:g}: {tier_credit(floor):g}" for floor in _TIER_FLOORS)
        + "\n  - otherwise: 0"
    )


def _order(task_id: str) -> tuple[int, int, str]:
    """Sort numeric IDs by number, before any other ID."""
    return (0, int(task_id), "") if task_id.isdigit() else (1, 0, task_id)


def _credit(result: TaskResult) -> float:
    return result.grade.effisql.credit if result.grade.WhichOneof("benchmark") == "effisql" else 0.0


def _failures(
    results: Mapping[str, Sequence[TaskResult]],
    trajectories: Mapping[tuple[str, int], Trajectory],
    instances: Mapping[str, EffiSqlInstance],
) -> list[_Failure]:
    """Summarize each failed train task, highest priority first.

    A task fails if its credit, the mean over its scored seeds, is below SOLVED_CREDIT (spec
    §6.3); a grade with no score, as after a grader error, says nothing and is left out. The
    summary is of the task's worst seed. Flagged tasks come first, then the lowest credits.
    """
    failures = []
    for task_id, task_results in results.items():
        scored = [result for result in task_results if result.grade.status in _SCORED]
        if not scored:
            continue
        credit = sum(map(_credit, scored)) / len(scored)
        if credit >= SOLVED_CREDIT:
            continue
        worst = min(scored, key=lambda result: (_credit(result), result.seed))
        flagged = any(is_flagged(result.grade.effisql) for result in scored)
        trajectory = trajectories.get((task_id, worst.seed))
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
        flags = _flags(result)
        if flags:
            parts.append(f"flags={flags}")
    return ", ".join(parts)


def _flags(result: TaskResult) -> str:
    details = result.grade.effisql
    flags = []
    if details.matched and details.HasField("alt_matched") and not details.alt_matched:
        flags.append("hard_coded")
    if details.literal_result:
        flags.append("literal_result")
    return "+".join(flags)


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


def _ledger_summary(experiments: Sequence[ExperimentRecord], train: frozenset[str]) -> str:
    if not experiments:
        return "(no experiments yet)"
    shown = experiments[-_LEDGER_EXPERIMENTS:]
    lines = []
    if len(shown) < len(experiments):
        lines.append(f"({len(experiments) - len(shown)} earlier experiments are not shown.)")
    for record in shown:
        lines.append(
            f"- {record.experiment_id}, round {record.round}: "
            f"{_hypothesis_text(record.hypothesis, train)}"
        )
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
            summary = f": {_cut(run.summary, _CHANGE_CHARS)}" if run.summary else ""
            lines.append(f"  Implementer: {exit_name}{summary}")
        failed = [Guard.Name(guard.guard) for guard in record.guards if not guard.passed]
        if failed:
            lines.append(f"  Failed guards: {', '.join(failed)}")
        for name in ("screening", "validation", "confirmation"):
            if record.HasField(name):
                phase: PhaseResult = getattr(record, name)
                lines.append(f"  {name.capitalize()}: {_phase_text(phase, name, train)}")
    return "\n".join(lines)


def _phase_text(phase: PhaseResult, name: str, train: frozenset[str]) -> str:
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
        flagged = [task for task in phase.flagged_instance_ids if task in train]
        if name == "screening" and flagged:
            parts.append(f"flagged tasks: {', '.join(sorted(flagged, key=_order))}")
        else:
            parts.append(f"{len(phase.flagged_instance_ids)} flagged tasks")
    return "; ".join(parts)


def _queue_summary(queued: Sequence[Hypothesis], train: frozenset[str]) -> str:
    if not queued:
        return "(none)"
    lines = [f"- {_hypothesis_text(hypothesis, train)}" for hypothesis in queued[:_QUEUED_SHOWN]]
    if len(queued) > _QUEUED_SHOWN:
        lines.append(f"({len(queued) - _QUEUED_SHOWN} more are queued.)")
    return "\n".join(lines)


def _hypothesis_text(hypothesis: Hypothesis, train: frozenset[str]) -> str:
    kind = ChangeKind.Name(hypothesis.change_kind).removeprefix("CHANGE_KIND_").lower()
    source = HypothesisSource.Name(hypothesis.source).removeprefix("HYPOTHESIS_SOURCE_").lower()
    targets = [task for task in hypothesis.target_instance_ids if task in train]
    text = (
        f"hypothesis {hypothesis.hypothesis_id} [{kind}, {source}] targeting "
        f"{_cut(hypothesis.failure_pattern, _PATTERN_CHARS)!r}"
    )
    if targets:
        text += f" (train tasks {', '.join(targets)})"
    return f"{text}: {_cut(hypothesis.change, _CHANGE_CHARS)}"


def _cut(text: str, limit: int) -> str:
    """Return ``text`` with its middle cut out if it is longer than ``limit`` characters."""
    text = text.strip()
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n[... {len(text) - 2 * half} characters cut ...]\n{text[-half:]}"
