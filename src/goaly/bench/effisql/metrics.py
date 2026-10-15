"""Effi-SQL metrics (spec §4.1, §6.3) and the paired comparison RSI decides on (spec §8.3).

A version's results form a :class:`GradeTable`: each task's grade in each of the same seeds.
Every metric comes in two scorings (:class:`Scoring`):

- official, as the official evaluator computes it, ignoring the hard-coding flags: a matched
  task's credit is g(τ) of its capped speedup, and any other task's is 0;
- checked, which RSI decides on: the grader's credit, as given, and a flagged task, hard_coded
  or literal_result, counts as unmatched, so it gets credit 0, is not covered and is left out
  of GMS.

R-VES and Coverage@2x always divide by every task. :func:`tier_credit` is g(τ), the one function
the grader and the official numbers share.
"""

import enum
import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Self

from goaly.contracts.gen.result_pb2 import TaskResult
from goaly.contracts.gen.task_pb2 import (
    GRADE_STATUS_GRADED,
    GRADE_STATUS_PREDICTION_FAILED,
    GRADE_STATUS_TIMEOUT,
    EffiSqlGrade,
    GradeStatus,
)
from goaly.harness.core.stats import (
    BOOTSTRAP_RESAMPLES,
    Interval,
    MeanStd,
    SignTest,
    bootstrap_ci,
    mean_std,
    seed_means,
    sign_test,
)

__all__ = [
    "REGRESSED_CREDIT",
    "SCORED_STATUSES",
    "SOLVED_CREDIT",
    "SPEEDUP_CAP",
    "TIERS",
    "Comparison",
    "GradeTable",
    "Metrics",
    "Scoring",
    "Summary",
    "by_difficulty",
    "compare",
    "delta_rves_ci",
    "flagged_tasks",
    "is_flagged",
    "markdown_table",
    "seed_metrics",
    "solved",
    "stably_solved",
    "summarize",
    "task_credits",
    "tier_credit",
]

SPEEDUP_CAP = 1000.0  # τ is capped here, by the grader and again by GMS.
SOLVED_CREDIT = 0.75  # A task with at least this credit is solved: matched with τ ≥ 1.2.
REGRESSED_CREDIT = 0.25  # A solved task regresses if the candidate's credit is at most this.

# g(τ)'s tiers, highest first: each tier's lowest τ and its credit; any lower τ earns 0.
TIERS = ((2.0, 1.0), (1.2, 0.75), (1.0, 0.5), (0.5, 0.25))
_CREDITS = frozenset({0.0, *(credit for _, credit in TIERS)})
_COVERED_SPEEDUP = 2.0
# The statuses whose grades score the prediction. Any other says nothing about it: GRADER_ERROR
# is the grader's own failure, to regrade first (task.proto), and UNSPECIFIED was never set.
SCORED_STATUSES = frozenset(
    {GRADE_STATUS_GRADED, GRADE_STATUS_PREDICTION_FAILED, GRADE_STATUS_TIMEOUT}
)
_DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2}
_NAMED_TASKS = 5  # An error names at most this many tasks of a set.


class Scoring(enum.StrEnum):
    """Whether the metrics apply the hard-coding flags (spec §4.1)."""

    OFFICIAL = "official"  # As the official evaluator computes them: g(τ), flags ignored.
    CHECKED = "checked"  # The grader's credit; a flagged task is unmatched. RSI decides on these.


@dataclass(frozen=True, slots=True)
class Metrics:
    """One seed's metrics over every task (spec §4.1)."""

    rves: float  # 100 times the mean credit.
    coverage_2x: float  # 100 times the share of tasks that are matched with τ ≥ 2.
    gms: float | None  # Geometric mean of the capped τ over matched tasks with τ > 0, if any.


@dataclass(frozen=True, slots=True)
class Summary:
    """A version's metrics under one scoring, as the report gives them (spec §6.3)."""

    task_count: int
    seed_count: int
    rves: MeanStd  # Over the seeds' values.
    rves_ci: Interval  # The bootstrap 95% CI, resampling tasks with their seed-mean credits.
    coverage_2x: MeanStd
    gms: MeanStd | None  # None if a seed has no matched task with τ > 0.
    solved_count: int
    stably_solved_count: int


@dataclass(frozen=True, slots=True)
class Comparison:
    """A candidate against the best, task by task, as spec §8.3 decides and PhaseResult records.

    The scores and the delta are fractions, mean credits or R-VES / 100, as PhaseResult records
    them; delta_rves_ci gives the delta's interval in R-VES points instead. The delta is rounded
    once from the exact difference, so a gain of exactly 5 points meets a min_delta of 0.05.
    """

    task_count: int
    best_score: float
    candidate_score: float
    delta: float  # candidate_score - best_score.
    sign: SignTest  # On each task's seed-mean credit; a win is a task the candidate does better.
    regressions: frozenset[str]  # Credit ≥ 0.75 under the best, ≤ 0.25 under the candidate.


@dataclass(frozen=True, slots=True)
class GradeTable:
    """One version's grades, ``grades[instance_id][seed]``: every task in the same seeds."""

    grades: Mapping[str, Mapping[int, EffiSqlGrade]]

    def __post_init__(self) -> None:
        # Keep a read-only copy, so that no later change to the input escapes these checks.
        copied = {
            task: MappingProxyType({seed: _copy(grade) for seed, grade in seeds.items()})
            for task, seeds in self.grades.items()
        }
        object.__setattr__(self, "grades", MappingProxyType(copied))
        if not self.grades:
            raise ValueError("the table has no tasks")
        first, first_seeds = next(iter(self.grades.items()))
        for task, seeds in self.grades.items():
            if not seeds:
                raise ValueError(f"task {task!r} has no seeds")
            if seeds.keys() != first_seeds.keys():
                raise ValueError(
                    f"task {task!r} has seeds {sorted(seeds)}, but task {first!r} has"
                    f" {sorted(first_seeds)}: every task needs the same seeds"
                )
            for seed, grade in seeds.items():
                if grade.credit not in _CREDITS:
                    raise ValueError(
                        f"task {task!r}, seed {seed}, has credit {grade.credit}, which no tier has"
                    )
                if grade.credit and not grade.matched:
                    raise ValueError(
                        f"task {task!r}, seed {seed}, is not matched but has credit {grade.credit}"
                    )
                if grade.matched and not grade.speedup >= 0:
                    raise ValueError(f"task {task!r}, seed {seed}, has speedup {grade.speedup}")

    @classmethod
    def from_results(cls, results: Iterable[TaskResult]) -> Self:
        """Build the table from a version's results.

        As result.proto says, the last line for a task and seed wins within a run. The same task
        and seed in two runs is ambiguous, and so is an error.
        """
        latest: dict[tuple[str, int], TaskResult] = {}
        for result in results:
            key = (result.instance_id, result.seed)
            earlier = latest.get(key)
            if earlier is not None and earlier.run_id != result.run_id:
                raise ValueError(
                    f"task {key[0]!r}, seed {key[1]}, has results from two runs:"
                    f" {earlier.run_id!r} and {result.run_id!r}"
                )
            latest[key] = result
        grades: dict[str, dict[int, EffiSqlGrade]] = {}
        for (task, seed), result in latest.items():
            grades.setdefault(task, {})[seed] = _effisql_grade(result)
        return cls(grades)

    @property
    def tasks(self) -> tuple[str, ...]:
        return tuple(self.grades)

    def subset(self, tasks: Iterable[str]) -> Self:
        """Return the table of just ``tasks``, in this table's order.

        For example, the best's table on a candidate's tasks, which compare needs.
        """
        wanted = set(tasks)
        if missing := wanted - self.grades.keys():
            raise ValueError(f"the table lacks {_names(missing)}")
        return type(self)({task: seeds for task, seeds in self.grades.items() if task in wanted})

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(sorted(next(iter(self.grades.values()))))


def tier_credit(speedup: float) -> float:
    """Return g(τ), a matched task's credit for its speedup τ: 1, 0.75, 0.5, 0.25 or 0."""
    if not speedup >= 0:
        raise ValueError(f"a speedup is at least 0, not {speedup}")
    return next((credit for lowest, credit in TIERS if speedup >= lowest), 0.0)


def is_flagged(grade: EffiSqlGrade) -> bool:
    """Return whether a hard-coding check flagged the prediction (spec §4.1).

    Both checks run only on a matched prediction, and not on an identity-cached grade, so their
    fields are set only on a match: alt_matched false is the hard_coded flag, and literal_result
    true the literal_result flag. A field left unset, by a check that did not run, flags nothing.
    hard_coded also requires the match itself, in case an unmatched grade has alt_matched false.
    """
    hard_coded = grade.matched and grade.HasField("alt_matched") and not grade.alt_matched
    return hard_coded or grade.literal_result


def seed_metrics(table: GradeTable, *, scoring: Scoring) -> dict[int, Metrics]:
    """Return each seed's R-VES, Coverage@2x and GMS."""
    scoring = Scoring(scoring)
    return {
        seed: _metrics([seeds[seed] for seeds in table.grades.values()], scoring)
        for seed in table.seeds
    }


def task_credits(table: GradeTable, *, scoring: Scoring) -> dict[str, float]:
    """Return each task's credit for the version: its mean over the seeds (spec §8.3)."""
    return seed_means(_seed_credits(table, Scoring(scoring)))


def solved(table: GradeTable, *, scoring: Scoring) -> frozenset[str]:
    """Return the tasks whose credit, the mean over seeds, is at least 0.75 (spec §6.3)."""
    credits = task_credits(table, scoring=Scoring(scoring))
    return frozenset(task for task, credit in credits.items() if credit >= SOLVED_CREDIT)


def stably_solved(table: GradeTable, *, scoring: Scoring) -> frozenset[str]:
    """Return the tasks whose credit is at least 0.75 in every seed (spec §6.3)."""
    credits = _seed_credits(table, Scoring(scoring))
    return frozenset(task for task, values in credits.items() if min(values) >= SOLVED_CREDIT)


def flagged_tasks(table: GradeTable) -> frozenset[str]:
    """Return the tasks flagged in any seed; RSI fails such a candidate outright (spec §8.2).

    A frozenset has no order: sort it to record it, as in the ledger.
    """
    return frozenset(
        task for task, seeds in table.grades.items() if any(map(is_flagged, seeds.values()))
    )


def summarize(
    table: GradeTable,
    *,
    scoring: Scoring,
    bootstrap_seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> Summary:
    """Summarize the version over its seeds, with R-VES's bootstrap 95% CI over tasks.

    The bootstrap takes the tasks sorted by ID, so the interval depends on the results alone,
    not on the order they were listed in.
    """
    scoring = Scoring(scoring)
    per_seed = seed_metrics(table, scoring=scoring).values()
    gms = [metrics.gms for metrics in per_seed if metrics.gms is not None]
    credits = task_credits(table, scoring=scoring)
    by_task = [credits[task] for task in sorted(credits)]
    return Summary(
        task_count=len(table.grades),
        seed_count=len(per_seed),
        rves=mean_std([metrics.rves for metrics in per_seed]),
        rves_ci=bootstrap_ci(by_task, _rves, seed=bootstrap_seed, resamples=resamples),
        coverage_2x=mean_std([metrics.coverage_2x for metrics in per_seed]),
        gms=mean_std(gms) if len(gms) == len(per_seed) else None,
        solved_count=len(solved(table, scoring=scoring)),
        stably_solved_count=len(stably_solved(table, scoring=scoring)),
    )


def compare(best: GradeTable, candidate: GradeTable, *, scoring: Scoring) -> Comparison:
    """Compare ``candidate`` with ``best`` on the same tasks; each has its own seeds.

    To compare on the candidate's tasks only, pass ``best.subset(candidate.tasks)``.
    """
    scoring = Scoring(scoring)
    before, after = _paired_credits(best, candidate, scoring)
    best_score, candidate_score = _mean_credit(best, scoring), _mean_credit(candidate, scoring)
    return Comparison(
        task_count=len(before),
        best_score=float(best_score),
        candidate_score=float(candidate_score),
        delta=float(candidate_score - best_score),
        sign=sign_test(before, after),
        regressions=frozenset(
            task
            for task, credit in before.items()
            if credit >= SOLVED_CREDIT and after[task] <= REGRESSED_CREDIT
        ),
    )


def delta_rves_ci(
    best: GradeTable,
    candidate: GradeTable,
    *,
    scoring: Scoring,
    bootstrap_seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> Interval:
    """Return the paired bootstrap 95% CI of the candidate's R-VES gain, in R-VES points.

    The points are 100 times Comparison.delta's fractions. It resamples tasks, each with its
    difference in seed-mean credit, sorted by task ID as in summarize.
    """
    before, after = _paired_credits(best, candidate, Scoring(scoring))
    differences = [after[task] - before[task] for task in sorted(before)]
    return bootstrap_ci(differences, _rves, seed=bootstrap_seed, resamples=resamples)


def by_difficulty(table: GradeTable, difficulty: Mapping[str, str]) -> dict[str, GradeTable]:
    """Split the table by task difficulty: easy, medium, hard, then other labels in order.

    ``difficulty`` maps each instance_id to its label, as EffiSqlTaskInput.difficulty gives it.
    """
    groups: dict[str, dict[str, Mapping[int, EffiSqlGrade]]] = {}
    for task, seeds in table.grades.items():
        if task not in difficulty:
            raise ValueError(f"task {task!r} has no difficulty")
        groups.setdefault(difficulty[task], {})[task] = seeds
    order = sorted(groups, key=lambda label: (_DIFFICULTY_ORDER.get(label, math.inf), label))
    return {label: GradeTable(groups[label]) for label in order}


def markdown_table(
    rows: Mapping[str, GradeTable],
    *,
    bootstrap_seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> str:
    """Return a Markdown table with a line for each row's official, then checked, metrics.

    R-VES, Coverage@2x and GMS show the mean ± standard deviation over seeds, the CI is
    R-VES's bootstrap 95% interval, and the last two columns count tasks.
    """
    lines = [
        "| | Scoring | Tasks | Seeds | R-VES | 95% CI | Coverage@2x | GMS | Solved"
        " | Stably solved |",
        "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for label, table in rows.items():
        for scoring in Scoring:
            summary = summarize(
                table, scoring=scoring, bootstrap_seed=bootstrap_seed, resamples=resamples
            )
            cells = [
                label.replace("|", r"\|"),
                scoring,
                summary.task_count,
                summary.seed_count,
                _format(summary.rves, 1),
                f"[{summary.rves_ci.low:.1f}, {summary.rves_ci.high:.1f}]",
                _format(summary.coverage_2x, 1),
                "n/a" if summary.gms is None else _format(summary.gms, 2),
                summary.solved_count,
                summary.stably_solved_count,
            ]
            lines.append(f"| {' | '.join(map(str, cells))} |")
    return "\n".join(lines) + "\n"


def _effisql_grade(result: TaskResult) -> EffiSqlGrade:
    where = f"task {result.instance_id!r}, seed {result.seed},"
    if not result.HasField("grade"):
        raise ValueError(f"{where} has no grade: metrics need a graded run")
    if result.grade.WhichOneof("benchmark") != "effisql":
        raise ValueError(f"{where} has no Effi-SQL grade")
    if result.grade.status not in SCORED_STATUSES:
        status = GradeStatus.Name(result.grade.status)
        raise ValueError(f"{where} is {status}: regrade it before computing metrics")
    return result.grade.effisql


def _score(grade: EffiSqlGrade, scoring: Scoring) -> tuple[float, float]:
    """Return the task's credit, and its capped τ if it counts as matched, otherwise 0."""
    # An unset speedup reads as 0: a match whose timing failed.
    speedup = min(grade.speedup, SPEEDUP_CAP) if grade.matched else 0.0
    match scoring:
        case Scoring.OFFICIAL:
            return tier_credit(speedup), speedup
        case Scoring.CHECKED:
            return (0.0, 0.0) if is_flagged(grade) else (grade.credit, speedup)
        case _:
            raise ValueError(f"{scoring!r} is not a valid Scoring")


def _metrics(grades: Sequence[EffiSqlGrade], scoring: Scoring) -> Metrics:
    scores = [_score(grade, scoring) for grade in grades]
    speedups = [speedup for _, speedup in scores if speedup > 0]
    covered = sum(speedup >= _COVERED_SPEEDUP for _, speedup in scores)
    return Metrics(
        rves=float(100 * _mean(credit for credit, _ in scores)),
        coverage_2x=100 * covered / len(scores),
        gms=statistics.geometric_mean(speedups) if speedups else None,
    )


def _seed_credits(table: GradeTable, scoring: Scoring) -> dict[str, list[float]]:
    return {
        task: [_score(grade, scoring)[0] for grade in seeds.values()]
        for task, seeds in table.grades.items()
    }


def _mean_credit(table: GradeTable, scoring: Scoring) -> Fraction:
    """Return the mean credit over every task and seed, which is the mean of the seed means."""
    return _mean(
        _score(grade, scoring)[0] for seeds in table.grades.values() for grade in seeds.values()
    )


def _mean(credits: Iterable[float]) -> Fraction:
    """Return the exact mean of tier credits, which are binary fractions such as 0.75."""
    exact = [Fraction(credit) for credit in credits]
    return sum(exact, Fraction(0)) / len(exact)


def _paired_credits(
    best: GradeTable, candidate: GradeTable, scoring: Scoring
) -> tuple[dict[str, float], dict[str, float]]:
    only = {
        "best": best.grades.keys() - candidate.grades.keys(),
        "candidate": candidate.grades.keys() - best.grades.keys(),
    }
    if any(only.values()):
        sides = "; ".join(
            f"only the {side} has {_names(tasks)}" for side, tasks in only.items() if tasks
        )
        raise ValueError(f"the best and the candidate need the same tasks, but {sides}")
    return task_credits(best, scoring=scoring), task_credits(candidate, scoring=scoring)


def _copy(grade: EffiSqlGrade) -> EffiSqlGrade:
    copied = EffiSqlGrade()
    copied.CopyFrom(grade)
    return copied


def _names(tasks: Iterable[str]) -> str:
    """Return up to five of ``tasks``, sorted and quoted, and how many more there are."""
    ordered = sorted(tasks)
    named = ", ".join(map(repr, ordered[:_NAMED_TASKS]))
    rest = len(ordered) - _NAMED_TASKS
    return f"{named} and {rest} more" if rest > 0 else named


def _rves(credits: Sequence[float]) -> float:
    return 100 * statistics.fmean(credits)


def _format(value: MeanStd, digits: int) -> str:
    mean = f"{value.mean:.{digits}f}"
    return mean if value.std is None else f"{mean} ± {value.std:.{digits}f}"
