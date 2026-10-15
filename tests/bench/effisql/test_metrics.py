"""Tests for goaly.bench.effisql.metrics, on constructed grades."""

import hashlib
import json
import math
import os
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from goaly.bench.effisql.metrics import (
    SPEEDUP_CAP,
    Comparison,
    GradeTable,
    Metrics,
    Scoring,
    by_difficulty,
    compare,
    delta_rves_ci,
    flagged_tasks,
    is_flagged,
    markdown_table,
    seed_metrics,
    solved,
    stably_solved,
    summarize,
    task_credits,
    tier_credit,
)
from goaly.contracts.gen.result_pb2 import TaskResult
from goaly.contracts.gen.task_pb2 import (
    GRADE_STATUS_GRADED,
    GRADE_STATUS_GRADER_ERROR,
    GRADE_STATUS_PREDICTION_FAILED,
    GRADE_STATUS_TIMEOUT,
    GRADE_STATUS_UNSPECIFIED,
    EffiSqlGrade,
    Grade,
    GradeStatus,
    SweGrade,
)
from goaly.harness.core.stats import Interval, MeanStd, SignTest

OFFICIAL = Scoring.OFFICIAL
CHECKED = Scoring.CHECKED


def grade(
    credit: float = 0.0,
    *,
    speedup: float | None = None,
    matched: bool | None = None,
    alt_matched: bool | None = None,
    literal_result: bool | None = None,
) -> EffiSqlGrade:
    """An Effi-SQL grade. It is matched if it has a speedup, unless ``matched`` says otherwise."""
    graded = EffiSqlGrade(
        credit=credit, matched=speedup is not None if matched is None else matched
    )
    if speedup is not None:
        graded.speedup = speedup
    if alt_matched is not None:
        graded.alt_matched = alt_matched
    if literal_result is not None:
        graded.literal_result = literal_result
    return graded


def timed(
    speedup: float, *, alt_matched: bool | None = None, literal_result: bool | None = None
) -> EffiSqlGrade:
    """A matched, timed grade with g(τ) as its credit, as the official evaluator scores it."""
    return grade(
        tier_credit(speedup),
        speedup=speedup,
        alt_matched=alt_matched,
        literal_result=literal_result,
    )


def hard_coded() -> EffiSqlGrade:
    """A constant answer, matched at the speedup cap and flagged by the robustness check.

    The grader's credit is then 0. The official evaluator would give it g(1000) = 1.
    """
    return grade(0.0, speedup=SPEEDUP_CAP, alt_matched=False)


def literal() -> EffiSqlGrade:
    """A constant answer over region or nation, which have the same rows at every scale.

    It passes the robustness check, but the plan check flags it as a literal result, so the
    grader's credit is 0. The official evaluator would give it g(1000) = 1.
    """
    return grade(0.0, speedup=SPEEDUP_CAP, alt_matched=True, literal_result=True)


def unmatched() -> EffiSqlGrade:
    return grade(0.0)


def timing_failed() -> EffiSqlGrade:
    """Matched, but the timing failed: τ = 0 and credit 0."""
    return grade(0.0, matched=True)


def one_seed(*grades: EffiSqlGrade) -> GradeTable:
    """Tasks "t0", "t1", ... with one grade each, in seed 0."""
    return GradeTable({f"t{index}": {0: graded} for index, graded in enumerate(grades)})


# The lowest speedup of each credit tier.
TIER_SPEEDUPS = {1.0: 2.0, 0.75: 1.2, 0.5: 1.0, 0.25: 0.5}


def seeded(credits: Mapping[str, Sequence[float]]) -> GradeTable:
    """Tasks with these credits in seeds 0, 1, ...: timed at the tier's speedup, or unmatched."""
    return GradeTable(
        {
            task: {
                seed: timed(TIER_SPEEDUPS[credit]) if credit else unmatched()
                for seed, credit in enumerate(values)
            }
            for task, values in credits.items()
        }
    )


def result(
    task: str,
    seed: int,
    graded: EffiSqlGrade,
    *,
    run: str = "run-1",
    status: GradeStatus.ValueType = GRADE_STATUS_GRADED,
) -> TaskResult:
    return TaskResult(
        run_id=run,
        instance_id=task,
        seed=seed,
        grade=Grade(instance_id=task, score=graded.credit, status=status, effisql=graded),
    )


def only_seed(table: GradeTable, *, scoring: Scoring) -> Metrics:
    (metrics,) = seed_metrics(table, scoring=scoring).values()
    return metrics


# tier_credit: g(τ).


@pytest.mark.parametrize(
    ("speedup", "credit"),
    [
        (0.0, 0.0),
        (0.4999, 0.0),
        (0.5, 0.25),
        (0.9999, 0.25),
        (1.0, 0.5),
        (1.1999, 0.5),
        (1.2, 0.75),
        (1.9999, 0.75),
        (2.0, 1.0),
        (SPEEDUP_CAP, 1.0),
        (155_857.4, 1.0),
    ],
)
def test_tier_credit_at_each_boundary(speedup: float, credit: float) -> None:
    assert tier_credit(speedup) == credit


@pytest.mark.parametrize("speedup", [-0.5, math.nan])
def test_tier_credit_rejects_a_speedup_that_cannot_be(speedup: float) -> None:
    with pytest.raises(ValueError, match="speedup"):
        tier_credit(speedup)


# is_flagged.


@pytest.mark.parametrize(
    ("graded", "flagged"),
    [
        pytest.param(timed(4.0), False, id="no check ran"),
        pytest.param(timed(4.0, alt_matched=True), False, id="robust"),
        pytest.param(timed(4.0, alt_matched=False), True, id="hard_coded"),
        pytest.param(timed(4.0, alt_matched=True, literal_result=False), False, id="both passed"),
        pytest.param(timed(4.0, literal_result=True), True, id="literal_result"),
        pytest.param(timed(4.0, alt_matched=True, literal_result=True), True, id="robust literal"),
        pytest.param(timed(4.0, alt_matched=False, literal_result=True), True, id="both flags"),
    ],
)
def test_is_flagged(graded: EffiSqlGrade, flagged: bool) -> None:
    assert is_flagged(graded) is flagged


# GradeTable.


def test_from_results_builds_every_task_in_every_seed() -> None:
    a1, b1, a0, b0 = timed(4.0), timed(1.0), timed(1.5), unmatched()

    table = GradeTable.from_results(
        [result("b", 1, b1), result("a", 1, a1), result("a", 0, a0), result("b", 0, b0)]
    )

    assert table.tasks == ("b", "a")
    assert table.seeds == (0, 1)
    assert table.grades == {"a": {0: a0, 1: a1}, "b": {0: b0, 1: b1}}


def test_from_results_keeps_the_last_line_of_a_run_for_a_task_and_seed() -> None:
    first, last = timed(1.0), timed(4.0)

    table = GradeTable.from_results([result("a", 0, first), result("a", 0, last)])

    assert table.grades == {"a": {0: last}}


def test_from_results_takes_a_regrade_after_a_grader_error() -> None:
    regraded = timed(4.0)

    table = GradeTable.from_results(
        [result("a", 0, unmatched(), status=GRADE_STATUS_GRADER_ERROR), result("a", 0, regraded)]
    )

    assert table.grades == {"a": {0: regraded}}


def test_from_results_rejects_a_task_and_seed_from_two_runs() -> None:
    with pytest.raises(ValueError, match="task 'a', seed 0, has results from two runs"):
        GradeTable.from_results([result("a", 0, timed(1.0)), result("a", 0, timed(1.0), run="r2")])


@pytest.mark.parametrize("status", [GRADE_STATUS_PREDICTION_FAILED, GRADE_STATUS_TIMEOUT])
def test_from_results_scores_a_failed_prediction(status: GradeStatus.ValueType) -> None:
    table = GradeTable.from_results([result("a", 0, unmatched(), status=status)])

    assert only_seed(table, scoring=CHECKED) == Metrics(rves=0.0, coverage_2x=0.0, gms=None)


@pytest.mark.parametrize(
    ("status", "message"),
    [
        (GRADE_STATUS_GRADER_ERROR, "GRADE_STATUS_GRADER_ERROR: regrade it"),
        (GRADE_STATUS_UNSPECIFIED, "GRADE_STATUS_UNSPECIFIED: regrade it"),
    ],
)
def test_from_results_rejects_a_grade_that_means_nothing(
    status: GradeStatus.ValueType, message: str
) -> None:
    with pytest.raises(ValueError, match=f"task 'a', seed 0, is {message}"):
        GradeTable.from_results([result("a", 0, unmatched(), status=status)])


def test_from_results_rejects_a_result_without_a_grade() -> None:
    with pytest.raises(ValueError, match="task 'a', seed 0, has no grade"):
        GradeTable.from_results([TaskResult(run_id="run-1", instance_id="a", seed=0)])


def test_from_results_rejects_a_grade_of_another_benchmark() -> None:
    swe = TaskResult(
        run_id="run-1",
        instance_id="a",
        seed=0,
        grade=Grade(instance_id="a", status=GRADE_STATUS_GRADED, swe=SweGrade()),
    )

    with pytest.raises(ValueError, match="task 'a', seed 0, has no Effi-SQL grade"):
        GradeTable.from_results([swe])


def test_a_table_needs_a_task() -> None:
    with pytest.raises(ValueError, match="no tasks"):
        GradeTable.from_results([])


def test_a_table_needs_every_task_in_the_same_seeds() -> None:
    with pytest.raises(ValueError, match=r"task 'b' has seeds \[1\], but task 'a' has \[0, 1\]"):
        GradeTable({"a": {0: timed(1.0), 1: timed(1.0)}, "b": {1: timed(1.0)}})


def test_a_table_needs_a_seed() -> None:
    with pytest.raises(ValueError, match="task 'a' has no seeds"):
        GradeTable({"a": {}})


@pytest.mark.parametrize("credit", [0.6, 75.0, -0.25, math.nan])
def test_a_table_rejects_a_credit_that_is_no_tier(credit: float) -> None:
    with pytest.raises(ValueError, match="task 'a', seed 0, has credit"):
        GradeTable({"a": {0: grade(credit, speedup=2.0)}})


def test_a_table_rejects_credit_for_an_unmatched_task() -> None:
    with pytest.raises(ValueError, match=r"task 'a', seed 0, is not matched but has credit 0\.5"):
        GradeTable({"a": {0: grade(0.5, matched=False)}})


@pytest.mark.parametrize("speedup", [-1.0, math.nan])
def test_a_table_rejects_a_match_with_a_speedup_that_cannot_be(speedup: float) -> None:
    with pytest.raises(ValueError, match="task 'a', seed 0, has speedup"):
        GradeTable({"a": {0: grade(1.0, speedup=speedup)}})


# The metrics of one seed (spec §4.1).


@pytest.mark.parametrize("scoring", [OFFICIAL, CHECKED])
def test_rves_is_100_times_the_mean_credit_over_every_task(scoring: Scoring) -> None:
    table = one_seed(timed(2.0), timed(1.2), timed(1.0), timed(0.5), timed(0.1), unmatched())

    assert only_seed(table, scoring=scoring).rves == 100 * 2.5 / 6


def test_checked_credit_is_the_graders_and_official_credit_g_of_the_speedup() -> None:
    # The two disagree here only to show which number each scoring reads.
    table = one_seed(grade(0.5, speedup=3.0))

    assert only_seed(table, scoring=CHECKED).rves == 50.0
    assert only_seed(table, scoring=OFFICIAL).rves == 100.0


def test_official_credit_is_g_of_the_capped_speedup_of_a_match_and_0_otherwise() -> None:
    # g(1000) for the first; no speedup for a match whose timing failed; no match for the rest.
    unmatched_but_timed = grade(0.0, speedup=1.5, matched=False)
    table = one_seed(hard_coded(), timing_failed(), unmatched(), unmatched_but_timed)

    assert only_seed(table, scoring=OFFICIAL).rves == 25.0


def test_coverage_counts_matched_tasks_with_a_speedup_of_2_over_every_task() -> None:
    table = one_seed(timed(2.0), timed(1.9999), timed(5.0), timing_failed(), unmatched())

    assert only_seed(table, scoring=OFFICIAL).coverage_2x == 40.0


def test_coverage_never_counts_an_unmatched_task() -> None:
    assert only_seed(one_seed(grade(0.0, speedup=5.0, matched=False)), scoring=OFFICIAL) == (
        Metrics(rves=0.0, coverage_2x=0.0, gms=None)
    )


def test_gms_is_the_geometric_mean_over_matched_tasks_with_a_speedup() -> None:
    zero = grade(0.0, speedup=0.0)  # A failed timing recorded as τ = 0.
    table = one_seed(timed(4.0), timed(0.25), timing_failed(), zero, unmatched())

    assert only_seed(table, scoring=OFFICIAL).gms == pytest.approx(1.0)


def test_gms_caps_the_speedup_at_1000() -> None:
    table = one_seed(timed(4_000.0), timed(10.0))

    assert only_seed(table, scoring=OFFICIAL).gms == pytest.approx(math.sqrt(SPEEDUP_CAP * 10))


def test_gms_is_none_without_a_matched_task_with_a_speedup() -> None:
    table = one_seed(timing_failed(), unmatched())

    assert only_seed(table, scoring=OFFICIAL) == Metrics(rves=0.0, coverage_2x=0.0, gms=None)


# Official and checked numbers.


@pytest.mark.parametrize("flagged", [hard_coded(), literal()], ids=["hard_coded", "literal"])
def test_checked_numbers_count_a_flagged_task_as_unmatched(flagged: EffiSqlGrade) -> None:
    # Officially the constant answer is a match at τ = 1000; checked, it is no match at all.
    table = one_seed(flagged, timed(4.0))

    official = only_seed(table, scoring=OFFICIAL)
    checked = only_seed(table, scoring=CHECKED)

    assert (official.rves, official.coverage_2x) == (100.0, 100.0)
    assert official.gms == pytest.approx(math.sqrt(SPEEDUP_CAP * 4))
    assert (checked.rves, checked.coverage_2x) == (50.0, 50.0)
    assert checked.gms == pytest.approx(4.0)


def test_checked_numbers_give_a_flagged_task_no_credit_even_if_its_grade_has_some() -> None:
    table = one_seed(timed(SPEEDUP_CAP, alt_matched=False), timed(4.0, literal_result=True))

    assert only_seed(table, scoring=CHECKED).rves == 0.0


def test_checked_numbers_keep_a_task_that_passed_both_checks() -> None:
    table = one_seed(timed(4.0, alt_matched=True, literal_result=False), timed(1.0))

    assert only_seed(table, scoring=CHECKED) == only_seed(table, scoring=OFFICIAL)


def test_flagged_tasks_are_those_flagged_in_any_seed() -> None:
    table = GradeTable(
        {
            "a": {0: timed(4.0), 1: hard_coded()},
            "b": {0: timed(4.0, alt_matched=True, literal_result=False), 1: timed(1.0)},
            "c": {0: literal(), 1: timed(4.0)},
        }
    )

    assert flagged_tasks(table) == {"a", "c"}


# Seeds (spec §6.3, §8.3).


def test_task_credit_is_its_mean_over_the_seeds() -> None:
    table = seeded({"a": [1.0, 0.5], "b": [0.25, 0.25], "c": [0.0, 0.75]})

    assert task_credits(table, scoring=OFFICIAL) == {"a": 0.75, "b": 0.25, "c": 0.375}


def test_task_credit_under_the_checked_scoring() -> None:
    table = GradeTable({"a": {0: hard_coded(), 1: timed(4.0)}})

    assert task_credits(table, scoring=OFFICIAL) == {"a": 1.0}
    assert task_credits(table, scoring=CHECKED) == {"a": 0.5}


def test_solved_uses_the_seed_mean_and_stably_solved_every_seed() -> None:
    table = seeded(
        {"a": [1.0, 0.5], "b": [0.75, 1.0], "c": [1.0, 0.25], "d": [0.75, 0.75], "e": [0.5, 0.5]}
    )

    assert solved(table, scoring=OFFICIAL) == {"a", "b", "d"}
    assert stably_solved(table, scoring=OFFICIAL) == {"b", "d"}


@pytest.mark.parametrize("flagged", [hard_coded(), literal()], ids=["hard_coded", "literal"])
def test_a_flagged_task_is_not_solved_under_the_checked_scoring(flagged: EffiSqlGrade) -> None:
    table = GradeTable({"a": {0: flagged}, "b": {0: timed(4.0)}})

    assert solved(table, scoring=OFFICIAL) == stably_solved(table, scoring=OFFICIAL) == {"a", "b"}
    assert solved(table, scoring=CHECKED) == stably_solved(table, scoring=CHECKED) == {"b"}


def test_summary_gives_mean_and_std_over_seeds() -> None:
    table = GradeTable(
        {
            "a": {0: timed(4.0), 1: timed(1.0)},
            "b": {0: timed(1.5), 1: timed(2.5)},
            "c": {0: unmatched(), 1: timed(0.5)},
            "d": {0: timed(1.25), 1: timed(1.25)},
        }
    )

    summary = summarize(table, scoring=OFFICIAL, bootstrap_seed=5)

    # Seed 0: credits 1, 0.75, 0, 0.75; seed 1: 0.5, 1, 0.25, 0.75.
    assert (summary.task_count, summary.seed_count) == (4, 2)
    assert summary.rves == MeanStd(mean=(62.5 + 62.5) / 2, std=0.0)
    assert summary.coverage_2x == MeanStd(mean=25.0, std=0.0)
    seed_gms = [math.prod([4.0, 1.5, 1.25]) ** (1 / 3), math.prod([1.0, 2.5, 0.5, 1.25]) ** (1 / 4)]
    assert summary.gms is not None
    assert summary.gms.mean == pytest.approx(statistics.fmean(seed_gms))
    assert summary.gms.std == pytest.approx(statistics.stdev(seed_gms))
    assert summary.solved_count == 3  # a (0.75), b (0.875) and d (0.75).
    assert summary.stably_solved_count == 2  # b and d: a has 0.5 in seed 1.


def test_summary_of_one_seed_has_no_standard_deviation() -> None:
    table = one_seed(timed(4.0), unmatched())

    summary = summarize(table, scoring=OFFICIAL, bootstrap_seed=0, resamples=10)

    assert summary.rves == MeanStd(mean=50.0, std=None)
    assert summary.gms is not None
    assert summary.gms.mean == pytest.approx(4.0)
    assert summary.gms.std is None


def test_summary_gms_is_none_if_a_seed_has_no_speedup() -> None:
    table = GradeTable({"a": {0: timed(4.0), 1: unmatched()}})

    assert summarize(table, scoring=OFFICIAL, bootstrap_seed=0, resamples=10).gms is None


def test_summary_rves_ci_resamples_tasks_by_their_seed_mean_credit() -> None:
    table = seeded({"a": [1.0, 0.5], "b": [0.25, 0.25], "c": [1.0, 1.0]})

    summary = summarize(table, scoring=OFFICIAL, bootstrap_seed=9, resamples=500)

    # The interval depends only on each task's seed-mean credit.
    assert (
        summary.rves_ci
        == summarize(
            seeded({"a": [0.75], "b": [0.25], "c": [1.0]}),
            scoring=OFFICIAL,
            bootstrap_seed=9,
            resamples=500,
        ).rves_ci
    )
    assert summary.rves_ci.low < summary.rves.mean < summary.rves_ci.high
    # Tasks with equal means, 0.75 each, give a point.
    point = summarize(
        seeded({"a": [0.5, 1.0], "b": [1.0, 0.5]}), scoring=OFFICIAL, bootstrap_seed=9
    )
    assert point.rves_ci == Interval(75.0, 75.0)


# Comparing two versions (spec §8.3).


def test_compare_counts_wins_and_losses_on_seed_mean_credit() -> None:
    best = seeded({"a": [0.5], "b": [1.0], "c": [0.25], "d": [0.75]})
    candidate = seeded({"a": [1.0, 0.5], "b": [1.0, 0.5], "c": [0.25, 0.25], "d": [1.0, 0.75]})

    comparison = compare(best, candidate, scoring=OFFICIAL)

    # a: 0.75 > 0.5; b: 0.75 < 1; c ties; d: 0.875 > 0.75.
    assert comparison.sign == SignTest(wins=2, losses=1, ties=1, p_value=0.5)
    assert comparison.best_score == 2.5 / 4
    assert comparison.candidate_score == 2.625 / 4


def test_compare_delta_is_exact_at_a_decision_threshold() -> None:
    # 30 tasks: the candidate gains six credit tiers in all, ΔR-VES exactly 5 points. Each mean
    # rounded on its own would give 0.049999999999999996, and fail EffiSqlThresholds' default
    # min_delta of 0.05.
    best = seeded({f"t{i}": [0.25 if i < 5 else 0.0] for i in range(30)})
    candidate = seeded({f"t{i}": [[1.0, 1.0, 0.75][i] if i < 3 else 0.0] for i in range(30)})

    comparison = compare(best, candidate, scoring=OFFICIAL)

    assert comparison.delta == 0.05


@pytest.mark.parametrize("flagged", [hard_coded(), literal()], ids=["hard_coded", "literal"])
def test_compare_checked_gives_a_flagged_task_no_credit(flagged: EffiSqlGrade) -> None:
    best = GradeTable({"a": {0: timed(1.0)}, "b": {0: timed(1.0)}})
    candidate = GradeTable({"a": {0: flagged}, "b": {0: timed(1.0)}})

    assert compare(best, candidate, scoring=OFFICIAL).sign.wins == 1
    checked = compare(best, candidate, scoring=CHECKED)
    assert checked.sign == SignTest(wins=0, losses=1, ties=1, p_value=1.0)
    assert checked.delta == -0.25


def test_a_regression_is_solved_under_the_best_and_at_most_025_under_the_candidate() -> None:
    best = seeded({"a": [0.75], "b": [1.0], "c": [1.0], "d": [1.0], "e": [0.5]})
    # c drops to 0.375 and stays above the bar; d's seeds average exactly 0.25; e was unsolved.
    candidate = seeded(
        {"a": [0.25, 0.25], "b": [0.0, 0.0], "c": [0.5, 0.25], "d": [0.5, 0.0], "e": [0.0, 0.0]}
    )

    comparison = compare(best, candidate, scoring=OFFICIAL)

    assert comparison.regressions == {"a", "b", "d"}


def test_compare_with_itself() -> None:
    table = seeded({"a": [1.0], "b": [0.25]})

    assert compare(table, table, scoring=CHECKED) == Comparison(
        best_score=0.625,
        candidate_score=0.625,
        delta=0.0,
        sign=SignTest(wins=0, losses=0, ties=2, p_value=1.0),
        regressions=frozenset(),
    )


def test_compare_needs_the_same_tasks() -> None:
    with pytest.raises(ValueError, match="the same tasks"):
        compare(seeded({"a": [1.0]}), seeded({"b": [1.0]}), scoring=CHECKED)


def test_delta_rves_ci_resamples_paired_tasks() -> None:
    best = seeded({"a": [0.0], "b": [0.25], "c": [0.5], "d": [0.75]})
    better = seeded({"a": [0.25], "b": [0.5], "c": [0.75], "d": [1.0]})

    # Every task gains one tier, so every resample does too: a paired bootstrap gives a point.
    assert delta_rves_ci(best, better, scoring=CHECKED, bootstrap_seed=0) == Interval(25.0, 25.0)
    mixed = seeded({"a": [1.0], "b": [0.25], "c": [0.0], "d": [0.75]})
    interval = delta_rves_ci(best, mixed, scoring=CHECKED, bootstrap_seed=0, resamples=2_000)
    assert interval == delta_rves_ci(
        best, mixed, scoring=CHECKED, bootstrap_seed=0, resamples=2_000
    )
    assert interval.low < 100 * (2.0 - 1.5) / 4 < interval.high


def test_delta_rves_ci_needs_the_same_tasks() -> None:
    with pytest.raises(ValueError, match="the same tasks"):
        delta_rves_ci(seeded({"a": [1.0]}), seeded({"b": [1.0]}), scoring=CHECKED, bootstrap_seed=0)


# Breakdowns and tables.


def test_by_difficulty_splits_the_tasks_in_a_fixed_order() -> None:
    table = seeded({"1": [1.0], "2": [0.5], "3": [0.0], "4": [0.75], "5": [0.25]})
    labels = {"1": "hard", "2": "easy", "3": "hard", "4": "extreme", "5": "medium"}

    groups = by_difficulty(table, labels)

    assert list(groups) == ["easy", "medium", "hard", "extreme"]
    assert {label: group.tasks for label, group in groups.items()} == {
        "easy": ("2",),
        "medium": ("5",),
        "hard": ("1", "3"),
        "extreme": ("4",),
    }
    assert groups["hard"].grades["1"] == table.grades["1"]


def test_by_difficulty_needs_every_tasks_label() -> None:
    with pytest.raises(ValueError, match="task '2' has no difficulty"):
        by_difficulty(seeded({"1": [1.0], "2": [1.0]}), {"1": "easy"})


def test_markdown_table_has_a_line_per_row_and_scoring() -> None:
    v0 = one_seed(timed(1.0), timed(1.0))
    final = GradeTable({task: {0: hard_coded(), 1: hard_coded()} for task in ("a", "b")})

    table = markdown_table({"v0": v0, "final": final}, bootstrap_seed=0, resamples=20)

    assert table == (
        "| | Scoring | Tasks | Seeds | R-VES | 95% CI | Coverage@2x | GMS | Solved"
        " | Stably solved |\n"
        "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|\n"
        "| v0 | official | 2 | 1 | 50.0 | [50.0, 50.0] | 0.0 | 1.00 | 0 | 0 |\n"
        "| v0 | checked | 2 | 1 | 50.0 | [50.0, 50.0] | 0.0 | 1.00 | 0 | 0 |\n"
        "| final | official | 2 | 2 | 100.0 ± 0.0 | [100.0, 100.0] | 100.0 ± 0.0 | 1000.00 ± 0.00"
        " | 2 | 2 |\n"
        "| final | checked | 2 | 2 | 0.0 ± 0.0 | [0.0, 0.0] | 0.0 ± 0.0 | n/a | 0 | 0 |\n"
    )


def test_markdown_table_escapes_a_pipe_in_a_label() -> None:
    table = markdown_table({"a|b": one_seed(timed(1.0))}, bootstrap_seed=0, resamples=1)

    assert "| a\\|b | official |" in table


# The gold scores (spec §4.1), from the dataset itself.

DATA_VARIABLE = "GOALY_EFFISQL_DATA"  # The path of effi-sql-pg.jsonl, which is never committed.
DATA_SHA256 = "2af3bdc9fe6c1197f23e31089660e7ce10fca6956b32e5970a3113f0c87e9d97"


@pytest.mark.integration
def test_gold_scores_from_the_datasets_own_timings() -> None:
    path = os.environ.get(DATA_VARIABLE)
    if not path:
        pytest.skip(f"set {DATA_VARIABLE} to the path of effi-sql-pg.jsonl")
    data = Path(path).read_bytes()
    assert hashlib.sha256(data).hexdigest() == DATA_SHA256, f"{path} is not effi-sql-pg.jsonl"
    tasks = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    # τ = base_time / fast_time, uncapped: the metrics must cap it themselves. A time of 0 counts
    # as 0.001 ms, as the grader counts it.
    speedups = {
        str(task["id"]): task["base_time"] / max(task["fast_time"], 0.001)
        for task in tasks
        if task["db"] == "tpch"
    }
    gold = GradeTable.from_results(
        result(task, 0, timed(speedup)) for task, speedup in speedups.items()
    )

    metrics = only_seed(gold, scoring=CHECKED)

    assert len(speedups) == 238
    assert only_seed(gold, scoring=OFFICIAL) == metrics  # No gold is flagged.
    assert round(metrics.rves, 1) == 88.3
    assert round(metrics.coverage_2x, 1) == 53.4
    assert metrics.gms is not None
    assert round(metrics.gms, 2) == 3.29
    assert round(statistics.geometric_mean(speedups.values()), 2) == 3.55  # Uncapped.
