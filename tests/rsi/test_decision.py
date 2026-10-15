"""Tests for goaly.rsi.decision: the Effi-SQL decision rules of spec §8.3."""

from collections.abc import Mapping, Sequence

import pytest
from pydantic import ValidationError

from goaly.contracts.gen.experiment_pb2 import (
    REJECTION_REASON_COST_GUARDRAIL,
    REJECTION_REASON_EFFICIENCY_CAP,
    REJECTION_REASON_HARD_CODED,
    REJECTION_REASON_INSUFFICIENT_GAIN,
    REJECTION_REASON_NO_SCREEN_GAIN,
    REJECTION_REASON_NOT_CONFIRMED,
    REJECTION_REASON_NOT_SIGNIFICANT,
    REJECTION_REASON_REGRESSION,
    DecisionThresholds,
    EffiSqlThresholds,
)
from goaly.contracts.gen.llm_call_pb2 import Usage
from goaly.contracts.gen.result_pb2 import TaskResult
from goaly.contracts.gen.task_pb2 import (
    GRADE_STATUS_GRADED,
    SPLIT_TRAIN,
    SPLIT_VAL,
    EffiSqlGrade,
    Grade,
)
from goaly.rsi.decision import (
    CostStats,
    DecisionConfig,
    Measured,
    confirm,
    describe,
    rank,
    screen,
    validate,
)

CONFIG = DecisionConfig()
VAL_TASKS = [f"{n}" for n in range(1, 31)]
_SPEEDUP = {0.25: 0.6, 0.5: 1.0, 0.75: 1.5, 1.0: 3.0}


def result(
    task: str,
    seed: int,
    credit: float,
    *,
    run_id: str,
    tokens: int = 1000,
    cost: float | None = 0.01,
    wall: float = 10.0,
    flagged: bool = False,
) -> TaskResult:
    effisql = EffiSqlGrade(credit=credit, matched=credit > 0 or flagged)
    if effisql.matched:
        effisql.speedup = _SPEEDUP.get(credit, 3.0)
    if flagged:
        effisql.alt_matched = False
        effisql.credit = 0.0
    usage = Usage(input_tokens=tokens // 2, output_tokens=tokens - tokens // 2)
    if cost is not None:
        usage.cost_usd = cost
    return TaskResult(
        run_id=run_id,
        instance_id=task,
        seed=seed,
        grade=Grade(status=GRADE_STATUS_GRADED, effisql=effisql),
        wall_time_s=wall,
        usage=usage,
    )


def measured(
    credits: Mapping[str, Sequence[float]],
    *,
    run_id: str = "run",
    seeds: Sequence[int] | None = None,
    flagged: frozenset[str] = frozenset(),
    tokens: int = 1000,
    cost: float | None = 0.01,
    wall: float = 10.0,
) -> Measured:
    """A version whose task ``t`` scores ``credits[t][i]`` in seed ``seeds[i]``."""
    results = [
        result(
            task,
            index if seeds is None else seeds[index],
            credit,
            run_id=run_id,
            tokens=tokens,
            cost=cost,
            wall=wall,
            flagged=task in flagged,
        )
        for task, values in credits.items()
        for index, credit in enumerate(values)
    ]
    return Measured.from_results(results)


def val(changes: Mapping[str, float] | None = None, *, seeds: int = 1) -> dict[str, list[float]]:
    """Every val task at credit 0.5 in each seed, except the tasks ``changes`` names."""
    table = {task: [0.5] * seeds for task in VAL_TASKS}
    for task, credit in (changes or {}).items():
        table[task] = [credit] * seeds
    return table


BEST = measured(val(seeds=2), run_id="best", seeds=[0, 1])
V0 = measured(val(seeds=2), run_id="v0", seeds=[0, 1])
# Four tasks go from 0.5 to 1.0: ΔR-VES = 2/30 = 6.7 points, 4 wins and no loss, p = 1/16.
FOUR_WINS = {"1": 1.0, "2": 1.0, "3": 1.0, "4": 1.0}


def test_a_clear_gain_passes_validation_and_records_what_it_used() -> None:
    verdict = validate(
        SPLIT_VAL, BEST, measured(val(FOUR_WINS), run_id="cand"), v0=V0, config=CONFIG
    )
    assert verdict.passed
    phase = verdict.phase
    assert phase.split == SPLIT_VAL
    assert (phase.task_count, phase.seed_count) == (30, 1)
    assert phase.delta == pytest.approx(2 / 30)
    assert (phase.wins, phase.losses, phase.regressions) == (4, 0, 0)
    assert phase.sign_test_p == pytest.approx(1 / 16)
    assert phase.effective_min_delta == pytest.approx(0.05)
    assert phase.token_ratio_vs_best == pytest.approx(1.0)
    assert phase.cost_ratio_vs_v0 == pytest.approx(1.0)
    assert list(phase.candidate_run_ids) == ["cand"]
    assert list(phase.best_run_ids) == ["best"]
    assert list(phase.v0_run_ids) == ["v0"]
    assert not verdict.guardrail


def test_a_gain_of_exactly_five_points_meets_the_threshold() -> None:
    # Six tasks from 0.5 to 0.75: exactly 1.5 credits over 30 tasks, with p = 1/64.
    changes = {f"{n}": 0.75 for n in range(1, 7)}
    verdict = validate(SPLIT_VAL, BEST, measured(val(changes)), v0=V0, config=CONFIG)
    assert verdict.phase.delta == 0.05
    assert verdict.passed


def test_a_positive_net_gain_with_too_many_regressions_is_rejected() -> None:
    best = measured(val({"29": 1.0, "30": 1.0}, seeds=2), seeds=[0, 1])
    # Eight tasks from 0.5 to 1.0, but two solved tasks drop to 0: Δ = 20 points, p = 0.055.
    gains = {f"{n}": 1.0 for n in range(1, 9)}
    candidate = measured(val({**gains, "29": 0.0, "30": 0.0}))
    verdict = validate(SPLIT_VAL, best, candidate, v0=V0, config=CONFIG)
    assert verdict.phase.delta > 0.05
    assert verdict.phase.sign_test_p <= 0.1
    assert verdict.phase.regressions == 2
    assert verdict.reasons == (REJECTION_REASON_REGRESSION,)


def test_one_regression_is_allowed() -> None:
    best = measured(val({"30": 1.0}, seeds=2), seeds=[0, 1])
    gains = {f"{n}": 1.0 for n in range(1, 9)}
    verdict = validate(SPLIT_VAL, best, measured(val({**gains, "30": 0.25})), v0=V0, config=CONFIG)
    assert verdict.phase.regressions == 1
    assert verdict.passed


def test_a_small_gain_is_insufficient_and_too_few_wins_are_not_significant() -> None:
    # Three tasks from 0.5 to 0.75: 2.5 points, and p = 1/8.
    changes = {"1": 0.75, "2": 0.75, "3": 0.75}
    verdict = validate(SPLIT_VAL, BEST, measured(val(changes)), v0=V0, config=CONFIG)
    assert verdict.reasons == (REJECTION_REASON_INSUFFICIENT_GAIN, REJECTION_REASON_NOT_SIGNIFICANT)


def test_a_gain_from_three_tasks_is_not_significant() -> None:
    # Three tasks from 0 to 1.0: 6.7 points, but p = 1/8.
    best = measured(val({"1": 0.0, "2": 0.0, "3": 0.0}, seeds=2), seeds=[0, 1])
    candidate = measured(val({"1": 1.0, "2": 1.0, "3": 1.0}))
    verdict = validate(SPLIT_VAL, best, candidate, v0=V0, config=CONFIG)
    assert verdict.reasons == (REJECTION_REASON_NOT_SIGNIFICANT,)


def test_a_gain_that_passes_validation_but_fails_confirmation_is_rejected() -> None:
    validation = measured(val(FOUR_WINS), run_id="val")
    assert validate(SPLIT_VAL, BEST, validation, v0=V0, config=CONFIG).passed
    # The confirmation seed keeps the four tasks at 0.5 and drops another to 0: over both seeds
    # the gain is (4 * 0.25 - 0.25) / 30 = 2.5 points, below 3.
    both = val(FOUR_WINS, seeds=2)
    for task in ("1", "2", "3", "4"):
        both[task] = [1.0, 0.5]
    both["5"] = [0.5, 0.0]
    candidate = measured(both, run_id="val", seeds=[0, 1])
    verdict = confirm(SPLIT_VAL, BEST, candidate, v0=V0, config=CONFIG, guardrail=False)
    assert verdict.phase.seed_count == 2
    assert verdict.phase.delta == pytest.approx(0.75 / 30)
    assert verdict.phase.effective_min_delta == pytest.approx(0.03)
    assert verdict.reasons == (REJECTION_REASON_NOT_CONFIRMED,)


def test_a_gain_that_holds_over_both_seeds_is_confirmed() -> None:
    both = val(seeds=2)
    for task in ("1", "2", "3", "4"):
        both[task] = [1.0, 0.75]
    verdict = confirm(
        SPLIT_VAL, BEST, measured(both, seeds=[0, 1]), v0=V0, config=CONFIG, guardrail=False
    )
    assert verdict.phase.delta == pytest.approx(1.5 / 30)
    assert verdict.passed


@pytest.mark.parametrize("stat", ["tokens", "wall"])
def test_the_cost_guardrail_doubles_the_gain_threshold(stat: str) -> None:
    tokens, wall = (1600, 10.0) if stat == "tokens" else (1000, 16.0)
    candidate = measured(val(FOUR_WINS), tokens=tokens, wall=wall)
    verdict = validate(SPLIT_VAL, BEST, candidate, v0=V0, config=CONFIG)
    assert verdict.guardrail
    assert verdict.phase.effective_min_delta == pytest.approx(0.10)
    assert verdict.reasons == (REJECTION_REASON_COST_GUARDRAIL,)
    # Eight wins, 13.3 points, clear the doubled threshold.
    gains = {f"{n}": 1.0 for n in range(1, 9)}
    assert validate(
        SPLIT_VAL, BEST, measured(val(gains), tokens=tokens, wall=wall), v0=V0, config=CONFIG
    ).passed


def test_the_cost_guardrail_doubles_the_confirmation_threshold() -> None:
    # Over both seeds four tasks gain 0.375 each: 5 points, above 3 but below the doubled 6.
    both = val(seeds=2)
    for task in ("1", "2", "3", "4"):
        both[task] = [1.0, 0.75]
    candidate = measured(both, seeds=[0, 1])
    carried = confirm(SPLIT_VAL, BEST, candidate, v0=V0, config=CONFIG, guardrail=True)
    assert carried.phase.effective_min_delta == pytest.approx(0.06)
    assert carried.reasons == (REJECTION_REASON_NOT_CONFIRMED,)
    # The guardrail also trips on the confirmation's own numbers.
    heavy = measured(both, seeds=[0, 1], tokens=2000)
    own = confirm(SPLIT_VAL, BEST, heavy, v0=V0, config=CONFIG, guardrail=False)
    assert own.guardrail
    assert own.reasons == (REJECTION_REASON_NOT_CONFIRMED,)


def test_a_ratio_at_the_guardrail_does_not_trip_it() -> None:
    candidate = measured(val(FOUR_WINS), tokens=1500, wall=15.0)
    verdict = validate(SPLIT_VAL, BEST, candidate, v0=V0, config=CONFIG)
    assert not verdict.guardrail
    assert verdict.passed


@pytest.mark.parametrize(("cost", "wall", "field"), [(0.031, 10.0, "cost"), (0.01, 30.5, "wall")])
def test_the_efficiency_cap_rejects_a_candidate_whatever_its_gain(
    cost: float, wall: float, field: str
) -> None:
    gains = {f"{n}": 1.0 for n in range(1, 21)}  # 33 points.
    candidate = measured(val(gains), cost=cost, wall=wall)
    verdict = validate(SPLIT_VAL, BEST, candidate, v0=V0, config=CONFIG)
    assert REJECTION_REASON_EFFICIENCY_CAP in verdict.reasons
    # Wall time trips the guardrail too, but 33 points clear the doubled threshold.
    assert verdict.reasons == (REJECTION_REASON_EFFICIENCY_CAP,)
    ratio = verdict.phase.cost_ratio_vs_v0 if field == "cost" else verdict.phase.time_ratio_vs_v0
    assert ratio == pytest.approx(3.1 if field == "cost" else 3.05)
    both = {task: [credit[0], credit[0]] for task, credit in val(gains).items()}
    twice = measured(both, seeds=[0, 1], cost=cost, wall=wall)
    confirmed = confirm(SPLIT_VAL, BEST, twice, v0=V0, config=CONFIG, guardrail=False)
    assert confirmed.reasons == (REJECTION_REASON_EFFICIENCY_CAP,)


def test_a_flagged_prediction_fails_the_candidate_in_every_phase() -> None:
    gains = {f"{n}": 1.0 for n in range(1, 21)}
    candidate = measured(val(gains), flagged=frozenset({"7"}))
    screened = screen(SPLIT_TRAIN, BEST, candidate)
    assert screened.reasons == (REJECTION_REASON_HARD_CODED,)
    assert list(screened.phase.flagged_instance_ids) == ["7"]
    validated = validate(SPLIT_VAL, BEST, candidate, v0=V0, config=CONFIG)
    assert validated.reasons == (REJECTION_REASON_HARD_CODED,)
    both = {task: [credit[0], credit[0]] for task, credit in val(gains).items()}
    twice = measured(both, seeds=[0, 1], flagged=frozenset({"7"}))
    confirmed = confirm(SPLIT_VAL, BEST, twice, v0=V0, config=CONFIG, guardrail=False)
    assert confirmed.reasons == (REJECTION_REASON_HARD_CODED,)


def test_the_screen_needs_a_net_gain() -> None:
    tasks = {"1": [0.0], "2": [1.0]}
    best = measured(tasks)
    assert screen(SPLIT_TRAIN, best, measured(tasks)).reasons == (REJECTION_REASON_NO_SCREEN_GAIN,)
    worse = measured({"1": [0.25], "2": [0.5]})
    assert screen(SPLIT_TRAIN, best, worse).reasons == (REJECTION_REASON_NO_SCREEN_GAIN,)
    better = screen(SPLIT_TRAIN, best, measured({"1": [0.25], "2": [1.0]}))
    assert better.passed
    assert not better.phase.HasField("effective_min_delta")
    assert not better.phase.HasField("cost_ratio_vs_v0")
    assert not better.phase.v0_run_ids
    assert "needed" not in describe("screen", better)


def test_rank_picks_the_largest_gain_and_the_first_of_equals() -> None:
    assert rank({"a": 0.06, "b": 0.10, "c": 0.07}) == ("b", ("a", "c"))
    assert rank({"a": 0.06, "b": 0.06}) == ("a", ("b",))
    assert rank({"a": 0.06}) == ("a", ())
    with pytest.raises(ValueError, match="at least one"):
        rank({})


def test_describe_gives_aggregates_and_never_a_task_id() -> None:
    candidate = measured(val({**FOUR_WINS, "27": 0.0}), flagged=frozenset({"28"}), tokens=1600)
    verdict = validate(SPLIT_VAL, BEST, candidate, v0=V0, config=CONFIG)
    text = describe("validation", verdict)
    assert text.startswith("validation: ΔR-VES")
    assert "1 flagged tasks" in text
    assert "cost guardrail tripped" in text
    assert "REJECTION_REASON_HARD_CODED" in text
    assert "28" not in text.replace("1.28", "")  # No ID, though a ratio may hold the digits.
    clean = validate(SPLIT_VAL, BEST, measured(val(FOUR_WINS)), v0=V0, config=CONFIG)
    assert describe("validation", clean).endswith("passed")


def test_thresholds_come_from_config() -> None:
    config = DecisionConfig(min_delta=0.1, confirm_min_delta=0.04, max_regressions=0)
    assert config.thresholds() == DecisionThresholds(
        cost_guardrail_ratio=1.5,
        efficiency_cap_ratio=3.0,
        effisql=EffiSqlThresholds(
            min_delta=0.1, sign_test_max_p=0.1, max_regressions=0, confirm_min_delta=0.04
        ),
    )
    verdict = validate(SPLIT_VAL, BEST, measured(val(FOUR_WINS)), v0=V0, config=config)
    assert verdict.phase.effective_min_delta == pytest.approx(0.1)
    assert verdict.reasons == (REJECTION_REASON_INSUFFICIENT_GAIN,)
    with pytest.raises(ValidationError):
        DecisionConfig(min_delta=-0.1)
    with pytest.raises(ValidationError):
        DecisionConfig(efficiency_cap_ratio=float("inf"))
    with pytest.raises(ValidationError):
        DecisionConfig(unknown=1)  # type: ignore[call-arg]  # Rejected: extra fields are forbidden.


def test_cost_stats_average_per_task_and_seed() -> None:
    results = [
        result("1", 0, 1.0, run_id="r", tokens=1000, cost=0.02, wall=4.0),
        result("1", 1, 1.0, run_id="r", tokens=3000, cost=0.04, wall=8.0),
    ]
    stats = CostStats.from_results(results)
    assert (stats.mean_tokens, stats.mean_wall_time_s) == (2000.0, 6.0)
    assert stats.mean_cost_usd == pytest.approx(0.03)
    results.append(result("2", 0, 1.0, run_id="r", cost=None))
    assert CostStats.from_results(results).mean_cost_usd is None
    with pytest.raises(ValueError, match="at least one"):
        CostStats.from_results([])


def test_a_ratio_without_a_basis_is_left_unset() -> None:
    best = measured(val(seeds=2), seeds=[0, 1], tokens=0, wall=0.0)
    v0 = measured(val(seeds=2), seeds=[0, 1], cost=None)
    candidate = measured(val(FOUR_WINS), tokens=5000, wall=99.0, cost=1.0)
    verdict = validate(SPLIT_VAL, best, candidate, v0=v0, config=CONFIG)
    for field in ("token_ratio_vs_best", "time_ratio_vs_best", "cost_ratio_vs_v0"):
        assert not verdict.phase.HasField(field)
    assert verdict.phase.time_ratio_vs_v0 == pytest.approx(9.9)
    assert verdict.reasons == (REJECTION_REASON_EFFICIENCY_CAP,)


def test_cost_stats_combine_weighted_by_count() -> None:
    first = CostStats(1000.0, 0.01, 10.0, count=30)
    second = CostStats(4000.0, 0.04, 40.0, count=10)
    both = CostStats.combine([first, second])
    assert both.count == 40
    assert both.mean_tokens == pytest.approx(1750.0)
    assert both.mean_cost_usd == pytest.approx(0.0175)
    assert both.mean_wall_time_s == pytest.approx(17.5)
    assert CostStats.combine([first, CostStats(1.0, None, 1.0)]).mean_cost_usd is None
    with pytest.raises(ValueError, match="at least one"):
        CostStats.combine([])


def test_measured_takes_the_evaluators_stats_over_the_results_own() -> None:
    results = [result("1", 0, 1.0, run_id="r", tokens=999, wall=99.0)]
    stats = CostStats(10.0, 0.5, 1.0)
    assert Measured.from_results(results, stats).stats is stats
    assert Measured.from_results(results).stats.mean_tokens == 999.0


def test_measured_names_each_run_once_in_order() -> None:
    results = [
        result("1", 0, 1.0, run_id="b"),
        result("2", 0, 1.0, run_id="a"),
        result("3", 0, 1.0, run_id="b"),
    ]
    assert Measured.from_results(results).run_ids == ("b", "a")
