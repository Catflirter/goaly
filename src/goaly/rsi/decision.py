"""The RSI decision rules (spec §8.2 steps 4-7, §8.3), as pure functions over T07's comparison.

Each phase compares a candidate with the lane's best on the same tasks, through
:func:`goaly.bench.effisql.metrics.compare` on the checked credit, and returns a :class:`Verdict`:
the PhaseResult the ledger records, and every rule the candidate failed, in the order checked.

- :func:`screen`: any prediction flagged ``hard_coded`` or ``literal_result`` fails, and so does
  no net gain (Δ ≤ 0).
- :func:`validate`, in this order: a flagged prediction; the efficiency cap, at most 3x v0's
  mean cost and mean wall time per task; a gain ΔR-VES of at least 5 points, or 10 once the cost
  guardrail tripped (above 1.5x the best's mean tokens or wall time per task); the one-sided
  sign test's p ≤ 0.1; at most 1 regression.
- :func:`confirm`: a flagged prediction; the efficiency cap; a gain of at least 3 points over
  both seeds, or 6 under the guardrail.
- :func:`rank`: of the candidates that passed validation, the largest Δ wins.

A task's credit is its mean over the version's seeds, so the best's table holds all its val
seeds, and the confirmation's candidate table both of the candidate's. The numbers are the ones
config sets (:class:`DecisionConfig`), and every PhaseResult records the threshold it applied.

Tokens, cost and wall time per task are :class:`CostStats`, which the evaluator reports from the
LLM proxy's records and the host's clock: files written in the solve container are untrusted
(spec §6.1, §9; ruling V27). A ratio whose basis is 0 or unknown is left unset, and its rule then
does not apply.

:func:`describe` puts a verdict into words with aggregates only, never a task ID, so the
decision_reason and the lessons drawn from it stay clean of per-task val data (spec §7).
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from goaly.bench.effisql.metrics import GradeTable, Scoring, compare, flagged_tasks
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
    PhaseResult,
    RejectionReason,
)
from goaly.contracts.gen.result_pb2 import TaskResult
from goaly.contracts.gen.task_pb2 import Split

__all__ = [
    "CostStats",
    "DecisionConfig",
    "Measured",
    "Verdict",
    "confirm",
    "describe",
    "rank",
    "screen",
    "validate",
]

_SCORING = Scoring.CHECKED  # RSI decides on the checked numbers (spec §6.3).
_GUARDRAIL_FACTOR = 2.0  # The cost guardrail doubles both Δ thresholds (spec §8.3).

type _Ratio = Literal[
    "token_ratio_vs_best", "time_ratio_vs_best", "cost_ratio_vs_v0", "time_ratio_vs_v0"
]
_VS_BEST: Final[tuple[_Ratio, ...]] = ("token_ratio_vs_best", "time_ratio_vs_best")
_VS_V0: Final[tuple[_Ratio, ...]] = ("cost_ratio_vs_v0", "time_ratio_vs_v0")
_LABELS: Final[dict[_Ratio, str]] = {
    "token_ratio_vs_best": "tokens {:.2f}x the best's",
    "time_ratio_vs_best": "wall time {:.2f}x the best's",
    "cost_ratio_vs_v0": "cost {:.2f}x v0's",
    "time_ratio_vs_v0": "wall time {:.2f}x v0's",
}


class DecisionConfig(BaseModel):
    """The Effi-SQL lane's decision thresholds, from config (spec §8.3). Gains are fractions of
    R-VES / 100, as PhaseResult keeps them: 0.05 is 5 points.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    min_delta: float = Field(default=0.05, ge=0, le=1)
    sign_test_max_p: float = Field(default=0.1, gt=0, le=1)
    max_regressions: int = Field(default=1, ge=0)
    confirm_min_delta: float = Field(default=0.03, ge=0, le=1)
    cost_guardrail_ratio: float = Field(default=1.5, gt=0)
    efficiency_cap_ratio: float = Field(default=3.0, gt=0)

    def thresholds(self) -> DecisionThresholds:
        """The thresholds as ExperimentRecord.thresholds records them."""
        return DecisionThresholds(
            cost_guardrail_ratio=self.cost_guardrail_ratio,
            efficiency_cap_ratio=self.efficiency_cap_ratio,
            effisql=EffiSqlThresholds(
                min_delta=self.min_delta,
                sign_test_max_p=self.sign_test_max_p,
                max_regressions=self.max_regressions,
                confirm_min_delta=self.confirm_min_delta,
            ),
        )


@dataclass(frozen=True, slots=True)
class CostStats:
    """A version's means per task and seed: tokens (input plus output, reasoning included), cost
    in USD, None if unknown, and wall time in seconds; over ``count`` tasks and seeds.

    The evaluator reports them from the proxy's records and the host's clock (V27).
    """

    mean_tokens: float
    mean_cost_usd: float | None
    mean_wall_time_s: float
    count: int = 1

    @classmethod
    def from_results(cls, results: Iterable[TaskResult]) -> "CostStats":
        """Average ``results``' usage and wall time, for results whose numbers the host itself
        measured, as in the one-process mode and the tests; never for a solve container's.
        """
        listed = list(results)
        if not listed:
            raise ValueError("cost statistics need at least one result")
        costs = [r.usage.cost_usd for r in listed if r.usage.HasField("cost_usd")]
        return cls(
            mean_tokens=math.fsum(r.usage.input_tokens + r.usage.output_tokens for r in listed)
            / len(listed),
            mean_cost_usd=math.fsum(costs) / len(listed) if len(costs) == len(listed) else None,
            mean_wall_time_s=math.fsum(r.wall_time_s for r in listed) / len(listed),
            count=len(listed),
        )

    @classmethod
    def combine(cls, parts: Sequence["CostStats"]) -> "CostStats":
        """Average ``parts``, each weighted by its count, as over a version's two val seeds."""
        if not parts:
            raise ValueError("cost statistics need at least one part")
        count = sum(part.count for part in parts)
        costs = [part.mean_cost_usd for part in parts]
        return cls(
            mean_tokens=math.fsum(p.mean_tokens * p.count for p in parts) / count,
            mean_cost_usd=None
            if None in costs
            else math.fsum(p.count * (p.mean_cost_usd or 0.0) for p in parts) / count,
            mean_wall_time_s=math.fsum(p.mean_wall_time_s * p.count for p in parts) / count,
            count=count,
        )


@dataclass(frozen=True, slots=True)
class Measured:
    """A version as one phase measured it: its grades, cost statistics and runs."""

    table: GradeTable
    stats: CostStats
    run_ids: tuple[str, ...]

    @classmethod
    def from_results(
        cls, results: Sequence[TaskResult], stats: CostStats | None = None
    ) -> "Measured":
        """Measure a version from its graded results, which name their runs, and the evaluator's
        ``stats``; without them, from the results' own numbers (CostStats.from_results).
        """
        return cls(
            table=GradeTable.from_results(results),
            stats=CostStats.from_results(results) if stats is None else stats,
            run_ids=tuple(dict.fromkeys(result.run_id for result in results)),
        )


@dataclass(frozen=True, slots=True)
class Verdict:
    """One phase's outcome: its PhaseResult, and the rules the candidate failed, in order."""

    phase: PhaseResult
    reasons: tuple[RejectionReason.ValueType, ...]
    guardrail: bool  # Whether the cost guardrail tripped in this phase.

    @property
    def passed(self) -> bool:
        return not self.reasons


def screen(split: Split.ValueType, best: Measured, candidate: Measured) -> Verdict:
    """Screen ``candidate`` against ``best`` on the same tasks: it needs a net gain, Δ > 0, and
    no flagged prediction (spec §8.2 step 4).
    """
    phase = _phase(split, best, candidate, v0=None)
    reasons: list[RejectionReason.ValueType] = []
    if phase.flagged_instance_ids:
        reasons.append(REJECTION_REASON_HARD_CODED)
    if not phase.delta > 0:
        reasons.append(REJECTION_REASON_NO_SCREEN_GAIN)
    return Verdict(phase, tuple(reasons), guardrail=False)


def validate(
    split: Split.ValueType,
    best: Measured,
    candidate: Measured,
    *,
    v0: Measured,
    config: DecisionConfig,
) -> Verdict:
    """Validate ``candidate`` against ``best`` on every val task, by the four rules of spec
    §8.3 that need no confirmation, after the hard-coding flags and the efficiency cap.
    """
    phase = _phase(split, best, candidate, v0=v0)
    guardrail = _guardrail(phase, config)
    phase.effective_min_delta = config.min_delta * (_GUARDRAIL_FACTOR if guardrail else 1.0)
    reasons = _outright(phase, config)
    if phase.delta < config.min_delta:
        reasons.append(REJECTION_REASON_INSUFFICIENT_GAIN)
    elif phase.delta < phase.effective_min_delta:
        reasons.append(REJECTION_REASON_COST_GUARDRAIL)
    if phase.sign_test_p > config.sign_test_max_p:
        reasons.append(REJECTION_REASON_NOT_SIGNIFICANT)
    if phase.regressions > config.max_regressions:
        reasons.append(REJECTION_REASON_REGRESSION)
    return Verdict(phase, tuple(reasons), guardrail)


def confirm(
    split: Split.ValueType,
    best: Measured,
    candidate: Measured,
    *,
    v0: Measured,
    config: DecisionConfig,
    guardrail: bool,
) -> Verdict:
    """Confirm ``candidate``, measured over its validation and confirmation seeds together: its
    gain must still be at least confirm_min_delta (spec §8.2 step 7).

    The threshold doubles if the guardrail tripped in the validation, ``guardrail``, or trips
    on the confirmation's own numbers.
    """
    phase = _phase(split, best, candidate, v0=v0)
    guardrail = guardrail or _guardrail(phase, config)
    factor = _GUARDRAIL_FACTOR if guardrail else 1.0
    phase.effective_min_delta = config.confirm_min_delta * factor
    reasons = _outright(phase, config)
    if phase.delta < phase.effective_min_delta:
        reasons.append(REJECTION_REASON_NOT_CONFIRMED)
    return Verdict(phase, tuple(reasons), guardrail)


def rank(deltas: Mapping[str, float]) -> tuple[str, tuple[str, ...]]:
    """Return the experiment with the largest validation Δ and the others, which it outranks,
    in their given order. Of equal Δs, the first given wins.
    """
    if not deltas:
        raise ValueError("ranking needs at least one candidate")
    winner = max(deltas, key=lambda experiment: deltas[experiment])  # max keeps the first.
    return winner, tuple(experiment for experiment in deltas if experiment != winner)


def describe(name: str, verdict: Verdict) -> str:
    """Put ``verdict`` into words, with aggregates only: never a task ID (spec §7)."""
    phase = verdict.phase
    parts = [
        f"{name}: ΔR-VES {100 * phase.delta:+.1f} points over {phase.task_count} tasks"
        f" x {phase.seed_count} seeds ({100 * phase.candidate_score:.1f} vs"
        f" {100 * phase.best_score:.1f})"
    ]
    if phase.HasField("effective_min_delta"):
        parts.append(f"needed {100 * phase.effective_min_delta:.1f}")
    parts.append(f"wins {phase.wins}, losses {phase.losses}, sign test p={phase.sign_test_p:.3f}")
    parts.append(f"{phase.regressions} regressions")
    if phase.flagged_instance_ids:
        parts.append(f"{len(phase.flagged_instance_ids)} flagged tasks")
    parts.extend(_LABELS[field].format(value) for field, value in _ratios(phase, *_LABELS))
    if verdict.guardrail:
        parts.append("cost guardrail tripped")
    failed = ", ".join(RejectionReason.Name(reason) for reason in verdict.reasons)
    parts.append(f"failed {failed}" if failed else "passed")
    return "; ".join(parts)


def _phase(
    split: Split.ValueType, best: Measured, candidate: Measured, *, v0: Measured | None
) -> PhaseResult:
    comparison = compare(best.table, candidate.table, scoring=_SCORING)
    phase = PhaseResult(
        split=split,
        task_count=comparison.task_count,
        seed_count=len(candidate.table.seeds),
        candidate_score=comparison.candidate_score,
        best_score=comparison.best_score,
        delta=comparison.delta,
        wins=comparison.sign.wins,
        losses=comparison.sign.losses,
        sign_test_p=comparison.sign.p_value,
        regressions=len(comparison.regressions),
        flagged_instance_ids=sorted(flagged_tasks(candidate.table)),
        candidate_run_ids=candidate.run_ids,
        best_run_ids=best.run_ids,
    )
    ours = candidate.stats
    _set_ratio(phase, "token_ratio_vs_best", ours.mean_tokens, best.stats.mean_tokens)
    _set_ratio(phase, "time_ratio_vs_best", ours.mean_wall_time_s, best.stats.mean_wall_time_s)
    if v0 is not None:
        _set_ratio(phase, "cost_ratio_vs_v0", ours.mean_cost_usd, v0.stats.mean_cost_usd)
        _set_ratio(phase, "time_ratio_vs_v0", ours.mean_wall_time_s, v0.stats.mean_wall_time_s)
        phase.v0_run_ids.extend(v0.run_ids)
    return phase


def _set_ratio(phase: PhaseResult, field: _Ratio, value: float | None, basis: float | None) -> None:
    if value is not None and basis is not None and basis > 0:
        setattr(phase, field, value / basis)


def _ratios(phase: PhaseResult, *fields: _Ratio) -> list[tuple[_Ratio, float]]:
    """The ratios among ``fields`` that the phase set, in that order."""
    return [(field, getattr(phase, field)) for field in fields if phase.HasField(field)]


def _guardrail(phase: PhaseResult, config: DecisionConfig) -> bool:
    return any(value > config.cost_guardrail_ratio for _, value in _ratios(phase, *_VS_BEST))


def _outright(phase: PhaseResult, config: DecisionConfig) -> list[RejectionReason.ValueType]:
    """The rules that fail a candidate whatever its gain: a flag, then the efficiency cap."""
    reasons: list[RejectionReason.ValueType] = []
    if phase.flagged_instance_ids:
        reasons.append(REJECTION_REASON_HARD_CODED)
    if any(value > config.efficiency_cap_ratio for _, value in _ratios(phase, *_VS_V0)):
        reasons.append(REJECTION_REASON_EFFICIENCY_CAP)
    return reasons
