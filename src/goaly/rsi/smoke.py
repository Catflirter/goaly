"""goaly-smoke's runner (T20's SmokeRunner seam): the lane's guards and evaluator, for one task.

T20's SmokeService refuses tasks outside train and calls past the limit, and pins the pack into a
commit; this runner then guards that commit and evaluates it on the one task, with one seed,
through the lane's Evaluator: the isolated two-phase path (spec §6.1), with EvalPhase.SMOKE,
whose grading goes ahead of every phase (T10's Priority.SMOKE). Both calls come from the
implementer's worker thread, so the evaluation runs on the lane's event loop and the thread waits.
"""

import asyncio
from pathlib import Path

from goaly.contracts.gen.experiment_pb2 import Guard
from goaly.contracts.gen.task_pb2 import (
    GRADE_STATUS_GRADED,
    GRADE_STATUS_PREDICTION_FAILED,
    GRADE_STATUS_TIMEOUT,
    SPLIT_TRAIN,
)
from goaly.contracts.gen.trajectory_pb2 import StopReason
from goaly.rsi.git import run_git
from goaly.rsi.implementer.smoke import SmokeOutcome
from goaly.rsi.round import EvalPhase, EvaluationRequest, Evaluator, GuardRunner

__all__ = ["EvaluatorSmokeRunner"]

_SCORED = frozenset({GRADE_STATUS_GRADED, GRADE_STATUS_PREDICTION_FAILED, GRADE_STATUS_TIMEOUT})


class EvaluatorSmokeRunner:
    """Runs goaly-smoke for the implementers of the lane whose event loop is ``loop``."""

    def __init__(
        self,
        evaluator: Evaluator,
        guards: GuardRunner,
        repo: Path,
        loop: asyncio.AbstractEventLoop,
        *,
        seed: int = 0,
        timeout_s: float = 1800.0,
    ) -> None:
        self._evaluator = evaluator
        self._guards = guards
        self._repo = repo
        self._loop = loop
        self._seed = seed
        self._timeout_s = timeout_s

    def guard(self, commit: str) -> str | None:
        """Run the step-3 guards on the pinned commit, against its parent, the worktree's HEAD."""
        parent = run_git(["rev-parse", f"{commit}^"], cwd=self._repo).strip()
        results = self._guards(f"smoke-{commit[:12]}", parent, commit)
        failed = [result for result in results if not result.passed]
        if not failed:
            return None
        return "; ".join(f"{Guard.Name(r.guard)}: {r.detail or 'failed'}" for r in failed)

    def run(self, commit: str, instance_id: str) -> SmokeOutcome:
        """Evaluate the commit on one train task, with SMOKE priority, and wait for it."""
        tree = run_git(["rev-parse", f"{commit}^{{tree}}"], cwd=self._repo).strip()
        run_id = f"smoke-{commit[:12]}-{instance_id}"
        request = EvaluationRequest(
            EvalPhase.SMOKE, commit, tree, SPLIT_TRAIN, (instance_id,), (self._seed,), run_id
        )
        future = asyncio.run_coroutine_threadsafe(self._evaluator.evaluate(request), self._loop)
        evaluation = future.result(timeout=self._timeout_s)
        result = next((r for r in evaluation.results if r.instance_id == instance_id), None)
        if result is None:
            return SmokeOutcome(run_id, None, "no result", 0, 0.0, "the evaluation returned none")
        stats = evaluation.stats
        scored = result.grade.status in _SCORED and result.grade.HasField("effisql")
        return SmokeOutcome(
            run_id=run_id,
            score=result.grade.effisql.credit if scored else None,
            stop_reason=StopReason.Name(result.stop_reason),
            steps=result.step_count,
            cost_usd=(stats.mean_cost_usd or 0.0) * stats.count,
            last_tool_error=result.error_message,
        )
