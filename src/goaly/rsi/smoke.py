"""goaly-smoke's runner (T20's SmokeRunner seam): the lane's guards and evaluator, for one task.

T20's SmokeService refuses tasks outside train and calls past the limit, and pins the pack into a
commit; this runner then guards that commit and evaluates it on the one task, with one seed,
through the lane's Evaluator: the isolated two-phase path (spec §6.1), with EvalPhase.SMOKE,
whose grading goes ahead of every phase (T10's Priority.SMOKE). Both calls come from the
implementer's worker thread, so the evaluation runs on the lane's event loop and the thread waits.

An evaluation that fails or times out is the infrastructure's failure, not the candidate's: it
raises SmokeInfrastructureError, a SandboxError, which T20 maps to IMPLEMENTER_EXIT_INFRASTRUCTURE,
so the experiment is aborted and its hypothesis queued again. A timed-out evaluation is
cancelled, and :meth:`EvaluatorSmokeRunner.cancel_all` cancels those in flight when the lane
stops, so no implementer thread waits on a loop that is going away.
"""

import asyncio
import concurrent.futures
import threading
from collections.abc import Collection
from pathlib import Path

from goaly.contracts.gen.experiment_pb2 import LANE_EFFISQL, Guard
from goaly.contracts.gen.task_pb2 import (
    GRADE_STATUS_GRADED,
    GRADE_STATUS_PREDICTION_FAILED,
    GRADE_STATUS_TIMEOUT,
    SPLIT_TRAIN,
)
from goaly.contracts.gen.trajectory_pb2 import StopReason
from goaly.rsi.git import run_git
from goaly.rsi.implementer.smoke import SmokeOutcome
from goaly.rsi.lanes import lane_info
from goaly.rsi.round import (
    EvalPhase,
    Evaluation,
    EvaluationRequest,
    Evaluator,
    GuardRunner,
    tasks_digest,
)
from goaly.rsi.sandbox import SandboxError

__all__ = ["EvaluatorSmokeRunner", "SmokeInfrastructureError"]

_SCORED = frozenset({GRADE_STATUS_GRADED, GRADE_STATUS_PREDICTION_FAILED, GRADE_STATUS_TIMEOUT})


class SmokeInfrastructureError(SandboxError):
    """goaly-smoke's evaluation failed or timed out: the experiment aborts (T20's mapping)."""


class EvaluatorSmokeRunner:
    """Runs goaly-smoke for the implementers of the lane whose event loop is ``loop``."""

    def __init__(
        self,
        evaluator: Evaluator,
        guards: GuardRunner,
        repo: Path,
        loop: asyncio.AbstractEventLoop,
        *,
        train_ids: Collection[str],
        seed: int = 0,
        timeout_s: float = 1800.0,
    ) -> None:
        self._evaluator = evaluator
        self._guards = guards
        self._repo = repo
        self._loop = loop
        self._train_ids = frozenset(train_ids)
        self._seed = seed
        self._timeout_s = timeout_s
        self._lock = threading.Lock()
        self._pending: set[concurrent.futures.Future[Evaluation]] = set()

    def cancel_all(self) -> int:
        """Cancel the evaluations in flight, as when the lane stops; return how many."""
        with self._lock:
            pending = list(self._pending)
        return sum(future.cancel() for future in pending)

    def guard(self, commit: str) -> str | None:
        """Run the step-3 guards on the pinned commit, against its parent, the worktree's HEAD."""
        parent = run_git(["rev-parse", f"{commit}^"], cwd=self._repo).strip()
        results = self._guards(f"smoke-{commit[:12]}", parent, commit)
        failed = [result for result in results if not result.passed]
        if not failed:
            return None
        return "; ".join(f"{Guard.Name(r.guard)}: {r.detail or 'failed'}" for r in failed)

    def run(self, commit: str, instance_id: str) -> SmokeOutcome:
        """Evaluate the commit on one train task, with SMOKE priority, and wait for it.

        Raises SmokeInfrastructureError if the evaluation fails or times out, and
        concurrent.futures.CancelledError if the lane cancelled it.
        """
        if instance_id not in self._train_ids:  # SmokeService refuses these first.
            return SmokeOutcome("", None, "refused", 0, 0.0, f"{instance_id} is not a train task")
        pack = lane_info(LANE_EFFISQL).pack_dirs[0]
        tree, pack_tree = run_git(
            ["rev-parse", f"{commit}^{{tree}}", f"{commit}:{pack}"], cwd=self._repo
        ).split()
        run_id = f"smoke-{commit[:12]}-{instance_id}"
        tasks, seeds = (instance_id,), (self._seed,)
        request = EvaluationRequest(
            EvalPhase.SMOKE,
            commit,
            tree,
            SPLIT_TRAIN,
            tasks,
            seeds,
            run_id,
            pack_tree=pack_tree,
            tasks_digest=tasks_digest(SPLIT_TRAIN, tasks, seeds),
        )
        future = asyncio.run_coroutine_threadsafe(self._evaluator.evaluate(request), self._loop)
        with self._lock:
            self._pending.add(future)
        try:
            evaluation = future.result(timeout=self._timeout_s)
        except concurrent.futures.CancelledError:
            raise
        except TimeoutError as error:
            future.cancel()
            raise SmokeInfrastructureError(
                f"goaly-smoke's evaluation {run_id} took over {self._timeout_s:g} s"
            ) from error
        except Exception as error:
            raise SmokeInfrastructureError(
                f"goaly-smoke's evaluation {run_id} failed: {type(error).__name__}: {error}"
            ) from error
        finally:
            with self._lock:
                self._pending.discard(future)
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
