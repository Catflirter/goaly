"""Tests for goaly.rsi.smoke: goaly-smoke through the lane's guards and evaluator."""

import asyncio
from pathlib import Path

import pytest

from goaly.bench.effisql.grading_pool import Priority
from goaly.contracts.gen.experiment_pb2 import GUARD_UNIT_TESTS, GuardResult
from goaly.contracts.gen.task_pb2 import GRADE_STATUS_GRADER_ERROR, SPLIT_TRAIN
from goaly.rsi.decision import CostStats
from goaly.rsi.round import EvalPhase, Evaluation, EvaluationRequest
from goaly.rsi.smoke import EvaluatorSmokeRunner
from tests.rsi.gitrepo import git, make_repo
from tests.rsi.round_world import ToyEvaluator, task_result

pytestmark = pytest.mark.anyio


def pinned(repo: Path) -> tuple[str, str]:
    """A commit on top of HEAD, as Workspace.pin makes, and HEAD."""
    head = git(repo, "rev-parse", "HEAD").strip()
    tree = git(repo, "rev-parse", "HEAD^{tree}").strip()
    return git(repo, "commit-tree", tree, "-p", head, "-mpin").strip(), head


async def test_a_smoke_run_is_one_train_task_with_smoke_priority(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo")
    commit, head = pinned(repo)
    evaluator = ToyEvaluator()
    guarded: list[tuple[str, str, str]] = []

    def guards(experiment_id: str, parent: str, candidate: str) -> list[GuardResult]:
        guarded.append((experiment_id, parent, candidate))
        return []

    runner = EvaluatorSmokeRunner(evaluator, guards, repo, asyncio.get_running_loop())
    assert await asyncio.to_thread(runner.guard, commit) is None
    assert guarded == [(f"smoke-{commit[:12]}", head, commit)]
    outcome = await asyncio.to_thread(runner.run, commit, "6")
    (request,) = evaluator.requests
    assert request.phase == EvalPhase.SMOKE
    assert request.priority == Priority.SMOKE
    assert (request.split, request.instance_ids, request.seeds) == (SPLIT_TRAIN, ("6",), (0,))
    assert request.tree == git(repo, "rev-parse", f"{commit}^{{tree}}").strip()
    assert outcome.score == 1.0
    assert outcome.cost_usd == pytest.approx(0.01)
    assert outcome.run_id == request.run_id


async def test_a_failed_guard_and_an_ungraded_task_are_reported(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo")
    commit, _ = pinned(repo)

    class Ungraded(ToyEvaluator):
        async def evaluate(self, request: EvaluationRequest) -> Evaluation:
            result = task_result(request.run_id, request.instance_ids[0], 0, 0.0, flagged=False)
            result.grade.status = GRADE_STATUS_GRADER_ERROR
            result.error_message = "psycopg: connection lost"
            return Evaluation([result], CostStats(10.0, None, 1.0))

    def guards(experiment_id: str, parent: str, candidate: str) -> list[GuardResult]:
        return [GuardResult(guard=GUARD_UNIT_TESTS, passed=False, detail="2 failed")]

    runner = EvaluatorSmokeRunner(Ungraded(), guards, repo, asyncio.get_running_loop())
    assert await asyncio.to_thread(runner.guard, commit) == "GUARD_UNIT_TESTS: 2 failed"
    outcome = await asyncio.to_thread(runner.run, commit, "1")
    assert outcome.score is None
    assert outcome.cost_usd == 0.0
    assert outcome.last_tool_error == "psycopg: connection lost"


async def test_a_smoke_run_without_its_result_says_so(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo")
    commit, _ = pinned(repo)

    class Empty(ToyEvaluator):
        async def evaluate(self, request: EvaluationRequest) -> Evaluation:
            return Evaluation([], CostStats(0.0, 0.0, 0.0))

    runner = EvaluatorSmokeRunner(Empty(), lambda *_: [], repo, asyncio.get_running_loop())
    outcome = await asyncio.to_thread(runner.run, commit, "1")
    assert outcome.score is None
    assert "returned none" in outcome.last_tool_error
