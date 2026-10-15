"""Tests for goaly.rsi.smoke: goaly-smoke through the lane's guards and evaluator."""

import asyncio
import concurrent.futures
import threading
from pathlib import Path

import pytest

from goaly.bench.effisql.grading_pool import Priority
from goaly.contracts.gen.experiment_pb2 import (
    GUARD_UNIT_TESTS,
    IMPLEMENTER_EXIT_INFRASTRUCTURE,
    IMPLEMENTER_EXIT_INTERRUPTED,
    GuardResult,
)
from goaly.contracts.gen.task_pb2 import GRADE_STATUS_GRADER_ERROR, SPLIT_TRAIN
from goaly.rsi.decision import CostStats
from goaly.rsi.implementer.run import exit_for_error
from goaly.rsi.round import EvalPhase, Evaluation, EvaluationRequest
from goaly.rsi.sandbox import SandboxError
from goaly.rsi.smoke import EvaluatorSmokeRunner, SmokeInfrastructureError
from tests.rsi.gitrepo import git, make_repo
from tests.rsi.round_world import TRAIN, ToyEvaluator, task_result

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

    runner = EvaluatorSmokeRunner(
        evaluator, guards, repo, asyncio.get_running_loop(), train_ids=TRAIN
    )
    assert await asyncio.to_thread(runner.guard, commit) is None
    assert guarded == [(f"smoke-{commit[:12]}", head, commit)]
    outcome = await asyncio.to_thread(runner.run, commit, "6", threading.Event())
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

    runner = EvaluatorSmokeRunner(
        Ungraded(), guards, repo, asyncio.get_running_loop(), train_ids=TRAIN
    )
    assert await asyncio.to_thread(runner.guard, commit) == "GUARD_UNIT_TESTS: 2 failed"
    outcome = await asyncio.to_thread(runner.run, commit, "1", threading.Event())
    assert outcome.score is None
    assert outcome.cost_usd == 0.0
    assert outcome.last_tool_error == "psycopg: connection lost"


async def test_a_smoke_run_without_its_result_says_so(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo")
    commit, _ = pinned(repo)

    class Empty(ToyEvaluator):
        async def evaluate(self, request: EvaluationRequest) -> Evaluation:
            return Evaluation([], CostStats(0.0, 0.0, 0.0))

    runner = EvaluatorSmokeRunner(
        Empty(), lambda *_: [], repo, asyncio.get_running_loop(), train_ids=TRAIN
    )
    outcome = await asyncio.to_thread(runner.run, commit, "1", threading.Event())
    assert outcome.score is None
    assert "returned none" in outcome.last_tool_error


async def test_a_failed_or_slow_smoke_evaluation_is_the_infrastructures(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo")
    commit, _ = pinned(repo)
    loop = asyncio.get_running_loop()
    failing = ToyEvaluator(smoke_error=True)
    runner = EvaluatorSmokeRunner(failing, lambda *_: [], repo, loop, train_ids=TRAIN)
    with pytest.raises(SmokeInfrastructureError, match="exited with 137"):
        await asyncio.to_thread(runner.run, commit, "1", threading.Event())
    assert issubclass(SmokeInfrastructureError, SandboxError)  # T20 maps it to INFRASTRUCTURE.
    assert exit_for_error(SmokeInfrastructureError("x")) == IMPLEMENTER_EXIT_INFRASTRUCTURE
    slow = ToyEvaluator(smoke_gate=asyncio.Event())
    runner = EvaluatorSmokeRunner(slow, lambda *_: [], repo, loop, train_ids=TRAIN, timeout_s=0.2)
    with pytest.raises(SmokeInfrastructureError, match="took over"):
        await asyncio.to_thread(runner.run, commit, "1", threading.Event())
    await asyncio.sleep(0.05)
    assert slow.in_flight == 0  # The timed-out evaluation was cancelled, not left running.


async def test_cancel_all_releases_a_waiting_implementer(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo")
    commit, _ = pinned(repo)
    stuck = ToyEvaluator(smoke_gate=asyncio.Event())
    runner = EvaluatorSmokeRunner(
        stuck, lambda *_: [], repo, asyncio.get_running_loop(), train_ids=TRAIN
    )
    waiting = asyncio.ensure_future(asyncio.to_thread(runner.run, commit, "1", threading.Event()))
    while not stuck.requests:
        await asyncio.sleep(0.01)
    assert runner.cancel_all() == 1
    with pytest.raises(asyncio.CancelledError):  # The thread's CancelledError, through to_thread.
        await waiting
    assert exit_for_error(concurrent.futures.CancelledError()) == IMPLEMENTER_EXIT_INTERRUPTED


async def test_a_task_outside_train_is_refused_here_too(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo")
    commit, _ = pinned(repo)
    evaluator = ToyEvaluator()
    runner = EvaluatorSmokeRunner(
        evaluator, lambda *_: [], repo, asyncio.get_running_loop(), train_ids=TRAIN
    )
    outcome = await asyncio.to_thread(runner.run, commit, "101", threading.Event())
    assert outcome.score is None
    assert "not a train task" in outcome.last_tool_error
    assert evaluator.requests == []


async def test_a_stopped_implementer_cancels_its_smoke_run(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo")
    commit, _ = pinned(repo)
    stuck = ToyEvaluator(smoke_gate=asyncio.Event())
    runner = EvaluatorSmokeRunner(
        stuck, lambda *_: [], repo, asyncio.get_running_loop(), train_ids=TRAIN
    )
    stop = threading.Event()
    waiting = asyncio.ensure_future(asyncio.to_thread(runner.run, commit, "1", stop))
    while not stuck.requests:
        await asyncio.sleep(0.01)
    stop.set()  # The implementer's run is ending.
    with pytest.raises(asyncio.CancelledError):
        await waiting
    await asyncio.sleep(0.05)
    assert stuck.in_flight == 0
