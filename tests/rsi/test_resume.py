"""Tests for goaly.rsi.resume: reconciling the best tag, aborting experiments, leftovers."""

from pathlib import Path

import pytest

from goaly.contracts.gen.experiment_pb2 import (
    DECISION_ABORTED,
    DECISION_ACCEPTED,
    DECISION_PENDING,
    LANE_EFFISQL,
    ExperimentRecord,
    PhaseResult,
)
from goaly.contracts.gen.llm_call_pb2 import CallPurpose, Usage
from goaly.rsi.resume import (
    LaneStateError,
    abort_experiments,
    merge_usage,
    reconcile_best,
    remove_worktrees,
)
from tests.rsi.gitrepo import git
from tests.rsi.round_world import World, hypothesis, make_world


@pytest.fixture
def world(tmp_path: Path) -> World:
    return make_world(tmp_path)


def commit(world: World, name: str, parent: str | None = None) -> str:
    """A commit on top of ``parent`` (the repository's HEAD by default), on no branch."""
    tree = git(world.repo, "rev-parse", "HEAD^{tree}").strip()
    base = parent or git(world.repo, "rev-parse", "HEAD").strip()
    return git(world.repo, "commit-tree", tree, "-p", base, f"-m{name}").strip()


def record(world: World, experiment_id: str, **fields: object) -> ExperimentRecord:
    entry = ExperimentRecord(
        experiment_id=experiment_id,
        lane=LANE_EFFISQL,
        round=1,
        parent_version=world.v0,
        hypothesis=hypothesis(f"h-{experiment_id}"),
        decision=DECISION_PENDING,
        **fields,  # type: ignore[arg-type]  # The record's own fields.
    )
    world.ledger.record_experiment(entry)
    return entry


def reconcile(world: World) -> str:
    return reconcile_best(
        world.ledger, world.workspace, LANE_EFFISQL, v0_commit=world.v0, now=world.clock()
    )


def test_an_agreeing_tag_is_left_alone(world: World) -> None:
    assert reconcile(world) == world.v0
    assert world.best() == world.v0


def test_a_tag_on_a_confirmed_candidate_records_its_acceptance(world: World) -> None:
    candidate = commit(world, "candidate")
    record(world, "e1", candidate_commit=candidate, confirmation=PhaseResult(delta=0.1))
    record(world, "e2", candidate_commit=commit(world, "other"))
    world.workspace.set_best(LANE_EFFISQL, candidate, expected=world.v0)  # Then the crash.
    assert reconcile(world) == candidate
    latest = world.ledger.latest_experiments(LANE_EFFISQL)
    assert latest["e1"].decision == DECISION_ACCEPTED
    assert latest["e1"].HasField("finished_at")
    assert latest["e2"].decision == DECISION_PENDING
    assert world.ledger.best_version(LANE_EFFISQL) == candidate


def test_a_missing_tag_is_set_to_the_ledgers_best(world: World) -> None:
    git(world.repo, "tag", "--delete", "best/effisql")
    assert reconcile(world) == world.v0
    assert world.best() == world.v0


def test_a_tag_behind_the_ledger_moves_forward(world: World) -> None:
    candidate = commit(world, "candidate")
    entry = record(world, "e1", candidate_commit=candidate)
    entry.decision = DECISION_ACCEPTED
    world.ledger.record_experiment(entry)
    assert reconcile(world) == candidate
    assert world.best() == candidate


def test_a_tag_no_crash_explains_is_an_error(world: World) -> None:
    stray = commit(world, "stray")
    git(world.repo, "tag", "--force", "best/effisql", stray)
    with pytest.raises(LaneStateError, match="neither"):
        reconcile(world)
    assert world.best() == stray


def test_abort_experiments_records_and_queues_them_again(world: World) -> None:
    record(world, "e1")
    record(world, "e2", candidate_commit=commit(world, "built"))
    usage = {CallPurpose.CALL_PURPOSE_RSI_IMPL: Usage(input_tokens=7, output_tokens=3)}
    aborted = abort_experiments(
        world.ledger,
        world.queue,
        LANE_EFFISQL,
        "the lane's deadline",
        now=world.clock(),
        uncommitted_only=False,
        usage=lambda _: usage,
    )
    assert [r.experiment_id for r in aborted] == ["e1", "e2"]
    latest = world.ledger.latest_experiments(LANE_EFFISQL)
    for name in ("e1", "e2"):
        assert latest[name].decision == DECISION_ABORTED
        assert latest[name].decision_reason == "aborted: the lane's deadline"
        assert latest[name].usage_by_purpose[0].usage.input_tokens == 7
    assert latest["e1"].implementer.exit == 8  # INTERRUPTED: cut before its commit.
    assert not latest["e2"].HasField("implementer")
    assert [h.hypothesis_id for h in world.queue.pending()] == ["h-e1", "h-e2"]
    assert world.ledger.pending_experiments(LANE_EFFISQL) == []


def test_merge_usage_replaces_a_purposes_total(world: World) -> None:
    entry = ExperimentRecord()
    merge_usage(entry, {CallPurpose.CALL_PURPOSE_RSI_IMPL: Usage(input_tokens=1)})
    merge_usage(entry, {CallPurpose.CALL_PURPOSE_RSI_IMPL: Usage(input_tokens=5)})
    assert len(entry.usage_by_purpose) == 1
    assert entry.usage_by_purpose[0].usage.input_tokens == 5


def test_remove_worktrees_clears_the_lanes_leftovers(world: World) -> None:
    world.workspace.create(LANE_EFFISQL, "e1")
    world.workspace.create(LANE_EFFISQL, "e2")
    assert remove_worktrees(world.workspace, LANE_EFFISQL) == 2
    assert world.workspace.worktrees(LANE_EFFISQL) == []
    assert "rsi/effisql/e1" in git(world.repo, "branch", "--list", "rsi/*")
