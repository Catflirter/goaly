"""Tests for goaly.rsi.implementer.smoke: goaly-smoke's host side, with a fake runner."""

from pathlib import Path

import pytest

from goaly.contracts.gen.experiment_pb2 import LANE_EFFISQL, SmokeTest
from goaly.rsi.implementer.smoke import SmokeService, mentions_smoke
from goaly.rsi.sandbox import SandboxError
from goaly.rsi.workspace import Workspace, Worktree
from tests.rsi.gitrepo import git, make_repo
from tests.rsi.implementer.fakes import FakeRunner

# Effi-SQL's splits are lists of small integer IDs; these stand for them.
TRAIN = ("3", "17", "205")
VAL = "31"
HELD_OUT = "44"
PACK_FILE = "src/goaly/harness/packs/effisql/prompt.md"


class Setup:
    def __init__(self, tmp_path: Path, runner: FakeRunner) -> None:
        self.workspace = Workspace(make_repo(tmp_path / "repo"), root=tmp_path / "worktrees")
        self.worktree: Worktree = self.workspace.create(LANE_EFFISQL, "exp-1")
        self.runner = runner
        self.service = SmokeService(runner, self.workspace, self.worktree, train_ids=TRAIN, limit=2)


@pytest.fixture
def setup(tmp_path: Path) -> Setup:
    return Setup(tmp_path, FakeRunner())


def test_a_train_task_runs_on_a_pin_of_the_pack_as_it_stands(setup: Setup) -> None:
    (setup.worktree.path / PACK_FILE).write_text("Look at the plan first.\n", encoding="utf-8")
    head = setup.workspace.head(setup.worktree)

    output = setup.service.handle("goaly-smoke 17")

    assert output == {
        "output": "goaly-smoke 17: score 0.75; stop reason STOP_REASON_SUBMITTED, 7 steps, "
        "cost $0.0123, last tool error: none\n",
        "returncode": 0,
        "exception_info": "",
    }
    ((commit, instance_id),) = setup.runner.ran
    assert instance_id == "17"
    assert setup.runner.guarded == [commit]  # The guards ran first, on the same commit.
    pinned = git(setup.worktree.path, "show", f"{commit}:{PACK_FILE}")
    assert pinned == "Look at the plan first.\n"
    assert setup.workspace.head(setup.worktree) == head  # The branch did not move.
    (test,) = setup.service.tests
    assert test == SmokeTest(
        instance_id="17",
        run_id="smoke-1",
        score=0.75,
        detail="stop reason STOP_REASON_SUBMITTED, 7 steps, cost $0.0123, last tool error: none",
    )


@pytest.mark.parametrize("instance_id", [VAL, HELD_OUT, "not-a-task"])
def test_val_and_held_out_tasks_are_refused(setup: Setup, instance_id: str) -> None:
    output = setup.service.handle(f"goaly-smoke {instance_id}")

    assert output["returncode"] == 1
    assert "train tasks only" in output["output"]
    assert setup.runner.guarded == []
    assert setup.runner.ran == []
    (test,) = setup.service.tests
    assert test.instance_id == instance_id
    assert not test.HasField("score")
    assert test.detail.startswith("refused: ")


def test_a_third_call_is_refused(setup: Setup) -> None:
    setup.service.handle("goaly-smoke 3")
    setup.service.handle(f"goaly-smoke {VAL}")  # Refused, and not counted.
    setup.service.handle("goaly-smoke 17")

    third = setup.service.handle("goaly-smoke 205")

    assert third["returncode"] == 1
    assert "limit of 2 smoke runs per hypothesis" in third["output"]
    assert [instance for _, instance in setup.runner.ran] == ["3", "17"]
    assert [test.instance_id for test in setup.service.tests] == ["3", VAL, "17", "205"]


def test_a_failed_guard_stops_the_run_and_counts(tmp_path: Path) -> None:
    setup = Setup(tmp_path, FakeRunner(guard_failure="the hard-coding scan found a gold window"))

    first = setup.service.handle("goaly-smoke 3")
    setup.service.handle("goaly-smoke 3")
    third = setup.service.handle("goaly-smoke 3")

    assert first["returncode"] == 1
    assert "a guard failed: the hard-coding scan found a gold window" in first["output"]
    assert setup.runner.ran == []
    assert len(setup.runner.guarded) == 2
    assert "limit" in third["output"]


def test_a_pack_that_cannot_be_pinned_is_refused(setup: Setup) -> None:
    (setup.worktree.pack_dirs[0] / "link").symlink_to("/etc/passwd")

    output = setup.service.handle("goaly-smoke 3")

    assert output["returncode"] == 1
    assert "the pack cannot be run" in output["output"]
    assert setup.runner.guarded == []


def test_a_failed_grading_has_no_score(tmp_path: Path) -> None:
    setup = Setup(tmp_path, FakeRunner(score=None))

    output = setup.service.handle("goaly-smoke 3")

    assert "score none (grading failed)" in output["output"]
    (test,) = setup.service.tests
    assert not test.HasField("score")
    assert test.run_id == "smoke-1"


def test_an_infrastructure_failure_is_raised(setup: Setup) -> None:
    setup.runner.error = SandboxError("docker failed")

    with pytest.raises(SandboxError):
        setup.service.handle("goaly-smoke 3")


@pytest.mark.parametrize(
    "command",
    [
        "goaly-smoke",
        "goaly-smoke 3 17",
        "goaly-smoke 3 | tail",
        "cd x && goaly-smoke 3",
        "goaly-smoke '3",
    ],
)
def test_any_other_use_is_explained_and_not_counted(setup: Setup, command: str) -> None:
    output = setup.service.handle(command)

    assert output["returncode"] == 2
    assert "Run it on its own, as `goaly-smoke <instance_id>`" in output["output"]
    assert setup.service.tests == []


@pytest.mark.parametrize(
    ("command", "mentions"),
    [
        ("goaly-smoke 3", True),
        ("echo x; goaly-smoke 3", True),
        ("ls goaly-smoke-notes.md", False),
        ("grep -r my-goaly-smoke .", False),
        ("ls", False),
    ],
)
def test_mentions_smoke(command: str, mentions: bool) -> None:
    assert mentions_smoke(command) is mentions
