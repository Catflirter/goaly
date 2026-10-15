"""Tests for goaly.rsi.guards.unit_tests and goaly.rsi.guards.run_guards, with a fake docker CLI.

tests/rsi/guards/test_unit_tests_docker.py runs the unit-test guard in real containers.
"""

from pathlib import Path

import pytest

from goaly.contracts.gen.experiment_pb2 import (
    GUARD_ALLOWED_PATHS,
    GUARD_CHECKSUMS,
    GUARD_HARD_CODING_SCAN,
    GUARD_PACK_CODE_ALLOWLIST,
    GUARD_UNIT_TESTS,
    LANE_EFFISQL,
    GuardResult,
)
from goaly.rsi.commit import commit_candidate
from goaly.rsi.guards import GuardsConfig, run_guards
from goaly.rsi.guards.checksums import frozen_checksums
from goaly.rsi.guards.gold_windows import GoldWindows
from goaly.rsi.guards.unit_tests import check_unit_tests
from goaly.rsi.sandbox import SandboxConfig, SandboxError
from goaly.rsi.workspace import Workspace, Worktree
from tests.rsi.fakedocker import FakeDocker
from tests.rsi.gitrepo import git, make_repo, write

PACK = "src/goaly/harness/packs/effisql"
PACK_TESTS = "tests/harness/packs/effisql"
GOLD = GoldWindows.build(
    ["SELECT 1 FROM partsupp WHERE ps_partkey = 42 AND ps_suppkey = 7 ORDER BY 1"], [], []
)


@pytest.fixture
def docker(tmp_path: Path) -> FakeDocker:
    docker = FakeDocker(tmp_path / "fake-docker")
    docker.script({"wait": {"stdout": "0\n"}, "logs": {"stdout": "2 passed in 0.10s\n"}})
    return docker


@pytest.fixture
def config(docker: FakeDocker) -> GuardsConfig:
    return GuardsConfig(sandbox=SandboxConfig(docker=str(docker.path), docker_timeout_s=30))


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    return Workspace(make_repo(tmp_path / "repo"), root=tmp_path / "worktrees")


@pytest.fixture
def worktree(workspace: Workspace) -> Worktree:
    return workspace.create(LANE_EFFISQL, "exp-1")


# The unit-test guard.


def unit_tests(snapshot: Path, config: GuardsConfig) -> tuple[bool, str]:
    result = check_unit_tests(
        snapshot, LANE_EFFISQL, experiment_id="exp-1", image="sandbox:1", config=config.sandbox
    )
    assert result.guard == GUARD_UNIT_TESTS
    return result.passed, result.detail


def test_the_packs_tests_run_read_only_in_a_fresh_sandbox(
    docker: FakeDocker, config: GuardsConfig, tmp_path: Path
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()

    assert unit_tests(snapshot, config) == (True, "2 passed in 0.10s")

    run_args = docker.calls()[0]["args"]
    mounts = [arg for arg in run_args if arg.startswith("--mount=")]
    assert mounts == [f"--mount=type=bind,source={snapshot},target={snapshot},readonly"]
    assert "--label=goaly.rsi.purpose=unit-tests" in run_args
    assert run_args[-1].endswith(f"python -m pytest -q -p no:cacheprovider {PACK_TESTS}")


@pytest.mark.parametrize(
    ("exit_code", "output", "detail"),
    [
        (1, "F.\n1 failed, 1 passed in 0.2s\n", "F.\n1 failed, 1 passed in 0.2s"),
        (5, "no tests ran in 0.01s\n", "no test of the pack ran: no tests ran in 0.01s"),
        (4, "ERROR: file or directory not found\n", "no test of the pack ran: ERROR: file or"),
        (124, "", "the pack's tests ran out of time after 600 s"),
    ],
)
def test_tests_that_fail_or_do_not_run_fail_the_guard(
    docker: FakeDocker,
    config: GuardsConfig,
    tmp_path: Path,
    exit_code: int,
    output: str,
    detail: str,
) -> None:
    docker.script({"wait": {"stdout": f"{exit_code}\n"}, "logs": {"stdout": output}})

    passed, found = unit_tests(tmp_path, config)

    assert not passed
    assert found.startswith(detail)


def test_a_long_report_keeps_its_end(
    docker: FakeDocker, config: GuardsConfig, tmp_path: Path
) -> None:
    report = "".join(f"line {n}\n" for n in range(100)) + "1 failed in 0.2s\n"
    docker.script({"wait": {"stdout": "1\n"}, "logs": {"stdout": report}})

    _, detail = unit_tests(tmp_path, config)

    assert detail.splitlines()[-1] == "1 failed in 0.2s"
    assert len(detail.splitlines()) == 40


# All the guards, in order.


def run(
    workspace: Workspace, commit: str, config: GuardsConfig, frozen: dict[str, str]
) -> list[GuardResult]:
    return run_guards(
        workspace,
        LANE_EFFISQL,
        experiment_id="exp-1",
        parent="best/effisql",
        candidate=commit,
        gold=GOLD,
        frozen=frozen,
        image="sandbox:1",
        config=config,
    )


@pytest.fixture
def frozen(workspace: Workspace) -> dict[str, str]:
    with workspace.snapshot("best/effisql") as snapshot:
        return frozen_checksums(snapshot)


def commit(workspace: Workspace, worktree: Worktree, files: dict[str, str]) -> str:
    write(worktree.path, files)
    made = commit_candidate(workspace, worktree, hypothesis_id="h1", summary="candidate")
    assert made is not None
    return made


def test_a_clean_candidate_passes_every_guard_in_order(
    workspace: Workspace, worktree: Worktree, config: GuardsConfig, frozen: dict[str, str]
) -> None:
    candidate = commit(workspace, worktree, {f"{PACK}/prompt.md": "Read the plan first.\n"})

    results = run(workspace, candidate, config, frozen)

    assert [result.guard for result in results] == [
        GUARD_ALLOWED_PATHS,
        GUARD_CHECKSUMS,
        GUARD_HARD_CODING_SCAN,
        GUARD_PACK_CODE_ALLOWLIST,
        GUARD_UNIT_TESTS,
    ]
    assert all(result.passed for result in results)


def test_the_guards_stop_at_the_first_failure(
    workspace: Workspace,
    worktree: Worktree,
    config: GuardsConfig,
    frozen: dict[str, str],
    docker: FakeDocker,
) -> None:
    candidate = commit(workspace, worktree, {f"{PACK}/policy.py": "import os\n"})

    results = run(workspace, candidate, config, frozen)

    assert [(result.guard, result.passed) for result in results] == [
        (GUARD_ALLOWED_PATHS, True),
        (GUARD_CHECKSUMS, True),
        (GUARD_HARD_CODING_SCAN, True),
        (GUARD_PACK_CODE_ALLOWLIST, False),
    ]
    assert docker.calls() == []  # The pack's tests never ran.


def test_a_change_outside_the_pack_stops_before_any_snapshot(
    workspace: Workspace, config: GuardsConfig, frozen: dict[str, str]
) -> None:
    write(workspace.repo, {"src/goaly/bench/runner.py": "changed\n"})
    candidate = commit_outside(workspace)

    results = run(workspace, candidate, config, frozen)

    assert [(result.guard, result.passed) for result in results] == [(GUARD_ALLOWED_PATHS, False)]


def commit_outside(workspace: Workspace) -> str:
    git(workspace.repo, "commit", "--quiet", "--all", "--message=outside the pack")
    return git(workspace.repo, "rev-parse", "HEAD").strip()


def test_the_guards_see_the_commit_not_the_live_pack(
    workspace: Workspace, worktree: Worktree, config: GuardsConfig, frozen: dict[str, str]
) -> None:
    clean = commit(workspace, worktree, {f"{PACK}/policy.py": "STEPS = 4\n"})
    # The live pack changes after the commit, as a sandbox still running could change it.
    write(worktree.path, {f"{PACK}/policy.py": "import os\n"})

    assert all(result.passed for result in run(workspace, clean, config, frozen))

    write(worktree.path, {f"{PACK}/policy.py": "import os\n"})
    dirty = commit(workspace, worktree, {})
    write(worktree.path, {f"{PACK}/policy.py": "STEPS = 4\n"})  # Clean again, uncommitted.

    assert not run(workspace, dirty, config, frozen)[3].passed


def test_a_docker_failure_is_raised_not_counted_against_the_candidate(
    workspace: Workspace,
    worktree: Worktree,
    config: GuardsConfig,
    frozen: dict[str, str],
    docker: FakeDocker,
) -> None:
    candidate = commit(workspace, worktree, {f"{PACK}/prompt.md": "Be faster.\n"})
    docker.script({"run": {"exit": 125, "stderr": "the daemon is gone"}})

    with pytest.raises(SandboxError, match="daemon is gone"):
        run(workspace, candidate, config, frozen)
