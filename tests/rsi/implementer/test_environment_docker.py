"""Integration tests for the implementer's sandbox on real Docker (spec §8.5, §8.8).

They need Docker and run only with ``pytest --integration``. Each test's experiment ID is its
own, so the check for leftover containers never sees anyone else's.
"""

import concurrent.futures
import secrets
import subprocess
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from goaly.contracts.gen.experiment_pb2 import LANE_EFFISQL, Hypothesis, ImplementerExit
from goaly.rsi.implementer import msa
from goaly.rsi.implementer.environment import SandboxEnvironment, SandboxStoppedError
from goaly.rsi.implementer.run import Implementer
from goaly.rsi.sandbox import LABEL_EXPERIMENT, SandboxConfig, ensure_image
from goaly.rsi.workspace import Workspace, Worktree
from tests.rsi.gitrepo import git, make_repo
from tests.rsi.implementer.fakes import NoLlm, ScriptedModel, submit

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
SECRET = f"sk-it-{secrets.token_hex(8)}"


@pytest.fixture(scope="module")
def image() -> str:
    return ensure_image(REPO_ROOT)


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    return Workspace(make_repo(tmp_path / "repo"), root=tmp_path / "worktrees")


@pytest.fixture
def worktree(workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> Iterator[Worktree]:
    monkeypatch.setenv("GOALY_API_KEY", SECRET)  # As in the lane process, which holds the key.
    worktree = workspace.create(LANE_EFFISQL, f"it-{secrets.token_hex(4)}")
    yield worktree
    assert containers_of(worktree) == [], "a sandbox container outlived its test"


def containers_of(worktree: Worktree) -> list[str]:
    label = f"--filter=label={LABEL_EXPERIMENT}={worktree.experiment_id}"
    return subprocess.run(  # noqa: S603  (docker with arguments built here)
        ["docker", "ps", "--all", "--quiet", label],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()


def test_a_write_outside_the_pack_fails_and_no_key_is_inside(
    image: str, worktree: Worktree
) -> None:
    with SandboxEnvironment(
        image=image, worktree=worktree, command_timeout_s=60, lifetime_s=600
    ) as sandbox:
        outside = sandbox.execute({"command": "touch src/goaly/bench/evil.py"})
        inside = sandbox.execute({"command": "touch src/goaly/harness/packs/effisql/ok.md"})
        tests = sandbox.execute({"command": "touch tests/harness/packs/effisql/test_ok.py"})
        environment = sandbox.execute({"command": "env; cat /proc/1/environ | tr '\\0' '\\n'"})
        commit = sandbox.execute({"command": "git add --all && git commit -qm x"})

    assert outside["returncode"] != 0
    assert "Read-only file system" in outside["output"]
    assert not (worktree.path / "src/goaly/bench/evil.py").exists()
    assert (inside["returncode"], tests["returncode"]) == (0, 0)
    assert (worktree.pack_dirs[0] / "ok.md").exists()
    assert environment["returncode"] == 0
    assert SECRET not in environment["output"]
    assert "GOALY_" not in environment["output"]
    assert commit["returncode"] != 0  # Only the host commits.
    assert containers_of(worktree) == []


def test_a_command_is_stopped_inside_the_container_at_its_timeout(
    image: str, worktree: Worktree
) -> None:
    with SandboxEnvironment(
        image=image, worktree=worktree, command_timeout_s=1, lifetime_s=600
    ) as sandbox:
        slow = sandbox.execute({"command": "sleep 30"})
        after = sandbox.execute({"command": "echo still here"})

    assert slow["returncode"] == 124
    assert "timed out" in slow["exception_info"]
    assert after == {"output": "still here\n", "returncode": 0, "exception_info": ""}


def test_a_sandbox_the_agent_stops_is_the_candidates_fault(image: str, worktree: Worktree) -> None:
    sandbox = SandboxEnvironment(
        image=image, worktree=worktree, command_timeout_s=10, lifetime_s=600
    )
    try:
        with pytest.raises(SandboxStoppedError):
            sandbox.execute({"command": "kill 1; sleep 5"})
    finally:
        sandbox.cleanup()


def test_a_background_process_with_open_output_does_not_hold_the_command(
    image: str, worktree: Worktree
) -> None:
    # On this Docker, exec ends with its command even when a setsid child keeps the output
    # open. Were it to wait, the host's timeout would end it as a timed-out command (see the
    # fake-docker test), never as a Docker failure.
    with SandboxEnvironment(
        image=image,
        worktree=worktree,
        command_timeout_s=5,
        lifetime_s=600,
        sandbox=SandboxConfig(docker_timeout_s=15),
    ) as sandbox:
        started = time.monotonic()
        held = sandbox.execute({"command": "setsid sleep 600 &"})
        elapsed = time.monotonic() - started
        after = sandbox.execute({"command": "echo still here"})

    assert held["returncode"] in (0, 124)
    assert elapsed < 5 + 5 + 15 + 5
    assert after["output"] == "still here\n"


def test_huge_output_is_read_as_it_comes_and_cut(image: str, worktree: Worktree) -> None:
    with SandboxEnvironment(
        image=image, worktree=worktree, command_timeout_s=60, lifetime_s=600
    ) as sandbox:
        output = sandbox.execute({"command": "head -c 300000000 /dev/zero | tr '\\0' a; echo"})

    assert output["returncode"] == 0
    assert output["output"].startswith("[... 299934")
    assert len(output["output"]) < 70_000


def test_a_scripted_run_in_the_sandbox_ends_as_one_commit(
    image: str, workspace: Workspace, worktree: Worktree, tmp_path: Path
) -> None:
    base = workspace.head(worktree)
    skill = "src/goaly/harness/packs/effisql/skills/a.md"
    model = ScriptedModel(
        "touch README.md",  # Refused by the sandbox; the agent sees the error and goes on.
        f"mkdir -p $(dirname {skill}) && echo 'Join first.' > {skill}",
        submit("Add a skill"),
    )
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        implementer = Implementer(
            NoLlm(),
            workspace,
            image=image,
            runs_dir=tmp_path / "runs",
            pool=pool,
            make_model=lambda experiment_id, stop: model,
        )
        done = implementer.run(Hypothesis(hypothesis_id="h-1"), worktree, threading.Event())

    assert done.run.exit == ImplementerExit.IMPLEMENTER_EXIT_SUBMITTED
    assert git(worktree.path, "rev-list", f"{base}..HEAD").split() == [done.commit]
    assert git(worktree.path, "show", "--name-only", "--format=", "HEAD").split() == [skill]
    assert msa.SUBMIT_MARKER not in done.run.summary
