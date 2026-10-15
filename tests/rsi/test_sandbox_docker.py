"""Integration tests for goaly.rsi.sandbox with real containers (spec §8.5, §8.8).

They need Docker and run only with ``pytest --integration``. Each test labels its containers
with an experiment ID of its own, so cleanup never touches anyone else's containers.
"""

import os
import secrets
import shlex
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from goaly.contracts.gen.experiment_pb2 import LANE_EFFISQL
from goaly.rsi.sandbox import (
    LABEL_EXPERIMENT,
    SandboxConfig,
    SandboxResult,
    ensure_image,
    remove_containers,
    run_once,
    run_options,
)
from goaly.rsi.workspace import Workspace, Worktree
from tests.rsi.gitrepo import git, make_repo

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def image() -> str:
    return ensure_image(REPO_ROOT)


@pytest.fixture
def worktree(tmp_path: Path) -> Iterator[Worktree]:
    workspace = Workspace(make_repo(tmp_path / "repo"), root=tmp_path / "worktrees")
    worktree = workspace.create(LANE_EFFISQL, f"it-{secrets.token_hex(4)}")
    yield worktree
    leftovers = containers_of(worktree)
    remove_containers(experiment_id=worktree.experiment_id)  # Even after a failed test.
    assert leftovers == [], "a sandbox container outlived its test"


def containers_of(worktree: Worktree) -> list[str]:
    label = f"--filter=label={LABEL_EXPERIMENT}={worktree.experiment_id}"
    return subprocess.run(  # noqa: S603  (docker with arguments built here)
        ["docker", "ps", "--all", "--quiet", label],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()


def run(
    image: str, worktree: Worktree, command: str, *, writable_packs: bool, timeout_s: float = 60
) -> SandboxResult:
    return run_once(
        image,
        worktree.path,
        LANE_EFFISQL,
        experiment_id=worktree.experiment_id,
        purpose="implementer" if writable_packs else "unit-tests",
        command=command,
        timeout_s=timeout_s,
        writable_packs=writable_packs,
    )


def test_a_write_outside_the_pack_fails_with_a_read_only_file_system(
    image: str, worktree: Worktree
) -> None:
    pack, pack_tests = worktree.pack_dirs

    result = run(
        image,
        worktree,
        "touch src/goaly/bench/evil.py; echo bench=$?\n"
        "touch /evil; echo root=$?\n"
        f"echo hi > {pack}/new.md; echo pack=$?\n"
        f"echo hi > {pack_tests}/test_new.py; echo pack_tests=$?\n",
        writable_packs=True,
    )

    assert "src/goaly/bench/evil.py': Read-only file system" in result.output
    assert "bench=1" in result.output
    assert "root=1" in result.output
    assert "pack=0" in result.output
    assert "pack_tests=0" in result.output
    assert not (worktree.path / "src/goaly/bench/evil.py").exists()
    assert (pack / "new.md").stat().st_uid == os.getuid()
    assert (pack_tests / "test_new.py").read_text(encoding="utf-8") == "hi\n"


def test_git_commit_fails_inside_the_sandbox(image: str, worktree: Worktree) -> None:
    head = git(worktree.path, "rev-parse", "HEAD")

    result = run(
        image,
        worktree,
        f"echo change > {worktree.pack_dirs[0]}/prompt.md\n"
        "git add --all && git commit --message=escape",
        writable_packs=True,
    )

    assert result.exit_code != 0
    assert "not a git repository" in result.output
    assert git(worktree.path, "rev-parse", "HEAD") == head


def test_the_containers_environment_holds_no_key(
    image: str, worktree: Worktree, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = f"sentinel-{secrets.token_hex(8)}"
    monkeypatch.setenv("GOALY_API_KEY", sentinel)
    monkeypatch.setenv("PGPASSWORD", sentinel)
    host_config = Path.home() / ".config"

    result = run(
        image,
        worktree,
        f"env; test -e {host_config} && echo HOST_CONFIG_VISIBLE; echo done",
        writable_packs=True,
    )

    assert result.exit_code == 0
    assert "done" in result.output
    assert "PYTHONPATH=" in result.output  # The environment was printed.
    assert sentinel not in result.output
    assert "GOALY_" not in result.output
    assert "PGPASSWORD" not in result.output
    assert "HOST_CONFIG_VISIBLE" not in result.output


def test_a_command_past_its_timeout_is_stopped(image: str, worktree: Worktree) -> None:
    started = time.monotonic()

    result = run(image, worktree, "sleep 300", writable_packs=False, timeout_s=2)

    assert result.exit_code == 124
    assert result.timed_out
    assert time.monotonic() - started < 60


def test_a_read_only_sandbox_cannot_write_the_pack(image: str, worktree: Worktree) -> None:
    pack = worktree.pack_dirs[0]

    result = run(image, worktree, f"echo hi > {pack}/new.md", writable_packs=False)

    assert result.exit_code == 1
    assert "Read-only file system" in result.output
    assert not (pack / "new.md").exists()


def test_a_read_only_sandbox_runs_the_packs_tests(image: str, worktree: Worktree) -> None:
    (worktree.pack_dirs[1] / "test_pack.py").write_text(
        "from goaly.harness.packs.effisql import policy\n\n\n"
        "def test_steps() -> None:\n    assert policy.STEPS == 3\n",
        encoding="utf-8",
    )

    result = run(
        image,
        worktree,
        "python -m pytest -q -p no:cacheprovider tests/harness/packs/effisql",
        writable_packs=False,
    )

    assert result.exit_code == 0, result.output
    assert "1 passed" in result.output
    assert not list(worktree.path.rglob("__pycache__"))


def test_leftover_containers_are_removed_by_label(image: str, worktree: Worktree) -> None:
    options = run_options(
        worktree.path,
        LANE_EFFISQL,
        experiment_id=worktree.experiment_id,
        purpose="implementer",
        writable_packs=True,
        uid=os.getuid(),
        gid=os.getgid(),
    )
    subprocess.run(  # noqa: S603  (docker with arguments built here)
        ["docker", "run", "--detach", *options, image, "sleep", "300"],  # noqa: S607
        check=True,
        capture_output=True,
    )
    assert len(containers_of(worktree)) == 1

    assert remove_containers(lane=LANE_EFFISQL, experiment_id=worktree.experiment_id) == 1


def test_a_file_past_the_size_limit_cannot_be_written(image: str, worktree: Worktree) -> None:
    pack = worktree.path / "src/goaly/harness/packs/effisql"
    config = SandboxConfig(max_file_bytes=1024 * 1024)

    result = run_once(
        image,
        worktree.path,
        LANE_EFFISQL,
        experiment_id=worktree.experiment_id,
        purpose="implementer",
        command=f"head -c 2000000 /dev/zero > {pack}/big; echo status=$?",
        timeout_s=60,
        writable_packs=True,
        config=config,
    )

    assert "status=0" not in result.output, result.output
    assert (pack / "big").stat().st_size <= 1024 * 1024


def test_the_implementer_sandbox_cannot_open_a_tcp_socket(image: str, worktree: Worktree) -> None:
    code = (
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('127.0.0.1', 13337), timeout=5)\n"
        "    print('tcp connected')\n"
        "except PermissionError:\n"
        "    print('tcp refused')\n"
    )

    result = run(image, worktree, f"python -c {shlex.quote(code)}", writable_packs=True)

    assert result.exit_code == 0, result.output
    assert "tcp refused" in result.output
