"""Tests for goaly.rsi.implementer.environment, against a fake docker CLI.

tests/rsi/implementer/test_environment_docker.py runs the same sandbox on real Docker.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from goaly.contracts.gen.experiment_pb2 import LANE_EFFISQL
from goaly.rsi.implementer import msa
from goaly.rsi.implementer.environment import SandboxEnvironment
from goaly.rsi.sandbox import SandboxConfig, SandboxError
from goaly.rsi.workspace import Workspace, Worktree
from tests.rsi.fakedocker import FakeDocker
from tests.rsi.gitrepo import make_repo

SECRET = "sk-live-never-in-docker"  # noqa: S105  (a fake key, to show it stays out)


@pytest.fixture
def docker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeDocker:
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)  # A key in the host process's environment.
    return FakeDocker(tmp_path / "docker")


@pytest.fixture
def worktree(tmp_path: Path) -> Worktree:
    workspace = Workspace(make_repo(tmp_path / "repo"), root=tmp_path / "worktrees")
    return workspace.create(LANE_EFFISQL, "exp-1")


def sandbox(docker: FakeDocker, **changes: Any) -> SandboxConfig:
    return SandboxConfig(docker=str(docker.path), **changes)


@pytest.fixture
def environment(docker: FakeDocker, worktree: Worktree) -> Iterator[SandboxEnvironment]:
    with SandboxEnvironment(
        image="goaly-rsi-sandbox:test",
        worktree=worktree,
        command_timeout_s=120,
        sandbox=sandbox(docker),
    ) as environment:
        yield environment


def test_one_long_lived_sandbox_holds_the_worktree_and_writable_packs(
    docker: FakeDocker, worktree: Worktree, environment: SandboxEnvironment
) -> None:
    (start,) = docker.calls()
    args = start["args"]
    assert args[:2] == ["run", "--detach"]
    name = args[2].removeprefix("--name=")
    assert name.startswith("goaly-rsi-effisql-exp-1-implementer-")
    assert environment.container_id == name
    assert args[-3:] == ["goaly-rsi-sandbox:test", "sleep", "infinity"]
    root = str(worktree.path)
    assert f"--mount=type=bind,source={root},target={root},readonly" in args
    for pack in worktree.pack_dirs:
        assert f"--mount=type=bind,source={pack},target={pack}" in args
    assert "--network=host" in args
    assert "--read-only" in args
    assert "--label=goaly.rsi.experiment=exp-1" in args
    assert "--label=goaly.rsi.purpose=implementer" in args
    assert [arg for arg in args if arg.startswith("--env=")] == [f"--env=PYTHONPATH={root}/src"]
    # Neither the container nor the docker CLI gets the host's key.
    assert SECRET not in str(docker.calls())


def test_each_command_runs_in_the_worktree_under_the_in_container_timeout(
    docker: FakeDocker, worktree: Worktree, environment: SandboxEnvironment
) -> None:
    docker.script({"exec": {"stdout": "x.py\n"}})

    output = environment.execute({"command": "ls src"})

    assert output == {"output": "x.py\n", "returncode": 0, "exception_info": ""}
    exec_call = docker.calls()[-1]["args"]
    assert exec_call == [
        "exec",
        f"--workdir={worktree.path}",
        environment.container_id,
        *("timeout", "--kill-after=5s", "120s", "bash", "-c", "ls src"),
    ]


def test_a_failing_command_is_an_output(
    docker: FakeDocker, environment: SandboxEnvironment
) -> None:
    docker.script(
        {"exec": {"stdout": "Read-only file system\n", "exit": 1}, "inspect": {"stdout": "true\n"}}
    )

    output = environment.execute({"command": "touch src/goaly/bench/x.py"})

    assert output["returncode"] == 1
    assert output["output"] == "Read-only file system\n"
    assert docker.commands()[-2:] == ["exec", "inspect"]


def test_a_command_that_runs_too_long_is_stopped_and_said_so(
    docker: FakeDocker, environment: SandboxEnvironment
) -> None:
    docker.script({"exec": {"stdout": "partial\n", "exit": 124}})

    output = environment.execute({"command": "sleep 999"})

    assert output["returncode"] == 124
    assert output["exception_info"] == "The command timed out after 120 s and was stopped."
    assert docker.commands()[-1] == "exec"  # A timeout is the command's, not the container's.


def test_a_stopped_container_is_a_sandbox_error(
    docker: FakeDocker, environment: SandboxEnvironment
) -> None:
    docker.script({"exec": {"exit": 137}, "inspect": {"stdout": "false\n"}})

    with pytest.raises(SandboxError, match="stopped"):
        environment.execute({"command": "ls"})


def test_a_hung_docker_exec_is_a_sandbox_error(docker: FakeDocker, worktree: Worktree) -> None:
    config = sandbox(docker, kill_after_s=0, docker_timeout_s=0.5)
    with SandboxEnvironment(
        image="image", worktree=worktree, command_timeout_s=0.1, sandbox=config
    ) as environment:
        docker.script({"exec": {"sleep": 5}})

        with pytest.raises(SandboxError, match="timed out"):
            environment.execute({"command": "ls"})
        docker.script({})


def test_output_that_starts_with_the_marker_finishes_the_run(
    docker: FakeDocker, environment: SandboxEnvironment
) -> None:
    docker.script({"exec": {"stdout": f"{msa.SUBMIT_MARKER}\nAdded a skill.\n"}})

    with pytest.raises(msa.Submitted) as finished:
        environment.execute({"command": "printf ..."})

    (message,) = finished.value.messages
    assert message["extra"] == {"exit_status": "Submitted", "submission": "Added a skill.\n"}


def test_long_output_keeps_its_end(docker: FakeDocker, worktree: Worktree) -> None:
    config = sandbox(docker, output_limit=10)
    with SandboxEnvironment(
        image="image", worktree=worktree, command_timeout_s=5, sandbox=config
    ) as environment:
        docker.script({"exec": {"stdout": "a" * 15 + "0123456789"}})

        output = environment.execute({"command": "cat big"})

    assert output["output"] == "[... 15 characters cut ...]\n0123456789"


def test_cleanup_removes_the_container_and_waits(docker: FakeDocker, worktree: Worktree) -> None:
    environment = SandboxEnvironment(
        image="image", worktree=worktree, command_timeout_s=5, sandbox=sandbox(docker)
    )
    name = environment.container_id

    environment.cleanup()
    environment.cleanup()  # Once is enough.
    del environment  # The base class would remove it again, in the background.

    assert docker.commands() == ["run", "rm"]
    assert docker.calls()[-1]["args"] == ["rm", "--force", name]


def test_a_command_after_cleanup_is_a_sandbox_error(docker: FakeDocker, worktree: Worktree) -> None:
    with SandboxEnvironment(
        image="image", worktree=worktree, command_timeout_s=5, sandbox=sandbox(docker)
    ) as environment:
        pass

    with pytest.raises(SandboxError, match="not running"):
        environment.execute({"command": "ls"})


def test_a_failed_removal_is_a_sandbox_error(docker: FakeDocker, worktree: Worktree) -> None:
    environment = SandboxEnvironment(
        image="image", worktree=worktree, command_timeout_s=5, sandbox=sandbox(docker)
    )
    docker.script({"rm": {"stderr": "daemon gone", "exit": 1}})

    with pytest.raises(SandboxError, match="docker rm failed"):
        environment.cleanup()


def test_a_sandbox_that_fails_to_start_is_removed_and_a_sandbox_error(
    docker: FakeDocker, worktree: Worktree
) -> None:
    docker.script({"run": {"stderr": "no such image", "exit": 125}})

    with pytest.raises(SandboxError, match=r"docker run failed \(125\): no such image"):
        SandboxEnvironment(
            image="missing", worktree=worktree, command_timeout_s=5, sandbox=sandbox(docker)
        )

    assert docker.commands() == ["run", "rm"]


def test_a_docker_cli_that_cannot_run_is_a_sandbox_error(worktree: Worktree) -> None:
    config = SandboxConfig(docker="/nonexistent/docker")

    with pytest.raises(SandboxError, match="cannot run"):
        SandboxEnvironment(image="image", worktree=worktree, command_timeout_s=5, sandbox=config)


def test_the_trajectory_records_the_sandbox_without_a_secret(
    environment: SandboxEnvironment,
) -> None:
    data = environment.serialize()

    config = data["info"]["config"]["environment"]
    assert config["image"] == "goaly-rsi-sandbox:test"
    assert "--read-only" in config["run_args"]
    assert SECRET not in str(data)


def test_goaly_smoke_is_answered_on_the_host_and_never_reaches_the_container(
    docker: FakeDocker, worktree: Worktree
) -> None:
    asked: list[str] = []

    def smoke(command: str) -> dict[str, Any]:
        asked.append(command)
        return {"output": "goaly-smoke 3: score 1\n", "returncode": 0, "exception_info": ""}

    with SandboxEnvironment(
        image="image", worktree=worktree, command_timeout_s=5, sandbox=sandbox(docker), smoke=smoke
    ) as environment:
        output = environment.execute({"command": "goaly-smoke 3"})
        environment.execute({"command": "ls"})

    assert asked == ["goaly-smoke 3"]
    assert output["output"] == "goaly-smoke 3: score 1\n"
    assert docker.commands() == ["run", "exec", "rm"]
