"""Tests for goaly.rsi.implementer.environment, against a fake docker CLI.

tests/rsi/implementer/test_environment_docker.py runs the same sandbox on real Docker.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from goaly.contracts.gen.experiment_pb2 import LANE_EFFISQL
from goaly.rsi.implementer import msa
from goaly.rsi.implementer.environment import SandboxEnvironment, SandboxStoppedError
from goaly.rsi.sandbox import SandboxConfig, SandboxError
from goaly.rsi.workspace import Workspace, Worktree
from tests.rsi.fakedocker import FakeDocker
from tests.rsi.gitrepo import make_repo

SECRET = "sk-live-never-in-docker"  # noqa: S105  (a fake key, to show it stays out)
LABELS = ["--filter=label=goaly.rsi.sandbox=1", "--filter=label=goaly.rsi.experiment=exp-1"]


@pytest.fixture
def docker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeDocker:
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)  # A key in the host process's environment.
    return FakeDocker(tmp_path / "docker")


@pytest.fixture
def worktree(tmp_path: Path) -> Worktree:
    workspace = Workspace(make_repo(tmp_path / "repo"), root=tmp_path / "worktrees")
    return workspace.create(LANE_EFFISQL, "exp-1")


def make(
    docker: FakeDocker, worktree: Worktree, *, command_timeout_s: float = 120, **config: Any
) -> SandboxEnvironment:
    return SandboxEnvironment(
        image="goaly-rsi-sandbox:test",
        worktree=worktree,
        command_timeout_s=command_timeout_s,
        lifetime_s=2700,
        sandbox=SandboxConfig(docker=str(docker.path), **config),
    )


@pytest.fixture
def environment(docker: FakeDocker, worktree: Worktree) -> Iterator[SandboxEnvironment]:
    with make(docker, worktree) as environment:
        yield environment


def test_one_long_lived_sandbox_holds_the_worktree_and_writable_packs(
    docker: FakeDocker, worktree: Worktree, environment: SandboxEnvironment
) -> None:
    leftovers, start = docker.calls()
    assert leftovers["args"] == ["ps", "--all", "--quiet", *LABELS]
    args = start["args"]
    assert args[:2] == ["run", "--detach"]
    name = args[2].removeprefix("--name=")
    assert name.startswith("goaly-rsi-effisql-exp-1-implementer-")
    assert environment.container_id == name
    # A finite lifetime: a container a crashed lane leaves behind ends by itself.
    assert args[-3:] == ["goaly-rsi-sandbox:test", "sleep", "2700"]
    root = str(worktree.path)
    assert f"--mount=type=bind,source={root},target={root},readonly" in args
    for pack in worktree.pack_dirs:
        assert f"--mount=type=bind,source={pack},target={pack}" in args
    assert "--read-only" in args
    assert "--label=goaly.rsi.experiment=exp-1" in args
    assert "--label=goaly.rsi.purpose=implementer" in args
    assert [arg for arg in args if arg.startswith("--env=")] == [f"--env=PYTHONPATH={root}/src"]
    # Neither the container nor the docker CLI gets the host's key.
    assert SECRET not in str(docker.calls())


def test_a_leftover_container_of_the_experiment_is_removed_before_the_start(
    docker: FakeDocker, worktree: Worktree
) -> None:
    docker.script({"ps": {"stdout": "0ld1\n"}})

    environment = make(docker, worktree)
    docker.script({})
    environment.cleanup()

    assert docker.commands()[:3] == ["ps", "rm", "run"]
    assert docker.calls()[1]["args"] == ["rm", "--force", "0ld1"]


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
        {
            "exec": {"stdout": "Read-only file system\n", "exit": 1},
            "inspect": {"stdout": "true 0 false\n"},
        }
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


@pytest.mark.parametrize(
    ("state", "reason"),
    [
        pytest.param("false 0 false", r"its init exited \(0\)", id="killed-its-init"),
        pytest.param("false 137 true", "it ran out of memory", id="out-of-memory"),
    ],
)
def test_a_container_that_stopped_by_itself_rejects_the_candidate(
    docker: FakeDocker, environment: SandboxEnvironment, state: str, reason: str
) -> None:
    docker.script({"exec": {"exit": 137}, "inspect": {"stdout": state}})

    with pytest.raises(SandboxStoppedError, match=reason):
        environment.execute({"command": "pkill sleep"})


@pytest.mark.parametrize(
    "inspect",
    [
        pytest.param({"stderr": "No such container", "exit": 1}, id="gone"),
        pytest.param({"stdout": "<no value>"}, id="no-state"),
    ],
)
def test_a_container_docker_cannot_account_for_is_a_sandbox_error(
    docker: FakeDocker, environment: SandboxEnvironment, inspect: dict[str, Any]
) -> None:
    docker.script({"exec": {"exit": 1}, "inspect": inspect})

    with pytest.raises(SandboxError):
        environment.execute({"command": "ls"})


def test_an_exec_held_open_past_its_time_is_a_timed_out_command(
    docker: FakeDocker, worktree: Worktree
) -> None:
    # As when the command left `setsid cmd &` behind, holding the exec's output open.
    with make(
        docker, worktree, command_timeout_s=0.1, kill_after_s=0.1, docker_timeout_s=0.5
    ) as environment:
        docker.script({"exec": {"sleep": 5}, "inspect": {"stdout": "true 0 false\n"}})

        output = environment.execute({"command": "setsid sleep 600 &"})
        docker.script({})

    assert output["returncode"] == 124
    assert "output stayed open" in output["exception_info"]
    assert "cmd > /tmp/log 2>&1 &" in output["exception_info"]


def test_a_hung_exec_that_docker_cannot_account_for_is_a_sandbox_error(
    docker: FakeDocker, worktree: Worktree
) -> None:
    with make(
        docker, worktree, command_timeout_s=0.1, kill_after_s=0.1, docker_timeout_s=0.5
    ) as environment:
        docker.script({"exec": {"sleep": 5}, "inspect": {"stderr": "daemon gone", "exit": 1}})

        with pytest.raises(SandboxError, match="docker inspect failed"):
            environment.execute({"command": "ls"})
        docker.script({})


def test_a_hung_docker_call_is_a_sandbox_error(docker: FakeDocker, worktree: Worktree) -> None:
    environment = make(docker, worktree, docker_timeout_s=0.5)
    docker.script({"rm": {"sleep": 5}})

    with pytest.raises(SandboxError, match="docker rm timed out"):
        environment.cleanup()


def test_a_docker_warning_does_not_spoil_the_state(
    docker: FakeDocker, environment: SandboxEnvironment
) -> None:
    docker.script(
        {
            "exec": {"stdout": "no such file\n", "exit": 1},
            "inspect": {"stdout": "true 0 false\n", "stderr": "WARNING: config unreadable\n"},
        }
    )

    output = environment.execute({"command": "cat missing"})

    assert output["returncode"] == 1


def test_output_that_starts_with_the_marker_finishes_the_run(
    docker: FakeDocker, environment: SandboxEnvironment
) -> None:
    docker.script({"exec": {"stdout": f"\n{msa.SUBMIT_MARKER}\nAdded a skill.\n"}})

    with pytest.raises(msa.Submitted) as finished:
        environment.execute({"command": "printf ..."})

    (message,) = finished.value.messages
    assert message["extra"] == {"exit_status": "Submitted", "submission": "Added a skill.\n"}


def test_only_the_end_of_long_output_is_kept(docker: FakeDocker, worktree: Worktree) -> None:
    with make(docker, worktree, output_limit=10) as environment:
        docker.script({"exec": {"stdout": "a" * 200_000 + "0123456789"}})

        output = environment.execute({"command": "cat big"})

    assert output["output"] == "[... 200000 bytes cut ...]\n0123456789"


def test_a_long_submission_still_finishes_the_run(docker: FakeDocker, worktree: Worktree) -> None:
    text = f"{msa.SUBMIT_MARKER}\n" + "x" * 50 + "the end."
    with make(docker, worktree, output_limit=10) as environment:
        docker.script({"exec": {"stdout": text}})

        with pytest.raises(msa.Submitted) as finished:
            environment.execute({"command": "printf ..."})

    submission = finished.value.messages[0]["extra"]["submission"]
    assert submission == f"[... {len(text) - 10} bytes cut ...]\nxxthe end."


def test_cleanup_removes_every_container_of_the_experiment_and_checks(
    docker: FakeDocker, worktree: Worktree
) -> None:
    environment = make(docker, worktree)
    name = environment.container_id

    environment.cleanup()
    environment.cleanup()  # No container to remove by name, but it sweeps by label again.
    del environment  # The base class would remove it again, in the background.

    assert docker.commands() == ["ps", "run", "rm", "ps", "ps", "ps", "ps"]
    calls = [call["args"] for call in docker.calls()]
    assert calls[2] == ["rm", "--force", name]
    assert calls[3] == calls[4] == ["ps", "--all", "--quiet", *LABELS]


def test_a_container_of_the_experiment_left_after_cleanup_is_a_sandbox_error(
    docker: FakeDocker, worktree: Worktree
) -> None:
    environment = make(docker, worktree)
    # It is listed, and stays listed after it is removed: as if another one kept starting.
    docker.script({"ps": {"stdout": "4f2a9c\n"}})

    with pytest.raises(SandboxError, match="remain: 4f2a9c"):
        environment.cleanup()


def test_a_command_after_cleanup_is_a_sandbox_error(docker: FakeDocker, worktree: Worktree) -> None:
    with make(docker, worktree) as environment:
        pass

    with pytest.raises(SandboxError, match="not running"):
        environment.execute({"command": "ls"})


def test_a_failed_removal_is_a_sandbox_error(docker: FakeDocker, worktree: Worktree) -> None:
    environment = make(docker, worktree)
    docker.script({"rm": {"stderr": "daemon gone", "exit": 1}})

    with pytest.raises(SandboxError, match="docker rm failed"):
        environment.cleanup()


def test_a_sandbox_that_fails_to_start_is_removed_and_a_sandbox_error(
    docker: FakeDocker, worktree: Worktree
) -> None:
    docker.script({"run": {"stderr": "no such image", "exit": 125}})

    with pytest.raises(SandboxError, match=r"docker run failed \(125\): no such image"):
        make(docker, worktree)

    assert docker.commands() == ["ps", "run", "rm"]


def test_a_docker_cli_that_cannot_run_is_a_sandbox_error(worktree: Worktree) -> None:
    with pytest.raises(SandboxError, match="cannot run"):
        SandboxEnvironment(
            image="image",
            worktree=worktree,
            command_timeout_s=5,
            lifetime_s=60,
            sandbox=SandboxConfig(docker="/nonexistent/docker"),
        )


def test_the_trajectory_records_the_sandbox_without_a_secret(
    environment: SandboxEnvironment,
) -> None:
    data = environment.serialize()

    config = data["info"]["config"]["environment"]
    assert config["image"] == "goaly-rsi-sandbox:test"
    assert "--read-only" in config["run_args"]
    assert SECRET not in str(data)


def test_a_failed_start_whose_removal_fails_too_is_still_the_start_error(
    docker: FakeDocker, worktree: Worktree, caplog: pytest.LogCaptureFixture
) -> None:
    docker.script({"run": {"exit": 125}, "rm": {"exit": 1}})

    with pytest.raises(SandboxError, match="docker run failed"):
        make(docker, worktree)

    assert "could not remove the sandbox container" in caplog.text


def test_a_docker_cli_gone_mid_run_is_a_sandbox_error(
    docker: FakeDocker, environment: SandboxEnvironment
) -> None:
    docker.path.rename(docker.path.with_name("moved"))
    try:
        with pytest.raises(SandboxError, match="cannot run"):
            environment.execute({"command": "ls"})
    finally:
        docker.path.with_name("moved").rename(docker.path)
