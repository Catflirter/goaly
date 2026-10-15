"""Unit tests for goaly.rsi.sandbox, with a fake docker CLI. tests/rsi/test_sandbox_docker.py
runs the real thing.
"""

import hashlib
import logging
import math
import os
from pathlib import Path
from typing import Any

import pytest

from goaly.contracts.gen.experiment_pb2 import LANE_EFFISQL, LANE_SWE
from goaly.rsi import sandbox
from goaly.rsi.sandbox import (
    SandboxConfig,
    SandboxError,
    SandboxResult,
    containers,
    ensure_image,
    image_tag,
    remove_containers,
    run_once,
    run_options,
    timed_shell,
)
from tests.rsi.fakedocker import FakeDocker

WORKTREE = Path("/home/u/goaly-rsi-worktrees/effisql/exp-1")
PACK = "/home/u/goaly-rsi-worktrees/effisql/exp-1/src/goaly/harness/packs/effisql"
PACK_TESTS = "/home/u/goaly-rsi-worktrees/effisql/exp-1/tests/harness/packs/effisql"

READ_ONLY_OPTIONS = [
    "--init",
    "--network=host",
    "--user=1001:1002",
    "--read-only",
    "--tmpfs=/tmp:rw,nosuid,nodev,noexec,size=1g",
    "--cap-drop=ALL",
    "--security-opt=no-new-privileges",
    "--pids-limit=512",
    "--ulimit=fsize=67108864:67108864",
    "--ulimit=nofile=1024:1024",
    "--memory=4g",
    "--memory-swap=4g",
    "--cpuset-cpus=0,1,12,13",
    "--log-driver=json-file",
    "--log-opt=max-size=1m",
    "--log-opt=max-file=2",
    "--label=goaly.rsi.sandbox=1",
    "--label=goaly.rsi.lane=effisql",
    "--label=goaly.rsi.experiment=exp-1",
    "--label=goaly.rsi.purpose=unit-tests",
    f"--workdir={WORKTREE}",
    f"--env=PYTHONPATH={WORKTREE}/src",
    f"--mount=type=bind,source={WORKTREE},target={WORKTREE},readonly",
]


@pytest.fixture
def docker(tmp_path: Path) -> FakeDocker:
    return FakeDocker(tmp_path / "fake-docker")


@pytest.fixture
def config(docker: FakeDocker) -> SandboxConfig:
    return SandboxConfig(docker=str(docker.path), docker_timeout_s=10)


# The container's options.


def test_a_read_only_sandbox_mounts_only_the_worktree_and_read_only() -> None:
    options = run_options(
        WORKTREE,
        LANE_EFFISQL,
        experiment_id="exp-1",
        purpose="unit-tests",
        writable_packs=False,
        uid=1001,
        gid=1002,
    )

    assert options == READ_ONLY_OPTIONS


def test_an_implementer_sandbox_can_write_only_the_lanes_two_pack_directories() -> None:
    options = run_options(
        WORKTREE,
        LANE_EFFISQL,
        experiment_id="exp-1",
        purpose="implementer",
        writable_packs=True,
        uid=1001,
        gid=1002,
    )

    expected = [
        option.replace("purpose=unit-tests", "purpose=implementer") for option in READ_ONLY_OPTIONS
    ]
    assert options == [
        *expected,
        f"--mount=type=bind,source={PACK},target={PACK}",
        f"--mount=type=bind,source={PACK_TESTS},target={PACK_TESTS}",
    ]


def test_the_options_come_from_config() -> None:
    config = SandboxConfig(
        cpus="2,3",
        memory="2g",
        pids_limit=64,
        tmp_size="256m",
        max_file_bytes=1024,
        max_open_files=64,
    )

    options = run_options(
        WORKTREE,
        LANE_SWE,
        experiment_id="exp-1",
        purpose="implementer",
        writable_packs=False,
        uid=0,
        gid=0,
        config=config,
    )

    assert "--cpuset-cpus=2,3" in options
    assert "--memory=2g" in options
    assert "--memory-swap=2g" in options
    assert "--pids-limit=64" in options
    assert "--tmpfs=/tmp:rw,nosuid,nodev,noexec,size=256m" in options
    assert "--label=goaly.rsi.lane=swe" in options
    assert "--ulimit=fsize=1024:1024" in options
    assert "--ulimit=nofile=64:64" in options


def test_no_host_variable_reaches_the_options(monkeypatch: pytest.MonkeyPatch) -> None:
    def options() -> list[str]:
        return run_options(
            WORKTREE,
            LANE_EFFISQL,
            experiment_id="exp-1",
            purpose="implementer",
            writable_packs=True,
            uid=1001,
            gid=1002,
        )

    before = options()
    monkeypatch.setenv("GOALY_API_KEY", "sk-not-a-real-key")
    monkeypatch.setenv("PGPASSWORD", "not-a-real-password")

    assert options() == before
    assert [option for option in before if option.startswith("--env")] == [
        f"--env=PYTHONPATH={WORKTREE}/src"
    ]
    assert not any("docker.sock" in option or ".config" in option for option in before)


@pytest.mark.parametrize(
    "worktree",
    [
        Path("relative/worktree"),
        Path("/home/u/a,b"),
        Path('/home/u/a"b'),
        Path("/home/u/a\nb"),
        Path("/home/u/../etc"),  # Docker would not resolve it the way a reader expects.
        Path("/home/u/a:b"),  # It would split PYTHONPATH in two.
    ],
)
def test_a_worktree_path_docker_could_misread_is_refused(worktree: Path) -> None:
    with pytest.raises(ValueError, match="worktree"):
        run_options(
            worktree,
            LANE_EFFISQL,
            experiment_id="exp-1",
            purpose="implementer",
            writable_packs=True,
            uid=1001,
            gid=1002,
        )


@pytest.mark.parametrize(
    ("experiment_id", "purpose", "problem"),
    [("exp 1", "unit-tests", "experiment_id"), ("exp-1", "unit tests", "purpose")],
)
def test_labels_take_only_valid_ids(experiment_id: str, purpose: str, problem: str) -> None:
    with pytest.raises(ValueError, match=problem):
        run_options(
            WORKTREE,
            LANE_EFFISQL,
            experiment_id=experiment_id,
            purpose=purpose,
            writable_packs=False,
            uid=1001,
            gid=1002,
        )


def test_a_main_repository_is_never_mounted(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()  # A worktree's .git is a file; only a main repository's is not.

    with pytest.raises(ValueError, match="main repository"):
        run_options(
            tmp_path,
            LANE_EFFISQL,
            experiment_id="exp-1",
            purpose="unit-tests",
            writable_packs=False,
            uid=1001,
            gid=1002,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cpus", "0;rm"),
        ("memory", "4 g"),
        ("tmp_size", "1g,exec"),
        ("pids_limit", 0),
        # A limit of 0 would turn the limit off.
        ("memory", "0g"),
        ("tmp_size", "0m"),
        ("kill_after_s", 0),
        ("max_file_bytes", 0),
        ("max_open_files", 0),
    ],
)
def test_config_refuses_values_docker_would_misread(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        SandboxConfig.model_validate({field: value})


@pytest.mark.parametrize("timeout_s", [0, -1, math.inf, math.nan])
def test_a_timeout_that_would_not_stop_the_command_is_refused(timeout_s: float) -> None:
    with pytest.raises(ValueError, match="timeout"):
        timed_shell(timeout_s)


def test_commands_run_under_an_in_container_timeout() -> None:
    assert timed_shell(120, SandboxConfig()) == ["timeout", "--kill-after=5s", "120s", "bash", "-c"]
    assert timed_shell(0.5, SandboxConfig(kill_after_s=2)) == [
        "timeout",
        "--kill-after=2s",
        "0.5s",
        "bash",
        "-c",
    ]


# One-shot runs.


def run(
    config: SandboxConfig,
    command: str = "pytest -q",
    timeout_s: float = 30,
    *,
    writable_packs: bool = False,
) -> SandboxResult:
    return run_once(
        "goaly-rsi-sandbox:abc",
        WORKTREE,
        LANE_EFFISQL,
        experiment_id="exp-1",
        purpose="implementer" if writable_packs else "unit-tests",
        command=command,
        timeout_s=timeout_s,
        writable_packs=writable_packs,
        config=config,
    )


def test_an_implementer_run_mounts_the_packs_writable(
    docker: FakeDocker, config: SandboxConfig
) -> None:
    docker.script({"wait": {"stdout": "0\n"}})

    run(config, writable_packs=True)

    run_args = docker.calls()[0]["args"]
    assert f"--mount=type=bind,source={PACK},target={PACK}" in run_args
    assert f"--mount=type=bind,source={PACK_TESTS},target={PACK_TESTS}" in run_args
    assert "--label=goaly.rsi.purpose=implementer" in run_args


def test_a_one_shot_run_waits_reads_the_output_and_removes_the_container(
    docker: FakeDocker, config: SandboxConfig
) -> None:
    docker.script(
        {"run": {"stdout": "abc123\n"}, "wait": {"stdout": "3\n"}, "logs": {"stdout": "boom\n"}}
    )

    result = run(config)

    assert result == SandboxResult(exit_code=3, output="boom\n", timed_out=False)
    assert docker.commands() == ["run", "wait", "logs", "rm"]
    run_args, wait_args, logs_args, rm_args = (call["args"] for call in docker.calls())
    name = run_args[2].removeprefix("--name=")
    assert run_args[:2] == ["run", "--detach"]
    assert name.startswith("goaly-rsi-effisql-exp-1-unit-tests-")
    host_user = f"--user={os.getuid()}:{os.getgid()}"
    assert run_args[3:-7] == [
        host_user if option.startswith("--user=") else option for option in READ_ONLY_OPTIONS
    ]
    assert run_args[-7:] == [
        "goaly-rsi-sandbox:abc",
        *timed_shell(30, config),
        "exec 2>&1\npytest -q",
    ]
    assert wait_args == ["wait", name]
    assert logs_args == ["logs", name]
    assert rm_args == ["rm", "--force", name]


def test_the_in_container_timeout_shows_as_timed_out(
    docker: FakeDocker, config: SandboxConfig
) -> None:
    docker.script({"wait": {"stdout": "124\n"}})

    result = run(config, timeout_s=1)

    assert result.exit_code == 124
    assert result.timed_out


@pytest.mark.parametrize(("wait_s", "timed_out"), [(0.6, True), (0, False)])
def test_a_kill_counts_as_a_timeout_only_once_the_timeout_has_passed(
    docker: FakeDocker, config: SandboxConfig, wait_s: float, timed_out: bool
) -> None:
    # 137 comes from timeout's KILL after the deadline, or from the kernel at any time.
    docker.script({"wait": {"stdout": "137\n", "sleep": wait_s}})

    result = run(config, timeout_s=0.5)

    assert result.exit_code == 137
    assert result.timed_out is timed_out


def test_messages_on_the_containers_stderr_are_kept(
    docker: FakeDocker, config: SandboxConfig
) -> None:
    docker.script(
        {
            "wait": {"stdout": "124\n"},
            "logs": {"stdout": "collected 3 items\n", "stderr": "timeout: sending signal KILL\n"},
        }
    )

    result = run(config)

    assert "collected 3 items" in result.output
    assert "timeout: sending signal KILL" in result.output


def test_the_wait_allows_the_timeout_the_kill_and_the_docker_margin(
    docker: FakeDocker, config: SandboxConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    docker.script({"wait": {"stdout": "0\n"}})
    limits: dict[str, float | None] = {}
    real_docker = sandbox._docker

    def spy(config: SandboxConfig, args: list[str], **options: Any) -> str:
        limits[args[0]] = options.get("timeout_s")
        return real_docker(config, args, **options)

    monkeypatch.setattr(sandbox, "_docker", spy)

    run(config.model_copy(update={"kill_after_s": 2}), timeout_s=30)

    assert limits["wait"] == 30 + 2 + 10


def test_an_interrupt_while_waiting_still_removes_the_container(
    docker: FakeDocker, config: SandboxConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_docker = sandbox._docker

    def interrupted(config: SandboxConfig, args: list[str], **options: Any) -> str:
        if args[0] == "wait":
            raise KeyboardInterrupt
        return real_docker(config, args, **options)

    monkeypatch.setattr(sandbox, "_docker", interrupted)

    with pytest.raises(KeyboardInterrupt):
        run(config)

    assert docker.commands() == ["run", "rm"]


def test_only_the_end_of_long_output_is_kept(docker: FakeDocker, tmp_path: Path) -> None:
    config = SandboxConfig(docker=str(docker.path), output_limit=10)
    docker.script({"wait": {"stdout": "1\n"}, "logs": {"stdout": "a" * 50 + "0123456789"}})

    result = run(config)

    assert result.output == "[... 50 characters cut ...]\n0123456789"


def test_a_container_that_does_not_start_is_a_sandbox_error_and_is_removed(
    docker: FakeDocker, config: SandboxConfig
) -> None:
    docker.script({"run": {"exit": 125, "stderr": "docker: Error response from daemon: boom"}})

    with pytest.raises(SandboxError, match="boom"):
        run(config)

    assert docker.commands() == ["run", "rm"]


def test_an_unreadable_exit_status_is_a_sandbox_error(
    docker: FakeDocker, config: SandboxConfig
) -> None:
    docker.script({"wait": {"stdout": "Error: unexpected\n"}})

    with pytest.raises(SandboxError, match="exit status"):
        run(config)

    assert docker.commands()[-1] == "rm"


def test_a_container_that_outlives_its_timeout_is_removed(docker: FakeDocker) -> None:
    # docker wait gets the command's timeout plus a margin; past that, docker itself is stuck.
    # The margin leaves the fake docker time to start even on a loaded machine.
    config = SandboxConfig(docker=str(docker.path), docker_timeout_s=3, kill_after_s=0.1)
    docker.script({"wait": {"sleep": 30}})

    with pytest.raises(SandboxError, match="docker wait timed out"):
        run(config, timeout_s=0.5)

    assert docker.commands() == ["run", "wait", "rm"]


def test_a_failed_removal_after_a_run_is_a_sandbox_error(
    docker: FakeDocker, config: SandboxConfig
) -> None:
    docker.script({"wait": {"stdout": "0\n"}, "rm": {"exit": 1, "stderr": "cannot remove"}})

    with pytest.raises(SandboxError, match="cannot remove"):
        run(config)


def test_the_docker_cli_gets_no_secret(
    docker: FakeDocker, config: SandboxConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GOALY_API_KEY", "sk-not-a-real-key")
    monkeypatch.setenv("PGPASSWORD", "not-a-real-password")
    monkeypatch.setenv("DOCKER_HOST", "unix:///run/docker.sock")
    docker.script({"wait": {"stdout": "0\n"}})

    run(config)

    for call in docker.calls():
        assert set(call["env"]) <= {"PATH", "HOME", "LANG", "DOCKER_HOST"}
        assert call["env"]["DOCKER_HOST"] == "unix:///run/docker.sock"


# Cleanup.


def test_remove_containers_finds_leftovers_by_label(
    docker: FakeDocker, config: SandboxConfig
) -> None:
    docker.script({"ps": {"stdout": "c1\nc2\n"}})

    removed = remove_containers(lane=LANE_EFFISQL, experiment_id="exp-1", config=config)

    assert removed == 2
    ps_args, rm_args = (call["args"] for call in docker.calls())
    assert ps_args == [
        "ps",
        "--all",
        "--quiet",
        "--filter=label=goaly.rsi.sandbox=1",
        "--filter=label=goaly.rsi.lane=effisql",
        "--filter=label=goaly.rsi.experiment=exp-1",
    ]
    assert rm_args == ["rm", "--force", "c1", "c2"]


def test_containers_lists_an_experiments_containers_running_or_not(
    docker: FakeDocker, config: SandboxConfig
) -> None:
    docker.script({"ps": {"stdout": "c1\n"}})

    found = containers(experiment_id="exp-1", purpose="implementer", config=config)

    assert found == ["c1"]
    assert docker.calls()[0]["args"] == [
        "ps",
        "--all",
        "--quiet",
        "--filter=label=goaly.rsi.sandbox=1",
        "--filter=label=goaly.rsi.experiment=exp-1",
        "--filter=label=goaly.rsi.purpose=implementer",
    ]


@pytest.mark.parametrize(("experiment_id", "purpose"), [("exp 1", None), ("exp-1", "a b")])
def test_containers_takes_only_valid_labels(
    config: SandboxConfig, experiment_id: str, purpose: str | None
) -> None:
    with pytest.raises(ValueError, match=r"experiment_id|purpose"):
        containers(experiment_id=experiment_id, purpose=purpose, config=config)


def test_remove_containers_with_no_leftovers_removes_nothing(
    docker: FakeDocker, config: SandboxConfig
) -> None:
    assert remove_containers(config=config) == 0
    assert docker.commands() == ["ps"]
    assert docker.calls()[0]["args"] == [
        "ps",
        "--all",
        "--quiet",
        "--filter=label=goaly.rsi.sandbox=1",
    ]


# The image.


def write_project(repo: Path) -> None:
    (repo / "docker").mkdir(parents=True)
    (repo / "docker" / "rsi-sandbox.Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    (repo / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (repo / "secret.env").write_text("never in the context\n", encoding="utf-8")


def test_the_image_is_tagged_with_the_hash_of_the_lock_and_the_dockerfile(tmp_path: Path) -> None:
    write_project(tmp_path)

    lock = hashlib.sha256(b"version = 1\n").hexdigest()
    recipe = hashlib.sha256(b"FROM scratch\n").hexdigest()
    expected = hashlib.sha256(f"uv.lock {lock}\nDockerfile {recipe}\n".encode()).hexdigest()
    assert image_tag(tmp_path) == f"goaly-rsi-sandbox:{expected}"

    (tmp_path / "docker" / "rsi-sandbox.Dockerfile").write_text("FROM debian\n", encoding="utf-8")
    assert image_tag(tmp_path) != f"goaly-rsi-sandbox:{expected}"


@pytest.mark.parametrize("missing", ["uv.lock", "pyproject.toml", "docker/rsi-sandbox.Dockerfile"])
def test_a_missing_project_file_is_a_sandbox_error(
    docker: FakeDocker, config: SandboxConfig, tmp_path: Path, missing: str
) -> None:
    write_project(tmp_path)
    (tmp_path / missing).unlink()

    with pytest.raises(SandboxError, match="cannot read"):
        ensure_image(tmp_path, config=config)


def test_a_missing_image_is_built_from_a_context_of_two_files(
    docker: FakeDocker, config: SandboxConfig, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    write_project(repo)
    recipe = hashlib.sha256(b"FROM scratch\n").hexdigest()

    tag = ensure_image(repo, config=config)

    assert tag == image_tag(repo)
    assert docker.commands() == ["image ls", "build"]
    build = docker.calls()[1]
    assert build["context"] == ["pyproject.toml", "uv.lock"]
    assert build["args"][:-1] == [
        "build",
        "--network=host",
        "--quiet",
        f"--file={repo / 'docker' / 'rsi-sandbox.Dockerfile'}",
        f"--tag={tag}",
        f"--label=goaly.rsi.dockerfile-sha256={recipe}",
    ]


def test_an_image_built_from_this_dockerfile_is_reused(
    docker: FakeDocker, config: SandboxConfig, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    write_project(repo)
    recipe = hashlib.sha256(b"FROM scratch\n").hexdigest()
    docker.script(
        {"image ls": {"stdout": "sha256:1\n"}, "image inspect": {"stdout": recipe + "\n"}}
    )

    ensure_image(repo, config=config)

    assert docker.commands() == ["image ls", "image inspect"]


def test_an_image_built_from_another_dockerfile_is_rebuilt(
    docker: FakeDocker, config: SandboxConfig, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    write_project(repo)
    docker.script(
        {"image ls": {"stdout": "sha256:1\n"}, "image inspect": {"stdout": "<no value>\n"}}
    )

    ensure_image(repo, config=config)

    assert docker.commands() == ["image ls", "image inspect", "build"]


def test_a_failed_build_is_a_sandbox_error(
    docker: FakeDocker, config: SandboxConfig, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    write_project(repo)
    docker.script({"build": {"exit": 1, "stderr": "no space left on device"}})

    with pytest.raises(SandboxError, match="no space left"):
        ensure_image(repo, config=config)


def test_a_missing_docker_cli_is_a_sandbox_error(tmp_path: Path) -> None:
    config = SandboxConfig(docker=str(tmp_path / "no-docker-here"))

    with pytest.raises(SandboxError, match="cannot run"):
        remove_containers(config=config)


def test_a_failed_cleanup_keeps_the_first_error_and_logs_the_second(
    docker: FakeDocker, config: SandboxConfig, caplog: pytest.LogCaptureFixture
) -> None:
    docker.script(
        {
            "run": {"exit": 125, "stderr": "the daemon refused"},
            "rm": {"exit": 1, "stderr": "no daemon"},
        }
    )

    with caplog.at_level(logging.WARNING), pytest.raises(SandboxError, match="refused"):
        run(config)

    assert "could not remove the sandbox container" in caplog.text
