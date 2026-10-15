"""Unit tests for goaly.bench.sandbox's solve mode, with a fake docker CLI.

tests/bench/test_sandbox_docker.py runs it in real containers; tests/rsi/test_sandbox.py tests
the implementer's sandbox, which shares this module's options, runs and cleanup.
"""

import os
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from goaly.bench.sandbox import (
    SandboxConfig,
    SandboxError,
    SandboxResult,
    SolveEnvironment,
    remove_solve_containers,
    run_container,
    run_solve,
    solve_options,
)
from tests.rsi.fakedocker import FakeDocker

HARNESS = Path("/srv/goaly/snapshots/abc/tree")
RUN_DIR = Path("/home/u/goaly/runs/val-1")
DATA = Path("/home/u/.cache/goaly/effisql/effi-sql-pg.runtime.jsonl")

SOLVE_OPTIONS = [
    "--init",
    "--network=host",
    "--user=1001:1002",
    "--read-only",
    "--tmpfs=/tmp:rw,nosuid,nodev,noexec,size=1g",
    "--cap-drop=ALL",
    "--security-opt=no-new-privileges",
    "--pids-limit=512",
    "--memory=4g",
    "--memory-swap=4g",
    "--cpuset-cpus=0,1,12,13",
    "--log-driver=json-file",
    "--log-opt=max-size=1m",
    "--log-opt=max-file=2",
    "--label=goaly.bench.solve=1",
    "--label=goaly.bench.run=val-1",
    f"--workdir={HARNESS}",
    f"--env=PYTHONPATH={HARNESS}/src",
    f"--mount=type=bind,source={HARNESS},target={HARNESS},readonly",
    f"--mount=type=bind,source={DATA},target={DATA},readonly",
    f"--mount=type=bind,source={RUN_DIR},target={RUN_DIR}",
]


def environment(**changes: object) -> SolveEnvironment:
    fields: dict[str, object] = {
        "base_url": "http://127.0.0.1:43123",
        "token": "run-token-0123456789",
        "pg_host": "127.0.0.1",
        "pg_ports": (55432, 55433),
        "pg_agent_password": "agent-password",
        "run_id": "val-1",
        "seed": 2,
    }
    return SolveEnvironment.model_validate({**fields, **changes})


# The container's options.


def test_the_solve_container_mounts_the_harness_and_data_read_only_and_the_run_dir_writable() -> (
    None
):
    options = solve_options(HARNESS, RUN_DIR, DATA, run_id="val-1", uid=1001, gid=1002)

    assert options == SOLVE_OPTIONS


@pytest.mark.parametrize(
    ("harness", "run_dir", "data", "problem"),
    [
        (HARNESS, HARNESS / "runs", DATA, "run directory"),  # The harness would be writable.
        (RUN_DIR / "tree", RUN_DIR, DATA, "harness"),
        (HARNESS, RUN_DIR, RUN_DIR / "tasks.jsonl", "data file"),
        (Path("relative"), RUN_DIR, DATA, "absolute"),
        (HARNESS, Path("/home/u/a,b"), DATA, "comma"),
    ],
)
def test_mounts_that_would_widen_the_sandbox_are_refused(
    harness: Path, run_dir: Path, data: Path, problem: str
) -> None:
    with pytest.raises(ValueError, match=problem):
        solve_options(harness, run_dir, data, run_id="val-1", uid=1001, gid=1002)


@pytest.mark.parametrize("run_id", ["", "a=b", "a,b", "-a", "a" * 129])
def test_a_run_id_docker_could_misread_as_a_label_is_refused(run_id: str) -> None:
    with pytest.raises(ValueError, match="run ID"):
        solve_options(HARNESS, RUN_DIR, DATA, run_id=run_id, uid=1001, gid=1002)


def test_the_options_hold_no_host_variable_and_no_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOALY_API_KEY", "sk-not-a-real-key")

    options = solve_options(HARNESS, RUN_DIR, DATA, run_id="val-1", uid=1001, gid=1002)

    assert [option for option in options if option.startswith("--env")] == [
        f"--env=PYTHONPATH={HARNESS}/src"
    ]
    assert not any(".config" in option or "docker.sock" in option for option in options)


# What the container may know.


def test_the_environment_holds_only_the_allowed_variables() -> None:
    assert environment().variables() == {
        "GOALY_BASE_URL": "http://127.0.0.1:43123",
        "GOALY_API_KEY": "run-token-0123456789",
        "GOALY_PG_HOST": "127.0.0.1",
        "GOALY_PG_PORTS": "55432,55433",
        "GOALY_PG_AGENT_PASSWORD": "agent-password",
        "GOALY_RUN_ID": "val-1",
        "GOALY_SEED": "2",
    }


def test_the_environment_never_shows_its_secrets() -> None:
    shown = repr(environment()) + str(environment())

    assert "run-token-0123456789" not in shown
    assert "agent-password" not in shown


@pytest.mark.parametrize(
    ("changes", "problem"),
    [
        ({"base_url": "https://api.goaly.ai/inference/v1"}, "base_url"),  # Never the gateway.
        ({"base_url": "http://10.0.0.5:43123"}, "base_url"),
        ({"base_url": "http://127.0.0.1"}, "base_url"),  # The proxy has a port of its own.
        ({"base_url": "http://:pw@127.0.0.1:43123"}, "base_url"),
        ({"base_url": "http://127.0.0.1:43123/v1"}, "base_url"),
        ({"base_url": "http://127.0.0.1:43123#x"}, "base_url"),
        ({"token": ""}, "token"),
        ({"token": "two\nlines"}, "token"),
        ({"pg_agent_password": "a\nb"}, "pg_agent_password"),
        ({"pg_host": ""}, "pg_host"),
        ({"pg_host": "db host"}, "pg_host"),
        ({"pg_ports": ()}, "pg_ports"),
        ({"pg_ports": (0,)}, "pg_ports"),
        ({"run_id": "a b"}, "run_id"),
        ({"seed": -1}, "seed"),
        ({"extra": "x"}, "extra"),
    ],
)
def test_an_environment_that_could_leak_or_misdirect_is_refused(
    changes: dict[str, object], problem: str
) -> None:
    with pytest.raises(ValidationError, match=problem):
        environment(**changes)


# A solve run.


@pytest.fixture
def docker(tmp_path: Path) -> FakeDocker:
    docker = FakeDocker(tmp_path / "fake-docker")
    docker.script({"wait": {"stdout": "0\n"}, "logs": {"stdout": "solved 30 tasks\n"}})
    return docker


@pytest.fixture
def config(docker: FakeDocker) -> SandboxConfig:
    return SandboxConfig(docker=str(docker.path), docker_timeout_s=30)


def solve(config: SandboxConfig, tmp_path: Path, timeout_s: float = 3600) -> SandboxResult:
    run_dir = tmp_path / "runs" / "val-1"
    run_dir.mkdir(parents=True)
    return run_solve(
        "goaly-rsi-sandbox:abc",
        HARNESS,
        run_dir,
        DATA,
        environment=environment(),
        command="goaly run --phase solve --run-dir runs/val-1",
        timeout_s=timeout_s,
        config=config,
    )


def test_a_solve_run_passes_its_secrets_in_a_private_file_and_removes_the_container(
    docker: FakeDocker, config: SandboxConfig, tmp_path: Path
) -> None:
    result = solve(config, tmp_path)

    assert result == SandboxResult(exit_code=0, output="solved 30 tasks\n", timed_out=False)
    assert docker.commands() == ["run", "wait", "logs", "rm"]
    run = docker.calls()[0]
    env_file = next(arg for arg in run["args"] if arg.startswith("--env-file="))
    assert run["env_file"]["mode"] == "0o600"
    assert sorted(run["env_file"]["text"].splitlines()) == sorted(
        f"{name}={value}" for name, value in environment().variables().items()
    )
    assert not Path(env_file.removeprefix("--env-file=")).exists()  # Gone once it started.
    for call in docker.calls():
        assert "run-token-0123456789" not in " ".join(call["args"])
        assert "agent-password" not in " ".join(call["args"])
    assert run["args"][-1] == "exec 2>&1\ngoaly run --phase solve --run-dir runs/val-1"
    assert (
        f"--mount=type=bind,source={tmp_path}/runs/val-1,target={tmp_path}/runs/val-1"
        in run["args"]
    )


def test_a_solve_run_is_bounded_by_its_timeout(
    docker: FakeDocker, config: SandboxConfig, tmp_path: Path
) -> None:
    docker.script({"wait": {"stdout": "124\n"}})

    result = solve(config, tmp_path, timeout_s=60)

    assert result.timed_out
    run_args = docker.calls()[0]["args"]
    assert run_args[-6:-1] == ["timeout", "--kill-after=5s", "60s", "bash", "-c"]


def test_a_failed_solve_run_still_removes_its_container_and_secrets(
    docker: FakeDocker, config: SandboxConfig, tmp_path: Path
) -> None:
    docker.script({"run": {"exit": 125, "stderr": "no such image"}})

    with pytest.raises(SandboxError, match="no such image"):
        solve(config, tmp_path)

    assert docker.commands() == ["run", "rm"]
    env_file = next(arg for arg in docker.calls()[0]["args"] if arg.startswith("--env-file="))
    assert not Path(env_file.removeprefix("--env-file=")).exists()


@pytest.mark.parametrize(
    "variables", [{"GOALY_X": "two\nlines"}, {"GOALY_X": "nul\0"}, {"lower": "x"}, {"A=B": "x"}]
)
def test_a_variable_an_env_file_cannot_hold_is_refused_before_docker_runs(
    docker: FakeDocker, config: SandboxConfig, variables: dict[str, str]
) -> None:
    with pytest.raises(ValueError, match="env file"):
        run_container(
            "sandbox:1",
            [],
            "true",
            name="goaly-solve-x",
            timeout_s=10,
            environment=variables,
            config=config,
        )

    assert docker.calls() == []


def test_leftover_solve_containers_are_removed_by_label(
    docker: FakeDocker, config: SandboxConfig
) -> None:
    docker.script({"ps": {"stdout": "c1\n"}})

    assert remove_solve_containers(run_id="val-1", config=config) == 1
    assert docker.calls()[0]["args"] == [
        "ps",
        "--all",
        "--quiet",
        "--filter=label=goaly.bench.solve=1",
        "--filter=label=goaly.bench.run=val-1",
    ]


def test_the_host_user_runs_the_container(
    docker: FakeDocker, config: SandboxConfig, tmp_path: Path
) -> None:
    solve(config, tmp_path)

    assert f"--user={os.getuid()}:{os.getgid()}" in docker.calls()[0]["args"]


def test_a_secret_value_is_never_shown_even_through_secretstr() -> None:
    env = environment(token=SecretStr("run-token-0123456789"))

    assert env.variables()["GOALY_API_KEY"] == "run-token-0123456789"
