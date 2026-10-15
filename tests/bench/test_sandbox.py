"""Unit tests for goaly.bench.sandbox's solve mode, with a fake docker CLI.

tests/bench/test_sandbox_docker.py runs it in real containers; tests/rsi/test_sandbox.py tests
the implementer's sandbox, which shares this module's options, runs and cleanup.
"""

import logging
import os
import re
import socket
import tempfile
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from goaly.bench.sandbox import (
    SECCOMP_PROFILE,
    SandboxConfig,
    SandboxError,
    SandboxResult,
    SolveEnvironment,
    run_container,
    run_solve,
    solve_options,
    stop_solve,
)
from tests.rsi.fakedocker import FakeDocker


@dataclass(frozen=True)
class Layout:
    """A run's files on the host, as the solve phase finds them."""

    checkout: Path
    run_dir: Path
    solve_dir: Path
    instances: Path
    proxy_dir: Path
    pg_dirs: tuple[Path, Path]

    def socket_dirs(self) -> list[Path]:
        return [self.proxy_dir, *self.pg_dirs]


def unix_socket(path: Path) -> socket.socket:
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(path))
    return listener


@pytest.fixture
def layout(tmp_path: Path) -> Iterator[Layout]:
    root = tmp_path.resolve()
    checkout = root / "snapshot" / "tree"
    (checkout / "src").mkdir(parents=True)
    run_dir = root / "runs" / "val-1"
    solve_dir = run_dir / "solve"
    solve_dir.mkdir(parents=True)
    for name in ("results.jsonl", "manifest.json", "llm_calls.jsonl"):
        (run_dir / name).write_text("{}\n", encoding="utf-8")  # Host-only, beside solve/.
    instances = root / "data" / "effi-sql-pg.runtime.jsonl"
    instances.parent.mkdir()
    instances.write_text('{"id": 1}\n', encoding="utf-8")
    proxy_dir = root / "proxy"
    pg_dirs = (root / "pg-solve-1", root / "pg-solve-2")
    sockets = []
    for directory in (proxy_dir, *pg_dirs):
        directory.mkdir()
    sockets.append(unix_socket(proxy_dir / "llm.sock"))
    for directory, port in zip(pg_dirs, (55432, 55433), strict=True):
        sockets.append(unix_socket(directory / f".s.PGSQL.{port}"))
        (directory / f".s.PGSQL.{port}.lock").write_text("123\n", encoding="utf-8")
    yield Layout(checkout, run_dir, solve_dir, instances, proxy_dir, pg_dirs)
    for listener in sockets:
        listener.close()


def environment(layout: Layout, **changes: object) -> SolveEnvironment:
    fields: dict[str, object] = {
        "base_url": "http://localhost",
        "token": "run-token-0123456789",
        "llm_uds": layout.proxy_dir / "llm.sock",
        "pg_socket_dirs": layout.pg_dirs,
        "pg_agent_password": "agent-password",
        "run_id": "val-1",
    }
    return SolveEnvironment.model_validate({**fields, **changes})


def options(layout: Layout, **changes: object) -> list[str]:
    arguments: dict[str, object] = {
        "checkout": layout.checkout,
        "solve_dir": layout.solve_dir,
        "instances_file": layout.instances,
        "socket_dirs": layout.socket_dirs(),
    }
    arguments.update(changes)
    return solve_options(
        arguments["checkout"],  # type: ignore[arg-type]
        arguments["solve_dir"],  # type: ignore[arg-type]
        arguments["instances_file"],  # type: ignore[arg-type]
        socket_dirs=arguments["socket_dirs"],  # type: ignore[arg-type]
        run_id="val-1",
        uid=1001,
        gid=1002,
    )


# The container's options.


def test_the_solve_container_can_write_only_its_solve_directory(layout: Layout) -> None:
    checkout, solve, data = layout.checkout, layout.solve_dir, layout.instances

    assert options(layout) == [
        "--init",
        "--network=host",
        "--user=1001:1002",
        "--read-only",
        "--tmpfs=/tmp:rw,nosuid,nodev,noexec,size=1g",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        f"--security-opt=seccomp={SECCOMP_PROFILE}",
        "--pids-limit=512",
        "--ulimit=fsize=67108864:67108864",
        "--ulimit=nofile=1024:1024",
        "--memory=4g",
        "--memory-swap=4g",
        "--cpuset-cpus=0,1,12,13",
        "--log-driver=json-file",
        "--log-opt=max-size=1m",
        "--log-opt=max-file=2",
        "--label=goaly.bench.solve=1",
        "--label=goaly.bench.run=val-1",
        f"--workdir={checkout}",
        f"--env=PYTHONPATH={checkout}/src",
        f"--mount=type=bind,source={checkout},target={checkout},readonly",
        f"--mount=type=bind,source={data},target={data},readonly",
        *(
            f"--mount=type=bind,source={directory},target={directory},readonly"
            for directory in layout.socket_dirs()
        ),
        f"--mount=type=bind,source={solve},target={solve}",
    ]


def test_without_an_instances_file_none_is_mounted(layout: Layout) -> None:
    mounts = [option for option in options(layout, instances_file=None) if "--mount" in option]

    assert len(mounts) == 1 + 3 + 1  # The checkout, the socket directories and solve/.


def test_the_profile_ships_with_the_package() -> None:
    assert SECCOMP_PROFILE.is_file()
    assert SECCOMP_PROFILE.is_absolute()


def test_the_run_directory_itself_is_never_the_solve_directory(layout: Layout) -> None:
    with pytest.raises(ValueError, match="<run>/solve/"):
        options(layout, solve_dir=layout.run_dir)


@pytest.mark.parametrize("name", ["results.jsonl", "manifest.json", "official", "llm_calls.jsonl"])
def test_a_solve_directory_holding_a_host_only_file_is_refused(layout: Layout, name: str) -> None:
    (layout.solve_dir / name).mkdir()

    with pytest.raises(ValueError, match=f"host-only {name}"):
        options(layout)


def test_a_checkout_with_git_metadata_is_refused(layout: Layout) -> None:
    (layout.checkout / ".git").write_text("gitdir: /home/u/goaly/.git/worktrees/x\n")

    with pytest.raises(ValueError, match=r"\.git"):
        options(layout)


def test_a_path_through_a_symlink_or_dot_dot_is_refused(layout: Layout, tmp_path: Path) -> None:
    link = layout.run_dir / "solve-link"
    link.symlink_to(layout.checkout / "src")

    with pytest.raises(ValueError, match="canonical"):
        options(layout, solve_dir=link)
    with pytest.raises(ValueError, match=r"\.\."):
        options(layout, solve_dir=layout.run_dir / ".." / "val-1" / "solve")


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        ("missing-solve", "cannot be resolved"),
        ("instances-directory", "not a regular file"),
        ("checkout-file", "not a directory"),
        ("relative", "absolute"),
        ("comma", "comma"),
    ],
)
def test_paths_that_do_not_fit_their_mount_are_refused(
    layout: Layout, tmp_path: Path, change: str, problem: str
) -> None:
    changes: dict[str, object] = {
        "missing-solve": {"solve_dir": layout.run_dir / "nope" / "solve"},
        "instances-directory": {"instances_file": layout.instances.parent},
        "checkout-file": {"checkout": layout.instances},
        "relative": {"checkout": Path("snapshot/tree")},
        "comma": {"checkout": tmp_path.resolve() / "a,b"},
    }[change]  # type: ignore[assignment]

    with pytest.raises(ValueError, match=problem):
        options(layout, **changes)


def test_mounts_that_overlap_are_refused(layout: Layout) -> None:
    inside = layout.checkout / "runs" / "solve"
    inside.mkdir(parents=True)

    with pytest.raises(ValueError, match="overlap"):
        options(layout, solve_dir=inside)  # It would make the checkout writable.
    with pytest.raises(ValueError, match="overlap"):
        options(layout, socket_dirs=[layout.checkout.parent])


def test_a_socket_directory_that_holds_other_files_is_refused(layout: Layout) -> None:
    (layout.pg_dirs[0] / "postgresql.conf").write_text("x\n", encoding="utf-8")

    with pytest.raises(ValueError, match=re.escape("postgresql.conf, not a socket")):
        options(layout)


def test_a_run_id_docker_could_misread_as_a_label_is_refused(layout: Layout) -> None:
    with pytest.raises(ValueError, match="run ID"):
        solve_options(
            layout.checkout,
            layout.solve_dir,
            None,
            socket_dirs=[],
            run_id="a=b",
            uid=1001,
            gid=1002,
        )


def test_the_options_hold_no_host_variable_and_no_secret(
    layout: Layout, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GOALY_API_KEY", "sk-not-a-real-key")

    found = options(layout)

    assert [option for option in found if option.startswith("--env")] == [
        f"--env=PYTHONPATH={layout.checkout}/src"
    ]
    assert not any(".config" in option or "docker.sock" in option for option in found)
    assert not any(str(layout.run_dir / name) in " ".join(found) for name in ("results.jsonl",))


# What the container may know.


def test_the_environment_holds_only_the_allowed_variables(layout: Layout) -> None:
    assert environment(layout).variables() == {
        "GOALY_BASE_URL": "http://localhost",
        "GOALY_API_KEY": "run-token-0123456789",
        "GOALY_LLM_UDS": str(layout.proxy_dir / "llm.sock"),
        "GOALY_PG_SOCKET_DIRS": f"{layout.pg_dirs[0]},{layout.pg_dirs[1]}",
        "GOALY_PG_AGENT_PASSWORD": "agent-password",
    }
    assert environment(layout).socket_dirs() == layout.socket_dirs()


def test_the_environment_never_shows_its_secrets(layout: Layout) -> None:
    shown = repr(environment(layout)) + str(environment(layout))

    assert "run-token-0123456789" not in shown
    assert "agent-password" not in shown


@pytest.mark.parametrize(
    ("changes", "problem"),
    [
        ({"base_url": "https://api.goaly.ai/inference/v1"}, "base_url"),  # Never the gateway.
        ({"base_url": "http://10.0.0.5:43123"}, "base_url"),
        ({"base_url": "http://127.0.0.1:port"}, "base_url"),
        ({"base_url": "unix:///run/proxy.sock"}, "base_url"),
        ({"base_url": "http://:pw@127.0.0.1:43123"}, "base_url"),
        ({"base_url": "http://127.0.0.1:43123/v1"}, "base_url"),
        ({"base_url": "http://127.0.0.1:43123#x"}, "base_url"),
        ({"token": ""}, "token"),
        ({"token": "two\nlines"}, "token"),
        ({"token": "lone\udcffsurrogate"}, "token"),  # It could not be written as UTF-8.
        ({"pg_agent_password": "a\nb"}, "pg_agent_password"),
        ({"llm_uds": Path("llm.sock")}, "llm_uds"),
        ({"llm_uds": Path("/")}, "llm_uds"),
        ({"pg_socket_dirs": ()}, "pg_socket_dirs"),
        ({"pg_socket_dirs": (Path("/a,b"), Path("/c"))}, "pg_socket_dirs"),
        ({"pg_ports": (55432, 55433)}, "pg_ports"),  # The socket directories replace them.
        ({"run_id": "a b"}, "run_id"),
        ({"pg_host": "127.0.0.1"}, "pg_host"),  # TCP is refused in the container anyway.
        ({"seed": 1}, "seed"),  # The seeds travel in tasks.json.
    ],
)
def test_an_environment_that_could_leak_or_misdirect_is_refused(
    layout: Layout, changes: dict[str, object], problem: str
) -> None:
    with pytest.raises(ValidationError, match=problem):
        environment(layout, **changes)


@pytest.mark.parametrize(
    "base_url", ["http://localhost", "http://127.0.0.1:43123", "http://[::1]:8080/"]
)
def test_the_proxy_url_can_be_any_loopback_http_base(layout: Layout, base_url: str) -> None:
    assert environment(layout, base_url=base_url).base_url == base_url


def test_a_secret_value_is_passed_through_secretstr(layout: Layout) -> None:
    env = environment(layout, token=SecretStr("run-token-0123456789"))

    assert env.variables()["GOALY_API_KEY"] == "run-token-0123456789"


# A solve run.


@pytest.fixture
def docker(tmp_path: Path) -> FakeDocker:
    docker = FakeDocker(tmp_path / "fake-docker")
    docker.script({"wait": {"stdout": "0\n"}, "logs": {"stdout": "solved 30 tasks\n"}})
    return docker


@pytest.fixture
def config(docker: FakeDocker) -> SandboxConfig:
    return SandboxConfig(docker=str(docker.path), docker_timeout_s=30)


ARGV = ["goaly", "run", "--phase", "solve", "--in-container", "--llm-share", "qwen=2 3"]


def solve(config: SandboxConfig, layout: Layout, timeout_s: float = 3600) -> SandboxResult:
    return run_solve(
        "goaly-rsi-sandbox:abc",
        layout.checkout,
        layout.solve_dir,
        layout.instances,
        environment=environment(layout),
        argv=ARGV,
        timeout_s=timeout_s,
        config=config,
    )


def test_a_solve_run_passes_its_secrets_in_a_private_file_and_removes_the_container(
    docker: FakeDocker, config: SandboxConfig, layout: Layout
) -> None:
    result = solve(config, layout)

    assert result == SandboxResult(exit_code=0, output="solved 30 tasks\n", timed_out=False)
    assert docker.commands() == ["run", "wait", "logs", "rm"]
    run = docker.calls()[0]
    env_file = next(arg for arg in run["args"] if arg.startswith("--env-file="))
    assert run["env_file"]["mode"] == "0o600"
    assert sorted(run["env_file"]["text"].splitlines()) == sorted(
        f"{name}={value}" for name, value in environment(layout).variables().items()
    )
    assert not Path(env_file.removeprefix("--env-file=")).exists()  # Gone once it started.
    for call in docker.calls():
        assert "run-token-0123456789" not in " ".join(call["args"])
        assert "agent-password" not in " ".join(call["args"])
    # The arguments are quoted for bash, so a value with a space stays one argument.
    assert run["args"][-1] == (
        "exec 2>&1\ngoaly run --phase solve --in-container --llm-share 'qwen=2 3'"
    )
    solve_dir = layout.solve_dir
    assert f"--mount=type=bind,source={solve_dir},target={solve_dir}" in run["args"]


def test_a_solve_run_is_bounded_by_its_timeout(
    docker: FakeDocker, config: SandboxConfig, layout: Layout
) -> None:
    docker.script({"wait": {"stdout": "124\n"}})

    result = solve(config, layout, timeout_s=60)

    assert result.timed_out
    run_args = docker.calls()[0]["args"]
    assert run_args[-6:-1] == ["timeout", "--kill-after=5s", "60s", "bash", "-c"]


def test_a_failed_solve_run_still_removes_its_containers_and_secrets(
    docker: FakeDocker, config: SandboxConfig, layout: Layout
) -> None:
    docker.script({"run": {"exit": 125, "stderr": "no such image"}})

    with pytest.raises(SandboxError, match="no such image"):
        solve(config, layout)

    # By name, then by the run's label, for a container the daemon was still creating.
    assert docker.commands() == ["run", "rm", "ps"]
    assert docker.calls()[2]["args"][-1] == "--filter=label=goaly.bench.run=val-1"
    env_file = next(arg for arg in docker.calls()[0]["args"] if arg.startswith("--env-file="))
    assert not Path(env_file.removeprefix("--env-file=")).exists()


def test_stop_solve_cancels_a_run_in_another_thread(
    docker: FakeDocker, config: SandboxConfig, layout: Layout
) -> None:
    # The fake docker wait returns only once a docker rm has run, as the real one does.
    docker.script(
        {
            "wait": {"until": "rm", "stdout": "137\n"},
            "ps": {"stdout": "c1\n"},
            "logs": {"exit": 1, "stderr": "Error: No such container"},
        }
    )
    outcome: list[BaseException | SandboxResult] = []

    def run() -> None:
        try:
            outcome.append(solve(config, layout))
        except BaseException as error:  # noqa: BLE001  (the test inspects it)
            outcome.append(error)

    worker = threading.Thread(target=run)
    worker.start()
    deadline = time.monotonic() + 20
    while "wait" not in docker.commands() and time.monotonic() < deadline:
        time.sleep(0.05)

    assert stop_solve("val-1", config=config) == 1

    worker.join(timeout=20)
    assert not worker.is_alive()
    assert isinstance(outcome[0], SandboxError)


def test_stop_solve_touches_only_its_own_run(docker: FakeDocker, config: SandboxConfig) -> None:
    docker.script({"ps": {"stdout": "c1\n"}})

    assert stop_solve("val-1", config=config) == 1
    assert docker.calls()[0]["args"] == [
        "ps",
        "--all",
        "--quiet",
        "--filter=label=goaly.bench.solve=1",
        "--filter=label=goaly.bench.run=val-1",
    ]
    with pytest.raises(ValueError, match="run ID"):
        stop_solve("", config=config)


def test_the_host_user_runs_the_container(
    docker: FakeDocker, config: SandboxConfig, layout: Layout
) -> None:
    solve(config, layout)

    assert f"--user={os.getuid()}:{os.getgid()}" in docker.calls()[0]["args"]


# The env file.


@pytest.mark.parametrize(
    "variables",
    [
        {"GOALY_X": "two\nlines"},
        {"GOALY_X": "nul\0"},
        {"GOALY_X": "cr\r"},
        {"GOALY_X": "lone\udcff"},
        {"lower": "x"},
        {"A=B": "x"},
    ],
)
def test_a_variable_an_env_file_cannot_hold_is_refused_before_anything_is_written(
    docker: FakeDocker, config: SandboxConfig, variables: dict[str, str], tmp_path: Path
) -> None:
    before = set(Path(tempfile.gettempdir()).glob("goaly-sandbox-env-*"))

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
    assert set(Path(tempfile.gettempdir()).glob("goaly-sandbox-env-*")) == before


def test_an_env_file_that_cannot_be_written_leaves_nothing_behind(
    docker: FakeDocker, config: SandboxConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_fdopen(descriptor: int, mode: str = "r", encoding: str | None = None) -> None:
        os.close(descriptor)
        raise OSError(28, "No space left on device")

    made: list[str] = []
    real_mkdtemp = tempfile.mkdtemp

    def recording_mkdtemp(prefix: str | None = None) -> str:
        path = real_mkdtemp(prefix=prefix)
        made.append(path)
        return path

    monkeypatch.setattr("goaly.bench.sandbox.tempfile.mkdtemp", recording_mkdtemp)
    monkeypatch.setattr("goaly.bench.sandbox.os.fdopen", failing_fdopen)

    with pytest.raises(OSError, match="No space"):
        run_container(
            "sandbox:1", [], "true", name="x", timeout_s=10, environment={"A": "b"}, config=config
        )

    assert made
    assert not any(Path(path).exists() for path in made)
    assert "run" not in docker.commands()  # Only the cleanup's harmless rm ran.


def test_an_env_file_that_cannot_be_deleted_is_logged_not_raised(
    docker: FakeDocker,
    config: SandboxConfig,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    docker.script({"wait": {"stdout": "0\n"}})
    real_rmdir = Path.rmdir
    kept: list[Path] = []

    def busy(self: Path) -> None:
        if self.name.startswith("goaly-sandbox-env-"):
            kept.append(self)
            raise OSError(16, "Device or resource busy")
        real_rmdir(self)

    monkeypatch.setattr(Path, "rmdir", busy)

    with caplog.at_level(logging.WARNING):
        result = run_container(
            "sandbox:1", [], "true", name="x", timeout_s=10, environment={"A": "b"}, config=config
        )

    assert result.exit_code == 0
    assert "could not delete a sandbox env file" in caplog.text
    for directory in kept:
        real_rmdir(directory)
