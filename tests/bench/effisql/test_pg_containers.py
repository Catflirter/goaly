import shutil
import subprocess
from pathlib import Path

import pytest

from goaly.bench.effisql.pg_containers import (
    CONTAINER_PREFIX,
    PGDATA,
    SPEC_LABEL,
    ContainerError,
    ContainerSpec,
    Mount,
    ensure,
    logs,
    remove,
    run_args,
    spec_digest,
    start,
    state,
    stop,
)

IMAGE = "postgres@sha256:" + "0" * 64


def _spec(**changes: object) -> ContainerSpec:
    values: dict[str, object] = {
        "name": f"{CONTAINER_PREFIX}test",
        "image": IMAGE,
        "port": 55440,
        "data_dir": Path("/data/pgdata"),
        "user": "1001:1001",
    }
    values.update(changes)
    return ContainerSpec(**values)  # type: ignore[arg-type]  # The test varies one field at a time.


def _option(args: list[str], flag: str) -> list[str]:
    return [args[index + 1] for index, arg in enumerate(args) if arg == flag]


def test_the_server_runs_detached_on_the_host_network_as_the_given_user() -> None:
    args = run_args(_spec())

    assert args[:3] == ["docker", "run", "--detach"]
    assert _option(args, "--name") == [f"{CONTAINER_PREFIX}test"]
    assert "--network=host" in args
    assert _option(args, "--user") == ["1001:1001"]


def test_shared_memory_is_dockers_default_size_as_in_the_official_container() -> None:
    assert _option(run_args(_spec()), "--shm-size") == ["64m"]


def test_the_image_comes_last_followed_only_by_the_port_and_loopback_listening() -> None:
    args = run_args(_spec())

    image_at = args.index(IMAGE)
    assert args[image_at + 1 :] == ["-c", "port=55440", "-c", "listen_addresses=127.0.0.1"]


def test_the_data_directory_is_bind_mounted_where_the_image_keeps_it() -> None:
    mounts = _option(run_args(_spec()), "--mount")

    assert mounts[0] == f"type=bind,source=/data/pgdata,target={PGDATA}"


def test_extra_mounts_follow_and_can_be_read_only() -> None:
    spec = _spec(
        mounts=(Mount(Path("/raw"), "/raw", read_only=True), Mount(Path("/x"), "/y")),
    )

    assert _option(run_args(spec), "--mount")[1:] == [
        "type=bind,source=/raw,target=/raw,readonly",
        "type=bind,source=/x,target=/y",
    ]


def test_cpuset_and_restart_policy_appear_only_when_set() -> None:
    plain = run_args(_spec())
    pinned = run_args(_spec(cpuset="2,3,14,15", restart="unless-stopped"))

    assert "--cpuset-cpus" not in plain
    assert "--restart" not in plain
    assert _option(pinned, "--cpuset-cpus") == ["2,3,14,15"]
    assert _option(pinned, "--restart") == ["unless-stopped"]


def test_environment_variables_are_passed_in_a_stable_order() -> None:
    args = run_args(_spec(env={"B": "2", "A": "1"}))

    assert _option(args, "--env") == ["A=1", "B=2"]


def test_every_container_carries_the_goaly_label_and_its_specs_digest() -> None:
    labels = _option(run_args(_spec()), "--label")

    assert labels == ["goaly.component=effisql-postgres", f"{SPEC_LABEL}={spec_digest(_spec())}"]


def test_the_digest_changes_with_anything_that_changes_the_server() -> None:
    digests = {
        spec_digest(_spec()),
        spec_digest(_spec(cpuset="6,7,18,19")),
        spec_digest(_spec(port=55441)),
        spec_digest(_spec(image="postgres@sha256:" + "1" * 64)),
    }

    assert len(digests) == 4
    assert spec_digest(_spec()) == spec_digest(_spec())


@pytest.mark.parametrize("name", ["postgres", "goaly-other", f"{CONTAINER_PREFIX}bad name"])
def test_only_names_in_our_namespace_are_accepted(name: str) -> None:
    with pytest.raises(ContainerError, match="name"):
        run_args(_spec(name=name))


@pytest.mark.parametrize("path", ["/a,b", "/a=b"])
def test_mount_paths_that_would_break_the_mount_syntax_are_refused(path: str) -> None:
    with pytest.raises(ContainerError, match="mount"):
        run_args(_spec(data_dir=Path(path)))


class FakeDocker:
    """Stands in for the docker CLI: answers by subcommand and records each call."""

    def __init__(self, *, status: str | None = "running", exit_code: str = "0") -> None:
        self.calls: list[list[str]] = []
        self.status = status
        self.exit_code = exit_code
        self.digest = spec_digest(_spec())

    def __call__(self, args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert args[0] == "/usr/bin/docker"
        call = args[1:]
        self.calls.append(call)
        if call[:2] == ["container", "inspect"]:
            if self.status is None:
                return subprocess.CompletedProcess(args, 1, "", "Error: No such container: x")
            if "ExitCode" in call[3]:
                value = self.exit_code
            elif SPEC_LABEL in call[3]:
                value = self.digest
            else:
                value = self.status
            return subprocess.CompletedProcess(args, 0, value + "\n", "")
        if call[0] == "logs":
            return subprocess.CompletedProcess(args, 0, "out line\n", "err line\n")
        if call[0] == "run" and "--fail" in call:
            return subprocess.CompletedProcess(args, 125, "", "docker: error\n")
        return subprocess.CompletedProcess(args, 0, "", "")


@pytest.fixture
def docker(monkeypatch: pytest.MonkeyPatch) -> FakeDocker:
    fake = FakeDocker()
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(subprocess, "run", fake)
    return fake


NAME = f"{CONTAINER_PREFIX}test"


def test_start_runs_the_built_command_line(docker: FakeDocker) -> None:
    start(_spec())

    assert docker.calls == [run_args(_spec())[1:]]


def test_state_reports_the_status_or_none_for_an_absent_container(docker: FakeDocker) -> None:
    assert state(NAME) == "running"
    docker.status = None
    assert state(NAME) is None


def test_stop_returns_the_servers_exit_code(docker: FakeDocker) -> None:
    docker.exit_code = "0"

    assert stop(NAME, timeout_s=60) == 0
    assert ["stop", "--time", "60", NAME] in docker.calls


def test_remove_skips_an_absent_container_and_can_force(docker: FakeDocker) -> None:
    remove(NAME, force=True)
    docker.status = None
    remove(NAME)

    assert [call for call in docker.calls if call[0] == "rm"] == [["rm", "--force", NAME]]


def test_logs_returns_both_streams(docker: FakeDocker) -> None:
    assert logs(NAME, lines=5) == "out line\nerr line"
    assert ["logs", "--tail", "5", NAME] in docker.calls


def test_a_failing_docker_command_raises_with_its_error(
    docker: FakeDocker, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 1, "", "daemon unavailable")

    monkeypatch.setattr(subprocess, "run", failing)

    with pytest.raises(ContainerError, match="exit code 1: daemon unavailable"):
        stop(NAME, timeout_s=1)
    with pytest.raises(ContainerError, match="inspect"):
        state(NAME)


def test_a_missing_docker_binary_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)

    with pytest.raises(ContainerError, match="not on PATH"):
        state(NAME)


def _verbs(docker: FakeDocker) -> list[str]:
    return [call[0] for call in docker.calls if call[:2] != ["container", "inspect"]]


def test_ensure_creates_a_missing_container(docker: FakeDocker) -> None:
    docker.status = None

    assert ensure(_spec()) == "created"
    assert _verbs(docker) == ["run"]


def test_ensure_leaves_a_matching_running_container_alone(docker: FakeDocker) -> None:
    assert ensure(_spec()) == "running"
    assert _verbs(docker) == []


def test_ensure_restarts_a_matching_stopped_container_keeping_it(docker: FakeDocker) -> None:
    docker.status = "exited"

    assert ensure(_spec()) == "started"
    assert _verbs(docker) == ["start"]


def test_ensure_replaces_a_container_whose_spec_changed(docker: FakeDocker) -> None:
    docker.digest = "stale"

    assert ensure(_spec()) == "replaced"
    assert _verbs(docker) == ["stop", "rm", "run"]
