"""Tests for scripts/effisql/pgenv.py, the environment's command line, loaded from its path."""

import argparse
import importlib.util
import os
from pathlib import Path
from types import ModuleType

import pytest

from goaly.bench.effisql.pg_instances import INSTANCES, InstanceStatus
from goaly.bench.effisql.pg_settings import PgEnvConfig

PGENV = Path(__file__).resolve().parents[3] / "scripts" / "effisql" / "pgenv.py"


@pytest.fixture
def pgenv(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    spec = importlib.util.spec_from_file_location("pgenv_under_test", PGENV)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    lines: list[str] = []
    monkeypatch.setattr(module, "_write", lines.append)
    setattr(module, "written", lines)  # noqa: B010  # What the command printed, for the tests.
    return module


def _status(
    name: str, *, accepting: bool = True, socket: bool = True, cpuset: str | None = None
) -> InstanceStatus:
    instance = next(i for i in INSTANCES if i.name == name)
    pinned = ",".join(str(cpu) for cpu in instance.cpus)
    return InstanceStatus(
        instance,
        "running",
        pinned if cpuset is None else cpuset,
        accepting=accepting,
        socket_dir=Path("/sockets") / name,
        socket_accepting=socket,
    )


def test_status_passes_only_when_every_instance_is_up_and_pinned_right(
    pgenv: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    healthy = [_status("solve-1"), _status("grade-1")]
    monkeypatch.setattr(pgenv, "instance_status", lambda config: healthy)
    assert pgenv._status(PgEnvConfig(data_root=tmp_path), check=False) == 0

    misplaced = [_status("solve-1", cpuset="2,3"), _status("grade-1")]
    monkeypatch.setattr(pgenv, "instance_status", lambda config: misplaced)
    assert pgenv._status(PgEnvConfig(data_root=tmp_path), check=False) == 1
    assert "WRONG: should be 2,3,14,15" in pgenv.written[2]

    down = [_status("solve-1", accepting=False)]
    monkeypatch.setattr(pgenv, "instance_status", lambda config: down)
    assert pgenv._status(PgEnvConfig(data_root=tmp_path), check=False) == 1


def test_status_names_each_socket_and_fails_when_one_does_not_answer(
    pgenv: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    statuses = [_status("solve-1"), _status("grade-1", socket=False)]
    monkeypatch.setattr(pgenv, "instance_status", lambda config: statuses)

    assert pgenv._status(PgEnvConfig(data_root=tmp_path), check=False) == 1
    assert pgenv.written[0].endswith("socket /sockets/solve-1")
    assert pgenv.written[1].endswith("socket /sockets/grade-1 (NOT accepting connections there)")


def test_stop_fails_when_a_server_did_not_shut_down_cleanly(
    pgenv: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    results = {"solve-1": "stopped", "grade-2": "stopped with exit code 137"}
    monkeypatch.setattr(pgenv, "stop_instances", lambda: results)

    assert pgenv._run(argparse.Namespace(command="stop"), PgEnvConfig(data_root=tmp_path)) == 1
    assert pgenv.written == ["solve-1: stopped", "grade-2: stopped with exit code 137"]


def test_a_host_without_our_cpus_fails_cleanly(
    pgenv: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse() -> None:
        raise OSError("Invalid argument")

    monkeypatch.setattr(pgenv, "confine_to_python_cpus", refuse)

    assert pgenv.main(["status"]) == 1  # Reported, not a traceback.
    assert os.sched_getaffinity(0)  # Nothing was changed for the test process.
