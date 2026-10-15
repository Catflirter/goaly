"""Tests for goaly.rsi.preflight, with fake docker, gateway and PostgreSQL."""

import logging
import os
import stat
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from goaly.contracts.gen.experiment_pb2 import LANE_EFFISQL
from goaly.harness.core.settings import Settings
from goaly.rsi import preflight
from goaly.rsi.preflight import (
    check_disk,
    check_docker,
    check_gateway,
    check_image,
    check_lane,
    check_postgres,
    cleanup,
    run_checks,
)

KEY = "sk-preflight-key"


def fake_docker(tmp_path: Path, script: str) -> str:
    docker = tmp_path / "docker"
    docker.write_text(f"#!/bin/sh\n{script}\n", encoding="utf-8")
    docker.chmod(docker.stat().st_mode | stat.S_IEXEC)
    return str(docker)


def test_docker_answers(tmp_path: Path) -> None:
    assert check_docker(fake_docker(tmp_path, "echo 27.1.0")) is None


@pytest.mark.parametrize(
    ("script", "problem"),
    [
        ("exit 1", "docker does not answer: exit 1"),
        ("true", "docker does not answer: exit 0"),
        ("sleep 5", "docker does not answer: TimeoutExpired"),
    ],
)
def test_docker_problems(tmp_path: Path, script: str, problem: str) -> None:
    assert check_docker(fake_docker(tmp_path, script), timeout_s=0.5) == problem


def test_docker_missing(tmp_path: Path) -> None:
    problem = check_docker(str(tmp_path / "nowhere"))
    assert problem == "docker does not answer: FileNotFoundError"


def settings() -> Settings:
    return Settings(api_key=SecretStr(KEY), base_url="https://gateway.invalid/v1")


def test_gateway_answers_with_the_key() -> None:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"data": []})

    assert check_gateway(settings(), transport=httpx.MockTransport(handle)) is None
    assert str(seen[0].url) == "https://gateway.invalid/v1/models"
    assert seen[0].headers["Authorization"] == f"Bearer {KEY}"


def test_gateway_problems_never_quote_the_reply() -> None:
    reply = httpx.MockTransport(lambda _: httpx.Response(401, text=KEY))
    assert check_gateway(settings(), transport=reply) == "the gateway answered HTTP 401"

    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(KEY)

    problem = check_gateway(settings(), transport=httpx.MockTransport(fail))
    assert problem == "the gateway does not answer: ConnectError"


def test_disk(tmp_path: Path) -> None:
    assert check_disk(tmp_path, 0) is None
    problem = check_disk(tmp_path, 1e9)
    assert problem is not None
    assert problem.endswith("below 1e+09 GB")


def test_image(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(preflight, "ensure_image", lambda repo: "tag")
    assert check_image(tmp_path) is None

    def fail(repo: Path) -> str:
        raise RuntimeError("build failed")

    monkeypatch.setattr(preflight, "ensure_image", fail)
    assert check_image(tmp_path) == "the sandbox image is not ready: RuntimeError"


def test_postgres_runs_the_solving_pools_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class Pool:
        def __init__(self, config: object, credentials: object) -> None:
            calls.append("init")

        async def preflight(self) -> None:
            calls.append("preflight")

    class Module:
        SolvingPool = Pool

        @staticmethod
        def PgEnvConfig() -> object:  # noqa: N802  (as T09 names it)
            return object()

        @staticmethod
        def load_credentials() -> object:
            return object()

    monkeypatch.setattr("importlib.import_module", lambda name: Module)
    assert check_postgres() is None
    assert calls == ["init", "preflight"]

    async def broken(self: Pool) -> None:
        raise OSError("instance solve-1 (port 55433): the agent role cannot log in")

    monkeypatch.setattr(Pool, "preflight", broken)
    assert check_postgres() == (
        "PostgreSQL is not ready: OSError: instance solve-1 (port 55433): the agent role "
        "cannot log in"
    )


def test_postgres_without_t09_is_a_problem(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> object:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("importlib.import_module", missing)
    problem = check_postgres()
    assert problem is not None
    assert problem.startswith("PostgreSQL is not ready: ModuleNotFoundError")


def test_cleanup_removes_containers_and_worktrees(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    calls: list[str] = []

    def containers(**kwargs: object) -> int:
        calls.append(f"containers {kwargs['lane']}")
        return 2

    def worktrees(workspace: object, lane: object) -> int:
        calls.append("worktrees")
        return 1

    monkeypatch.setattr(preflight, "remove_containers", containers)
    monkeypatch.setattr(preflight, "remove_worktrees", worktrees)
    monkeypatch.setattr(preflight, "Workspace", lambda repo, root: object())
    assert cleanup(LANE_EFFISQL, tmp_path, None) is None
    assert calls == [f"containers {LANE_EFFISQL}", "worktrees"]
    assert "removed a crashed lane's leftovers" in caplog.text

    def fail(**_: object) -> int:
        raise RuntimeError("docker is down")

    monkeypatch.setattr(preflight, "remove_containers", fail)
    assert cleanup(LANE_EFFISQL, tmp_path, None) == "cleanup failed: RuntimeError: docker is down"


def test_run_checks_collects_problems_and_survives_a_broken_check() -> None:
    def broken() -> str | None:
        raise ValueError("bug")

    problems = run_checks([("a", lambda: None), ("b", lambda: "b is down"), ("c", broken)])
    assert problems == ["b is down", "c check failed: ValueError"]


def test_check_lane_runs_cleanup_first_and_every_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    order: list[str] = []
    monkeypatch.setattr(preflight, "cleanup", lambda *a: order.append("cleanup"))
    monkeypatch.setattr(preflight, "check_docker", lambda **kw: order.append("docker"))
    monkeypatch.setattr(preflight, "check_image", lambda repo: order.append("image"))
    monkeypatch.setattr(preflight, "load_settings", settings)
    monkeypatch.setattr(preflight, "check_gateway", lambda s, t: order.append("gateway"))
    monkeypatch.setattr(preflight, "check_postgres", lambda t: order.append("postgres"))
    monkeypatch.setattr(preflight, "check_disk", lambda p, g: "disk is full")
    assert check_lane(LANE_EFFISQL, tmp_path, None) == ["disk is full"]
    assert order == ["cleanup", "docker", "image", "gateway", "postgres"]
    assert os.environ.get("GOALY_API_KEY") is None  # Nothing leaked into the environment.
