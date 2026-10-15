"""The supervisor's checks before every start of a lane (spec §8.7).

:func:`check_lane` first removes what a crashed lane left behind, its sandbox containers (by
T18's labels) and its worktrees, then checks, in order:

- Docker answers;
- the sandbox image exists, built from the current Dockerfile (built now if it is not);
- the model gateway answers ``GET /models`` with the key;
- PostgreSQL: T09's ``SolvingPool.preflight()`` on the solving instances;
- the repository's disk has room.

It returns every problem found, in words that hold no secret; the supervisor counts a start
with any problem as a crash. Each check has a timeout, so none can hang the supervisor.
"""

import asyncio
import importlib
import logging
import shutil
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict, Field

from goaly.contracts.gen.experiment_pb2 import Lane
from goaly.harness.core.settings import Settings, load_settings
from goaly.rsi.resume import remove_worktrees
from goaly.rsi.sandbox import SandboxConfig, ensure_image, remove_containers
from goaly.rsi.workspace import Workspace

__all__ = [
    "PreflightConfig",
    "check_disk",
    "check_docker",
    "check_gateway",
    "check_lane",
    "check_postgres",
    "cleanup",
]

_logger = logging.getLogger(__name__)

type Check = Callable[[], str | None]


class PreflightConfig(BaseModel):
    """The checks' limits."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_free_gb: float = Field(default=20.0, ge=0)
    docker_timeout_s: float = Field(default=60.0, gt=0)
    gateway_timeout_s: float = Field(default=20.0, gt=0)
    postgres_timeout_s: float = Field(default=60.0, gt=0)


def cleanup(lane: Lane.ValueType, repo: Path, worktree_root: Path | None) -> str | None:
    """Remove the lane's leftover sandbox containers and worktrees; return a problem or None."""
    try:
        containers = remove_containers(lane=lane, config=SandboxConfig())
        worktrees = remove_worktrees(Workspace(repo, root=worktree_root), lane)
    except Exception as error:  # noqa: BLE001  (any failure is a failed start)
        return f"cleanup failed: {type(error).__name__}: {error}"
    if containers or worktrees:
        _logger.info(
            "removed a crashed lane's leftovers",
            extra={"containers": containers, "worktrees": worktrees},
        )
    return None


def check_docker(docker: str = "docker", timeout_s: float = 60.0) -> str | None:
    try:
        done = subprocess.run(  # noqa: S603  (docker, with fixed arguments)
            [docker, "info", "--format", "{{.ServerVersion}}"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"docker does not answer: {type(error).__name__}"
    if done.returncode != 0 or not done.stdout.strip():
        return f"docker does not answer: exit {done.returncode}"
    return None


def check_image(repo: Path) -> str | None:
    try:
        ensure_image(repo)
    except Exception as error:  # noqa: BLE001  (any failure is a failed start)
        return f"the sandbox image is not ready: {type(error).__name__}"
    return None


def check_gateway(
    settings: Settings, timeout_s: float = 20.0, transport: httpx.BaseTransport | None = None
) -> str | None:
    """Ask the gateway for its models with the key; the reply's text is never quoted."""
    headers = {"Authorization": f"Bearer {settings.api_key.get_secret_value()}"}
    try:
        with httpx.Client(timeout=timeout_s, transport=transport, trust_env=False) as client:
            response = client.get(f"{settings.base_url}/models", headers=headers)
    except httpx.HTTPError as error:
        return f"the gateway does not answer: {type(error).__name__}"
    if response.status_code != httpx.codes.OK:
        return f"the gateway answered HTTP {response.status_code}"
    return None


def check_postgres(timeout_s: float = 60.0) -> str | None:
    """Run T09's preflight on the solving instances, as the agent role."""
    try:
        sql_env = importlib.import_module("goaly.bench.effisql.sql_env")
        pg_settings = importlib.import_module("goaly.bench.effisql.pg_settings")
        pool = sql_env.SolvingPool(pg_settings.PgEnvConfig(), pg_settings.load_credentials())
        asyncio.run(asyncio.wait_for(pool.preflight(), timeout_s))
    except Exception as error:  # noqa: BLE001  (any failure is a failed start)
        # T09's messages name instances and roles, never a password.
        return f"PostgreSQL is not ready: {type(error).__name__}: {error}"
    return None


def check_disk(path: Path, min_free_gb: float) -> str | None:
    free_gb = shutil.disk_usage(path).free / 1e9
    if free_gb < min_free_gb:
        return f"only {free_gb:.1f} GB free on {path}, below {min_free_gb:g} GB"
    return None


def run_checks(checks: Sequence[tuple[str, Check]]) -> list[str]:
    """Run ``checks`` in order and return their problems; a check that raises is a problem."""
    problems = []
    for name, check in checks:
        try:
            problem = check()
        except Exception as error:  # noqa: BLE001  (a broken check is a failed start)
            problem = f"{name} check failed: {type(error).__name__}"
        if problem is not None:
            problems.append(problem)
    return problems


def check_lane(
    lane: Lane.ValueType,
    repo: Path,
    worktree_root: Path | None,
    config: PreflightConfig = PreflightConfig(),  # noqa: B008  (frozen)
) -> list[str]:
    """Clean up after the lane, then run every check; return the problems, empty if none."""

    def gateway() -> str | None:
        return check_gateway(load_settings(), config.gateway_timeout_s)

    return run_checks(
        [
            ("cleanup", lambda: cleanup(lane, repo, worktree_root)),
            ("docker", lambda: check_docker(timeout_s=config.docker_timeout_s)),
            ("image", lambda: check_image(repo)),
            ("gateway", gateway),
            ("postgres", lambda: check_postgres(config.postgres_timeout_s)),
            ("disk", lambda: check_disk(repo, config.min_free_gb)),
        ]
    )
