"""The Docker containers that run the environment's PostgreSQL servers (spec §1.3, §6.2).

Each server runs the pinned postgres image on the host network, the only mode that works on
this machine, and listens on loopback only. It runs as the invoking user, so the data directory
it writes stays ours to copy. Only containers named with :data:`CONTAINER_PREFIX` are ever
started, stopped or removed here; no other container on this machine is touched.
"""

import re
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "CONTAINER_PREFIX",
    "LABEL",
    "PGDATA",
    "SHM_SIZE",
    "ContainerError",
    "ContainerSpec",
    "Mount",
    "logs",
    "remove",
    "run_args",
    "start",
    "state",
    "stop",
]

CONTAINER_PREFIX = "goaly-effisql-"
LABEL = "goaly.component=effisql-postgres"
PGDATA = "/var/lib/postgresql/data"  # Where the postgres image keeps its data directory.
# The official TPC-H container sets no --shm-size, so Docker's default applies. We pin that
# default, so a different Docker cannot change the shared memory that parallel queries use.
SHM_SIZE = "64m"

_NAME = re.compile(rf"{re.escape(CONTAINER_PREFIX)}[a-z0-9][a-z0-9-]*")


class ContainerError(Exception):
    """A container could not be started, inspected or stopped; the message says why."""


@dataclass(frozen=True)
class Mount:
    """A host path bind-mounted into the container."""

    source: Path
    target: str
    read_only: bool = False


@dataclass(frozen=True)
class ContainerSpec:
    """One PostgreSQL server container.

    ``env`` must never hold a secret, because ``docker inspect`` shows it: pass a secret as a
    file in a read-only mount instead, such as the image's ``POSTGRES_PASSWORD_FILE``.
    """

    name: str
    image: str
    port: int
    data_dir: Path
    user: str  # "uid:gid"
    cpuset: str | None = None
    restart: str | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    mounts: tuple[Mount, ...] = ()


def run_args(spec: ContainerSpec) -> list[str]:
    """Return the ``docker run`` command line that starts the server ``spec`` describes."""
    _check_name(spec.name)
    args = ["docker", "run", "--detach", "--name", spec.name, "--network=host"]
    args += ["--shm-size", SHM_SIZE, "--user", spec.user, "--label", LABEL]
    if spec.cpuset is not None:
        args += ["--cpuset-cpus", spec.cpuset]
    if spec.restart is not None:
        args += ["--restart", spec.restart]
    for key, value in sorted(spec.env.items()):
        args += ["--env", f"{key}={value}"]
    for mount in (Mount(spec.data_dir, PGDATA), *spec.mounts):
        args += ["--mount", _mount_arg(mount)]
    # The server settings stay the image's defaults, as in the official container: only the
    # port and the listening address are set, and neither changes how queries run.
    args += [spec.image, "-c", f"port={spec.port}", "-c", "listen_addresses=127.0.0.1"]
    return args


def start(spec: ContainerSpec) -> None:
    """Create and start the container."""
    _docker(run_args(spec)[1:])


def state(name: str) -> str | None:
    """Return the container's state, such as "running" or "exited", or None if it is absent."""
    _check_name(name)
    completed = _docker(
        ["container", "inspect", "--format", "{{.State.Status}}", name], check=False
    )
    if completed.returncode != 0:
        if "no such container" in completed.stderr.lower():
            return None
        raise ContainerError(f"docker container inspect {name} failed: {_tail(completed.stderr)}")
    return completed.stdout.strip()


def stop(name: str, *, timeout_s: int) -> int:
    """Stop the container and return the server's exit code; 0 means a clean shutdown.

    docker stop sends the image's stop signal, SIGINT, PostgreSQL's fast shutdown, which
    writes a shutdown checkpoint. Only after ``timeout_s`` seconds does Docker kill it.
    """
    _check_name(name)
    _docker(["stop", "--time", str(timeout_s), name])
    inspected = _docker(["container", "inspect", "--format", "{{.State.ExitCode}}", name])
    return int(inspected.stdout.strip())


def remove(name: str, *, force: bool = False) -> None:
    """Remove the container if it exists; ``force`` also kills it if it is running."""
    if state(name) is None:
        return
    _docker(["rm", *(["--force"] if force else []), name])


def logs(name: str, *, lines: int = 30) -> str:
    """Return the last ``lines`` lines of the container's log, which never holds a password."""
    _check_name(name)
    completed = _docker(["logs", "--tail", str(lines), name], check=False)
    return (completed.stdout + completed.stderr).strip()


def _docker(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    executable = shutil.which("docker")
    if executable is None:
        raise ContainerError("docker is not on PATH")
    # The arguments are built here from validated names, paths and constants.
    completed = subprocess.run(  # noqa: S603
        [executable, *args], check=False, capture_output=True, text=True
    )
    if check and completed.returncode != 0:
        raise ContainerError(
            f"docker {args[0]} failed with exit code {completed.returncode}: "
            f"{_tail(completed.stderr)}"
        )
    return completed


def _check_name(name: str) -> None:
    if not _NAME.fullmatch(name):
        raise ContainerError(
            f"container name {name!r} is outside our namespace: it must match {_NAME.pattern}"
        )


def _mount_arg(mount: Mount) -> str:
    source = str(mount.source)
    for path in (source, mount.target):
        if "," in path or "=" in path:
            raise ContainerError(f"cannot bind-mount {path!r}: ',' and '=' break the mount syntax")
    return f"type=bind,source={source},target={mount.target}" + (
        ",readonly" if mount.read_only else ""
    )


def _tail(text: str, limit: int = 500) -> str:
    text = text.strip()
    return text if len(text) <= limit else "…" + text[-limit:]
