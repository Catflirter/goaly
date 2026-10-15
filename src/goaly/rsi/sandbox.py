"""The Docker sandbox the implementer and the unit-test guard run in (spec §8.5, §8.8).

A sandbox container mounts one worktree read-only, at its own path, and with ``writable_packs``
the lane's two pack directories read-write; nothing else from the host is mounted. No git
metadata is mounted, so nothing inside can commit; a main repository, whose ``.git`` is a
directory, is refused. A worktree's ``.git`` file stays visible and names the main repository's
path, which is harmless, since that path is not mounted. The container runs as the host user with
a read-only root, a tmpfs ``/tmp``, no capabilities, ``no-new-privileges``, and process, memory,
CPU, file-size and open-file limits (spec §6.2). Every limit must be above 0, since docker reads
0 as no limit. The file-size limit bounds each file the packs can hold, not their total, which
the guards bound. It gets no host environment variable, so no key, and the docker CLI itself gets
none but its own. Host networking is the only mode that works on this machine (spec §1.3), so the
container can reach every port on localhost.

Every command runs under ``timeout`` inside the container, because stopping ``docker exec`` on
the host leaves the command running. Containers carry labels, so leftovers of a crashed lane can
be found and removed, and a one-shot run removes its container before it returns.
"""

import hashlib
import logging
import math
import os
import re
import secrets
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from goaly.contracts.gen.experiment_pb2 import Lane
from goaly.rsi.ids import check_id
from goaly.rsi.lanes import lane_info

__all__ = [
    "DOCKERFILE",
    "IMAGE_REPOSITORY",
    "LABEL_EXPERIMENT",
    "LABEL_LANE",
    "LABEL_PURPOSE",
    "LABEL_SANDBOX",
    "SandboxConfig",
    "SandboxError",
    "SandboxResult",
    "containers",
    "ensure_image",
    "image_tag",
    "remove_containers",
    "run_once",
    "run_options",
    "timed_shell",
]

IMAGE_REPOSITORY = "goaly-rsi-sandbox"
DOCKERFILE = Path("docker/rsi-sandbox.Dockerfile")  # Relative to the repository root.
LABEL_SANDBOX = "goaly.rsi.sandbox"  # "1" on every sandbox container.
LABEL_LANE = "goaly.rsi.lane"
LABEL_EXPERIMENT = "goaly.rsi.experiment"
LABEL_PURPOSE = "goaly.rsi.purpose"  # Such as "implementer" or "unit-tests".
_LABEL_DOCKERFILE = "goaly.rsi.dockerfile-sha256"  # On the image: the Dockerfile it came from.

_CPUSET = re.compile(r"\d+(-\d+)?(,\d+(-\d+)?)*")
_SIZE = re.compile(r"[1-9]\d*[bkmg]")  # Never 0, which docker and tmpfs read as no limit.
_TIMED_OUT = 124  # timeout's exit status once it has stopped the command.
_KILLED = 137  # 128 + SIGKILL: after --kill-after, or when the kernel killed the command.

_logger = logging.getLogger(__name__)


class SandboxError(Exception):
    """Docker failed. This says nothing about the candidate: its experiment is aborted, not
    rejected.
    """


class SandboxConfig(BaseModel):
    """The sandbox's limits and the docker CLI's timeouts, from config."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    docker: str = "docker"
    cpus: str = "0,1,12,13"  # Physical cores 0 and 1, which the sandboxes share (spec §6.2).
    memory: str = "4g"
    tmp_size: str = "1g"
    pids_limit: int = Field(default=512, gt=0)
    max_file_bytes: int = Field(default=64 * 1024 * 1024, gt=0)  # The largest file it can write.
    max_open_files: int = Field(default=1024, gt=0)
    # From timeout's TERM to its KILL; never 0, which would turn the KILL off.
    kill_after_s: float = Field(default=5, gt=0)
    docker_timeout_s: float = Field(default=120, gt=0)  # For each docker call but the command.
    build_timeout_s: float = Field(default=1800, gt=0)
    output_limit: int = Field(default=64 * 1024, gt=0)  # The characters of output kept: the end.

    @field_validator("cpus")
    @classmethod
    def _check_cpus(cls, value: str) -> str:
        if not _CPUSET.fullmatch(value):
            raise ValueError("must be a cpuset such as 0,1,12,13")
        return value

    @field_validator("memory", "tmp_size")
    @classmethod
    def _check_size(cls, value: str) -> str:
        if not _SIZE.fullmatch(value):
            raise ValueError("must be a size such as 4g or 512m")
        return value


@dataclass(frozen=True)
class SandboxResult:
    """How a one-shot command ended."""

    exit_code: int
    # Its standard output and error, interleaved; only the end if it was long.
    output: str
    # True if the in-container timeout stopped it.
    timed_out: bool


def run_options(
    worktree: Path,
    lane: Lane.ValueType,
    *,
    experiment_id: str,
    purpose: str,
    writable_packs: bool,
    uid: int,
    gid: int,
    config: SandboxConfig = SandboxConfig(),  # noqa: B008  (frozen, so one default is safe)
) -> list[str]:
    """Return the ``docker run`` options for a sandbox on ``worktree``, which ``lane`` works in.

    The worktree, a candidate's or a snapshot of one, is mounted read-only at its own path; with
    ``writable_packs`` the lane's two pack directories are mounted read-write over it, and they
    must exist. The image and the command follow these options. Raises ValueError for a worktree
    path docker could misread, a main repository, or an invalid label.
    """
    check_id(experiment_id, "experiment_id")
    check_id(purpose, "purpose")
    info = lane_info(lane)
    root = _mount_path(worktree)
    if (worktree / ".git").is_dir():
        raise ValueError(f"{worktree} is a main repository; mounting it would expose its .git")
    options = [
        "--init",  # Reaps zombies and passes signals on.
        "--network=host",
        f"--user={uid}:{gid}",
        "--read-only",
        f"--tmpfs=/tmp:rw,nosuid,nodev,noexec,size={config.tmp_size}",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        f"--pids-limit={config.pids_limit}",
        f"--ulimit=fsize={config.max_file_bytes}:{config.max_file_bytes}",
        f"--ulimit=nofile={config.max_open_files}:{config.max_open_files}",
        f"--memory={config.memory}",
        f"--memory-swap={config.memory}",  # No swap beyond the memory limit.
        f"--cpuset-cpus={config.cpus}",
        # The log is how a one-shot run reads its output; it is capped so output cannot fill
        # the disk.
        "--log-driver=json-file",
        "--log-opt=max-size=1m",
        "--log-opt=max-file=2",
        f"--label={LABEL_SANDBOX}=1",
        f"--label={LABEL_LANE}={info.name}",
        f"--label={LABEL_EXPERIMENT}={experiment_id}",
        f"--label={LABEL_PURPOSE}={purpose}",
        f"--workdir={root}",
        f"--env=PYTHONPATH={root}/src",
        f"--mount=type=bind,source={root},target={root},readonly",
    ]
    if writable_packs:
        for pack in info.pack_dirs:
            path = f"{root}/{pack}"
            options.append(f"--mount=type=bind,source={path},target={path}")
    return options


def timed_shell(timeout_s: float, config: SandboxConfig = SandboxConfig()) -> list[str]:  # noqa: B008
    """Return the in-container command prefix that runs a bash command for at most
    ``timeout_s``: timeout sends TERM, then KILL ``config.kill_after_s`` later.

    Raises ValueError unless ``timeout_s`` is a finite number of seconds above 0: timeout reads 0
    as no limit.
    """
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError(f"the timeout must be finite and above 0 s; got {timeout_s!r}")
    return ["timeout", f"--kill-after={config.kill_after_s:g}s", f"{timeout_s:g}s", "bash", "-c"]


def run_once(
    image: str,
    worktree: Path,
    lane: Lane.ValueType,
    *,
    experiment_id: str,
    purpose: str,
    command: str,
    timeout_s: float,
    writable_packs: bool = False,
    config: SandboxConfig = SandboxConfig(),  # noqa: B008
) -> SandboxResult:
    """Run the bash ``command`` in a fresh sandbox on ``worktree``, as the host user, and return
    how it ended.

    The container is removed before this returns. If removing it fails while another error is
    already on its way, that failure is only logged, and :func:`remove_containers`, which a lane
    runs when it starts, is the backstop. A run with ``writable_packs`` is the implementer's; any
    other run that evaluates a candidate takes a snapshot of its commit, never the live worktree
    (goaly.rsi.workspace.Workspace.snapshot). Raises SandboxError if docker fails or hangs, and
    ValueError for a timeout that would not stop the command. A command that fails or times out is
    a result, not an error.
    """
    options = run_options(
        worktree,
        lane,
        experiment_id=experiment_id,
        purpose=purpose,
        writable_packs=writable_packs,
        uid=os.getuid(),
        gid=os.getgid(),
        config=config,
    )
    name = f"goaly-rsi-{lane_info(lane).name}-{experiment_id}-{purpose}-{secrets.token_hex(4)}"
    shell = [*timed_shell(timeout_s, config), f"exec 2>&1\n{command}"]
    try:
        _docker(config, ["run", "--detach", f"--name={name}", *options, image, *shell])
        started = time.monotonic()  # The container, and its timeout, have started.
        status = _docker(
            config,
            ["wait", name],
            timeout_s=timeout_s + config.kill_after_s + config.docker_timeout_s,
        )
        elapsed = time.monotonic() - started
        exit_code = _exit_code(status)
        # The command's stderr goes to its stdout; timeout's and the init's messages do not.
        output = _docker(config, ["logs", name], merge_stderr=True)
    except BaseException:
        _remove_quietly(config, name)
        raise
    _docker(config, ["rm", "--force", name])
    timed_out = exit_code == _TIMED_OUT or (exit_code == _KILLED and elapsed >= timeout_s)
    return SandboxResult(exit_code, _tail(output, config.output_limit), timed_out)


def remove_containers(
    *,
    lane: Lane.ValueType | None = None,
    experiment_id: str | None = None,
    config: SandboxConfig = SandboxConfig(),  # noqa: B008
) -> int:
    """Remove every sandbox container, or a lane's or an experiment's, running or not, and
    return how many there were. Containers without the sandbox label are never touched.
    """
    found = containers(lane=lane, experiment_id=experiment_id, config=config)
    if found:
        _docker(config, ["rm", "--force", *found])
    return len(found)


def containers(
    *,
    lane: Lane.ValueType | None = None,
    experiment_id: str | None = None,
    purpose: str | None = None,
    config: SandboxConfig = SandboxConfig(),  # noqa: B008
) -> list[str]:
    """Return the IDs of every sandbox container, running or not, or of those a lane, an
    experiment or a purpose has. An experiment's implementer has stopped for good once this
    lists none of its containers.
    """
    filters = [f"--filter=label={LABEL_SANDBOX}=1"]
    if lane is not None:
        filters.append(f"--filter=label={LABEL_LANE}={lane_info(lane).name}")
    if experiment_id is not None:
        check_id(experiment_id, "experiment_id")
        filters.append(f"--filter=label={LABEL_EXPERIMENT}={experiment_id}")
    if purpose is not None:
        check_id(purpose, "purpose")
        filters.append(f"--filter=label={LABEL_PURPOSE}={purpose}")
    return _docker(config, ["ps", "--all", "--quiet", *filters]).split()


def image_tag(repo: Path) -> str:
    """Return the sandbox image's tag for ``repo``: ``goaly-rsi-sandbox:<hash>``, where the hash
    covers uv.lock and the Dockerfile, so that the tag names what the image was built from.

    The apt packages, git and ripgrep, are not pinned, so two builds with one tag can differ by
    Debian's updates to them. Raises SandboxError if either file cannot be read.
    """
    lock = _sha256(repo / "uv.lock")
    recipe = _sha256(repo / DOCKERFILE)
    both = hashlib.sha256(f"uv.lock {lock}\nDockerfile {recipe}\n".encode()).hexdigest()
    return f"{IMAGE_REPOSITORY}:{both}"


def ensure_image(
    repo: Path,
    *,
    config: SandboxConfig = SandboxConfig(),  # noqa: B008
) -> str:
    """Return the sandbox image's tag for ``repo``, building the image first unless it exists
    and came from the current Dockerfile.

    The build context holds only pyproject.toml and uv.lock. Raises SandboxError if the build
    fails, or if a file it needs cannot be read.
    """
    tag = image_tag(repo)
    dockerfile = repo / DOCKERFILE
    recipe = _sha256(dockerfile)
    if _image_recipe(config, tag) == recipe:
        return tag
    with tempfile.TemporaryDirectory(prefix="goaly-rsi-sandbox-") as context:
        for name in ("pyproject.toml", "uv.lock"):
            try:
                shutil.copyfile(repo / name, Path(context) / name)
            except OSError as error:
                raise SandboxError(f"cannot read {repo / name}: {error}") from error
        _docker(
            config,
            [
                "build",
                "--network=host",
                "--quiet",
                f"--file={dockerfile}",
                f"--tag={tag}",
                f"--label={_LABEL_DOCKERFILE}={recipe}",
                context,
            ],
            timeout_s=config.build_timeout_s,
        )
    return tag


def _image_recipe(config: SandboxConfig, tag: str) -> str | None:
    """The hash of the Dockerfile that built the image ``tag``, or None if it does not exist."""
    if not _docker(config, ["image", "ls", "--quiet", tag]).strip():
        return None
    label = f'{{{{index .Config.Labels "{_LABEL_DOCKERFILE}"}}}}'
    return _docker(config, ["image", "inspect", f"--format={label}", tag]).strip()


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise SandboxError(f"cannot read {path}: {error}") from error


def _mount_path(worktree: Path) -> str:
    """``worktree`` as a --mount value, refusing anything docker's CSV parsing could misread, a
    ``..`` that docker would resolve, and a ``:``, which would split PYTHONPATH.
    """
    path = str(worktree)
    misread = any(char in ',":' or not char.isprintable() for char in path)
    if not worktree.is_absolute() or misread or ".." in worktree.parts:
        raise ValueError(
            f"the worktree path must be absolute, without .., commas, colons or quotes: {path!r}"
        )
    return path


def _docker(
    config: SandboxConfig,
    args: list[str],
    *,
    timeout_s: float | None = None,
    merge_stderr: bool = False,
) -> str:
    """Run the docker CLI and return its output, with its stderr if ``merge_stderr``; raise
    SandboxError if it fails or hangs.
    """
    limit = config.docker_timeout_s if timeout_s is None else timeout_s
    try:
        completed = subprocess.run(  # noqa: S603  (the docker CLI, with arguments built here)
            [config.docker, *args],
            env=_docker_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT if merge_stderr else subprocess.PIPE,
            timeout=limit,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise SandboxError(f"docker {args[0]} timed out after {limit:g} s") from None
    except OSError as error:
        raise SandboxError(f"cannot run {config.docker}: {error}") from error
    if completed.returncode != 0:
        reason = (completed.stderr or completed.stdout).decode("utf-8", "replace").strip()
        raise SandboxError(f"docker {args[0]} failed ({completed.returncode}): {reason}")
    return completed.stdout.decode("utf-8", "replace")


def _docker_environment() -> dict[str, str]:
    """The docker CLI's environment: what it needs to find the daemon, and no secret."""
    environment = {
        name: value
        for name, value in os.environ.items()
        if name in {"PATH", "HOME"} or name.startswith("DOCKER_")
    }
    environment["LANG"] = "C.UTF-8"
    return environment


def _exit_code(status: str) -> int:
    try:
        return int(status.strip())
    except ValueError:
        raise SandboxError(f"docker wait gave no exit status: {status!r}") from None


def _remove_quietly(config: SandboxConfig, name: str) -> None:
    """Remove the container while another error is on its way; log a failure, never raise."""
    try:
        _docker(config, ["rm", "--force", name])
    except SandboxError:
        _logger.warning("could not remove the sandbox container", extra={"container": name})


def _tail(output: str, limit: int) -> str:
    if len(output) <= limit:
        return output
    return f"[... {len(output) - limit} characters cut ...]\n{output[-limit:]}"
