"""The Docker sandbox that candidate code runs in (spec §6.1, §8.5, §8.8): one builder, two uses.

- **goaly.rsi's sandboxes** (goaly.rsi.sandbox): the implementer's and the unit-test guard's, on
  a candidate's worktree.
- **The solve phase of goaly run** (spec §6.1, §8.8): :func:`solve_options` and :func:`run_solve`
  run a pack's policy away from the key, the grader and gold. The container sees the candidate's
  checkout and the gold-free instances file, read-only; the LLM proxy's and the solving
  PostgreSQL instances' socket directories, read-only; and ``<run>/solve/``, its only writable
  mount. The host-only files of the run, such as the graded results, the manifest and the
  proxy's call records, are never mounted. Besides PYTHONPATH, which points at the checkout's
  ``src``, the environment holds only what :class:`SolveEnvironment` allows.

The builder lives in goaly.bench, so that goaly run can use it without loading goaly.rsi (spec
§3.1).

Every container runs as the host user with ``--network=host``, the only mode that works on this
machine (spec §1.3). Host networking would put it next to every service on the host's loopback,
some of them unauthenticated, so every container also runs under :data:`SECCOMP_PROFILE`:
Docker's default seccomp profile, with ``socket()`` also refused for IP, netlink and packet
sockets and the 32-bit ``socketcall`` refused, which leaves no way to open a TCP or UDP socket
(V28). Unix sockets still work: the solve container reaches the proxy and PostgreSQL through
socket files in the directories mounted into it. What remains is the host's abstract Unix socket
namespace, which a shared network namespace shares too; nothing on this machine listens there.

Each container gets a read-only root, a tmpfs ``/tmp``, no capabilities, ``no-new-privileges``,
and process, memory, CPU, file-size and open-file limits. Every limit must be above 0, since
docker reads 0 as no limit. The file-size limit bounds each file a writable mount can hold, not
their total, which the guards bound. No host environment variable reaches it, and the docker CLI
itself gets none but its own. Each command runs under ``timeout`` inside the container, because
stopping ``docker exec`` on the host leaves the command running. A one-shot run removes its
container before it returns. Containers carry labels, so that leftovers of a crash can be found
and removed.
"""

import contextlib
import hashlib
import logging
import math
import os
import re
import secrets
import shlex
import shutil
import stat
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

__all__ = [
    "DOCKERFILE",
    "HOST_ONLY_NAMES",
    "IMAGE_REPOSITORY",
    "LABEL_RUN",
    "LABEL_SOLVE",
    "SECCOMP_PROFILE",
    "SandboxConfig",
    "SandboxError",
    "SandboxResult",
    "SolveEnvironment",
    "bind",
    "ensure_image",
    "hardened_options",
    "image_tag",
    "list_containers",
    "mount_path",
    "remove_containers",
    "run_container",
    "run_solve",
    "solve_options",
    "stop_solve",
    "timed_shell",
]

IMAGE_REPOSITORY = "goaly-rsi-sandbox"
DOCKERFILE = Path("docker/rsi-sandbox.Dockerfile")  # Relative to the repository root.
LABEL_SOLVE = "goaly.bench.solve"  # "1" on every solve container.
LABEL_RUN = "goaly.bench.run"
# Docker's default profile with IP, netlink and packet sockets refused too (V28). It is derived from
# Docker 29.7.1's, and tests/bench/test_seccomp.py checks it against that one.
SECCOMP_PROFILE = Path(__file__).with_name("sandbox-seccomp.json")
# The host-only files of a run directory (V27): a directory that holds one is never the solve
# directory.
HOST_ONLY_NAMES = frozenset({"results.jsonl", "manifest.json", "official", "llm_calls.jsonl"})
_SOLVE_DIR = "solve"  # <run>/solve/: the solve container's only writable mount.
_LABEL_DOCKERFILE = "goaly.rsi.dockerfile-sha256"  # On the image: the Dockerfile it came from.

_CPUSET = re.compile(r"\d+(-\d+)?(,\d+(-\d+)?)*")
_SIZE = re.compile(r"[1-9]\d*[bkmg]")  # Never 0, which docker and tmpfs read as no limit.
_LABEL_VALUE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_VARIABLE = re.compile(r"[A-Z_][A-Z0-9_]*")
_LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})
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
    # The CPUs our Python processes keep to, away from the PostgreSQL instances' (spec §6.2):
    # T09's PYTHON_CPUS. TODO(T09): take the default from it once pg_instances lands.
    cpus: str = "0,1,12,13"
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
    # True if the in-container timeout stopped it; also for a command that itself exits 124.
    timed_out: bool


def mount_path(path: Path, what: str = "path") -> str:
    """``path`` as a --mount value. Raises ValueError, naming ``what``, for a relative path, one
    that docker's CSV parsing could misread, a ``..`` that docker would resolve, or a ``:``, which
    would split PYTHONPATH.
    """
    text = str(path)
    misread = any(char in ',":' or not char.isprintable() for char in text)
    if not path.is_absolute() or misread or ".." in path.parts:
        raise ValueError(
            f"the {what} must be absolute, without .., commas, colons or quotes: {text!r}"
        )
    return text


def bind(path: Path, *, writable: bool = False, what: str = "path") -> str:
    """The --mount option that binds ``path`` at its own path, read-only unless ``writable``."""
    root = mount_path(path, what)
    return f"--mount=type=bind,source={root},target={root}" + ("" if writable else ",readonly")


def hardened_options(
    *,
    uid: int,
    gid: int,
    workdir: Path,
    labels: Mapping[str, str],
    config: SandboxConfig,
) -> list[str]:
    """The options every sandbox container starts with, before its mounts: the hardening, the
    ``labels``, and ``workdir`` with its ``src`` on PYTHONPATH.
    """
    root = mount_path(workdir, "workdir")
    return [
        "--init",  # Reaps zombies and passes signals on.
        "--network=host",
        f"--user={uid}:{gid}",
        "--read-only",
        f"--tmpfs=/tmp:rw,nosuid,nodev,noexec,size={config.tmp_size}",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        f"--security-opt=seccomp={SECCOMP_PROFILE}",  # No IP sockets (V28).
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
        *(f"--label={key}={value}" for key, value in labels.items()),
        f"--workdir={root}",
        f"--env=PYTHONPATH={root}/src",
    ]


def timed_shell(timeout_s: float, config: SandboxConfig = SandboxConfig()) -> list[str]:  # noqa: B008
    """Return the in-container command prefix that runs a bash command for at most
    ``timeout_s``: timeout sends TERM, then KILL ``config.kill_after_s`` later.

    Raises ValueError unless ``timeout_s`` is a finite number of seconds above 0: timeout reads 0
    as no limit.
    """
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError(f"the timeout must be finite and above 0 s; got {timeout_s!r}")
    return ["timeout", f"--kill-after={config.kill_after_s:g}s", f"{timeout_s:g}s", "bash", "-c"]


def run_container(
    image: str,
    options: list[str],
    command: str,
    *,
    name: str,
    timeout_s: float,
    environment: Mapping[str, str] | None = None,
    config: SandboxConfig = SandboxConfig(),  # noqa: B008  (frozen, so one default is safe)
) -> SandboxResult:
    """Run the bash ``command`` once in a container of ``image`` started with ``options``, and
    return how it ended.

    The container is removed before this returns. If removing it fails while another error is
    already on its way, that failure is only logged, and removal by label, which a lane runs when
    it starts, is the backstop. ``environment``'s variables reach the container through a private
    env file, deleted as soon as the container has started, never through the command line.
    Raises SandboxError if docker fails or hangs, or after ``timeout_s`` plus the docker
    timeouts, and ValueError for a timeout that would not stop the command. A command that fails
    or times out is a result, not an error.
    """
    shell = [*timed_shell(timeout_s, config), f"exec 2>&1\n{command}"]
    variables = dict(environment or {})
    _check_variables(variables)
    try:
        env_file = _env_file(variables)
        try:
            env_options = [] if env_file is None else [f"--env-file={env_file}"]
            _docker(
                config, ["run", "--detach", f"--name={name}", *options, *env_options, image, *shell]
            )
        finally:
            _discard(env_file)  # The container holds its variables now.
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
    labels: Mapping[str, str],
    *,
    config: SandboxConfig = SandboxConfig(),  # noqa: B008
) -> int:
    """Remove every container, running or not, that carries all of ``labels``, and return how
    many there were.
    """
    containers = list_containers(labels, config=config)
    if containers:
        _docker(config, ["rm", "--force", *containers])
    return len(containers)


def list_containers(
    labels: Mapping[str, str],
    *,
    config: SandboxConfig = SandboxConfig(),  # noqa: B008
) -> list[str]:
    """Return the IDs of every container, running or not, that carries all of ``labels``."""
    filters = [f"--filter=label={key}={value}" for key, value in labels.items()]
    return _docker(config, ["ps", "--all", "--quiet", *filters]).split()


# The solve phase (spec §6.1, §8.8).


class SolveEnvironment(BaseModel):
    """The only variables a solve container gets (spec §6.1, V24, V28): never the real key, a
    grader's credentials or ``~/.config``.

    They reach the container as GOALY_BASE_URL, GOALY_API_KEY, GOALY_LLM_UDS,
    GOALY_PG_SOCKET_DIRS (joined with ``,``, in solving order) and GOALY_PG_AGENT_PASSWORD.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # The HTTP base of the LLM proxy on the host, which holds the real key, such as
    # http://localhost. The client sends its requests through llm_uds, so only the URL's form
    # matters (V28).
    base_url: str
    # The run's dummy token, which only the proxy accepts.
    token: SecretStr
    # The proxy's Unix socket, in a per-run directory of its own that is mounted read-only.
    llm_uds: Path
    # The solving instances' socket directories, in solving order, each mounted read-only. The
    # agent role reaches them with SELECT on tpch alone.
    pg_socket_dirs: tuple[Path, ...]
    pg_agent_password: SecretStr
    # The run, which labels its containers; it travels to goaly run as an argument.
    run_id: str

    @field_validator("base_url")
    @classmethod
    def _check_base_url(cls, value: str) -> str:
        parts = urlsplit(value)
        try:
            parts.port  # noqa: B018  (it raises for a port that is not a number)
        except ValueError:
            local = False
        else:
            local = parts.scheme == "http" and parts.hostname in _LOOPBACK
        extra = "@" in parts.netloc or parts.path not in ("", "/") or parts.query or parts.fragment
        if not local or extra:
            raise ValueError("must be the LLM proxy on this machine, such as http://localhost")
        return value

    @field_validator("token", "pg_agent_password")
    @classmethod
    def _check_secret(cls, value: SecretStr) -> SecretStr:
        if not _env_value(value.get_secret_value()):
            raise ValueError("must be one non-empty line of UTF-8")
        return value

    @field_validator("llm_uds")
    @classmethod
    def _check_socket(cls, value: Path) -> Path:
        mount_path(value, "LLM proxy socket")
        if value.parent == value:
            raise ValueError("must be a socket file in a directory of its own")
        return value

    @field_validator("pg_socket_dirs")
    @classmethod
    def _check_socket_dirs(cls, value: tuple[Path, ...]) -> tuple[Path, ...]:
        if not value:
            raise ValueError("must name one or more socket directories")
        for directory in value:
            mount_path(directory, "PostgreSQL socket directory")
        return value

    @field_validator("run_id")
    @classmethod
    def _check_run_id(cls, value: str) -> str:
        if not _LABEL_VALUE.fullmatch(value):
            raise ValueError(f"must match {_LABEL_VALUE.pattern}")
        return value

    def socket_dirs(self) -> list[Path]:
        """The directories the container reaches its sockets in: the proxy's, then PostgreSQL's."""
        return [self.llm_uds.parent, *self.pg_socket_dirs]

    def variables(self) -> dict[str, str]:
        """The container's environment, by variable name."""
        return {
            "GOALY_BASE_URL": self.base_url,
            "GOALY_API_KEY": self.token.get_secret_value(),
            "GOALY_LLM_UDS": str(self.llm_uds),
            "GOALY_PG_SOCKET_DIRS": ",".join(map(str, self.pg_socket_dirs)),
            "GOALY_PG_AGENT_PASSWORD": self.pg_agent_password.get_secret_value(),
        }


def solve_options(
    checkout: Path,
    solve_dir: Path,
    instances_file: Path | None,
    *,
    socket_dirs: Sequence[Path],
    run_id: str,
    uid: int,
    gid: int,
    config: SandboxConfig = SandboxConfig(),  # noqa: B008
) -> list[str]:
    """Return the ``docker run`` options of the solve container for ``run_id`` (spec §8.8, V27).

    Each path is mounted at its own path, so the paths the command names work inside too:

    - ``checkout``, the candidate's snapshot, read-only, pack directories included. It may hold
      no ``.git``.
    - ``instances_file``, the gold-free file of the run's instances, read-only, if there is one.
    - ``socket_dirs``, the proxy's and the solving instances' socket directories, read-only. Each
      may hold only sockets and PostgreSQL's lock files.
    - ``solve_dir``, ``<run>/solve/``, the only writable mount. It holds no host-only file
      (:data:`HOST_ONLY_NAMES`).

    Nothing else from the host is mounted. Every path must exist and be canonical, as
    ``Path.resolve`` gives it, so that what is checked is what docker mounts. Raises ValueError
    for a path that breaks one of these rules, or that docker could misread, for mounts that
    overlap, and for an invalid run ID.
    """
    if not _LABEL_VALUE.fullmatch(run_id):
        raise ValueError(f"the run ID must match {_LABEL_VALUE.pattern}; got {run_id!r}")
    _check_canonical(checkout, "checkout", is_directory=True)
    _check_canonical(solve_dir, "solve directory", is_directory=True)
    if instances_file is not None:
        _check_canonical(instances_file, "instances file", is_directory=False)
    for directory in socket_dirs:
        _check_canonical(directory, "socket directory", is_directory=True)
    if os.path.lexists(checkout / ".git"):
        raise ValueError(f"the checkout {checkout} holds a .git; mount a snapshot of the commit")
    if solve_dir.name != _SOLVE_DIR:
        raise ValueError(f"the solve directory must be <run>/{_SOLVE_DIR}/; got {solve_dir}")
    host_only = sorted(entry.name for entry in solve_dir.iterdir() if entry.name in HOST_ONLY_NAMES)
    if host_only:
        raise ValueError(f"the solve directory {solve_dir} holds host-only {', '.join(host_only)}")
    mounts = [checkout, solve_dir, *([instances_file] if instances_file else []), *socket_dirs]
    for first, second in ((a, b) for i, a in enumerate(mounts) for b in mounts[i + 1 :]):
        if first.is_relative_to(second) or second.is_relative_to(first):
            raise ValueError(f"the mounts {first} and {second} overlap")
    for directory in socket_dirs:
        _check_socket_dir(directory)
    labels = {LABEL_SOLVE: "1", LABEL_RUN: run_id}
    return [
        *hardened_options(uid=uid, gid=gid, workdir=checkout, labels=labels, config=config),
        bind(checkout, what="checkout"),
        *([bind(instances_file, what="instances file")] if instances_file else []),
        *(bind(directory, what="socket directory") for directory in socket_dirs),
        bind(solve_dir, writable=True, what="solve directory"),
    ]


def run_solve(
    image: str,
    checkout: Path,
    solve_dir: Path,
    instances_file: Path | None,
    *,
    environment: SolveEnvironment,
    argv: Sequence[str],
    timeout_s: float,
    config: SandboxConfig = SandboxConfig(),  # noqa: B008
) -> SandboxResult:
    """Run ``argv``, such as ``["goaly", *request.goaly_args(...)]``, in a fresh solve container,
    as the host user, for at most ``timeout_s``, and return how it ended.

    The mounts are :func:`solve_options`'s, with ``environment``'s socket directories. The
    container is removed before this returns. To cancel a run from another thread, as an async
    launcher does when it is cancelled, call :func:`stop_solve` with its run ID: this then
    raises SandboxError. Raises SandboxError if docker fails or hangs, and ValueError for mounts
    that break solve_options' rules.
    """
    options = solve_options(
        checkout,
        solve_dir,
        instances_file,
        socket_dirs=environment.socket_dirs(),
        run_id=environment.run_id,
        uid=os.getuid(),
        gid=os.getgid(),
        config=config,
    )
    name = f"goaly-solve-{environment.run_id}-{secrets.token_hex(4)}"
    try:
        return run_container(
            image,
            options,
            shlex.join(argv),
            name=name,
            timeout_s=timeout_s,
            environment=environment.variables(),
            config=config,
        )
    except BaseException:
        # A docker run that timed out can leave a container the daemon was still creating.
        with contextlib.suppress(SandboxError):
            stop_solve(environment.run_id, config=config)
        raise


def stop_solve(run_id: str, *, config: SandboxConfig = SandboxConfig()) -> int:  # noqa: B008
    """Stop and remove every container of the solve run ``run_id``, running or not, and return
    how many there were. It cancels a :func:`run_solve` of that run in another thread, and clears
    a crashed run's leftovers. Other runs' containers are never touched.
    """
    if not _LABEL_VALUE.fullmatch(run_id):
        raise ValueError(f"the run ID must match {_LABEL_VALUE.pattern}; got {run_id!r}")
    return remove_containers({LABEL_SOLVE: "1", LABEL_RUN: run_id}, config=config)


def _check_canonical(path: Path, what: str, *, is_directory: bool) -> None:
    """Refuse a path that does not exist, is not canonical, or is not of the kind expected."""
    mount_path(path, what)
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"the {what} {path} cannot be resolved: {error}") from error
    if resolved != path:
        raise ValueError(f"the {what} {path} is not canonical; pass {resolved}")
    mode = path.stat().st_mode
    if not (stat.S_ISDIR(mode) if is_directory else stat.S_ISREG(mode)):
        kind = "a directory" if is_directory else "a regular file"
        raise ValueError(f"the {what} {path} is not {kind}")


def _check_socket_dir(directory: Path) -> None:
    """Refuse a socket directory that holds anything but sockets and PostgreSQL's lock files,
    so that mounting it shows the container nothing else of the host.
    """
    for entry in directory.iterdir():
        mode = entry.lstat().st_mode
        lock = stat.S_ISREG(mode) and entry.name.startswith(".s.PGSQL.")
        if not (stat.S_ISSOCK(mode) or (lock and entry.name.endswith(".lock"))):
            raise ValueError(f"the socket directory {directory} holds {entry.name}, not a socket")


# The image.


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


# The docker CLI.


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise SandboxError(f"cannot read {path}: {error}") from error


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


def _env_value(value: str) -> bool:
    """Whether ``value`` can be one line of an env file: non-empty, one line, valid UTF-8."""
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:  # A lone surrogate.
        return False
    return bool(value) and "\n" not in value and "\r" not in value and "\0" not in value


def _check_variables(variables: Mapping[str, str]) -> None:
    for name, value in variables.items():
        if not _VARIABLE.fullmatch(name) or not (value == "" or _env_value(value)):
            raise ValueError(f"cannot pass the variable {name} in an env file")


def _env_file(variables: Mapping[str, str]) -> Path | None:
    """Write ``variables`` to a new file only this user can read, in a directory of its own. If
    writing fails, nothing is left behind.
    """
    if not variables:
        return None
    path = Path(tempfile.mkdtemp(prefix="goaly-sandbox-env-")) / "env"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.writelines(f"{name}={value}\n" for name, value in variables.items())
    except BaseException:
        _discard(path)
        raise
    return path


def _discard(env_file: Path | None) -> None:
    """Delete the env file and its directory; log a failure, never raise."""
    if env_file is None:
        return
    try:
        env_file.unlink(missing_ok=True)
        env_file.parent.rmdir()
    except OSError:
        _logger.warning("could not delete a sandbox env file", extra={"path": str(env_file)})


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
