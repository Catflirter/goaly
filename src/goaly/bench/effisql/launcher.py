"""Effi-SQL's solve launcher (spec §6.1, ruling V27): the solve phase, in its container.

Core plans an isolated run, takes the cache hits and grades on the host; it calls the launcher
with the tasks left to solve (goaly.harness.core.isolation's SolveRequest). The launcher:

1. stages the request's inputs directory, ``<run>/inputs/``, on the host: the gold-free task
   file, the schema snapshot and the plans the agents see (layout.SolveInputs);
2. starts T03's key-injecting proxy with a fresh token, its per-request records going to the
   run's host-side call log;
3. runs ``python -m goaly`` with the request's arguments in T18's solve container, which gets
   the checkout and the inputs read-only, ``<run>/solve/`` writable, and only the proxy's URL, the
   token and the agent role's password;
4. stops the container and the proxy, and raises unless the container exited 0.

Cancelled, it removes the container before the proxy stops and the cancellation goes on.
"""

import asyncio
import fcntl
import logging
import math
import os
import shutil
import subprocess
import tarfile
import tempfile
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import SecretStr

from goaly.bench.effisql.data import write_runtime_file
from goaly.bench.effisql.harness import EffiSqlPaths, RunCallLog
from goaly.bench.effisql.layout import SolveInputs
from goaly.bench.effisql.pg_instances import SOLVING
from goaly.bench.effisql.pg_settings import AgentCredentials, PgEnvConfig, load_agent_credentials
from goaly.bench.sandbox import (
    SandboxResult,
    SolveEnvironment,
    ensure_image,
    run_solve,
    stop_solve,
)
from goaly.contracts.gen.llm_call_pb2 import LlmCallRecord
from goaly.harness.core.atomic import write_text_atomically
from goaly.harness.core.jsonl_writer import JsonlWriter
from goaly.harness.core.llm import CallRecorder
from goaly.harness.core.llm_proxy import LlmProxy
from goaly.harness.core.settings import load_settings

__all__ = [
    "DockerSandbox",
    "EffiSqlLauncher",
    "SolveContainerError",
    "SolveRequestLike",
    "checkout_of",
    "container_argv",
    "hold_solving_instances",
    "proxy_socket",
    "snapshot_checkout",
    "stage_inputs",
]

_logger = logging.getLogger(__name__)

# A task's wall budget in v0 (config.yaml), which the container's timeout is sized by.
_TASK_WALL_S = 600.0
# How long a container may run beyond its tasks' budgets: start-up, the image, the last writes.
_SLACK_S = 600.0


class SolveContainerError(RuntimeError):
    """The solve container did not exit 0; the message gives its status and its log."""


class SolveRequestLike(Protocol):
    """What the launcher reads of core's SolveRequest (goaly.harness.core.isolation)."""

    @property
    def run_id(self) -> str: ...

    @property
    def pack_dir(self) -> Path | None: ...

    @property
    def solve_dir(self) -> Path: ...

    @property
    def llm_calls(self) -> Path: ...

    @property
    def inputs_dir(self) -> Path: ...

    @property
    def instances_file(self) -> Path | None: ...

    @property
    def tasks(self) -> tuple[tuple[str, int], ...]: ...

    @property
    def max_tasks(self) -> int: ...

    @property
    def task_timeout_s(self) -> float | None: ...

    def goaly_args(
        self, *, solve_dir: Path, inputs_dir: Path, instances_file: Path | None = None
    ) -> list[str]: ...


class Gateway(Protocol):
    """The proxy, as the launcher uses it: LlmProxy, or a test's stand-in.

    Its container_env() names the run's token (GOALY_API_KEY), the client's base URL
    (GOALY_BASE_URL) and, since the container opens no IP socket (V28), the Unix socket it
    listens on (GOALY_LLM_UDS), in a per-run directory of its own.
    """

    def container_env(self) -> dict[str, str]: ...


class Sandbox(Protocol):
    """Runs the solve container: T18's docker sandbox, or a test's stand-in."""

    async def prepare(self, checkout: Path) -> None:
        """Make ready what the container needs, such as its image."""

    async def run(
        self,
        checkout: Path,
        request: SolveRequestLike,
        argv: list[str],
        staged_instances: Path | None,
        environment: SolveEnvironment,
        timeout_s: float,
    ) -> SandboxResult:
        """Run ``argv`` in a solve container and return how it ended."""

    async def stop(self, run_id: str) -> None:
        """Stop and remove the run's solve containers, as after a cancellation."""


class DockerSandbox:
    """T18's solve sandbox (goaly.bench.sandbox), with the image built once per run."""

    def __init__(self) -> None:
        self._image: str | None = None

    async def prepare(self, checkout: Path) -> None:
        self._image = await asyncio.to_thread(ensure_image, checkout)

    async def run(
        self,
        checkout: Path,
        request: SolveRequestLike,
        argv: list[str],
        staged_instances: Path | None,
        environment: SolveEnvironment,
        timeout_s: float,
    ) -> SandboxResult:
        if self._image is None:
            raise RuntimeError("prepare() builds the image first")
        return await asyncio.to_thread(
            run_solve,
            self._image,
            checkout,
            request.solve_dir,
            None,  # A staged organizers' file lies in inputs_dir, which mounts it already.
            inputs_dir=request.inputs_dir,
            environment=environment,
            argv=argv,
            timeout_s=timeout_s,
        )

    async def stop(self, run_id: str) -> None:
        await asyncio.to_thread(stop_solve, run_id)


def _open_proxy(
    recorder: CallRecorder, run_id: str, uds: Path
) -> AbstractAsyncContextManager[Gateway]:
    """T03's proxy for run ``run_id``: it records every request to ``recorder``, on the host,
    and listens on the Unix socket ``uds``, which the container reaches (V28)."""
    return LlmProxy(load_settings(), recorder, run_id=run_id, uds=uds)


# The longest Unix-socket path the kernel takes, less a margin (sockaddr_un.sun_path is 108).
_UDS_MAX = 100


def proxy_socket(run_root: Path, run_id: str) -> Path:
    """Where the run's proxy listens: ``<run>/proxy/llm.sock``, beside solve/, or a short per-run
    directory under /tmp when that path is too long for a Unix socket."""
    socket = run_root / "proxy" / "llm.sock"
    if len(os.fsencode(socket)) <= _UDS_MAX:
        return socket
    return Path(tempfile.gettempdir()) / f"goaly-proxy-{run_id}" / "llm.sock"


def _solving_socket_dirs() -> tuple[Path, ...]:
    """The solving instances' socket directories on the host, in SOLVING order (T09, V28).

    TODO(T09 #26): pg_instances.socket_dirs(PgEnvConfig(), SOLVING), once it is in; this is its
    documented default, ``~/data/effisql/sockets/<name>``.
    """
    root = PgEnvConfig().data_root / "sockets"
    return tuple(root / instance.name for instance in SOLVING)


def _grading_lock_dir() -> Path:
    """Where grader processes lock each instance they grade on (T10): ``~/.cache/goaly/grading``.

    TODO(T10 #36): grading_service.default_lock_dir(), once the stack has it.
    """
    return Path.home() / ".cache" / "goaly" / "grading"


@contextmanager
def hold_solving_instances(lock_dir: Path, names: Sequence[str]) -> Iterator[None]:
    """Share the solving instances' grading locks while a solve phase uses them.

    A grader that grades on every instance (`goaly run --grading-instances all`, for T23's final
    runs) locks each one exclusively, so it cannot start while any solve phase holds these, and a
    solve phase cannot start while it grades. Solve phases share the locks with each other.
    """
    lock_dir.mkdir(parents=True, exist_ok=True)
    handles = []
    try:
        for name in names:
            handle = (lock_dir / f"{name}.lock").open("a", encoding="utf-8")
            handles.append(handle)
            try:
                fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SolveContainerError(
                    f"{name} is grading (goaly run --grading-instances all): solve once it is done"
                ) from None
        yield
    finally:
        for handle in handles:
            handle.close()


@dataclass(frozen=True)
class EffiSqlLauncher:
    """Effi-SQL's SolveLauncher; see the module docstring. Tests replace its parts."""

    paths: Callable[[], EffiSqlPaths] = EffiSqlPaths.default
    sandbox: Callable[[], Sandbox] = DockerSandbox
    proxy: Callable[[CallRecorder, str, Path], AbstractAsyncContextManager[Gateway]] = _open_proxy
    # Only the agent role's password: the container never gets the admin's or the grader's.
    credentials: Callable[[], AgentCredentials] = load_agent_credentials
    # The solving instances' Unix-socket directories, which the container mounts (V28).
    socket_dirs: Callable[[], tuple[Path, ...]] = _solving_socket_dirs
    # Where T10's graders lock the instances they grade on.
    lock_dir: Callable[[], Path] = _grading_lock_dir

    async def __call__(self, request: SolveRequestLike) -> None:
        if not request.tasks:
            raise ValueError("a solve request names at least one task")
        for what, path in (("solve", request.solve_dir), ("inputs", request.inputs_dir)):
            if path != path.resolve():  # The sandbox mounts each at its host path, as named.
                raise ValueError(f"the {what} directory must be a canonical path: {path}")
        inputs = SolveInputs(request.inputs_dir)
        staged = await asyncio.to_thread(stage_inputs, inputs, self.paths(), request.instances_file)
        checkout = await asyncio.to_thread(
            snapshot_checkout, checkout_of(request.pack_dir), request.solve_dir.parent / "checkout"
        )
        credentials = await asyncio.to_thread(self.credentials)
        sandbox = self.sandbox()
        await sandbox.prepare(checkout)
        argv = container_argv(request, staged)
        timeout_s = _timeout(len(request.tasks), request.max_tasks, request.task_timeout_s)
        log = request.solve_dir.parent / "solve-container.log"  # Host-side, beside solve/.
        uds = proxy_socket(request.solve_dir.parent, request.run_id)
        async with (
            self._recorder(request.llm_calls) as recorder,
            self.proxy(recorder, request.run_id, uds) as gateway,
        ):
            env = gateway.container_env()
            environment = SolveEnvironment(
                base_url=env["GOALY_BASE_URL"],
                token=SecretStr(env["GOALY_API_KEY"]),
                llm_uds=Path(env["GOALY_LLM_UDS"]),
                pg_socket_dirs=self.socket_dirs(),
                pg_agent_password=credentials.agent_password,
                run_id=request.run_id,
            )
            with hold_solving_instances(self.lock_dir(), [i.name for i in SOLVING]):
                try:
                    result = await sandbox.run(
                        checkout, request, argv, staged, environment, timeout_s
                    )
                except asyncio.CancelledError:
                    await asyncio.shield(sandbox.stop(request.run_id))
                    raise
        await asyncio.to_thread(_write_log, log, result)
        _logger.info(
            "the solve container exited",
            extra={"run_id": request.run_id, "exit_code": result.exit_code},
        )
        if result.exit_code != 0:
            why = "timed out" if result.timed_out else f"exited {result.exit_code}"
            raise SolveContainerError(f"the solve container {why}; its output is in {log}")

    @asynccontextmanager
    async def _recorder(self, path: Path) -> AsyncIterator[CallRecorder]:
        """The run's host-side call log, which the proxy's records go to."""
        async with JsonlWriter[LlmCallRecord](path) as writer:
            yield RunCallLog(writer)


def snapshot_checkout(checkout: Path, target: Path) -> Path:
    """A copy of ``checkout``'s tracked files as they are now, without .git, at ``target``.

    The solve container mounts a checkout with no .git (T18). The copy is of the working tree's
    tracked files, uncommitted changes included, so it holds what the host hashed for the
    manifest. A checkout without .git, such as an RSI snapshot, is used as it is.
    """
    if not (checkout / ".git").exists():
        return checkout.resolve(strict=True)
    if target.exists():
        shutil.rmtree(target)
    stash = _git(checkout, "stash", "create").strip()  # Empty for a clean working tree.
    tree = stash or "HEAD"
    target.mkdir(parents=True)
    archive = target.parent / f".{target.name}.tar"
    try:
        _git(checkout, "archive", "--format=tar", f"--output={archive}", tree)
        with tarfile.open(archive) as tar:
            tar.extractall(target, filter="data")
    finally:
        archive.unlink(missing_ok=True)
    return target.resolve(strict=True)


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(  # noqa: S603 - fixed git arguments.
        ["git", "-C", str(repo), *args],  # noqa: S607 - git on PATH.
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    if completed.returncode != 0:
        raise SolveContainerError(f"git {args[0]} failed: {completed.stderr.strip()[-500:]}")
    return completed.stdout


def checkout_of(pack_dir: Path | None) -> Path:
    """The checkout that holds ``pack_dir``, ``<checkout>/src/goaly/harness/packs/<name>``."""
    if pack_dir is None:
        raise ValueError("an isolated Effi-SQL run needs its pack's directory")
    checkout = pack_dir.resolve().parents[4]
    if not (checkout / "src" / "goaly" / "harness" / "packs").is_dir():
        raise ValueError(f"{pack_dir} is not a pack in a goaly checkout")
    return checkout


def stage_inputs(
    inputs: SolveInputs, paths: EffiSqlPaths, instances_file: Path | None
) -> Path | None:
    """Write the inputs directory: the gold-free tasks, the snapshot and the plans.

    The tasks are the organizers' file's, if the run has one, or the runtime copy's, and only
    their gold-free fields are kept whatever the source holds. Return the staged task file for
    an organizers' run, which the container's --instances names; None otherwise.
    """
    inputs.root.mkdir(parents=True, exist_ok=True)
    source = paths.tasks_file if instances_file is None else instances_file
    write_runtime_file(source, inputs.tasks)
    shutil.copyfile(paths.snapshot, inputs.snapshot)
    shutil.copyfile(paths.plans, inputs.plans)
    return None if instances_file is None else inputs.tasks


def container_argv(request: SolveRequestLike, staged_instances: Path | None) -> list[str]:
    """What the container runs: ``python -m goaly`` with the request's arguments, as goaly is on
    the image's PYTHONPATH, not installed."""
    args = request.goaly_args(
        solve_dir=request.solve_dir, inputs_dir=request.inputs_dir, instances_file=staged_instances
    )
    return ["python", "-m", "goaly", *args]


def _timeout(tasks: int, max_tasks: int, task_timeout_s: float | None) -> float:
    """The container's absolute limit: each round of tasks gets its wall budget and half again."""
    per_task = _TASK_WALL_S if task_timeout_s is None else task_timeout_s
    return math.ceil(tasks / max(max_tasks, 1)) * per_task * 1.5 + _SLACK_S


def _write_log(path: Path, result: SandboxResult) -> None:
    header = f"exit code {result.exit_code}; timed out: {result.timed_out}\n"
    write_text_atomically(path, header + result.output)
