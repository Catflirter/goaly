"""The implementer's sandbox as a mini-swe-agent environment (spec §8.8).

``SandboxEnvironment`` is a DockerEnvironment whose one long-lived container is T18's sandbox
(goaly.rsi.sandbox): the worktree read-only at its own path, the lane's two pack directories
read-write, host networking, the host user, a read-only root with a tmpfs /tmp, no capabilities,
no host environment variable and so no key. T18's one-shot ``run_once`` cannot keep a container
across commands, so this module adds the smallest extension on T18's argument builder: ``docker
run --detach ... sleep <lifetime>``, then one ``docker exec`` per command under T18's in-container
``timeout``, then a synchronous ``docker rm --force``. The lifetime is finite, so a container a
crashed lane leaves behind still ends; a lane removes leftovers by label when it starts.

Removal is by label too: :meth:`SandboxEnvironment.cleanup` removes every container of the
experiment and then checks that none is left, so host git never reads the worktree while a
sandbox could still change it (spec §8.8, Finishing).

A command's output is read as it comes, and only its first line and its last ``output_limit``
bytes are kept, so no command can fill the lane process's memory.

The docker CLI runs with only the variables it needs to find the daemon, as in T18's runner. A
failure of Docker itself raises SandboxError, and the experiment is aborted. A container that
stopped on its own, which only something run inside it can make it do, raises SandboxStoppedError,
and the candidate is rejected.
"""

import contextlib
import logging
import os
import secrets
import signal
import subprocess
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import TracebackType
from typing import IO, Any, Final, Self, cast

from goaly.rsi.implementer import msa
from goaly.rsi.implementer.smoke import Pause, is_smoke_call
from goaly.rsi.lanes import lane_info
from goaly.rsi.sandbox import (
    SandboxConfig,
    SandboxError,
    containers,
    remove_containers,
    run_options,
    timed_shell,
)
from goaly.rsi.workspace import Worktree

__all__ = ["PURPOSE", "HostCommand", "SandboxEnvironment", "SandboxStoppedError"]

PURPOSE: Final = "implementer"  # The sandbox's purpose label.
_TIMED_OUT: Final = 124  # timeout's exit status once it has stopped the command.
_HEAD_BYTES: Final = 4096  # How much of an output's start is kept, for its first line.
_CHUNK: Final = 64 * 1024

_logger = logging.getLogger(__name__)

# Answers, on the host, a command the sandbox must not run, as that command's output; it may
# pause the sandbox meanwhile.
type HostCommand = Callable[[str, Pause], dict[str, Any]]


class SandboxStoppedError(Exception):
    """The sandbox container stopped by itself, as when a command killed its init or it ran out
    of memory. Something the agent ran did it, so the candidate is rejected, not aborted.
    """


class SandboxEnvironment(msa.DockerEnvironment):
    """Runs the agent's bash commands in one sandbox container on ``worktree``.

    The container starts when the environment is made, and lives at most ``lifetime_s``;
    :meth:`cleanup` removes it before it returns. Use the environment as a context manager. Each
    command runs in a fresh bash in the worktree, for at most ``command_timeout_s``, except that a
    goaly-smoke call goes to ``smoke`` on the host (goaly.rsi.implementer.smoke), which may stop
    the container and start a fresh one meanwhile. Raises
    SandboxError if docker fails or hangs, and SandboxStoppedError if the container stopped by
    itself.
    """

    def __init__(
        self,
        *,
        image: str,
        worktree: Worktree,
        command_timeout_s: float,
        lifetime_s: float,
        sandbox: SandboxConfig = SandboxConfig(),  # noqa: B008  (frozen, so one default is safe)
        smoke: HostCommand | None = None,
    ) -> None:
        self._worktree = worktree
        self._smoke = smoke
        self._sandbox = sandbox
        self._command_timeout_s = command_timeout_s
        self._name = ""  # Each container started gets a name of its own.
        options = run_options(
            worktree.path,
            worktree.lane,
            experiment_id=worktree.experiment_id,
            purpose=PURPOSE,
            writable_packs=True,
            uid=os.getuid(),
            gid=os.getgid(),
            config=sandbox,
        )
        # The base class keeps these for the trajectory, and starts the container.
        super().__init__(
            image=image,
            cwd=str(worktree.path),
            timeout=int(command_timeout_s),
            executable=sandbox.docker,
            run_args=options,
            container_timeout=f"{lifetime_s:.0f}",
            interpreter=timed_shell(command_timeout_s, sandbox),
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.cleanup()

    def _start_container(self) -> None:
        # A container of the experiment that is still there could change the worktree.
        remove_containers(experiment_id=self._worktree.experiment_id, config=self._sandbox)
        lane = lane_info(self._worktree.lane).name
        self._name = (
            f"goaly-rsi-{lane}-{self._worktree.experiment_id}-{PURPOSE}-{secrets.token_hex(4)}"
        )
        options = list(self.config.run_args)
        command = ["run", "--detach", f"--name={self._name}", *options, self.config.image]
        try:
            self._docker([*command, "sleep", self.config.container_timeout])
        except SandboxError:
            self._remove_quietly()  # docker may have made the container before it failed.
            raise
        self.container_id = self._name
        _logger.info(
            "started the implementer's sandbox",
            extra={"container": self._name, "experiment_id": self._worktree.experiment_id},
        )

    def execute(
        self, action: dict[str, Any], cwd: str = "", *, timeout: int | None = None
    ) -> dict[str, Any]:
        """Run the action's bash command in the container and return its output.

        A command that fails or times out is an output for the agent. A command whose output
        starts with the line msa.SUBMIT_MARKER, and that succeeds, raises msa.Submitted with the
        rest of its output. Raises SandboxError if docker fails or hangs, and SandboxStoppedError if
        the container is no longer running.
        """
        if self._smoke is not None and is_smoke_call(action.get("command", "")):
            return self._smoke(action["command"], self._paused)
        container = self.container_id
        if container is None:
            raise SandboxError("the sandbox container is not running")
        limit = self._command_timeout_s if timeout is None else timeout
        command = [
            "exec",
            f"--workdir={cwd or self.config.cwd}",
            container,
            *timed_shell(limit, self._sandbox),
            action.get("command", ""),
        ]
        try:
            returncode, kept = self._stream(command, timeout_s=limit + self._sandbox.kill_after_s)
        except _HostTimeout as timeout_error:
            # Docker hung, or a process the command left behind holds its output open.
            self._check_running(container)
            return {
                "output": "",
                "returncode": _TIMED_OUT,
                "exception_info": (
                    f"The command's output stayed open for {timeout_error.limit_s:g} s, so it was "
                    "stopped. A process started in the background must send its output to a "
                    "file, as in `cmd > /tmp/log 2>&1 &`."
                ),
            }
        output: dict[str, Any] = {
            "output": kept.text(),
            "returncode": returncode,
            "exception_info": "",
        }
        if returncode == _TIMED_OUT:
            output["exception_info"] = f"The command timed out after {limit:g} s and was stopped."
        elif returncode != 0:
            self._check_running(container)  # A command's failure, or the container's?
        if returncode == 0 and kept.first_line().strip() == msa.SUBMIT_MARKER:
            submission = kept.after_first_line()
            raise msa.Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )
        return output

    def cleanup(self) -> None:
        """Remove every container of the experiment, wait until they are gone, and check that
        none is left (spec §8.8, Finishing), even if none was started. Raises SandboxError if
        docker fails or a container remains.
        """
        if self.container_id is not None:
            self._docker(["rm", "--force", self.container_id])
            self.container_id = None
        # Always, by label: a start that failed may still have left a container behind.
        experiment_id = self._worktree.experiment_id
        remove_containers(experiment_id=experiment_id, config=self._sandbox)
        left = containers(experiment_id=experiment_id, config=self._sandbox)
        if left:
            raise SandboxError(f"sandbox containers of the experiment remain: {', '.join(left)}")

    @contextmanager
    def _paused(self) -> Iterator[None]:
        """Stop and remove the sandbox for the block, then start a fresh one, whatever happens."""
        self.cleanup()
        try:
            yield
        finally:
            self._start_container()

    def __del__(self) -> None:
        """Do nothing: the base class would remove the container in the background. Leftovers of
        a crashed run are found by label (goaly.rsi.sandbox.remove_containers).
        """

    def _remove_quietly(self) -> None:
        try:
            self._docker(["rm", "--force", self._name])
        except SandboxError:
            _logger.warning(
                "could not remove the sandbox container", extra={"container": self._name}
            )

    def _check_running(self, container: str) -> None:
        """Raise SandboxStoppedError if the container stopped by itself, or SandboxError if docker
        cannot say.
        """
        state_format = "--format={{.State.Running}} {{.State.ExitCode}} {{.State.OOMKilled}}"
        state = self._docker(["inspect", state_format, container]).split()
        if len(state) != 3:
            raise SandboxError(f"docker inspect gave no state for {container}: {state!r}")
        running, exit_code, oom_killed = state
        if running == "true":
            return
        if oom_killed == "true":
            reason = "it ran out of memory"
        else:
            reason = f"its init exited ({exit_code})"
        raise SandboxStoppedError(f"the sandbox container {container} stopped: {reason}")

    def _docker(self, args: list[str]) -> str:
        """Run a docker command that must succeed, and return its standard output; its error
        output only explains a failure.
        """
        try:
            completed = subprocess.run(  # noqa: S603  (the docker CLI, with arguments built here)
                [self._sandbox.docker, *args],
                env=_docker_environment(),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=self._sandbox.docker_timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            limit = self._sandbox.docker_timeout_s
            raise SandboxError(f"docker {args[0]} timed out after {limit:g} s") from None
        except OSError as error:
            raise SandboxError(f"cannot run {self._sandbox.docker}: {error}") from error
        if completed.returncode != 0:
            reason = completed.stderr.decode("utf-8", "replace").strip()
            raise SandboxError(f"docker {args[0]} failed ({completed.returncode}): {reason}")
        return completed.stdout.decode("utf-8", "replace")

    def _stream(self, args: list[str], *, timeout_s: float) -> tuple[int, "_BoundedOutput"]:
        """Run ``docker exec`` for at most ``timeout_s`` plus docker's own timeout, reading its
        output as it comes, and return its exit status and what was kept of its output.

        Raises _HostTimeout once that time is up, with the exec's process group killed: docker
        exec waits for its output to close, which a process the command left in the background
        can hold open past the in-container timeout.
        """
        limit = timeout_s + self._sandbox.docker_timeout_s
        try:
            process = subprocess.Popen(  # noqa: S603  (the docker CLI, with arguments built here)
                [self._sandbox.docker, *args],
                env=_docker_environment(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # Its own process group, to kill as one.
            )
        except OSError as error:
            raise SandboxError(f"cannot run {self._sandbox.docker}: {error}") from error
        stdout = cast("IO[bytes]", process.stdout)  # Not None: stdout=PIPE.
        output = _BoundedOutput(self._sandbox.output_limit)
        reader = threading.Thread(target=output.drain, args=(stdout.fileno(),), daemon=True)
        reader.start()
        try:
            returncode = process.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise _HostTimeout(limit) from None
        finally:
            reader.join()
            stdout.close()
        return returncode, output


class _HostTimeout(Exception):  # noqa: N818  (not an error of its own: execute() decides)
    """docker exec did not end within its time on the host."""

    def __init__(self, limit_s: float) -> None:
        super().__init__(f"docker exec did not end within {limit_s:g} s")
        self.limit_s = limit_s


class _BoundedOutput:
    """A command's output, of which only the start and the last ``limit`` bytes are kept."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._head = bytearray()
        self._tail = bytearray()
        self._total = 0

    def drain(self, descriptor: int) -> None:
        """Read the pipe ``descriptor`` to its end, keeping what fits."""
        while chunk := os.read(descriptor, _CHUNK):
            self._total += len(chunk)
            if len(self._head) < _HEAD_BYTES:
                self._head += chunk[: _HEAD_BYTES - len(self._head)]
            self._tail += chunk
            if len(self._tail) > self._limit:
                del self._tail[: len(self._tail) - self._limit]

    def first_line(self) -> str:
        """The output's first non-blank line, which may be cut off at its start's limit."""
        return self._head.decode("utf-8", "replace").lstrip().partition("\n")[0]

    def after_first_line(self) -> str:
        """Everything after the first line; only the end of it if the output was long."""
        if self._total > len(self._tail):
            return self.text()
        return self._tail.decode("utf-8", "replace").lstrip().partition("\n")[2]

    def text(self) -> str:
        """The output, or its last ``limit`` bytes with a note of how much was cut."""
        text = self._tail.decode("utf-8", "replace")
        cut = self._total - len(self._tail)
        return f"[... {cut} bytes cut ...]\n{text}" if cut else text


def _docker_environment() -> dict[str, str]:
    """The docker CLI's environment: what it needs to find the daemon, and no secret."""
    environment = {
        name: value
        for name, value in os.environ.items()
        if name in {"PATH", "HOME"} or name.startswith("DOCKER_")
    }
    environment["LANG"] = "C.UTF-8"
    return environment
