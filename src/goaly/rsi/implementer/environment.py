"""The implementer's sandbox as a mini-swe-agent environment (spec §8.8).

``SandboxEnvironment`` is a DockerEnvironment whose one long-lived container is T18's sandbox
(goaly.rsi.sandbox): the worktree read-only at its own path, the lane's two pack directories
read-write, host networking, the host user, a read-only root with a tmpfs /tmp, no capabilities,
no host environment variable and so no key. T18's one-shot ``run_once`` cannot keep a container
across commands, so this module adds the smallest extension on T18's argument builder: ``docker
run --detach ... sleep infinity``, then one ``docker exec`` per command under T18's in-container
``timeout``, then a synchronous ``docker rm --force``.

The docker CLI runs with only the variables it needs to find the daemon, as in T18's runner. A
failure of Docker itself, as opposed to a command that fails, raises SandboxError: the experiment
is aborted, not rejected.
"""

import logging
import os
import secrets
import subprocess
from types import TracebackType
from typing import Any, Final, Self

from goaly.rsi.implementer import msa
from goaly.rsi.lanes import lane_info
from goaly.rsi.sandbox import SandboxConfig, SandboxError, run_options, timed_shell
from goaly.rsi.workspace import Worktree

__all__ = ["PURPOSE", "SandboxEnvironment"]

PURPOSE: Final = "implementer"  # The sandbox's purpose label.
_TIMED_OUT: Final = 124  # timeout's exit status once it has stopped the command.

_logger = logging.getLogger(__name__)


class SandboxEnvironment(msa.DockerEnvironment):
    """Runs the agent's bash commands in one sandbox container on ``worktree``.

    The container starts when the environment is made, and :meth:`cleanup` removes it before it
    returns; use the environment as a context manager. Each command runs in a fresh bash in the
    worktree, for at most ``command_timeout_s``. Raises SandboxError if docker fails or hangs.
    """

    def __init__(
        self,
        *,
        image: str,
        worktree: Worktree,
        command_timeout_s: float,
        sandbox: SandboxConfig = SandboxConfig(),  # noqa: B008  (frozen, so one default is safe)
    ) -> None:
        self._worktree = worktree
        self._sandbox = sandbox
        self._command_timeout_s = command_timeout_s
        self._name = (
            f"goaly-rsi-{lane_info(worktree.lane).name}-{worktree.experiment_id}-{PURPOSE}-"
            f"{secrets.token_hex(4)}"
        )
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
            container_timeout="infinity",
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

        A command that fails or times out is an output for the agent. A command that ends the run
        (see msa.SUBMIT_MARKER) raises msa.Submitted. Raises SandboxError if docker fails, hangs,
        or the container is no longer running.
        """
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
        completed = self._run(command, timeout_s=limit + self._sandbox.kill_after_s)
        text = completed.stdout.decode("utf-8", "replace")
        output: dict[str, Any] = {
            "output": _tail(text, self._sandbox.output_limit),
            "returncode": completed.returncode,
            "exception_info": "",
        }
        if completed.returncode == _TIMED_OUT:
            output["exception_info"] = f"The command timed out after {limit:g} s and was stopped."
        elif completed.returncode != 0:
            self._check_running(container)  # A command's failure, or the container's?
        self._check_finished(output)
        return output

    def cleanup(self) -> None:
        """Remove the container and wait until it is gone. Raises SandboxError if docker fails."""
        if self.container_id is None:
            return
        self._docker(["rm", "--force", self.container_id])
        self.container_id = None

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
        state = self._docker(["inspect", "--format={{.State.Running}}", container])
        if state.strip() != "true":
            raise SandboxError(f"the sandbox container {container} stopped")

    def _docker(self, args: list[str]) -> str:
        """Run a docker command that must succeed, and return its output."""
        completed = self._run(args, timeout_s=0)
        if completed.returncode != 0:
            reason = completed.stdout.decode("utf-8", "replace").strip()
            raise SandboxError(f"docker {args[0]} failed ({completed.returncode}): {reason}")
        return completed.stdout.decode("utf-8", "replace")

    def _run(self, args: list[str], *, timeout_s: float) -> subprocess.CompletedProcess[bytes]:
        """Run the docker CLI for at most ``timeout_s`` plus docker's own timeout."""
        limit = timeout_s + self._sandbox.docker_timeout_s
        try:
            return subprocess.run(  # noqa: S603  (the docker CLI, with arguments built here)
                [self._sandbox.docker, *args],
                env=_docker_environment(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=limit,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise SandboxError(f"docker {args[0]} timed out after {limit:g} s") from None
        except OSError as error:
            raise SandboxError(f"cannot run {self._sandbox.docker}: {error}") from error


def _docker_environment() -> dict[str, str]:
    """The docker CLI's environment: what it needs to find the daemon, and no secret."""
    environment = {
        name: value
        for name, value in os.environ.items()
        if name in {"PATH", "HOME"} or name.startswith("DOCKER_")
    }
    environment["LANG"] = "C.UTF-8"
    return environment


def _tail(output: str, limit: int) -> str:
    if len(output) <= limit:
        return output
    return f"[... {len(output) - limit} characters cut ...]\n{output[-limit:]}"
