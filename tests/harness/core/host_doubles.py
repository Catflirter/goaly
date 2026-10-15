"""A test double of a benchmark's solve launcher, which runs the solve container's command here."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from goaly.harness.core import cli
from goaly.harness.core.isolation import SolveRequest
from goaly.harness.core.registry import Registry


@dataclass
class InProcessLauncher:
    """Runs a SolveRequest's in-container command in this process and its event loop.

    The "container" is this process, so this isolates nothing: it is for tests, which check
    what the host does with what a solve phase leaves. ``registry`` is the one the container
    side runs with, set once it is built. ``after`` runs once the solve phase has, such as to
    tamper with what it wrote; ``fails`` makes the launch fail instead, as a container that
    exits with an error does.
    """

    budgets_file: Path
    registry: Registry = field(default_factory=Registry)
    after: Callable[[SolveRequest], None] | None = None
    fails: bool = False
    requests: list[SolveRequest] = field(default_factory=list)

    async def __call__(self, request: SolveRequest) -> None:
        self.requests.append(request)
        if self.fails:
            raise RuntimeError("the solve container exited with status 1")
        argv = request.goaly_args(
            solve_dir=request.solve_dir,
            inputs_dir=request.inputs_dir,
            instances_file=request.instances_file,
        )
        await cli.run(argv, self.registry, budgets_file=self.budgets_file)
        if self.after is not None:
            self.after(request)
