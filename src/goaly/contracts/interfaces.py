"""Behavioral interfaces: the Protocols the layers call each other through (spec §3.3).

bench implements the environments and the benchmark adapters, and the capability packs implement
the policies and the tools. core and the runner call them only through these Protocols, so
harness unit tests can use fakes, with no database or container.

gRPC-ready: every method takes and returns only contract messages (``goaly.contracts.gen``) or
plain values, never framework objects. A gRPC-backed implementation could then replace an
in-process one without changing any caller. The one exception is a tool's arguments, which stay
pydantic models (see Tool). tests/contracts/test_interfaces.py enforces the rule.
"""

from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from typing import Protocol

from pydantic import BaseModel

from goaly.contracts.gen.environment_pb2 import ShellResult, SqlResult
from goaly.contracts.gen.task_pb2 import Grade, Prediction, Split, TaskInput
from goaly.contracts.gen.trajectory_pb2 import ToolResult

__all__ = [
    "Benchmark",
    "Environment",
    "JsonValue",
    "Policy",
    "ShellEnvironment",
    "SqlEnvironment",
    "Tool",
]

# A JSON value, as json.loads returns it.
type JsonValue = dict[str, JsonValue] | list[JsonValue] | str | int | float | bool | None


class Environment(Protocol):
    """The runtime environment of one task: what the agent's tools act on.

    bench implements it, as a PostgreSQL sandbox (SqlEnvironment) or a Docker container
    (ShellEnvironment); harness unit tests use fakes. It belongs to whoever created it, normally
    the context manager that Benchmark.provision returns, which closes it.
    """

    async def close(self) -> None:
        """Release the environment's resources. Calling it again does nothing."""


class SqlEnvironment(Environment, Protocol):
    """A task's sandbox database, which BIRD-CRITIC's tools query: bench's PostgreSQL sandbox."""

    async def execute(self, sql: str, *, timeout_s: float) -> SqlResult:
        """Run ``sql``, stopping it after ``timeout_s`` seconds, and return its result.

        A statement that fails or times out gives a result with ``error`` set, not an exception.
        An exception means that the environment itself failed.
        """


class ShellEnvironment(Environment, Protocol):
    """A task's container, which SWE-fficiency's tools work in: bench's Docker environment.

    Paths are inside the container, where the repository is at /testbed.
    """

    async def exec(self, command: str, *, timeout_s: float) -> ShellResult:
        """Run ``command`` in a shell, stopping it after ``timeout_s`` seconds.

        A command that fails or times out gives a result, with ``exit_code`` or ``timed_out``
        saying so, not an exception. An exception means that the environment itself failed.
        """

    async def read_file(self, path: str) -> str:
        """Return the text of the file at ``path``. Raise FileNotFoundError if there is none."""

    async def write_file(self, path: str, content: str) -> None:
        """Write ``content`` to the file at ``path``, replacing the file if it exists."""


class Tool[ArgsT: BaseModel, EnvT: Environment](Protocol):
    """A tool the agent can call. Capability packs implement tools, and core's executor runs them.

    The executor validates the model's arguments against ``args_model``, calls :meth:`run`, and
    fills in the rest of the result (spec §5.1). Each tool declares its own argument model and
    the environment it needs, and a tool that works in any environment fits every pack.
    ``name``, ``description`` and ``args_model`` are read-only: a class attribute can provide each.

    The arguments are a pydantic model: the one exception to the rule that interfaces take only
    contract messages and plain values (spec §3.3). The OpenAI tools API takes JSON Schema, which
    pydantic generates, and GLM edits tools during RSI with no protobuf code to regenerate.
    """

    @property
    def name(self) -> str:
        """The name the model calls the tool by."""

    @property
    def description(self) -> str:
        """What the tool does, as the model reads it."""

    @property
    def args_model(self) -> type[ArgsT]:
        """The model of the tool's arguments; its JSON Schema is the tool's parameters."""

    async def run(self, args: ArgsT, env: EnvT) -> ToolResult:
        """Run the tool on validated ``args`` in ``env``.

        Set ``content``, the text the model sees, and ``is_error``; the executor fills in the
        other fields. A tool may raise: the executor turns the exception into an error result.
        """


class Policy[EnvT: Environment, CtxT](Protocol):
    """How a capability pack solves one task. Each pack implements one, and the runner calls it.

    ``CtxT`` is core's per-task context, the agent loop's AgentContext. It is a type parameter
    because contracts cannot import harness.
    """

    async def solve(self, task: TaskInput, env: EnvT, ctx: CtxT) -> Prediction:
        """Solve ``task`` in ``env`` and return the prediction; ``env`` stays the caller's.

        When a budget runs out, return the fallback prediction, with ``fallback`` set (spec §5.4).
        """


class Benchmark[InstanceT, EnvT: Environment](Protocol):
    """A benchmark adapter: it loads tasks, provisions their environments and grades predictions.

    bench implements one per benchmark, and the runner drives it. Adding a benchmark takes an
    adapter and a capability pack, and core does not change (spec §3.3).

    ``InstanceT`` is the benchmark's own record of a task and may hold gold answers. Only the
    copy that :meth:`agent_input` makes reaches the agent, so the types keep gold answers away
    from it.
    """

    @property
    def name(self) -> str:
        """The adapter's name, as --bench and RunManifest.benchmark give it, e.g. "birdcritic"."""

    def load_instances(self, split: Split.ValueType) -> Sequence[InstanceT]:
        """Return the instances in ``split``."""

    def provision(self, inst: InstanceT) -> AbstractAsyncContextManager[EnvT]:
        """Create the environment to solve ``inst`` in. Leaving the context closes it."""

    def agent_input(self, inst: InstanceT) -> TaskInput:
        """Copy out what the agent may see of ``inst``, which is never a gold answer."""

    async def grade(self, inst: InstanceT, prediction: Prediction) -> Grade:
        """Grade ``prediction`` against ``inst``'s gold answers.

        Grading uses its own environment, never the one the task was solved in (spec §4.1).
        """

    def to_official(self, inst: InstanceT, prediction: Prediction) -> dict[str, JsonValue]:
        """Convert ``prediction`` to one record of the official submission format.

        The record keeps the official field names exactly, such as BIRD's ``pred_sqls``.
        """
