"""Behavioral interfaces: the Protocols the layers call each other through (spec §3.3).

bench implements the environments and the benchmark adapters, and the capability packs implement
the policies and the tools. core and the runner call them only through these Protocols, so
harness unit tests can use fakes, with no database or container.

gRPC-ready: every annotation that is not a type parameter is a contract message
(``goaly.contracts.gen``) or a plain value, never a framework object, so a gRPC-backed
implementation could replace an in-process one without changing any caller. The type parameters
stand for types the implementers choose: ``EnvT`` is always one of the environment Protocols,
while ``CtxT`` (core's AgentContext) and ``InstanceT`` (a benchmark's own record) are in-process
types. Among the concrete annotations, a tool's pydantic arguments are the one documented
exception (see Tool). tests/contracts/test_interfaces.py enforces the rule.
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
    "Instance",
    "JsonValue",
    "Policy",
    "ShellEnvironment",
    "SqlEnvironment",
    "Tool",
]

# A JSON value, as json.loads returns it. It is built from dict and list, which are invariant, so
# build an official record inline, e.g. with list(...) inside the dict literal: a list[str] built
# beforehand is not a list[JsonValue].
type JsonValue = dict[str, JsonValue] | list[JsonValue] | str | int | float | bool | None


class Environment(Protocol):
    """The runtime environment of one task: what the agent's tools act on.

    bench implements it, as a PostgreSQL sandbox (SqlEnvironment) or a Docker container
    (ShellEnvironment); harness unit tests use fakes. It belongs to whoever created it, normally
    the context manager that Benchmark.provision returns, which closes it.

    A statement or command that fails gives a result. A call raises EnvironmentFailure
    (goaly.contracts.errors) when the environment itself fails, such as a dead database
    connection or container, and only then.

    A cancelled call stops the statement or command it started before CancelledError
    propagates, and the environment stays usable.
    """

    async def close(self) -> None:
        """Release the environment's resources. Calling it again does nothing.

        It completes even while its caller is being cancelled: implementations shield it.
        """


class SqlEnvironment(Environment, Protocol):
    """A task's sandbox database, which Effi-SQL's tools query: bench's PostgreSQL sandbox.

    ``provision`` has already run any setup SQL, and each statement's effects persist for the
    environment's lifetime. A read-only environment, such as Effi-SQL's solving sandbox,
    reports an attempt to write as an SQL error result.
    """

    async def execute(self, sql: str, *, timeout_s: float) -> SqlResult:
        """Run ``sql``, stopping it after ``timeout_s`` seconds, and return its result.

        ``sql`` holds one statement or more; the result is the last statement's. A statement
        that fails or times out gives a result with ``error`` set, not an exception. Only a
        failure of the environment itself raises EnvironmentFailure.
        """


class ShellEnvironment(Environment, Protocol):
    """A task's container, which SWE-fficiency's tools work in: bench's Docker environment.

    Paths are inside the container. Each ``exec`` runs in a fresh ``bash -c`` whose working
    directory is the task's root, /testbed for SWE-fficiency, and nothing persists between
    calls: no ``cd`` and no exported variable. A relative path resolves against the root.
    """

    async def exec(self, command: str, *, timeout_s: float) -> ShellResult:
        """Run ``command`` with ``bash -c``, stopping it after ``timeout_s`` seconds.

        The environment may cap the output, and then sets ``truncated``. A command that fails or
        times out gives a result, with ``exit_code`` or ``timed_out`` saying so, not an
        exception. Only a failure of the environment itself raises EnvironmentFailure.
        """

    async def read_file(self, path: str) -> str:
        """Return the whole text of the file at ``path``, never truncated.

        The file is decoded as UTF-8, with replacement characters for invalid bytes. Raise
        FileNotFoundError if there is no such file.
        """

    async def write_file(self, path: str, content: str) -> None:
        """Write ``content`` to the file at ``path`` as UTF-8, replacing the file if it exists.

        Missing parent directories are created.
        """


class Tool[ArgsT: BaseModel, EnvT: Environment](Protocol):
    """A tool the agent can call. Capability packs implement tools, and core's executor runs them.

    The executor validates the model's arguments against ``args_model``, calls :meth:`run`, and
    fills in the rest of the result (spec §5.1). Each tool declares its own argument model and
    the environment it needs, and a tool that works in any environment fits every pack.
    ``name``, ``description`` and ``args_model`` are read-only: a class attribute can provide each.

    ``ArgsT`` is invariant, so a list of different tools is a ``Sequence[Tool[Any, E]]``, with an
    explicit ``E``. core registers each one through a generic
    ``bind[A: BaseModel, E: Environment](tool: Tool[A, E]) -> BoundTool[E]``, which erases the
    argument type but validates the model's JSON and runs the tool while ``A`` is still concrete.

    The arguments are a pydantic model: among the concrete annotations, the one documented
    exception to the gRPC-ready rule (spec §3.3). The OpenAI tools API takes JSON Schema, which
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

        Set ``content``, the text the model sees. A tool ends the episode by returning a result
        with ``submission`` set: the agent loop stops, and the policy decides whether to return
        it. The executor fills in the other fields. A statement or command that ran and failed is
        a normal result; a tool that cannot do its job raises instead. The executor re-raises
        EnvironmentFailure and turns every other Exception into an error result, with
        ``is_error`` set; cancellation (asyncio.CancelledError, a BaseException) always
        propagates.
        """


class Policy[EnvT: Environment, CtxT](Protocol):
    """How a capability pack solves one task. Each pack implements one, and the runner calls it.

    ``CtxT`` is core's per-task context, the agent loop's AgentContext. It is a type parameter
    because contracts cannot import harness.
    """

    async def solve(self, task: TaskInput, env: EnvT, ctx: CtxT) -> Prediction:
        """Solve ``task`` in ``env`` and return the prediction; ``env`` stays the caller's.

        The task carries no instance ID, and the runner sets the prediction's ``instance_id``.

        A budget running out is not an exception: return the fallback prediction, with
        ``fallback`` set (spec §5.4). Let EnvironmentFailure propagate, and an LLM failure that
        outlasts the client's retries; the runner records them as TASK_ERROR_TYPE_ENVIRONMENT
        and TASK_ERROR_TYPE_LLM.
        """


class Instance(Protocol):
    """A benchmark's own record of one task, as its adapter loads it.

    The runner identifies a task by its ``instance_id``, because the TaskInput the agent sees
    carries no ID: an ID can give the answer away, as an SWE-fficiency ID names the upstream
    pull request whose diff is the gold patch. A frozen dataclass with an ``instance_id`` field
    satisfies this Protocol.
    """

    @property
    def instance_id(self) -> str:
        """The benchmark's ID for the task."""


class Benchmark[InstanceT: Instance, EnvT: Environment](Protocol):
    """A benchmark adapter: it loads tasks, provisions their environments and grades predictions.

    bench implements one per benchmark, and the runner drives it. Adding a benchmark takes an
    adapter and a capability pack, and core does not change (spec §3.3).

    ``InstanceT`` is the benchmark's own record of a task and may hold gold answers. Only the
    copy that :meth:`agent_input` makes reaches the agent, so the types keep gold answers away
    from it. The runner identifies each instance by its ``instance_id``.

    ``provision`` and ``grade`` may wait for the resources they need, such as database
    connections, CPU slots or grading instances. The adapter owns those pools; the runner only
    bounds how many tasks run at once (spec §6.2).

    The organizers' task file (``goaly run --instances``) reaches an adapter when it is
    constructed, and SPLIT_EXTERNAL then selects its tasks. Runs on SPLIT_EXTERNAL are not
    graded; they still get one official record per task.
    """

    @property
    def name(self) -> str:
        """The adapter's name, as --bench and RunManifest.benchmark give it, e.g. "effisql"."""

    def load_instances(self, split: Split.ValueType) -> Sequence[InstanceT]:
        """Return the instances in ``split``.

        For SPLIT_EXTERNAL, an adapter constructed with an organizer task file returns that
        file's tasks, and an adapter constructed without one raises ValueError.
        """

    def provision(self, inst: InstanceT) -> AbstractAsyncContextManager[EnvT]:
        """Create the environment to solve ``inst`` in.

        The context closes the environment however it exits: normally, on an exception or on
        cancellation. It never suppresses an exception raised in its body. If the environment
        cannot be created, entering the context raises EnvironmentFailure, which the runner
        records as TASK_ERROR_TYPE_ENVIRONMENT.
        """

    def agent_input(self, inst: InstanceT) -> TaskInput:
        """Copy out what the agent may see of ``inst``, which is never a gold answer."""

    async def grade(self, inst: InstanceT, prediction: Prediction) -> Grade:
        """Grade ``prediction`` for ``inst`` with the benchmark's grader.

        Effi-SQL compares the prediction's result with ``base_sql``'s and needs no gold answer;
        SWE runs the task's tests and compares the workload's speedup with the expert patch's.
        Grading uses its own environment, never the one the task was solved in (spec §4.1).
        Grading trouble does not raise: the returned Grade's ``status`` says what went wrong, such
        as a prediction that failed to run, a timeout, or the grader's own failure.
        Runs on SPLIT_EXTERNAL are not graded, so callers never call ``grade`` for them.

        A prediction with no benchmark branch set, as after a crash, is graded as the
        benchmark's empty answer, the one :meth:`to_official` writes for it: Effi-SQL's
        ``base_sql`` unchanged, which gets the credit of its measured identity τ, or an empty SWE
        patch, scored 1/gold.
        """

    def to_official(self, inst: InstanceT, prediction: Prediction) -> dict[str, JsonValue]:
        """Convert ``prediction`` to one record of the official submission format.

        The record keeps the official field names exactly. A prediction with no benchmark branch
        set, as after a crash, gets the benchmark's empty answer: Effi-SQL's ``base_sql``
        unchanged, or an empty SWE patch. Every task therefore still gets one official record.
        """
