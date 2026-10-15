"""Fakes of the goaly.contracts.interfaces Protocols, for unit tests.

They need no database, container or model, as spec §3.3 asks of harness unit tests. Each is a
small, deterministic implementation that records how it was called.
tests/contracts/test_interfaces.py proves, under mypy and at runtime, that each conforms to its
Protocol.

pytest puts the repository root on sys.path (``pythonpath`` in pyproject.toml), so any test can
import them, e.g. ``from tests.fakes import FakeSqlEnvironment``.
"""

import asyncio
import posixpath
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass

from google.protobuf.struct_pb2 import ListValue, Value
from pydantic import BaseModel

from goaly.contracts.errors import EnvironmentFailure
from goaly.contracts.gen.environment_pb2 import ShellResult, SqlResult
from goaly.contracts.gen.task_pb2 import (
    GRADE_STATUS_GRADED,
    SPLIT_TRAIN,
    SPLIT_VAL,
    BirdPrediction,
    BirdTaskInput,
    Grade,
    Prediction,
    Split,
    SwePrediction,
    TaskInput,
)
from goaly.contracts.gen.trajectory_pb2 import ToolResult
from goaly.contracts.interfaces import Environment, JsonValue

# The exit status a shell gives for a command it cannot find.
COMMAND_NOT_FOUND = 127
# PostgreSQL's error for a statement that ran past its statement_timeout.
STATEMENT_TIMEOUT = "canceling statement due to statement timeout"


class FakeEnvironment:
    """An Environment with nothing in it.

    Each call of its subclasses takes the canned result's ``elapsed_s`` of real time, and stops
    at the call's timeout. A call cancelled midway stops at once, is recorded in ``stopped`` and
    leaves the environment usable. Once closed, the environment refuses every call.
    """

    def __init__(self) -> None:
        self.closed = False
        self.stopped: list[str] = []

    async def close(self) -> None:
        self.closed = True

    def _ensure_open(self) -> None:
        if self.closed:
            raise RuntimeError("the environment is closed")

    async def _take(self, call: str, elapsed_s: float, timeout_s: float) -> bool:
        """Spend the time ``call`` takes, up to ``timeout_s``; return whether it timed out."""
        try:
            await asyncio.sleep(min(elapsed_s, timeout_s))
        except asyncio.CancelledError:
            self.stopped.append(call)
            raise
        return elapsed_s > timeout_s


class FakeSqlEnvironment(FakeEnvironment):
    """A SqlEnvironment that answers each statement from a table of canned results.

    A statement missing from the table gets a result with ``error`` set, as a failing statement
    does. A canned EnvironmentFailure is raised instead, as when the database dies. ``executed``
    records each statement with its timeout.
    """

    def __init__(self, results: Mapping[str, SqlResult | EnvironmentFailure] | None = None) -> None:
        super().__init__()
        self.results = dict(results or {})
        self.executed: list[tuple[str, float]] = []

    async def execute(self, sql: str, *, timeout_s: float) -> SqlResult:
        self._ensure_open()
        self.executed.append((sql, timeout_s))
        canned = self.results.get(sql, SqlResult(error=f"no canned result for {sql!r}"))
        if isinstance(canned, EnvironmentFailure):
            raise canned
        if await self._take(sql, canned.elapsed_s, timeout_s):
            return SqlResult(error=STATEMENT_TIMEOUT, elapsed_s=timeout_s)
        result = SqlResult()
        result.CopyFrom(canned)  # A copy: the caller cannot change the table.
        return result


class FakeShellEnvironment(FakeEnvironment):
    """A ShellEnvironment with canned command results and files kept in memory.

    A command missing from the table exits with status 127, as an unknown command does, and a
    canned EnvironmentFailure is raised instead, as when the container dies. Relative paths
    resolve against ``root``. ``commands`` records each command with its timeout, and
    ``files`` maps absolute paths to contents.
    """

    def __init__(
        self,
        results: Mapping[str, ShellResult | EnvironmentFailure] | None = None,
        files: Mapping[str, str] | None = None,
        *,
        root: str = "/testbed",
    ) -> None:
        super().__init__()
        self.root = root
        self.results = dict(results or {})
        self.files = {self._resolve(path): content for path, content in (files or {}).items()}
        self.commands: list[tuple[str, float]] = []

    def _resolve(self, path: str) -> str:
        return posixpath.normpath(posixpath.join(self.root, path))

    async def exec(self, command: str, *, timeout_s: float) -> ShellResult:
        self._ensure_open()
        self.commands.append((command, timeout_s))
        unknown = f"no canned result for {command!r}"
        canned = self.results.get(command, ShellResult(exit_code=COMMAND_NOT_FOUND, stderr=unknown))
        if isinstance(canned, EnvironmentFailure):
            raise canned
        if await self._take(command, canned.elapsed_s, timeout_s):
            return ShellResult(timed_out=True, elapsed_s=timeout_s)
        result = ShellResult()
        result.CopyFrom(canned)  # A copy: the caller cannot change the table.
        return result

    async def read_file(self, path: str) -> str:
        self._ensure_open()
        if self._resolve(path) not in self.files:
            raise FileNotFoundError(path)
        return self.files[self._resolve(path)]

    async def write_file(self, path: str, content: str) -> None:
        self._ensure_open()
        self.files[self._resolve(path)] = content


class EchoArgs(BaseModel):
    """EchoTool's arguments."""

    text: str


class EchoTool:
    """A Tool that answers with its ``text`` argument. It works in any environment."""

    name = "echo"
    description = "Reply with the given text."
    args_model = EchoArgs

    async def run(self, args: EchoArgs, env: Environment) -> ToolResult:
        return ToolResult(content=args.text)


class DoNothingPolicy:
    """A Policy that submits the do-nothing answer: BIRD's issue_sql unchanged, or no SWE patch.

    It works in any environment and with any context. ``tasks`` records each task it solved.
    Like every policy, it leaves the prediction's ``instance_id`` to the runner.
    """

    def __init__(self) -> None:
        self.tasks: list[TaskInput] = []

    async def solve(self, task: TaskInput, env: Environment, ctx: object) -> Prediction:
        self.tasks.append(task)
        if task.HasField("swe"):
            return Prediction(swe=SwePrediction())
        return Prediction(bird=BirdPrediction(sqls=task.bird.issue_sql))


@dataclass(frozen=True)
class ToyInstance:
    """A ToyBenchmark task: the benchmark's own record, so it holds the gold answer.

    Being a frozen dataclass with an ``instance_id``, it satisfies the Instance Protocol.
    """

    instance_id: str
    split: Split.ValueType
    query: str
    issue_sql: str
    gold_sql: str


# What a toy task's base query returns in its solving environment: one row, one column.
TOY_BASE_RESULT = SqlResult(
    columns=["n"], rows=[ListValue(values=[Value(number_value=3)])], row_count=1
)
TOY_INSTANCES = (
    ToyInstance(
        instance_id="toy-1",
        split=SPLIT_TRAIN,
        query="How many users are there?",
        issue_sql="SELECT count(*) FROM user",
        gold_sql="SELECT count(*) FROM users",
    ),
    ToyInstance(  # Its issue SQL is already right, so the do-nothing answer passes.
        instance_id="toy-2",
        split=SPLIT_TRAIN,
        query="What are the users' names?",
        issue_sql="SELECT name FROM users",
        gold_sql="SELECT name FROM users",
    ),
    ToyInstance(
        instance_id="toy-3",
        split=SPLIT_VAL,
        query="Who joined first?",
        issue_sql="SELECT min(name) FROM users",
        gold_sql="SELECT name FROM users ORDER BY id LIMIT 1",
    ),
)


class ToyBenchmark:
    """A Benchmark of BIRD-style SQL tasks held in memory, each solved in a FakeSqlEnvironment.

    A task's solving environment answers only the task's base query (its issue SQL), with
    TOY_BASE_RESULT, so it reveals nothing of the gold answer, which only ``grade`` uses. A
    prediction is correct if its last statement is the gold SQL. ``environments`` records each
    environment provisioned.
    """

    name = "toy"

    def __init__(self, instances: Sequence[ToyInstance] = TOY_INSTANCES) -> None:
        self.instances = tuple(instances)
        self.environments: list[FakeSqlEnvironment] = []

    def load_instances(self, split: Split.ValueType) -> list[ToyInstance]:
        return [inst for inst in self.instances if inst.split == split]

    @asynccontextmanager
    async def provision(self, inst: ToyInstance) -> AsyncIterator[FakeSqlEnvironment]:
        env = FakeSqlEnvironment({inst.issue_sql: TOY_BASE_RESULT})
        self.environments.append(env)
        try:
            yield env
        finally:
            await env.close()

    def agent_input(self, inst: ToyInstance) -> TaskInput:
        bird = BirdTaskInput(db_id="toy", query=inst.query, issue_sql=[inst.issue_sql])
        return TaskInput(bird=bird)  # No instance ID: the agent sees no ID (Instance).

    async def grade(self, inst: ToyInstance, prediction: Prediction) -> Grade:
        sqls = prediction.bird.sqls
        correct = len(sqls) > 0 and sqls[-1] == inst.gold_sql
        return Grade(
            instance_id=inst.instance_id,
            score=float(correct),
            correct=correct,
            status=GRADE_STATUS_GRADED,
        )

    def to_official(self, inst: ToyInstance, prediction: Prediction) -> dict[str, JsonValue]:
        # A prediction with no branch, as after a crash, gets the empty answer: the base query.
        return {
            "instance_id": inst.instance_id,
            "pred_sqls": list(prediction.bird.sqls)
            if prediction.HasField("bird")
            else [inst.issue_sql],
        }
