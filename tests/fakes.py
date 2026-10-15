"""Fakes of the goaly.contracts.interfaces Protocols, for unit tests.

They need no database, container or model, as spec §3.3 asks of harness unit tests. Each is a
small, deterministic implementation that records how it was called.
tests/contracts/test_interfaces.py proves, under mypy and at runtime, that each conforms to its
Protocol.

pytest imports tests/ as a namespace package, so any test can import them, e.g.
``from tests.fakes import FakeSqlEnvironment``.
"""

from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass

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


class FakeEnvironment:
    """An Environment with nothing in it. Once closed, its subclasses refuse every call."""

    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True

    def _ensure_open(self) -> None:
        if self.closed:
            raise RuntimeError("the environment is closed")


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
        result = SqlResult()
        result.CopyFrom(canned)  # A copy: the caller cannot change the table.
        return result


class FakeShellEnvironment(FakeEnvironment):
    """A ShellEnvironment with canned command results and files kept in memory.

    A command missing from the table exits with status 127, as an unknown command does, and a
    canned EnvironmentFailure is raised instead, as when the container dies. ``commands``
    records each command with its timeout, and ``files`` maps paths to contents.
    """

    def __init__(
        self,
        results: Mapping[str, ShellResult | EnvironmentFailure] | None = None,
        files: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.results = dict(results or {})
        self.files = dict(files or {})
        self.commands: list[tuple[str, float]] = []

    async def exec(self, command: str, *, timeout_s: float) -> ShellResult:
        self._ensure_open()
        self.commands.append((command, timeout_s))
        unknown = f"no canned result for {command!r}"
        canned = self.results.get(command, ShellResult(exit_code=COMMAND_NOT_FOUND, stderr=unknown))
        if isinstance(canned, EnvironmentFailure):
            raise canned
        result = ShellResult()
        result.CopyFrom(canned)  # A copy: the caller cannot change the table.
        return result

    async def read_file(self, path: str) -> str:
        self._ensure_open()
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    async def write_file(self, path: str, content: str) -> None:
        self._ensure_open()
        self.files[path] = content


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
    """

    def __init__(self) -> None:
        self.tasks: list[TaskInput] = []

    async def solve(self, task: TaskInput, env: Environment, ctx: object) -> Prediction:
        self.tasks.append(task)
        if task.HasField("swe"):
            return Prediction(instance_id=task.instance_id, swe=SwePrediction())
        sqls = task.bird.issue_sql
        return Prediction(instance_id=task.instance_id, bird=BirdPrediction(sqls=sqls))


@dataclass(frozen=True)
class ToyInstance:
    """A ToyBenchmark task: the benchmark's own record, so it holds the gold answer."""

    instance_id: str
    split: Split.ValueType
    query: str
    issue_sql: str
    gold_sql: str


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

    A task's environment runs only the task's gold SQL, which returns no rows. A prediction is
    correct if its last statement is the gold SQL. ``environments`` records each environment
    provisioned.
    """

    name = "toy"

    def __init__(self, instances: Sequence[ToyInstance] = TOY_INSTANCES) -> None:
        self.instances = tuple(instances)
        self.environments: list[FakeSqlEnvironment] = []

    def load_instances(self, split: Split.ValueType) -> list[ToyInstance]:
        return [inst for inst in self.instances if inst.split == split]

    @asynccontextmanager
    async def provision(self, inst: ToyInstance) -> AsyncIterator[FakeSqlEnvironment]:
        env = FakeSqlEnvironment({inst.gold_sql: SqlResult()})
        self.environments.append(env)
        try:
            yield env
        finally:
            await env.close()

    def agent_input(self, inst: ToyInstance) -> TaskInput:
        bird = BirdTaskInput(db_id="toy", query=inst.query, issue_sql=[inst.issue_sql])
        return TaskInput(instance_id=inst.instance_id, bird=bird)

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
        return {"instance_id": inst.instance_id, "pred_sqls": list(prediction.bird.sqls)}
