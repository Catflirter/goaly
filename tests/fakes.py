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
from goaly.contracts.gen.manifest_pb2 import RunManifest
from goaly.contracts.gen.task_pb2 import (
    GRADE_STATUS_GRADED,
    SPLIT_EXTERNAL,
    SPLIT_TRAIN,
    SPLIT_VAL,
    EffiSqlGrade,
    EffiSqlPrediction,
    EffiSqlTaskInput,
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


class SubmitArgs(BaseModel):
    """SubmitTool's arguments."""

    sql: str


class SubmitTool:
    """A Tool that ends the episode by submitting its ``sql`` argument as an Effi-SQL answer.

    It works in any environment. Like any tool that submits, it returns a result with
    ``submission`` set.
    """

    name = "submit"
    description = "Submit the optimized query as the answer."
    args_model = SubmitArgs

    async def run(self, args: SubmitArgs, env: Environment) -> ToolResult:
        submission = Prediction(effisql=EffiSqlPrediction(sql=args.sql))
        return ToolResult(content="Submitted.", submission=submission)


class DoNothingPolicy:
    """A Policy that submits the do-nothing answer: the base query unchanged, or no SWE patch.

    It works in any environment and with any context. ``tasks`` records each task it solved.
    Like every policy, it leaves the prediction's ``instance_id`` to the runner.
    """

    def __init__(self) -> None:
        self.tasks: list[TaskInput] = []

    async def solve(self, task: TaskInput, env: Environment, ctx: object) -> Prediction:
        self.tasks.append(task)
        if task.HasField("swe"):
            return Prediction(swe=SwePrediction())
        return Prediction(effisql=EffiSqlPrediction(sql=task.effisql.base_sql))


@dataclass(frozen=True)
class ToyInstance:
    """A ToyBenchmark task: the benchmark's own record, so it holds the gold answer.

    A task from an organizer task file has no gold answer, so its ``gold_sql`` is None. Being a
    frozen dataclass with an ``instance_id``, a ToyInstance satisfies the Instance Protocol.
    """

    instance_id: str
    split: Split.ValueType
    base_sql: str
    gold_sql: str | None


# What a toy task's base query returns in its solving environment: one row, one column.
TOY_BASE_RESULT = SqlResult(
    columns=["n"], rows=[ListValue(values=[Value(number_value=3)])], row_count=1
)
TOY_INSTANCES = (
    ToyInstance(
        instance_id="toy-1",
        split=SPLIT_TRAIN,
        base_sql="SELECT count(*) FROM users WHERE id IN (SELECT id FROM users)",
        gold_sql="SELECT count(id) FROM users",  # id is the key, so it is never NULL.
    ),
    ToyInstance(  # Its base query is already as fast as it gets.
        instance_id="toy-2",
        split=SPLIT_TRAIN,
        base_sql="SELECT count(*) FROM orders",
        gold_sql="SELECT count(*) FROM orders",
    ),
    ToyInstance(
        instance_id="toy-3",
        split=SPLIT_VAL,
        base_sql="SELECT count(*) FROM (SELECT DISTINCT user_id FROM orders) AS buyers",
        gold_sql="SELECT count(DISTINCT user_id) FROM orders",
    ),
)

# The toy dataset's stand-in SHA-256, as Benchmark.manifest_fields reports a dataset file's hash.
TOY_DATASET_SHA256 = "0123456789abcdef" * 4


class ToyBenchmark:
    """A Benchmark of Effi-SQL-style tasks held in memory, each solved in a FakeSqlEnvironment.

    ``external`` stands in for an organizer task file: ``load_instances(SPLIT_EXTERNAL)``
    returns it, or raises ValueError if there is none. A task's solving environment answers
    only the task's base query, with TOY_BASE_RESULT, so it reveals nothing of the gold rewrite,
    which only ``grade`` uses. ``grade`` times nothing: the gold rewrite earns R-VES's top credit
    of 1, the base query unchanged earns 0.5, as a speedup of 1 does, and any other query does
    not match. A prediction with no branch set, as after a crash, stands for the empty answer,
    the base query unchanged, in both ``grade`` and ``to_official``. Its manifest fields name a
    fixed dataset revision and hash and the "dev" grading mode. ``environments`` records each
    environment provisioned.
    """

    name = "toy"

    def __init__(
        self,
        instances: Sequence[ToyInstance] = TOY_INSTANCES,
        *,
        external: Sequence[ToyInstance] | None = None,
    ) -> None:
        self.instances = tuple(instances)
        self.external = None if external is None else tuple(external)
        self.environments: list[FakeSqlEnvironment] = []

    def load_instances(self, split: Split.ValueType) -> list[ToyInstance]:
        if split != SPLIT_EXTERNAL:
            return [inst for inst in self.instances if inst.split == split]
        if self.external is None:
            raise ValueError("this toy benchmark was constructed without a task file")
        return list(self.external)

    @asynccontextmanager
    async def provision(self, inst: ToyInstance) -> AsyncIterator[FakeSqlEnvironment]:
        env = FakeSqlEnvironment({inst.base_sql: TOY_BASE_RESULT})
        self.environments.append(env)
        try:
            yield env
        finally:
            await env.close()

    def agent_input(self, inst: ToyInstance) -> TaskInput:
        # No instance ID and no gold: the agent sees only what Instance and agent_input allow.
        return TaskInput(effisql=EffiSqlTaskInput(db="toy", base_sql=inst.base_sql))

    async def grade(self, inst: ToyInstance, prediction: Prediction) -> Grade:
        sql = self._answer(inst, prediction)
        if sql == inst.gold_sql:
            credit = 1.0
        elif sql == inst.base_sql:
            credit = 0.5
        else:
            credit = 0.0
        matched = credit > 0
        return Grade(
            instance_id=inst.instance_id,
            score=credit,
            correct=matched,
            status=GRADE_STATUS_GRADED,
            effisql=EffiSqlGrade(matched=matched, credit=credit),
        )

    def manifest_fields(self) -> RunManifest:
        return RunManifest(
            dataset_revision="toy-revision-1",
            dataset_sha256=TOY_DATASET_SHA256,
            environment_settings={"grading_mode": "dev"},
        )

    def to_official(self, inst: ToyInstance, prediction: Prediction) -> dict[str, JsonValue]:
        return {"instance_id": inst.instance_id, "sql": self._answer(inst, prediction)}

    @staticmethod
    def _answer(inst: ToyInstance, prediction: Prediction) -> str:
        """The SQL ``prediction`` submits, or the base query unchanged if it has no branch."""
        return prediction.effisql.sql if prediction.HasField("effisql") else inst.base_sql
