"""Fakes for Effi-SQL runs without a database, a model or a grader: the FakeLLM dry run.

:func:`fake_wiring` gives the Effi-SQL factories synthetic inputs, a fake client whose every
task submits :data:`SUBMITTED`, fake solving instances and a grader that records what it grades.
Every task is synthetic, one per ID of our committed split.
"""

import json
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from goaly.bench.effisql.grading import Mode
from goaly.bench.effisql.grading_pool import Priority
from goaly.bench.effisql.harness import BindsLlm, EffiSqlPaths, EffiSqlWiring
from goaly.bench.effisql.splits import load_splits
from goaly.contracts.gen.environment_pb2 import SqlResult
from goaly.contracts.gen.task_pb2 import GRADE_STATUS_GRADED, EffiSqlGrade, Grade, Prediction
from goaly.contracts.interfaces import JsonValue, SqlEnvironment
from goaly.harness.core.llm import BoundLlm, CallRecorder, CallTags
from tests.bench.effisql.synthetic import task
from tests.fakes import FakeSqlEnvironment
from tests.harness.core.agent_fakes import FakeLLM, reply, tool_call

# What every fake task submits: a comment the official form drops, then the query.
SUBMITTED = "-- final query\nSELECT count(*)\nFROM orders"
OFFICIAL_FORM = "SELECT count(*)\nFROM orders"

SNAPSHOT: dict[str, JsonValue] = {
    "db": "tpch",
    "server_version": "16.4",
    "tables": [
        {
            "name": "orders",
            "row_count": 4500000,
            "columns": [{"name": "o_orderkey", "type": "integer", "nullable": False}],
            "primary_key": ["o_orderkey"],
            "foreign_keys": [],
            "indexes": [],
        }
    ],
    "ddl": "CREATE TABLE orders (o_orderkey integer PRIMARY KEY);",
}


def write_inputs(root: Path) -> EffiSqlPaths:
    """Write synthetic inputs under ``root``: a task for each split ID, a snapshot and plans."""
    splits = load_splits()
    ids = [*splits.train, *splits.val, *splits.heldout]
    tasks = root / "tasks.jsonl"
    records = [task(id=int(i), base_sql=" ".join(["SELECT", i, "FROM orders"])) for i in ids]
    tasks.write_text(
        "".join(json.dumps({**record, "optimized_sql": None}) + "\n" for record in records),
        encoding="utf-8",
    )
    snapshot = root / "schema_snapshot.json"
    snapshot.write_text(json.dumps(SNAPSHOT), encoding="utf-8")
    plans = root / "plans.json"
    plans.write_text("{}", encoding="utf-8")
    return EffiSqlPaths(tasks, snapshot, plans, root / "grader")


class FakeClient:
    """An LLM client whose every bound model submits SUBMITTED at its first call."""

    def __init__(self) -> None:
        self.bound: list[tuple[str, CallTags]] = []

    def bind(self, *, model: str, tags: CallTags) -> BoundLlm:
        self.bound.append((model, tags))
        return FakeLLM([reply(tool_call("submit", {"sql": SUBMITTED}))])


class FakeSolving:
    """Solving instances that give each task an empty fake SQL environment."""

    def __init__(self) -> None:
        self.preflights = 0
        self.environments = 0

    async def preflight(self) -> None:
        self.preflights += 1

    @asynccontextmanager
    async def environment(self) -> AsyncIterator[SqlEnvironment]:
        self.environments += 1
        yield FakeSqlEnvironment({OFFICIAL_FORM: SqlResult(row_count=1)})


@dataclass
class FakeGrader:
    """A grader that grades every prediction matched, with credit 0.5, and records each one."""

    graded: list[tuple[str, str, bool, Mode, Priority]] = field(default_factory=list)
    closed: bool = False

    async def grade(
        self,
        inst: object,
        prediction: Prediction,
        mode: Mode,
        *,
        priority: Priority = Priority.RSI,
        on_plan: Callable[[str, JsonValue], None] | None = None,
    ) -> Grade:
        self.graded.append(
            (prediction.instance_id, prediction.effisql.sql, prediction.fallback, mode, priority)
        )
        if on_plan is not None:
            on_plan(prediction.instance_id, {"Node Type": "Aggregate"})
        return Grade(
            instance_id=prediction.instance_id,
            score=0.5,
            correct=True,
            status=GRADE_STATUS_GRADED,
            effisql=EffiSqlGrade(matched=True, credit=0.5),
        )

    def close(self) -> None:
        self.closed = True


@dataclass
class FakeWiring:
    """The fakes a fake_wiring() run uses, for the test to inspect."""

    wiring: EffiSqlWiring
    client: FakeClient
    solving: FakeSolving
    grader: FakeGrader
    shares: list[Mapping[str, int]]


def fake_wiring(root: Path) -> FakeWiring:
    paths = write_inputs(root)
    client, solving, grader = FakeClient(), FakeSolving(), FakeGrader()
    shares: list[Mapping[str, int]] = []

    async def llm(
        recorder: CallRecorder, run_shares: Mapping[str, int]
    ) -> AbstractAsyncContextManager[BindsLlm]:
        shares.append(run_shares)

        @asynccontextmanager
        async def opened() -> AsyncIterator[BindsLlm]:
            yield client

        return opened()

    async def open_solving() -> FakeSolving:
        return solving

    wiring = EffiSqlWiring(
        paths=lambda: paths,
        llm=llm,
        solving=open_solving,
        grader=lambda _root, _check: grader,
    )
    return FakeWiring(wiring, client, solving, grader, shares)
