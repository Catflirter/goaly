"""Tests for goaly.bench.effisql.benchmark: the Effi-SQL adapter over its parts."""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest

from goaly.bench.effisql.adapter import to_official
from goaly.bench.effisql.benchmark import EffiSqlBenchmark, TaskSource, grading_mode
from goaly.bench.effisql.data import EffiSqlInstance
from goaly.bench.effisql.grading import Mode
from goaly.bench.effisql.inputs import SchemaSnapshot
from goaly.bench.effisql.splits import EffiSqlSplits
from goaly.contracts.gen.task_pb2 import (
    GRADE_STATUS_GRADED,
    GRADE_STATUS_PREDICTION_FAILED,
    SPLIT_EXTERNAL,
    SPLIT_HELDOUT,
    SPLIT_TRAIN,
    SPLIT_UNSPECIFIED,
    SPLIT_VAL,
    EffiSqlPrediction,
    Grade,
    Prediction,
)
from goaly.contracts.interfaces import Benchmark, SqlEnvironment
from tests.bench.effisql.fake_run import SNAPSHOT
from tests.bench.effisql.synthetic import task
from tests.fakes import FakeSqlEnvironment

pytestmark = pytest.mark.anyio

SPLITS = EffiSqlSplits(seed=1, source_sha256="x", train=("1", "2"), val=("3",), heldout=("4",))


def inst(i: int, base_sql: str = "SELECT 1 FROM orders") -> EffiSqlInstance:
    return EffiSqlInstance.model_validate(task(id=i, base_sql=base_sql))


class Recorder:
    def __init__(self) -> None:
        self.graded: list[Prediction] = []

    async def __call__(self, instance: EffiSqlInstance, prediction: Prediction) -> Grade:
        self.graded.append(prediction)
        return Grade(instance_id=instance.instance_id, status=GRADE_STATUS_GRADED, score=1)


def benchmark(**fields: object) -> EffiSqlBenchmark:
    defaults: dict[str, object] = {
        "tasks": [inst(i) for i in (2, 1, 3, 4)],
        "splits": SPLITS,
        "snapshot": SchemaSnapshot.model_validate_json(json.dumps(SNAPSHOT)),
        "plans": {},
        "source": TaskSource(sha256="abc", revision="rev", notes=("note",)),
        "mode": Mode.DEV,
    }
    return EffiSqlBenchmark(**{**defaults, **fields})  # type: ignore[arg-type]


def test_it_is_a_benchmark() -> None:
    adapter: Benchmark[EffiSqlInstance, SqlEnvironment] = benchmark()

    assert adapter.name == "effisql"


@pytest.mark.parametrize(
    ("split", "mode"),
    [
        (SPLIT_TRAIN, Mode.DEV),
        (SPLIT_VAL, Mode.DEV),
        (SPLIT_HELDOUT, Mode.OFFICIAL),
        (SPLIT_EXTERNAL, None),
    ],
)
def test_the_grading_mode_follows_the_split(split: int, mode: Mode | None) -> None:
    assert grading_mode(split) is mode  # type: ignore[arg-type]


def test_an_unspecified_split_has_no_grading_mode() -> None:
    with pytest.raises(ValueError, match="SPLIT_UNSPECIFIED"):
        grading_mode(SPLIT_UNSPECIFIED)


class TestInstances:
    def test_a_split_loads_its_tasks_in_its_order(self) -> None:
        adapter = benchmark()

        assert [i.instance_id for i in adapter.load_instances(SPLIT_TRAIN)] == ["1", "2"]
        assert [i.instance_id for i in adapter.load_instances(SPLIT_HELDOUT)] == ["4"]

    def test_a_task_file_that_lacks_a_split_task_is_refused(self) -> None:
        with pytest.raises(ValueError, match="lacks tasks of the split: 3"):
            benchmark(tasks=[inst(1)]).load_instances(SPLIT_VAL)

    def test_external_runs_take_the_organizers_tasks(self) -> None:
        adapter = benchmark(external=[inst(9)])

        assert [i.instance_id for i in adapter.load_instances(SPLIT_EXTERNAL)] == ["9"]
        with pytest.raises(ValueError, match="--instances"):
            benchmark().load_instances(SPLIT_EXTERNAL)

    def test_the_agent_sees_the_task_but_no_gold(self) -> None:
        task_input = benchmark().agent_input(inst(1))

        assert task_input.effisql.base_sql == "SELECT 1 FROM orders"
        assert "orders" in task_input.effisql.schema
        assert "synthetic gold" not in str(task_input)


class TestProvisionAndGrade:
    async def test_provision_opens_a_solving_environment(self) -> None:
        opened = []

        @asynccontextmanager
        async def environment() -> AsyncIterator[SqlEnvironment]:
            opened.append(1)
            yield FakeSqlEnvironment()

        async with benchmark(provision_environment=environment).provision(inst(1)):
            assert opened == [1]

    def test_a_grade_only_run_provisions_nothing(self) -> None:
        with pytest.raises(RuntimeError, match="only grades"):
            benchmark().provision(inst(1))

    async def test_a_solve_only_run_grades_nothing(self) -> None:
        with pytest.raises(RuntimeError, match="grades nothing"):
            await benchmark().grade(inst(1), Prediction())

    async def test_the_official_form_is_graded_with_the_fallback_flag(self) -> None:
        grader = Recorder()
        prediction = Prediction(
            instance_id="1",
            fallback=True,
            effisql=EffiSqlPrediction(sql="-- why\nSELECT 2\nFROM orders;"),
        )

        grade = await benchmark(grade_function=grader).grade(inst(1), prediction)

        assert grade.status == GRADE_STATUS_GRADED
        assert [(p.instance_id, p.effisql.sql, p.fallback) for p in grader.graded] == [
            ("1", "SELECT 2\nFROM orders", True)
        ]

    async def test_a_prediction_without_an_answer_is_graded_as_base_sql(self) -> None:
        grader = Recorder()

        await benchmark(grade_function=grader).grade(inst(1, "SELECT 7 FROM orders"), Prediction())

        assert grader.graded[0].effisql.sql == "SELECT 7 FROM orders"

    async def test_an_answer_with_no_official_form_fails_without_grading(self) -> None:
        grader = Recorder()
        prediction = Prediction(effisql=EffiSqlPrediction(sql="SELECT 1 FROM t; SELECT 2 FROM t"))

        grade = await benchmark(grade_function=grader).grade(inst(1), prediction)

        assert grade.status == GRADE_STATUS_PREDICTION_FAILED
        assert grade.instance_id == "1"
        assert "more than one statement" in grade.feedback
        assert not grade.correct
        assert grade.HasField("effisql")
        assert grader.graded == []


class TestRecords:
    def test_manifest_fields(self) -> None:
        fields = benchmark(environment_settings={"grading_priority": "rsi"}).manifest_fields()

        assert fields.dataset_revision == "rev"
        assert list(fields.dataset_notes) == ["note"]
        assert fields.dataset_sha256 == "abc"
        assert dict(fields.environment_settings) == {
            "grading_priority": "rsi",
            "grading_mode": "dev",
        }
        assert not fields.run_id
        assert not fields.harness_hash

    def test_an_ungraded_run_names_no_mode(self) -> None:
        fields = benchmark(mode=None).manifest_fields()

        assert fields.environment_settings["grading_mode"] == "none"

    def test_the_official_record_is_the_adapters(self) -> None:
        prediction = Prediction(effisql=EffiSqlPrediction(sql="SELECT 2 FROM orders"))

        assert benchmark().to_official(inst(1), prediction) == to_official(inst(1), prediction)
