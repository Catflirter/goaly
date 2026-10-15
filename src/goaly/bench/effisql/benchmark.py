"""Effi-SQL's benchmark adapter (spec §3.3, §4.1): its tasks, environments, grades and records.

:class:`EffiSqlBenchmark` implements the Benchmark Protocol over the parts the other modules
build: the tasks of a task file (data), our splits less the exclusions (splits), the agent's
view of a task (inputs), a task's read-only SQL environment (T09's SolvingPool, given as a
provisioner), the grader (T10, given as a grade function) and the official record (adapter).

Grading follows the official record: a prediction is graded as its official form, the query both
blocks of its record carry, and a query with no official form fails, as the official runner
would fail it. The grading mode follows the split: held-out runs are graded in official mode,
train and val in dev mode, and SPLIT_EXTERNAL is never graded (spec §4.1, §6.2).
"""

from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field

from goaly.bench.effisql.adapter import official_answer, to_official
from goaly.bench.effisql.data import EffiSqlInstance
from goaly.bench.effisql.grading import Mode
from goaly.bench.effisql.inputs import SchemaSnapshot, agent_input
from goaly.bench.effisql.splits import EffiSqlSplits
from goaly.contracts.gen.manifest_pb2 import RunManifest
from goaly.contracts.gen.task_pb2 import (
    GRADE_STATUS_PREDICTION_FAILED,
    SPLIT_EXTERNAL,
    SPLIT_HELDOUT,
    SPLIT_TRAIN,
    SPLIT_VAL,
    EffiSqlGrade,
    EffiSqlPrediction,
    Grade,
    Prediction,
    Split,
    TaskInput,
)
from goaly.contracts.interfaces import JsonValue, SqlEnvironment

__all__ = [
    "BENCHMARK_NAME",
    "EffiSqlBenchmark",
    "GradeFunction",
    "Provisioner",
    "TaskSource",
    "grading_mode",
]

BENCHMARK_NAME = "effisql"

# Grades a prediction, as EffiSqlGrader.benchmark_grade returns it.
type GradeFunction = Callable[[EffiSqlInstance, Prediction], Awaitable[Grade]]
# Opens a task's SQL environment, as SolvingPool.environment does.
type Provisioner = Callable[[], AbstractAsyncContextManager[SqlEnvironment]]


def grading_mode(split: Split.ValueType) -> Mode | None:
    """The grader's mode for a run on ``split``; None for SPLIT_EXTERNAL, which is not graded.

    Held-out runs, v0's baseline and the final runs, are graded in official mode; train and val
    runs, including RSI's and the final val runs, in dev mode (spec §4.1, §6.2).
    """
    if split == SPLIT_HELDOUT:
        return Mode.OFFICIAL
    if split in (SPLIT_TRAIN, SPLIT_VAL):
        return Mode.DEV
    if split == SPLIT_EXTERNAL:
        return None
    raise ValueError(f"no grading mode for {Split.Name(split)}")


@dataclass(frozen=True)
class TaskSource:
    """Where the tasks come from, for the run manifest.

    ``sha256`` is that of the task file in the official format (RunManifest.dataset_sha256): the
    dataset for our splits, even when the tasks load from its gold-free copy, or the organizers'
    file. ``revision`` is the dataset's revision, empty for a file from outside.
    """

    sha256: str
    revision: str = ""
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, eq=False)
class EffiSqlBenchmark:
    """The Effi-SQL adapter; see the module docstring.

    ``tasks`` are those of the task file our splits draw on, and ``splits`` our splits with the
    exclusions dropped; ``external`` are the tasks of an organizers' file, if the run has one.
    ``provision`` and ``grade_function`` are None where the run does not need them: a grade
    phase provisions nothing, and a solve phase grades nothing.
    """

    tasks: Sequence[EffiSqlInstance]
    splits: EffiSqlSplits
    snapshot: SchemaSnapshot
    plans: Mapping[str, str]
    source: TaskSource
    mode: Mode | None
    provision_environment: Provisioner | None = None
    grade_function: GradeFunction | None = None
    external: Sequence[EffiSqlInstance] | None = None
    environment_settings: Mapping[str, str] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return BENCHMARK_NAME

    def load_instances(self, split: Split.ValueType) -> Sequence[EffiSqlInstance]:
        """The tasks of ``split``, in its order; for SPLIT_EXTERNAL, the organizers' file's."""
        if split == SPLIT_EXTERNAL:
            if self.external is None:
                raise ValueError("this run has no task file of its own (--instances)")
            return tuple(self.external)
        by_id = {inst.instance_id: inst for inst in self.tasks}
        ids = self.splits.ids(split)
        missing = [instance_id for instance_id in ids if instance_id not in by_id]
        if missing:
            raise ValueError(f"the task file lacks tasks of the split: {', '.join(missing)}")
        return tuple(by_id[instance_id] for instance_id in ids)

    def provision(self, inst: EffiSqlInstance) -> AbstractAsyncContextManager[SqlEnvironment]:
        """A read-only session on a solving instance, for any task: they all run on tpch."""
        if self.provision_environment is None:
            raise RuntimeError("this run provisions no environments: it only grades")
        return self.provision_environment()

    def agent_input(self, inst: EffiSqlInstance) -> TaskInput:
        return agent_input(inst, self.snapshot, self.plans)

    async def grade(self, inst: EffiSqlInstance, prediction: Prediction) -> Grade:
        """Grade ``prediction``'s official form; one with none fails as PREDICTION_FAILED."""
        if self.grade_function is None:
            raise RuntimeError("this run grades nothing: its grade phase grades it")
        answer = official_answer(inst, prediction)
        if answer.query is None:
            return Grade(
                instance_id=inst.instance_id,
                status=GRADE_STATUS_PREDICTION_FAILED,
                feedback=f"the answer has no official form: {answer.problem}",
                effisql=EffiSqlGrade(),
            )
        graded = Prediction(
            instance_id=inst.instance_id,
            fallback=prediction.fallback,
            effisql=EffiSqlPrediction(sql=answer.query),
        )
        return await self.grade_function(inst, graded)

    def manifest_fields(self) -> RunManifest:
        settings = {**self.environment_settings, "grading_mode": _mode_name(self.mode)}
        return RunManifest(
            dataset_revision=self.source.revision,
            dataset_notes=self.source.notes,
            dataset_sha256=self.source.sha256,
            environment_settings=settings,
        )

    def to_official(self, inst: EffiSqlInstance, prediction: Prediction) -> dict[str, JsonValue]:
        return to_official(inst, prediction)


def _mode_name(mode: Mode | None) -> str:
    return "none" if mode is None else mode.value
