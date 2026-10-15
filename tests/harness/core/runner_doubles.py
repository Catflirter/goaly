"""Test doubles for the runner: a scripted benchmark, a scripted policy and a stub task context.

They build on tests/fakes.py. Each instance's behaviour is scripted by its ID: a provisioning,
solving, grading or conversion failure, a delay, or a wait on an event. The benchmark counts the
tasks that hold an environment at once, which is how many tasks run at once.
"""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from goaly.contracts.gen.llm_call_pb2 import Usage
from goaly.contracts.gen.task_pb2 import (
    SPLIT_TRAIN,
    EffiSqlPrediction,
    Grade,
    Prediction,
    Split,
    TaskInput,
)
from goaly.contracts.gen.trajectory_pb2 import STOP_REASON_SUBMITTED, Step, Trajectory
from goaly.contracts.interfaces import Environment, JsonValue
from goaly.harness.core.runner import RunIdentity
from tests.fakes import FakeSqlEnvironment, ToyBenchmark, ToyInstance


def base_sql(n: int) -> str:
    """The base query of toy task ``n``."""
    return f"SELECT {n} FROM slow"  # noqa: S608 - Test data, never run.


def toy_instances(count: int, split: Split.ValueType = SPLIT_TRAIN) -> list[ToyInstance]:
    """``count`` toy tasks, ``t0`` onwards: each base query differs, and so does each gold."""
    return [
        ToyInstance(instance_id=f"t{n}", split=split, base_sql=base_sql(n), gold_sql=f"SELECT {n}")
        for n in range(count)
    ]


class ScriptedBenchmark(ToyBenchmark):
    """A ToyBenchmark whose agent input, provisioning, grading and conversion fail for the
    instances named.

    Grading an instance sets its event in ``grade_started``, then waits for its event in
    ``grade_waits``.

    ``running`` counts the tasks that hold an environment now, and ``most_running`` the most at
    once. ``graded`` records every prediction graded, and ``official`` every one converted.
    """

    def __init__(
        self,
        instances: Sequence[ToyInstance],
        *,
        external: Sequence[ToyInstance] | None = None,
        input_errors: dict[str, Exception] | None = None,
        provision_errors: dict[str, Exception] | None = None,
        provision_delay_s: float = 0.0,
        grade_errors: dict[str, Exception] | None = None,
        grade_started: dict[str, asyncio.Event] | None = None,
        grade_waits: dict[str, asyncio.Event] | None = None,
        official_errors: dict[str, Exception] | None = None,
    ) -> None:
        super().__init__(instances, external=external)
        self.input_errors = input_errors or {}
        self.provision_errors = provision_errors or {}
        self.provision_delay_s = provision_delay_s
        self.grade_errors = grade_errors or {}
        self.grade_started = grade_started or {}
        self.grade_waits = grade_waits or {}
        self.official_errors = official_errors or {}
        self.running = 0
        self.most_running = 0
        self.graded: list[Prediction] = []
        self.official: list[Prediction] = []

    @asynccontextmanager
    async def provision(self, inst: ToyInstance) -> AsyncIterator[FakeSqlEnvironment]:
        self.running += 1
        self.most_running = max(self.most_running, self.running)
        try:
            await asyncio.sleep(self.provision_delay_s)
            if inst.instance_id in self.provision_errors:
                raise self.provision_errors[inst.instance_id]
            async with super().provision(inst) as env:
                yield env
        finally:
            self.running -= 1

    def agent_input(self, inst: ToyInstance) -> TaskInput:
        if inst.instance_id in self.input_errors:
            raise self.input_errors[inst.instance_id]
        return super().agent_input(inst)

    async def grade(self, inst: ToyInstance, prediction: Prediction) -> Grade:
        recorded = Prediction()
        recorded.CopyFrom(prediction)
        self.graded.append(recorded)
        await asyncio.sleep(0)
        if inst.instance_id in self.grade_started:
            self.grade_started[inst.instance_id].set()
        if inst.instance_id in self.grade_waits:
            await self.grade_waits[inst.instance_id].wait()
        if inst.instance_id in self.grade_errors:
            raise self.grade_errors[inst.instance_id]
        return await super().grade(inst, prediction)

    def to_official(self, inst: ToyInstance, prediction: Prediction) -> dict[str, JsonValue]:
        recorded = Prediction()
        recorded.CopyFrom(prediction)
        self.official.append(recorded)
        if prediction.HasField("effisql") and inst.instance_id in self.official_errors:
            raise self.official_errors[inst.instance_id]
        return super().to_official(inst, prediction)


class StubContext:
    """A task context, as the runner's context factory makes it: its trajectory is canned."""

    def __init__(self, inst: ToyInstance, seed: int, run: RunIdentity) -> None:
        self.instance_id = inst.instance_id
        self.seed = seed
        self.run = run
        self.entered = False
        self.exited = False
        self.trajectory_error: Exception | None = None
        self._trajectory = Trajectory(stop_reason=STOP_REASON_SUBMITTED)

    @property
    def recorded(self) -> Trajectory:
        """The trajectory it gives the runner, which a test may change before the run."""
        return self._trajectory

    def take_step(self, input_tokens: int = 100, output_tokens: int = 20) -> None:
        """Record one model call, as the agent loop would."""
        usage = Usage(input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=0.001)
        self._trajectory.steps.append(Step(index=len(self._trajectory.steps), usage=usage))
        total = self._trajectory.usage
        total.input_tokens += input_tokens
        total.output_tokens += output_tokens
        total.cost_usd += 0.001

    def trajectory(self) -> Trajectory:
        if self.trajectory_error is not None:
            raise self.trajectory_error
        return self._trajectory


@dataclass
class StubContexts:
    """A context factory that records every context it makes."""

    made: list[StubContext] = field(default_factory=list)
    setup: Callable[[StubContext], None] | None = None

    @asynccontextmanager
    async def __call__(
        self, inst: ToyInstance, seed: int, run: RunIdentity
    ) -> AsyncIterator[StubContext]:
        ctx = StubContext(inst, seed, run)
        if self.setup is not None:
            self.setup(ctx)
        self.made.append(ctx)
        ctx.entered = True
        try:
            yield ctx
        finally:
            ctx.exited = True


type Action = Callable[[TaskInput, Environment, StubContext], Awaitable[Prediction | None]]


class ScriptedPolicy:
    """A policy that takes two steps and submits the base query unchanged, or does as scripted.

    ``actions`` maps an instance ID to what to do instead; an action that returns None falls
    back to the default answer. ``solved`` records each (instance ID, seed) it was called for.
    """

    def __init__(self, actions: dict[str, Action] | None = None, *, delay_s: float = 0.0) -> None:
        self.actions = actions or {}
        self.delay_s = delay_s
        self.solved: list[tuple[str, int]] = []

    async def solve(self, task: TaskInput, env: Environment, ctx: StubContext) -> Prediction:
        self.solved.append((ctx.instance_id, ctx.seed))
        ctx.take_step()
        await asyncio.sleep(self.delay_s)
        ctx.take_step()
        action = self.actions.get(ctx.instance_id)
        prediction = None if action is None else await action(task, env, ctx)
        if prediction is None:
            prediction = Prediction(effisql=EffiSqlPrediction(sql=task.effisql.base_sql))
        return prediction


def raising(error: Exception) -> Action:
    """An action that raises ``error``."""

    async def act(task: TaskInput, env: Environment, ctx: StubContext) -> Prediction | None:
        raise error

    return act


def sleeping(seconds: float) -> Action:
    """An action that sleeps, then gives the default answer."""

    async def act(task: TaskInput, env: Environment, ctx: StubContext) -> Prediction | None:
        await asyncio.sleep(seconds)
        return None

    return act


def waiting(event: asyncio.Event) -> Action:
    """An action that waits for ``event``, then gives the default answer."""

    async def act(task: TaskInput, env: Environment, ctx: StubContext) -> Prediction | None:
        await event.wait()
        return None

    return act


def answering(sql: str) -> Action:
    """An action that submits ``sql``."""

    async def act(task: TaskInput, env: Environment, ctx: StubContext) -> Prediction | None:
        return Prediction(effisql=EffiSqlPrediction(sql=sql))

    return act
