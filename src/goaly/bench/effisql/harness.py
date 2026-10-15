"""Effi-SQL's harness wiring (spec §6.1, §6.4): what `goaly run --bench effisql` opens for a run.

- :func:`open_harness` is the harness factory of the v0 pack, for the solve phase and the
  one-process mode: it loads the tasks and the agent's inputs, the pack (its config.yaml and
  prompt.md, and its policy, imported here and nowhere else), the LLM client with the run's call
  log, the solving pool, and a grader it starts only if the run grades.
- :func:`open_grading` is the grading factory of the grade phase: the same tasks and the grader,
  and no pack code at all.

The executor model is bound per task (:func:`context_factory`): the pack never sees the client,
the settings or a key, only the AgentContext core's loop calls through. :func:`classify_error`
says which LLM failures are the gateway's and which end the task. What reaches out of the
process, the LLM client, the PostgreSQL instances and the grader's workers, comes from an
:class:`EffiSqlWiring`, which tests replace with fakes.
"""

import asyncio
import dataclasses
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Self

import yaml
from pydantic import BaseModel, ConfigDict

import goaly.bench.effisql
import goaly.harness
from goaly.bench.effisql.benchmark import (
    EffiSqlBenchmark,
    GradeFunction,
    TaskSource,
    grading_mode,
)
from goaly.bench.effisql.data import (
    DATASET_REVISION,
    DATASET_SHA256,
    RUNTIME_FILE,
    EffiSqlInstance,
    default_cache_dir,
    load_instances,
)
from goaly.bench.effisql.grading import InstanceCheck, Mode
from goaly.bench.effisql.grading_pool import GRADING_INSTANCES, Priority
from goaly.bench.effisql.grading_service import EffiSqlGrader
from goaly.bench.effisql.inputs import load_plans, load_schema_snapshot
from goaly.bench.effisql.pg_settings import MAIN_DB, PgEnvConfig, load_credentials
from goaly.bench.effisql.splits import load_exclusions, load_splits
from goaly.bench.effisql.sql_env import SolvingPool
from goaly.contracts.gen.llm_call_pb2 import CALL_PURPOSE_EXEC, LlmCallRecord
from goaly.contracts.gen.manifest_pb2 import ModelSettings, RunManifest
from goaly.contracts.gen.result_pb2 import TASK_ERROR_TYPE_LLM
from goaly.contracts.gen.task_pb2 import SPLIT_EXTERNAL, SPLIT_HELDOUT, Grade, Prediction, Split
from goaly.contracts.interfaces import JsonValue, SqlEnvironment
from goaly.harness.core.agent import AgentContext
from goaly.harness.core.budgets import Budgets
from goaly.harness.core.context_window import ContextWindow
from goaly.harness.core.jsonl_writer import JsonlWriter
from goaly.harness.core.llm import BoundLlm, CallRecorder, CallTags, LlmClient
from goaly.harness.core.llm_errors import ConnectionFailed, RateLimited, ServerError, TimedOut
from goaly.harness.core.llm_models import EXECUTOR_MODEL, QWEN_PROFILE, ModelParams
from goaly.harness.core.manifest import content_hash, read_manifest, sha256_file
from goaly.harness.core.registry import (
    Grading,
    GradingOptions,
    Harness,
    HarnessOptions,
    RunnableGrading,
    RunnableHarness,
)
from goaly.harness.core.runner import ContextFactory, RunIdentity, error_classifier
from goaly.harness.core.settings import load_settings

__all__ = [
    "PACK_DIR",
    "BindsLlm",
    "EffiSqlPaths",
    "EffiSqlWiring",
    "PackRunConfig",
    "RunCallLog",
    "SolvesTasks",
    "classify_error",
    "context_factory",
    "grading_priority",
    "load_pack",
    "open_grading",
    "open_harness",
]

_logger = logging.getLogger(__name__)

# The v0 pack's directory, found without importing it: only open_harness imports pack code.
PACK_DIR = Path(goaly.harness.__file__).resolve().parent / "packs" / "effisql"
_BENCH_DIR = Path(goaly.bench.effisql.__file__).resolve().parent

# The gateway's outages, after the client's retries: no property of the harness, so the task is
# an infrastructure failure, never cached, and worth running again. Every other LLM failure,
# context overflow, an invalid response or a refused request, ends the task as a crash, as
# ImplementerExit does for the implementer (experiment.proto).
_GATEWAY_OUTAGES = (RateLimited, ServerError, TimedOut, ConnectionFailed)
classify_error = error_classifier(dict.fromkeys(_GATEWAY_OUTAGES, TASK_ERROR_TYPE_LLM))


class PackRunConfig(BaseModel):
    """A pack's config.yaml: what core takes from it, and the pack's own section.

    ``model`` overrides the executor profile's request parameters for every call, ``budgets``
    and ``context`` go to each task's AgentContext, and ``pack`` goes to the pack's
    build_policy, which validates it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: ModelParams
    budgets: Budgets
    context: ContextWindow = ContextWindow()
    pack: dict[str, JsonValue] = {}


def load_pack(pack_dir: Path) -> tuple[PackRunConfig, str]:
    """Read ``pack_dir``'s config.yaml and prompt.md. Raise ValueError if the config is invalid."""
    loaded = yaml.safe_load((pack_dir / "config.yaml").read_text(encoding="utf-8"))
    config = PackRunConfig.model_validate(loaded)
    return config, (pack_dir / "prompt.md").read_text(encoding="utf-8")


@dataclass(frozen=True)
class EffiSqlPaths:
    """Where a run's inputs are: the tasks, the agent's schema and plans, and the base cache."""

    # The gold-free runtime copy of the dataset, which our splits' tasks load from (spec §8.5).
    tasks_file: Path
    # T09's schema snapshot of tpch, and its plan cache.
    snapshot: Path
    plans: Path
    # The grader's caches, one directory per mode; dev mode's holds the setup's base cache.
    grading_cache_root: Path

    @classmethod
    def default(cls) -> Self:
        """The host's paths: the dataset cache and T09's data root."""
        pg = PgEnvConfig()
        return cls(
            tasks_file=default_cache_dir() / RUNTIME_FILE,
            snapshot=pg.snapshot_path(MAIN_DB),
            plans=pg.plans_path,
            grading_cache_root=pg.data_root / "grader",
        )


class BindsLlm(Protocol):
    """An LLM client that binds a model and tags: LlmClient, or a test's fake."""

    def bind(self, *, model: str, tags: CallTags) -> BoundLlm: ...


class SolvesTasks(Protocol):
    """The solving instances: SolvingPool, or a test's fake."""

    async def preflight(self) -> None: ...

    def environment(self) -> AbstractAsyncContextManager[SqlEnvironment]: ...


class GradesPredictions(Protocol):
    """The grader: EffiSqlGrader, or a test's fake."""

    def benchmark_grade(self, mode: Mode, *, priority: Priority) -> GradeFunction: ...

    def close(self) -> None: ...


async def _real_llm(recorder: CallRecorder, shares: Mapping[str, int]) -> LlmClient:
    settings = await asyncio.to_thread(load_settings)
    profile = QWEN_PROFILE.model_copy()  # The pack's model params override it per call.
    return LlmClient(settings, recorder, profiles=[profile], shares=_full_shares(shares))


def _full_shares(shares: Mapping[str, int]) -> dict[str, int]:
    """``shares``, with a nominal 1 for a model without one: the exec purpose cannot call it."""
    return {EXECUTOR_MODEL: 1, "glm-5.3": 1, **shares}


async def _real_solving() -> SolvingPool:
    credentials = await asyncio.to_thread(load_credentials)
    return SolvingPool(PgEnvConfig(), credentials)


def _real_grader(cache_root: Path, check: InstanceCheck | None) -> EffiSqlGrader:
    return EffiSqlGrader.on_instances(GRADING_INSTANCES, cache_root=cache_root, check=check)


@dataclass(frozen=True)
class EffiSqlWiring:
    """What the Effi-SQL factories open: the real ones by default; tests pass fakes."""

    paths: Callable[[], EffiSqlPaths] = EffiSqlPaths.default
    # Opens the LLM client, given the run's call log and this process's shares.
    llm: Callable[
        [CallRecorder, Mapping[str, int]],
        Awaitable[AbstractAsyncContextManager[BindsLlm]],
    ] = _real_llm
    solving: Callable[[], Awaitable[SolvesTasks]] = _real_solving
    # Starts the grader, given the cache root and the check each grading instance passes first.
    grader: Callable[[Path, InstanceCheck | None], GradesPredictions] = _real_grader
    # The check of a grading instance before each evaluation: T09's catalog fingerprint (spec
    # §8.5), once it is an InstanceCheck; until then None, and no instance is checked.
    instance_check: InstanceCheck | None = None
    pack_dir: Path = PACK_DIR


class RunCallLog:
    """The run's LLM call log, a CallRecorder that queues each record on the run's writer.

    It never raises: a record that cannot be queued is logged and dropped, as the client's
    contract asks, so a call never loses its reply to its log.
    """

    def __init__(self, writer: JsonlWriter[LlmCallRecord]) -> None:
        self._writer = writer

    def record(self, record: LlmCallRecord) -> None:
        try:
            self._writer.put(record)
        except Exception:
            _logger.exception("an LLM call record was lost", extra={"call_id": record.call_id})


def context_factory(
    client: BindsLlm, config: PackRunConfig
) -> ContextFactory[EffiSqlInstance, AgentContext]:
    """Each task's AgentContext: the executor model bound to the task, and the pack's budgets."""

    @asynccontextmanager
    async def open_context(
        inst: EffiSqlInstance, seed: int, run: RunIdentity
    ) -> AsyncIterator[AgentContext]:
        tags = CallTags(
            purpose=CALL_PURPOSE_EXEC, run_id=run.run_id, instance_id=inst.instance_id, seed=seed
        )
        llm = client.bind(model=EXECUTOR_MODEL, tags=tags)
        yield AgentContext(llm, budgets=config.budgets, params=config.model, window=config.context)

    return open_context


def grading_priority(split: Split.ValueType | None, budget_phase: str | None) -> Priority:
    """The grading instances' priority for a run (spec §6.2), by its phase of the schedule."""
    if budget_phase == "baseline":
        return Priority.BASELINE_HELDOUT if split == SPLIT_HELDOUT else Priority.BASELINE
    if budget_phase == "final":
        return Priority.FINAL
    return Priority.RSI


@dataclass(frozen=True)
class _Tasks:
    """What both factories load: the tasks, their splits and the agent's inputs."""

    benchmark: EffiSqlBenchmark
    namespace: str


def _load_tasks(
    paths: EffiSqlPaths,
    instances_file: Path | None,
    mode: Mode | None,
    environment_settings: Mapping[str, str],
) -> _Tasks:
    tasks = load_instances(paths.tasks_file)
    external = None if instances_file is None else load_instances(instances_file)
    if instances_file is None:
        source = TaskSource(
            sha256=DATASET_SHA256,
            revision=DATASET_REVISION,
            notes=(
                "only the 238 tpch tasks are in our splits",
                "the tasks load from the dataset's gold-free runtime copy",
            ),
        )
    else:
        source = TaskSource(sha256=sha256_file(instances_file))
    benchmark = EffiSqlBenchmark(
        tasks=tasks,
        splits=load_splits().without(load_exclusions()),
        snapshot=load_schema_snapshot(paths.snapshot),
        plans=load_plans(paths.plans),
        source=source,
        mode=mode,
        external=external,
        environment_settings=dict(environment_settings),
    )
    fingerprint = content_hash({"bench/effisql": _BENCH_DIR})[:16]
    namespace = "" if mode is None else f"{mode.value}-{fingerprint}"
    return _Tasks(benchmark, namespace)


class _LazyGrader:
    """Starts the grader's workers on the first grading only: a solve phase grades nothing."""

    def __init__(self, start: Callable[[], GradesPredictions], mode: Mode, priority: Priority):
        self._start = start
        self._mode = mode
        self._priority = priority
        self._grader: GradesPredictions | None = None
        self._lock = asyncio.Lock()

    async def __call__(self, inst: EffiSqlInstance, prediction: Prediction) -> Grade:
        async with self._lock:
            if self._grader is None:
                self._grader = await asyncio.to_thread(self._start)
        return await self._grader.benchmark_grade(self._mode, priority=self._priority)(
            inst, prediction
        )

    async def close(self) -> None:
        if self._grader is not None:
            await asyncio.to_thread(self._grader.close)


@asynccontextmanager
async def open_harness(
    options: HarnessOptions, wiring: EffiSqlWiring | None = None
) -> AsyncIterator[RunnableHarness]:
    """The v0 pack's harness for one run; the factory registered as effisql's "v0"."""
    wiring = EffiSqlWiring() if wiring is None else wiring
    # The pack's code, imported here only, so that a grade phase never loads it (spec V24).
    from goaly.harness.packs.effisql.policy import build_policy

    config, prompt = await asyncio.to_thread(load_pack, wiring.pack_dir)
    policy = build_policy(config.pack, prompt)
    split = options.split
    mode = Mode.DEV if split is None else grading_mode(split)
    priority = grading_priority(split, options.budget_phase)
    paths = wiring.paths()
    settings = {"grading_priority": priority.name.lower()}
    loaded = await asyncio.to_thread(_load_tasks, paths, options.instances_file, mode, settings)
    solving = await wiring.solving()
    await solving.preflight()
    async with AsyncExitStack() as stack:
        grader = None
        if mode is not None and split != SPLIT_EXTERNAL:
            grader = _LazyGrader(
                lambda: wiring.grader(paths.grading_cache_root, wiring.instance_check),
                mode,
                priority,
            )
            stack.push_async_callback(grader.close)
        calls = await stack.enter_async_context(JsonlWriter[LlmCallRecord](options.files.llm_calls))
        client = await stack.enter_async_context(
            await wiring.llm(RunCallLog(calls), options.llm_shares)
        )
        benchmark = _with(loaded.benchmark, solving.environment, grader)
        yield Harness(
            benchmark=benchmark,
            policy=policy,
            contexts=context_factory(client, config),
            classify=classify_error,
            pack_dir=wiring.pack_dir,
            models=(_model_settings(config.model),),
            cache_namespace=loaded.namespace,
        )


@asynccontextmanager
async def open_grading(
    options: GradingOptions, wiring: EffiSqlWiring | None = None
) -> AsyncIterator[RunnableGrading]:
    """Effi-SQL's grading for a grade phase; it imports no pack code (spec V24).

    The mode follows the run's split, whatever the solve phase wrote: its records are
    untrusted, and a manifest that names another mode is refused.
    """
    wiring = EffiSqlWiring() if wiring is None else wiring
    manifest = await asyncio.to_thread(read_manifest, options.files.manifest)
    mode = grading_mode(manifest.split)
    if mode is None:
        raise ValueError("a run on SPLIT_EXTERNAL is not graded")
    recorded = manifest.environment_settings.get("grading_mode")
    if recorded not in (None, mode.value):
        raise ValueError(
            f"the run's manifest names grading mode {recorded!r}, but its split grades in"
            f" {mode.value!r} mode"
        )
    priority = _recorded_priority(manifest)
    paths = wiring.paths()
    loaded = await asyncio.to_thread(_load_tasks, paths, None, mode, {})
    grader = _LazyGrader(
        lambda: wiring.grader(paths.grading_cache_root, wiring.instance_check), mode, priority
    )
    try:
        yield Grading(_with(loaded.benchmark, None, grader), cache_namespace=loaded.namespace)
    finally:
        await grader.close()


def _recorded_priority(manifest: RunManifest) -> Priority:
    """The priority the solve phase recorded; RSI's if it recorded none it may use."""
    name = manifest.environment_settings.get("grading_priority", "").upper()
    if name in Priority.__members__ and Priority[name] != Priority.SETUP:
        return Priority[name]
    return Priority.RSI


def _with(
    benchmark: EffiSqlBenchmark,
    provision: Callable[[], AbstractAsyncContextManager[SqlEnvironment]] | None,
    grader: _LazyGrader | None,
) -> EffiSqlBenchmark:
    return dataclasses.replace(benchmark, provision_environment=provision, grade_function=grader)


def _model_settings(params: ModelParams) -> ModelSettings:
    profile = QWEN_PROFILE.with_overrides(params)
    settings = ModelSettings(model=profile.model, thinking=bool(profile.params.enable_thinking))
    if profile.params.temperature is not None:
        settings.temperature = profile.params.temperature
    if profile.params.top_p is not None:
        settings.top_p = profile.params.top_p
    if profile.params.top_k is not None:
        settings.top_k = profile.params.top_k
    if profile.params.max_tokens is not None:
        settings.max_tokens = profile.params.max_tokens
    return settings
