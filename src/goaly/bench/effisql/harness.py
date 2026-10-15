"""Effi-SQL's harness wiring (spec §6.1, §6.4): what `goaly run --bench effisql` opens for a run.

- :func:`open_harness` is the harness factory of the v0 pack, for the solve phase and the
  one-process mode: it loads the tasks and the agent's inputs, the pack (its config.yaml and
  prompt.md, and its policy, imported here and nowhere else), the LLM client with the run's call
  log, the solving pool, and a grader it starts only if the run grades.
- :func:`open_host` is the trusted host side (core's HostBenchmark): the same tasks and the
  grader, and no pack code at all.

The executor model is bound per task (:func:`context_factory`): the pack never sees the client,
the settings or a key, only the AgentContext core's loop calls through. :func:`classify_error`
says which LLM failures are the gateway's and which end the task. What reaches out of the
process, the LLM client, the PostgreSQL instances and the grader's workers, comes from an
:class:`EffiSqlWiring`, which tests replace with fakes.
"""

import asyncio
import dataclasses
import json
import logging
import os
import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Self

import httpx
import yaml
from pydantic import BaseModel, ConfigDict, SecretStr

import goaly.bench.effisql
import goaly.harness
from goaly.bench.effisql.benchmark import (
    EffiSqlBenchmark,
    TaskSource,
    grading_mode,
)
from goaly.bench.effisql.data import (
    DATASET_REVISION,
    DATASET_SHA256,
    RUNTIME_FILE,
    ChecksumMismatchError,
    EffiSqlInstance,
    default_cache_dir,
    fetch_dataset,
    load_instances,
    write_runtime_file,
)
from goaly.bench.effisql.grading import InstanceCheck, Mode
from goaly.bench.effisql.grading_pool import (
    GRADING_INSTANCES,
    SETUP_INSTANCES,
    GradingInstance,
    Priority,
)
from goaly.bench.effisql.grading_service import EffiSqlGrader
from goaly.bench.effisql.inputs import load_plans, load_schema_snapshot
from goaly.bench.effisql.layout import SolveInputs
from goaly.bench.effisql.pg_instances import SOLVING
from goaly.bench.effisql.pg_settings import (
    MAIN_DB,
    AgentCredentials,
    PgEnvConfig,
    load_agent_credentials,
)
from goaly.bench.effisql.splits import load_exclusions, load_splits
from goaly.bench.effisql.sql_env import SolvingPool, SqlEnvOptions
from goaly.contracts.errors import EnvironmentFailure
from goaly.contracts.gen.llm_call_pb2 import CALL_PURPOSE_EXEC, LlmCallRecord
from goaly.contracts.gen.manifest_pb2 import ModelSettings, RunManifest
from goaly.contracts.gen.result_pb2 import TASK_ERROR_TYPE_LLM
from goaly.contracts.gen.task_pb2 import SPLIT_HELDOUT, Grade, Prediction, Split
from goaly.contracts.interfaces import JsonValue, SqlEnvironment
from goaly.harness.core.agent import AgentContext
from goaly.harness.core.budgets import Budgets
from goaly.harness.core.context_window import ContextWindow
from goaly.harness.core.grading import GRADING_TASK
from goaly.harness.core.isolation import SolveLauncher
from goaly.harness.core.jsonl_writer import JsonlWriter
from goaly.harness.core.llm import BoundLlm, CallRecorder, CallTags, LlmClient
from goaly.harness.core.llm_errors import ConnectionFailed, RateLimited, ServerError, TimedOut
from goaly.harness.core.llm_models import (
    DEFAULT_PROFILES,
    EXECUTOR_MODEL,
    QWEN_PROFILE,
    ModelParams,
)
from goaly.harness.core.manifest import content_hash, read_manifest, sha256_file
from goaly.harness.core.registry import (
    Harness,
    HarnessOptions,
    HostBenchmark,
    HostOptions,
    RunnableHarness,
    RunnableHost,
)
from goaly.harness.core.runner import ContextFactory, RunIdentity, error_classifier
from goaly.harness.core.settings import load_settings

__all__ = [
    "GRADING_INSTANCE_SETS",
    "PACK_DIR",
    "RUNTIME_SHA256",
    "SOLVING_STATEMENT_TIMEOUT_S",
    "BindsLlm",
    "EffiSqlPaths",
    "EffiSqlWiring",
    "LazyGrader",
    "LoadedTasks",
    "PackRunConfig",
    "PlanLog",
    "RunCallLog",
    "SolvesTasks",
    "classify_error",
    "context_factory",
    "grading_priority",
    "load_pack",
    "load_tasks",
    "model_settings",
    "named_priority",
    "open_harness",
    "open_host",
    "pack_models",
    "recorded_priority",
    "with_parts",
]

_logger = logging.getLogger(__name__)

# The v0 pack's directory, found without importing it: only open_harness imports pack code.
PACK_DIR = Path(goaly.harness.__file__).resolve().parent / "packs" / "effisql"
_BENCH_DIR = Path(goaly.bench.effisql.__file__).resolve().parent
# Every model the client has a profile of; each needs a share.
_ALL_MODELS = tuple(profile.model for profile in DEFAULT_PROFILES)
# The sha256 of the runtime copy of the pinned dataset (data.write_runtime_file), which is
# deterministic; the task file of our splits must be it.
RUNTIME_SHA256 = "f5145f80e557194c455ad4e8eb42ba3e95c54cdebb0787425d1e80a90de63412"

# The gateway's outages, after the client's retries: no property of the harness, so the task is
# an infrastructure failure, never cached, and worth running again. Every other LLM failure,
# context overflow, an invalid response or a refused request, ends the task as a crash, as
# ImplementerExit does for the implementer (experiment.proto).
_GATEWAY_OUTAGES = (RateLimited, ServerError, TimedOut, ConnectionFailed)
classify_error = error_classifier([(outage, TASK_ERROR_TYPE_LLM) for outage in _GATEWAY_OUTAGES])


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
    # The sha256 the task file must have: RUNTIME_SHA256, the runtime copy of the pinned
    # dataset; None not to check, as for a test's synthetic tasks.
    tasks_sha256: str | None = None
    # The pinned dataset the runtime copy derives from when it is missing; None if it cannot.
    dataset_cache: Path | None = None

    @classmethod
    def default(cls) -> Self:
        """The host's paths: the dataset cache and T09's data root."""
        pg = PgEnvConfig()
        cache = default_cache_dir()
        return cls(
            tasks_file=cache / RUNTIME_FILE,
            snapshot=pg.snapshot_path(MAIN_DB),
            plans=pg.plans_path,
            grading_cache_root=pg.data_root / "grader",
            tasks_sha256=RUNTIME_SHA256,
            dataset_cache=cache,
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

    async def grade(
        self,
        inst: EffiSqlInstance,
        prediction: Prediction,
        mode: Mode,
        *,
        priority: Priority = ...,
        on_plan: Callable[[str, JsonValue], None] | None = ...,
    ) -> Grade: ...

    def close(self) -> None: ...


class PlanLog:
    """The host-only audit of every graded prediction's EXPLAIN (FORMAT JSON) plan, for T23.

    Each plan T10's grader took goes to ``<run>/plans/seed-<k>.jsonl`` as a line
    ``{"instance_id": ..., "plan": ...}``. The solve container never sees the directory.
    """

    def __init__(self, run_root: Path) -> None:
        self._dir = run_root / "plans"
        self._lock = threading.Lock()

    def record(self, instance_id: str, plan: JsonValue) -> None:
        """Write ``plan``, under the seed core's GRADING_TASK names, the task being graded; a
        plan of no known task is logged."""
        task = GRADING_TASK.get()
        if task is None or task[0] != instance_id:
            _logger.warning("a graded plan of no known task", extra={"instance_id": instance_id})
            return
        line = json.dumps({"instance_id": instance_id, "plan": plan}, ensure_ascii=False)
        with self._lock:
            self._dir.mkdir(parents=True, exist_ok=True)
            with (self._dir / f"seed-{task[1]}.jsonl").open("a", encoding="utf-8") as file:
                file.write(line + "\n")


async def _real_llm(recorder: CallRecorder, shares: Mapping[str, int]) -> LlmClient:
    settings = await asyncio.to_thread(load_settings)
    profile = QWEN_PROFILE.model_copy()  # The pack's model params override it per call.
    # In the solve container no IP socket opens (V28): the proxy is on a Unix socket.
    uds = os.environ.get("GOALY_LLM_UDS")
    transport = None if not uds else httpx.AsyncHTTPTransport(uds=uds)
    return LlmClient(
        settings, recorder, profiles=[profile], shares=_full_shares(shares), transport=transport
    )


def _full_shares(shares: Mapping[str, int]) -> dict[str, int]:
    """``shares``, which must give the executor model one, and a nominal 1 for every other
    model: the exec purpose cannot call it, but the client needs a share for each."""
    if EXECUTOR_MODEL not in shares:
        raise ValueError(f"this process has no share of {EXECUTOR_MODEL}, which the pack calls")
    return {**dict.fromkeys(_ALL_MODELS, 1), **shares}


# The agent's statement timeout, a capacity rule of the shared solving instances (spec §6.2): the
# environment runs each statement for at most this, whatever the pack asks (T09's cap).
SOLVING_STATEMENT_TIMEOUT_S = 120.0


async def _real_solving() -> SolvingPool:
    """The solving instances, as the agent role only: no admin or grader password is read."""
    credentials = await asyncio.to_thread(load_agent_credentials)
    return SolvingPool(
        PgEnvConfig(), credentials, SqlEnvOptions(timeout_s=SOLVING_STATEMENT_TIMEOUT_S)
    )


async def _container_solving() -> SolvingPool:
    """The solving instances as the container's environment names them (V28): their Unix-socket
    directories, since the container opens no IP socket, and the agent role's password alone."""
    try:
        dirs = [Path(entry) for entry in os.environ["GOALY_PG_SOCKET_DIRS"].split(",")]
        password = os.environ["GOALY_PG_AGENT_PASSWORD"]
    except KeyError as error:
        raise EnvironmentFailure(f"the solve container lacks {error.args[0]}") from None
    if len(dirs) != len(SOLVING) or not all(path.is_absolute() for path in dirs):
        raise EnvironmentFailure(
            f"GOALY_PG_SOCKET_DIRS must name {len(SOLVING)} absolute directories, in SOLVING order"
        )
    credentials = AgentCredentials(agent_password=SecretStr(password))
    options = SqlEnvOptions(timeout_s=SOLVING_STATEMENT_TIMEOUT_S)
    sockets = {instance.name: path for instance, path in zip(SOLVING, dirs, strict=True)}
    return SolvingPool(PgEnvConfig(), credentials, options, socket_dirs=sockets)


# Which instances a run grades on (--grading-instances): the 3 grading instances, or all 5 for
# T23's final runs, which T10's per-instance locks keep apart from any solve phase.
GRADING_INSTANCE_SETS: Mapping[str, tuple[GradingInstance, ...]] = {
    "grading": GRADING_INSTANCES,
    "all": SETUP_INSTANCES,
}


def _real_grader(
    cache_root: Path, check: InstanceCheck | None, instances: str = "grading"
) -> EffiSqlGrader:
    return EffiSqlGrader.on_instances(
        GRADING_INSTANCE_SETS[instances], cache_root=cache_root, check=check
    )


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
    # Starts the grader, given the cache root, the check each grading instance passes first and
    # the set of instances it grades on (GRADING_INSTANCE_SETS).
    grader: Callable[[Path, InstanceCheck | None, str], GradesPredictions] = _real_grader
    # The check of a grading instance before each evaluation: T09's catalog fingerprint (spec
    # §8.5), once it is an InstanceCheck; until then None, and no instance is checked.
    instance_check: InstanceCheck | None = None
    pack_dir: Path = PACK_DIR

    @classmethod
    def in_container(cls, inputs_dir: Path) -> Self:
        """The solve container's wiring: its inputs, and its LLM and database environment.

        The LLM client's settings come from the environment, the proxy's URL and the run's
        token, and the database's from GOALY_PG_*, for the agent role only (spec §6.1).
        """
        inputs = SolveInputs(inputs_dir)
        paths = EffiSqlPaths(
            tasks_file=inputs.tasks,
            snapshot=inputs.snapshot,
            plans=inputs.plans,
            grading_cache_root=inputs.root / "no-grading",
        )
        return cls(paths=lambda: paths, solving=_container_solving, grader=_no_grader)


def _no_grader(
    cache_root: Path, check: InstanceCheck | None, instances: str = "grading"
) -> GradesPredictions:
    raise RuntimeError("the solve container grades nothing")


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
class LoadedTasks:
    """What both factories load: the tasks, their splits and the agent's inputs."""

    benchmark: EffiSqlBenchmark
    namespace: str


def load_tasks(
    paths: EffiSqlPaths,
    instances_file: Path | None,
    mode: Mode | None,
    environment_settings: Mapping[str, str],
) -> LoadedTasks:
    tasks = load_instances(_checked_tasks_file(paths))
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
    # The grading identity: the mode, the grader's code and, for dev mode, the base cache it
    # grades against, so a remeasured setup never hands out grades of the old one.
    fingerprint = content_hash({"bench/effisql": _BENCH_DIR})[:16]
    if mode is None:
        namespace = ""
    elif mode is Mode.DEV:
        namespace = f"dev-{fingerprint}-{_base_cache_digest(paths.grading_cache_root)}"
    else:
        namespace = f"{mode.value}-{fingerprint}"
    return LoadedTasks(benchmark, namespace)


def _checked_tasks_file(paths: EffiSqlPaths) -> Path:
    """The task file, derived from the pinned dataset if missing, and checked against its sha.

    A stale or changed runtime copy would load other tasks under the pinned dataset's name in
    the manifest, so it is refused: ChecksumMismatchError, a DatasetError.
    """
    if not paths.tasks_file.exists() and paths.dataset_cache is not None:
        write_runtime_file(fetch_dataset(paths.dataset_cache), paths.tasks_file)
    if paths.tasks_sha256 is not None:
        actual = sha256_file(paths.tasks_file)
        if actual != paths.tasks_sha256:
            raise ChecksumMismatchError(
                f"{paths.tasks_file} has sha256 {actual}, not the runtime copy's"
                f" {paths.tasks_sha256}; delete it to derive it again from the pinned dataset"
            )
    return paths.tasks_file


def _base_cache_digest(cache_root: Path) -> str:
    """A digest of the dev base cache the setup measured, which dev grades depend on."""
    dev = cache_root / Mode.DEV.value
    return content_hash({"dev": dev})[:16] if dev.is_dir() else "nobase"


class LazyGrader:
    """Starts the grader's workers on the first grading only: a solve phase grades nothing.

    With ``plans``, each plan the grader takes is written to the run's plan audit.
    """

    def __init__(
        self,
        start: Callable[[], GradesPredictions],
        mode: Mode,
        priority: Priority,
        plans: PlanLog | None = None,
    ):
        self._start = start
        self._mode = mode
        self._priority = priority
        self._plans = plans
        self._grader: GradesPredictions | None = None
        self._lock = asyncio.Lock()

    async def __call__(self, inst: EffiSqlInstance, prediction: Prediction) -> Grade:
        async with self._lock:
            if self._grader is None:
                self._grader = await asyncio.to_thread(self._start)
        on_plan = None if self._plans is None else self._plans.record
        return await self._grader.grade(
            inst, prediction, self._mode, priority=self._priority, on_plan=on_plan
        )

    async def close(self) -> None:
        if self._grader is not None:
            await asyncio.to_thread(self._grader.close)


def _inputs_dir(options: HarnessOptions) -> Path:
    if options.inputs_dir is None:
        raise ValueError("the solve container needs its inputs directory (--inputs)")
    return options.inputs_dir


@asynccontextmanager
async def open_harness(
    options: HarnessOptions, wiring: EffiSqlWiring | None = None
) -> AsyncIterator[RunnableHarness]:
    """The v0 pack's harness for one run; the factory registered as effisql's "v0"."""
    if wiring is None:
        wiring = (
            EffiSqlWiring.in_container(_inputs_dir(options))
            if options.in_container
            else EffiSqlWiring()
        )
    # The pack's code, imported here only, so that a grade phase never loads it (spec V24).
    from goaly.harness.packs.effisql.policy import build_policy

    config, prompt = await asyncio.to_thread(load_pack, wiring.pack_dir)
    policy = build_policy(config.pack, prompt)
    split = options.split
    mode = grading_mode(split)
    priority = (
        named_priority(options.grading_priority)
        if options.grading_priority is not None
        else grading_priority(split, options.budget_phase)
    )
    paths = wiring.paths()
    settings = {"grading_priority": priority.name.lower()}
    loaded = await asyncio.to_thread(load_tasks, paths, options.instances_file, mode, settings)
    solving = await wiring.solving()
    await solving.preflight()
    async with AsyncExitStack() as stack:
        grader = None
        if mode is not None:
            instances = _grading_instances(options.grading_instances)
            grader = LazyGrader(
                lambda: wiring.grader(paths.grading_cache_root, wiring.instance_check, instances),
                mode,
                priority,
                PlanLog(options.files.root),
            )
            stack.push_async_callback(grader.close)
        calls = await stack.enter_async_context(JsonlWriter[LlmCallRecord](options.files.llm_calls))
        client = await stack.enter_async_context(
            await wiring.llm(RunCallLog(calls), options.llm_shares)
        )
        benchmark = with_parts(loaded.benchmark, solving.environment, grader)
        yield Harness(
            benchmark=benchmark,
            policy=policy,
            contexts=context_factory(client, config),
            classify=classify_error,
            cache_namespace=loaded.namespace,
        )


@asynccontextmanager
async def open_host(
    options: HostOptions,
    wiring: EffiSqlWiring | None = None,
    launcher: SolveLauncher | None = None,
) -> AsyncIterator[RunnableHost]:
    """Effi-SQL's trusted host side for one run: its benchmark, which grades on the host and
    imports no pack code (spec V24, V27).

    The mode follows the split core gives: held-out official, train and val dev. The grading
    instances' priority follows the budget phase or, for a grade phase without one, what the
    run's host-written manifest recorded; a manifest that names another mode is refused.
    ``launcher`` solves the run's tasks in the solve container: EffiSqlLauncher by default.
    """
    from goaly.bench.effisql.launcher import EffiSqlLauncher

    wiring = EffiSqlWiring() if wiring is None else wiring
    mode = grading_mode(options.split)
    manifest = (
        await asyncio.to_thread(read_manifest, options.files.manifest)
        if await asyncio.to_thread(options.files.manifest.exists)
        else None
    )
    recorded = None if manifest is None else manifest.environment_settings.get("grading_mode")
    if recorded not in (None, _mode_name(mode)):
        raise ValueError(
            f"the run's manifest names grading mode {recorded!r}, but its split grades in"
            f" {_mode_name(mode)!r} mode"
        )
    if options.grading_priority is not None:
        priority = named_priority(options.grading_priority)
    elif options.budget_phase is None and manifest is not None:
        priority = recorded_priority(manifest)
    else:
        priority = grading_priority(options.split, options.budget_phase)
    instances = _grading_instances(options.grading_instances)
    paths = wiring.paths()
    settings = {"grading_priority": priority.name.lower()}
    loaded = await asyncio.to_thread(load_tasks, paths, options.instances_file, mode, settings)
    grader = None
    if mode is not None:
        grader = LazyGrader(
            lambda: wiring.grader(paths.grading_cache_root, wiring.instance_check, instances),
            mode,
            priority,
            PlanLog(options.files.root),
        )
    try:
        yield HostBenchmark(
            with_parts(loaded.benchmark, None, grader),
            cache_namespace=loaded.namespace,
            launcher=EffiSqlLauncher(paths=wiring.paths) if launcher is None else launcher,
        )
    finally:
        if grader is not None:
            await grader.close()


def _mode_name(mode: Mode | None) -> str:
    return "none" if mode is None else mode.value


def named_priority(name: str) -> Priority:
    """The priority --grading-priority names: smoke, baseline, baseline_heldout, rsi or final."""
    member = name.upper()
    if member not in Priority.__members__ or Priority[member] is Priority.SETUP:
        allowed = ", ".join(p.name.lower() for p in Priority if p is not Priority.SETUP)
        raise ValueError(f"no grading priority {name!r}; the priorities are: {allowed}")
    return Priority[member]


def _grading_instances(name: str) -> str:
    if name not in GRADING_INSTANCE_SETS:
        raise ValueError(f"--grading-instances is one of {', '.join(GRADING_INSTANCE_SETS)}")
    return name


def recorded_priority(manifest: RunManifest) -> Priority:
    """The priority the solve phase recorded; RSI's if it recorded none it may use."""
    name = manifest.environment_settings.get("grading_priority", "").upper()
    if name in Priority.__members__ and Priority[name] != Priority.SETUP:
        return Priority[name]
    return Priority.RSI


def with_parts(
    benchmark: EffiSqlBenchmark,
    provision: Callable[[], AbstractAsyncContextManager[SqlEnvironment]] | None,
    grader: LazyGrader | None,
) -> EffiSqlBenchmark:
    return dataclasses.replace(benchmark, provision_environment=provision, grade_function=grader)


def pack_models(pack_dir: Path = PACK_DIR) -> tuple[ModelSettings, ...]:
    """The settings of the models the pack calls, from its config.yaml, for Registry.add.

    A config that cannot be read gives none and a warning, so it never blocks a command that
    does not run the pack, such as a grade phase or --help; a run of the pack fails on it.
    """
    try:
        config, _ = load_pack(pack_dir)
    except (OSError, ValueError, yaml.YAMLError):
        _logger.warning("the pack's config.yaml cannot be read", extra={"pack": str(pack_dir)})
        return ()
    return (model_settings(config.model),)


def model_settings(params: ModelParams) -> ModelSettings:
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
