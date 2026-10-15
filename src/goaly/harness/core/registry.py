"""The registry of what `goaly run` can run: benchmarks, each with the harnesses that solve it.

A harness is a benchmark adapter together with a pack's policy and the context its policy runs
with. The composition root (``goaly.cli``) registers each harness under its benchmark's name and
its own: T11 registers Effi-SQL's v0 pack, and tests register the toy benchmark. Adding a
benchmark registers it here; core does not change (spec §3.3).

A benchmark also has a host side, which the composition root registers with add_benchmark(): a
HostBenchmark, the benchmark adapter alone, with the launcher of its isolated solve phase
(spec §6.1, V27). The host plans and grades a run with it, and never imports pack code, so a
harness factory imports its pack's modules inside itself, never at the top of the module that
registers it. What the host needs to know of a harness, its pack's directory and its models, is
therefore given when the harness is registered, not by its factory.

A harness factory is an async context manager, so it owns what the harness needs for the run,
such as the LLM client and the writer of the run's LLM call log, and closes them after it. It
binds the harness's types, the benchmark's instance and environment and the policy's context,
inside: the CLI sees only RunnableHarness.
"""

from collections.abc import Callable, Collection, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from goaly.contracts.gen.manifest_pb2 import ModelSettings, RunManifest
from goaly.contracts.gen.task_pb2 import Split
from goaly.contracts.interfaces import Benchmark, Environment, Instance, Policy
from goaly.harness.core.eval_cache import EvalCache
from goaly.harness.core.grading import HostRunner, SolveOptions
from goaly.harness.core.isolation import SolveLauncher
from goaly.harness.core.runner import (
    ContextFactory,
    ErrorClassifier,
    RunFiles,
    RunLock,
    Runner,
    RunSummary,
    TaskContext,
    classify_error,
)

__all__ = [
    "Harness",
    "HarnessFactory",
    "HarnessOptions",
    "HostBenchmark",
    "HostFactory",
    "HostOptions",
    "RegisteredHarness",
    "Registry",
    "RegistryError",
    "RunnableHarness",
    "RunnableHost",
    "RunsHost",
    "RunsTasks",
]


class RegistryError(LookupError):
    """No benchmark or harness is registered under a name; the message lists those there are."""


@dataclass(frozen=True)
class HarnessOptions:
    """What `goaly run` gives a harness factory: in the solve container, or with --in-process."""

    # The run's directory: in the container, the solve directory. The LLM call log belongs in
    # files.llm_calls.
    files: RunFiles
    # The organizers' task file (--instances), for the adapter's SPLIT_EXTERNAL; or None.
    instances_file: Path | None
    # This process's share of each model's concurrency budget (llm_budgets); a model with no
    # share must not be called. The LLM client enforces it.
    llm_shares: Mapping[str, int]
    # The split the run's tasks come from; SPLIT_EXTERNAL for --instances. It sets the grading
    # mode of a benchmark whose mode depends on it.
    split: Split.ValueType
    # The schedule's phase whose concurrency shares apply (--budget-phase), as a benchmark may
    # order its own work by it; None if the command named none.
    budget_phase: str | None
    # True in the solve container (--in-container), where the LLM and the databases are reached
    # as the container's environment says.
    in_container: bool = False
    # In the solve container, the read-only directory of inputs its benchmark's launcher staged
    # (--inputs); None elsewhere.
    inputs_dir: Path | None = None
    # How --in-process grades: as HostOptions's.
    grading_priority: str | None = None
    grading_instances: str = "grading"


class RunsTasks(Protocol):
    """A Runner, whatever its types."""

    async def run(
        self,
        run_dir: Path,
        manifest: RunManifest,
        *,
        split: Split.ValueType,
        seeds: Sequence[int],
        instance_ids: Collection[str] | None = None,
        resume: bool = False,
        retry_errors: bool = False,
        lock: RunLock | None = None,
        grading: bool = True,
        tasks: Collection[tuple[str, int]] | None = None,
    ) -> RunSummary:
        """Run the tasks and record them, as Runner.run does."""


class RunnableHarness(Protocol):
    """A Harness, whatever its types: what the CLI needs to describe a run and run it."""

    @property
    def benchmark_name(self) -> str:
        """The benchmark adapter's name."""

    @property
    def gold_sha256(self) -> str:
        """The SHA-256 of the gold-answer files outside the dataset; empty if there are none."""

    @property
    def cache_namespace(self) -> str:
        """The grading identity its runs' evaluation cache is kept under; empty for none."""

    def runner(
        self, *, cache: EvalCache | None, max_tasks: int, task_timeout_s: float | None
    ) -> RunsTasks:
        """Return the Runner of this harness, with these options."""


@dataclass(frozen=True)
class Harness[InstanceT: Instance, EnvT: Environment, CtxT: TaskContext]:
    """A benchmark adapter, with the pack's policy that solves its tasks and their contexts."""

    benchmark: Benchmark[InstanceT, EnvT]
    policy: Policy[EnvT, CtxT]
    contexts: ContextFactory[InstanceT, CtxT]
    classify: ErrorClassifier = classify_error
    # For --in-process, which grades with this harness's benchmark: as HostBenchmark's, the
    # SHA-256 of the gold-answer files outside the dataset, and the grading identity that keys
    # the evaluation cache. The two must equal the benchmark's HostBenchmark's.
    gold_sha256: str = ""
    cache_namespace: str = ""

    @property
    def benchmark_name(self) -> str:
        """The benchmark adapter's name."""
        return self.benchmark.name

    def runner(
        self, *, cache: EvalCache | None, max_tasks: int, task_timeout_s: float | None
    ) -> Runner[InstanceT, EnvT, CtxT]:
        """Return the Runner of this harness, with these options."""
        return Runner(
            self.benchmark,
            self.policy,
            self.contexts,
            classify=self.classify,
            cache=cache,
            max_tasks=max_tasks,
            task_timeout_s=task_timeout_s,
        )


# Opens a harness for one run, and closes what it holds when the run is over.
type HarnessFactory = Callable[[HarnessOptions], AbstractAsyncContextManager[RunnableHarness]]


@dataclass(frozen=True)
class HostOptions:
    """What `goaly run`, on the host, gives a benchmark's host factory."""

    # The run's directory on the host.
    files: RunFiles
    # The organizers' task file (--instances), for the adapter's SPLIT_EXTERNAL; or None.
    instances_file: Path | None
    # As HarnessOptions.split; on --phase grade, the run's, from its manifest.
    split: Split.ValueType
    # As HarnessOptions.budget_phase.
    budget_phase: str | None
    # The priority of the run's grading among the grader's work (--grading-priority), such as
    # "smoke"; None for the one its split and budget phase give. The benchmark checks the name.
    grading_priority: str | None = None
    # Which database instances grade (--grading-instances): "grading", those kept for it, or
    # "all", the solving ones too, while no solve phase uses them.
    grading_instances: str = "grading"


class RunsHost(Protocol):
    """A HostRunner, whatever its types."""

    async def run(
        self,
        run_dir: Path,
        manifest: RunManifest,
        *,
        split: Split.ValueType,
        seeds: list[int],
        instance_ids: Collection[str] | None,
        solve: SolveOptions,
        resume: bool = False,
        retry_errors: bool = False,
        lock: RunLock | None = None,
        grading: bool = True,
    ) -> RunSummary:
        """Solve the run in the container, then grade it, as HostRunner.run does."""

    async def grade(
        self, run_dir: Path, *, lock: RunLock | None = None, retry_errors: bool = False
    ) -> RunSummary:
        """Grade the run's solved tasks, as HostRunner.grade does."""


class RunnableHost(Protocol):
    """A HostBenchmark, whatever its types: what the host side of a run needs."""

    @property
    def benchmark_name(self) -> str:
        """The benchmark adapter's name."""

    @property
    def cache_namespace(self) -> str:
        """The grading identity the evaluation cache is kept under; empty for none."""

    @property
    def gold_sha256(self) -> str:
        """The SHA-256 of the gold-answer files outside the dataset; empty if there are none."""

    @property
    def launcher(self) -> SolveLauncher | None:
        """The launcher of the benchmark's isolated solve phase; None if it has none."""

    def host_runner(self, *, cache: EvalCache | None) -> RunsHost:
        """Return the host side of this benchmark's runs, with this cache."""


@dataclass(frozen=True)
class HostBenchmark[InstanceT: Instance, EnvT: Environment]:
    """A benchmark's trusted host side: its adapter alone, which loads no pack code, and the
    launcher of its isolated solve phase (spec §6.1, V27).

    The host plans a run with it, takes the cached evaluations, has ``launcher`` solve the rest
    in the solve container, and grades every task.
    """

    benchmark: Benchmark[InstanceT, EnvT]
    # The grading identity of the benchmark's runs, which keys the evaluation cache (EvalCache's
    # namespace), such as f"{mode}-{bench_fingerprint}"; empty if they cannot use a cache.
    cache_namespace: str = ""
    # The SHA-256, in hex, of the gold-answer files its grader uses that are not part of the
    # dataset, for the manifest; empty if there are none. The dataset's own fields come from
    # the benchmark's manifest_fields() alone.
    gold_sha256: str = ""
    # Runs the isolated solve phase of a run's tasks (isolation.SolveRequest); None for a
    # benchmark that has none, which only --in-process can run.
    launcher: SolveLauncher | None = None

    @property
    def benchmark_name(self) -> str:
        """The benchmark adapter's name."""
        return self.benchmark.name

    def host_runner(self, *, cache: EvalCache | None) -> HostRunner[InstanceT, EnvT]:
        """Return the host side of this benchmark's runs, with this cache."""
        return HostRunner(self.benchmark, launcher=self.launcher, cache=cache)


# Opens a benchmark's host side for one run, and closes what it holds when the run is over.
type HostFactory = Callable[[HostOptions], AbstractAsyncContextManager[RunnableHost]]


@dataclass(frozen=True)
class RegisteredHarness:
    """A harness as it is registered: what the host knows of it without importing its pack."""

    name: str
    factory: HarnessFactory
    # The pack's directory, which the harness hash covers with core; None for a harness without
    # one, such as a test's.
    pack_dir: Path | None = None
    # The settings of every model the pack calls, for the manifest.
    models: tuple[ModelSettings, ...] = ()


class Registry:
    """The harnesses `goaly run` can run, by benchmark name, then by harness name."""

    def __init__(self) -> None:
        self._harnesses: dict[str, dict[str, RegisteredHarness]] = {}
        self._hosts: dict[str, HostFactory] = {}

    def add(
        self,
        benchmark: str,
        harness: str,
        factory: HarnessFactory,
        *,
        pack_dir: Path | None = None,
        models: Sequence[ModelSettings] = (),
    ) -> None:
        """Register ``factory`` as ``benchmark``'s harness ``harness``, a name it has once.

        ``pack_dir`` is the directory of the pack it runs, and ``models`` the settings of every
        model the pack calls: the host hashes the one and records the other without importing
        the pack.
        """
        harnesses = self._harnesses.setdefault(benchmark, {})
        if harness in harnesses:
            raise ValueError(f"{benchmark!r} already has a harness {harness!r}")
        harnesses[harness] = RegisteredHarness(harness, factory, pack_dir, tuple(models))

    def add_benchmark(self, benchmark: str, factory: HostFactory) -> None:
        """Register ``factory`` as the host side of ``benchmark``, which has one."""
        if benchmark in self._hosts:
            raise ValueError(f"{benchmark!r} already has a host side")
        self._hosts[benchmark] = factory

    def host(self, benchmark: str) -> HostFactory:
        """Return the factory of ``benchmark``'s host side, or raise RegistryError."""
        if benchmark not in self._hosts:
            known = ", ".join(sorted(self._hosts))
            there = f"the benchmarks that have one are: {known}" if known else "none has one"
            raise RegistryError(f"no host side for the benchmark {benchmark!r}; {there}")
        return self._hosts[benchmark]

    def benchmarks(self) -> list[str]:
        """The benchmarks registered with a harness, in name order."""
        return sorted(self._harnesses)

    def harnesses(self, benchmark: str) -> list[str]:
        """The names of ``benchmark``'s harnesses, in order."""
        return sorted(self._registered(benchmark))

    def harness(self, benchmark: str, harness: str | None) -> RegisteredHarness:
        """Return ``benchmark``'s harness ``harness``, as it was registered.

        Without a name, return the benchmark's only harness. Raise RegistryError, listing the
        names there are, if there is no such benchmark or harness, or no name for several.
        """
        harnesses = self._registered(benchmark)
        names = ", ".join(sorted(harnesses))
        if harness is None:
            if len(harnesses) > 1:
                raise RegistryError(f"{benchmark!r} has several harnesses: {names}; name one")
            return next(iter(harnesses.values()))
        if harness not in harnesses:
            raise RegistryError(
                f"{benchmark!r} has no harness {harness!r}; its harnesses are: {names}"
            )
        return harnesses[harness]

    def _registered(self, benchmark: str) -> dict[str, RegisteredHarness]:
        if benchmark not in self._harnesses:
            known = ", ".join(self.benchmarks())
            there = f"the benchmarks are: {known}" if known else "none is registered"
            raise RegistryError(f"no benchmark {benchmark!r}; {there}")
        return self._harnesses[benchmark]
