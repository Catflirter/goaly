"""The registry of what `goaly run` can run: benchmarks, each with the harnesses that solve it.

A harness is a benchmark adapter together with a pack's policy and the context its policy runs
with. The composition root (``goaly.cli``) registers each harness under its benchmark's name and
its own: T11 registers Effi-SQL's v0 pack, and tests register the toy benchmark. Adding a
benchmark registers it here; core does not change (spec §3.3).

The grade phase of a two-phase run (spec V24) needs the benchmark alone: a Grading, which the
composition root registers with add_benchmark(). Opening one must not import any pack code, so
a harness factory imports its pack's modules inside itself, never at the top of the module that
registers it.

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
from goaly.harness.core.grading import Grader
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
    "Dataset",
    "GradesRuns",
    "Grading",
    "GradingFactory",
    "GradingOptions",
    "Harness",
    "HarnessFactory",
    "HarnessOptions",
    "Registry",
    "RegistryError",
    "RunnableGrading",
    "RunnableHarness",
    "RunsTasks",
]


class RegistryError(LookupError):
    """No benchmark or harness is registered under a name; the message lists those there are."""


@dataclass(frozen=True)
class Dataset:
    """Where a benchmark's tasks came from, as the run manifest records it."""

    # The revision of the dataset, such as the commit of its Hugging Face repository.
    revision: str = ""
    # How the loader changed the dataset; empty if it changed nothing.
    notes: tuple[str, ...] = ()
    # The SHA-256 of gold-answer files outside the dataset, in hex; empty if there are none.
    gold_sha256: str = ""


@dataclass(frozen=True)
class HarnessOptions:
    """What `goaly run` gives a harness factory."""

    # The run's directory. The LLM call log belongs in files.llm_calls.
    files: RunFiles
    # The organizers' task file (--instances), for the adapter's SPLIT_EXTERNAL; or None.
    instances_file: Path | None
    # This process's share of each model's concurrency budget (llm_budgets); a model with no
    # share must not be called. The LLM client enforces it.
    llm_shares: Mapping[str, int]
    # The split the run takes its tasks from: SPLIT_EXTERNAL for --instances; None if unknown, as
    # in a test. A benchmark may grade by it, e.g. Effi-SQL's held-out runs in official mode.
    split: Split.ValueType | None = None
    # The schedule's phase whose shares apply (--budget-phase): baseline, rsi or final; None if
    # unknown. A benchmark may schedule its grading by it.
    budget_phase: str | None = None
    # True only under --phase solve --in-container, in the solve container (T06's interface).
    in_container: bool = False
    # The read-only inputs directory the container's launcher staged (--inputs); None outside it.
    inputs_dir: Path | None = None


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
        lock: RunLock | None = None,
        grading: bool = True,
    ) -> RunSummary:
        """Run the tasks and record them, as Runner.run does."""


class RunnableHarness(Protocol):
    """A Harness, whatever its types: what the CLI needs to describe a run and run it."""

    @property
    def benchmark_name(self) -> str:
        """The benchmark adapter's name."""

    @property
    def pack_dir(self) -> Path | None:
        """The pack's directory, which the harness hash covers with core."""

    @property
    def models(self) -> Sequence[ModelSettings]:
        """The settings of every model the pack calls."""

    @property
    def dataset(self) -> Dataset:
        """Where the benchmark's tasks came from."""

    @property
    def cache_namespace(self) -> str:
        """The grading identity its runs' evaluation cache is kept under; empty for none."""

    def manifest_fields(self) -> RunManifest:
        """The benchmark's part of the run's manifest (Benchmark.manifest_fields)."""

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
    # The pack's directory, for the harness hash; None for a harness without one, such as a test's.
    pack_dir: Path | None = None
    # The settings of every model the pack calls, for the manifest.
    models: tuple[ModelSettings, ...] = ()
    dataset: Dataset = Dataset()
    # The grading identity of its runs, which keys its evaluation cache (EvalCache's namespace),
    # such as f"{mode}-{bench_fingerprint}"; empty if it cannot use a cache.
    cache_namespace: str = ""

    @property
    def benchmark_name(self) -> str:
        """The benchmark adapter's name."""
        return self.benchmark.name

    def manifest_fields(self) -> RunManifest:
        """The benchmark's part of the run's manifest (Benchmark.manifest_fields)."""
        return self.benchmark.manifest_fields()

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
class GradingOptions:
    """What `goaly run --phase grade` gives a benchmark's grading factory."""

    # The run's directory.
    files: RunFiles


class GradesRuns(Protocol):
    """A Grader, whatever its types."""

    async def grade(self, run_dir: Path, *, lock: RunLock | None = None) -> RunSummary:
        """Grade the run's ungraded tasks, as Grader.grade does."""


class RunnableGrading(Protocol):
    """A Grading, whatever its types: what the grade phase needs."""

    @property
    def benchmark_name(self) -> str:
        """The benchmark adapter's name."""

    @property
    def cache_namespace(self) -> str:
        """The grading identity the evaluation cache is kept under; empty for none."""

    def grader(self, *, cache: EvalCache | None) -> GradesRuns:
        """Return the Grader of this benchmark, with this cache."""


@dataclass(frozen=True)
class Grading[InstanceT: Instance, EnvT: Environment]:
    """A benchmark adapter alone, as the grade phase uses it: its grader, and no pack code."""

    benchmark: Benchmark[InstanceT, EnvT]
    # As Harness.cache_namespace: the grading identity that keys the evaluation cache.
    cache_namespace: str = ""

    @property
    def benchmark_name(self) -> str:
        """The benchmark adapter's name."""
        return self.benchmark.name

    def grader(self, *, cache: EvalCache | None) -> Grader[InstanceT, EnvT]:
        """Return the Grader of this benchmark, with this cache."""
        return Grader(self.benchmark, cache=cache)


# Opens a benchmark for one grade phase, and closes what it holds when it is over.
type GradingFactory = Callable[[GradingOptions], AbstractAsyncContextManager[RunnableGrading]]


class Registry:
    """The harnesses `goaly run` can run, by benchmark name, then by harness name."""

    def __init__(self) -> None:
        self._harnesses: dict[str, dict[str, HarnessFactory]] = {}
        self._gradings: dict[str, GradingFactory] = {}

    def add(self, benchmark: str, harness: str, factory: HarnessFactory) -> None:
        """Register ``factory`` as ``benchmark``'s harness ``harness``, a name it has once."""
        harnesses = self._harnesses.setdefault(benchmark, {})
        if harness in harnesses:
            raise ValueError(f"{benchmark!r} already has a harness {harness!r}")
        harnesses[harness] = factory

    def add_benchmark(self, benchmark: str, factory: GradingFactory) -> None:
        """Register ``factory`` as the grading of ``benchmark``, which has one."""
        if benchmark in self._gradings:
            raise ValueError(f"{benchmark!r} already has a grading")
        self._gradings[benchmark] = factory

    def grading(self, benchmark: str) -> GradingFactory:
        """Return the factory of ``benchmark``'s grading, or raise RegistryError."""
        if benchmark not in self._gradings:
            known = ", ".join(sorted(self._gradings))
            there = f"the benchmarks that grade are: {known}" if known else "none is registered"
            raise RegistryError(f"no benchmark {benchmark!r} to grade with; {there}")
        return self._gradings[benchmark]

    def benchmarks(self) -> list[str]:
        """The benchmarks registered with a harness, in name order."""
        return sorted(self._harnesses)

    def harnesses(self, benchmark: str) -> list[str]:
        """The names of ``benchmark``'s harnesses, in order."""
        return sorted(self._registered(benchmark))

    def factory(self, benchmark: str, harness: str | None) -> HarnessFactory:
        """Return the factory of ``benchmark``'s harness ``harness``.

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

    def _registered(self, benchmark: str) -> dict[str, HarnessFactory]:
        if benchmark not in self._harnesses:
            known = ", ".join(self.benchmarks())
            there = f"the benchmarks are: {known}" if known else "none is registered"
            raise RegistryError(f"no benchmark {benchmark!r}; {there}")
        return self._harnesses[benchmark]
