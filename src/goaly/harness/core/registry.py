"""The registry of what `goaly run` can run: benchmarks, each with the harnesses that solve it.

A harness is a benchmark adapter together with a pack's policy and the context its policy runs
with. The composition root (``goaly.cli``) registers each harness under its benchmark's name and
its own: T11 registers Effi-SQL's v0 pack, and tests register the toy benchmark. Adding a
benchmark registers it here; core does not change (spec §3.3).

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
    "Registry",
    "RegistryError",
    "RunnableHarness",
    "RunsTasks",
]


class RegistryError(LookupError):
    """No benchmark or harness is registered under a name; the message lists those there are."""


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
    # The pack's directory, for the harness hash; None for a harness without one, such as a test's.
    pack_dir: Path | None = None
    # The settings of every model the pack calls, for the manifest.
    models: tuple[ModelSettings, ...] = ()
    # The SHA-256, in hex, of the gold-answer files its grader uses that are not part of the
    # dataset, for the manifest; empty if there are none. The dataset's own fields come from the
    # benchmark's manifest_fields() alone.
    gold_sha256: str = ""
    # The grading identity of its runs, which keys its evaluation cache (EvalCache's namespace),
    # such as f"{mode}-{bench_fingerprint}"; empty if it cannot use a cache.
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


class Registry:
    """The harnesses `goaly run` can run, by benchmark name, then by harness name."""

    def __init__(self) -> None:
        self._harnesses: dict[str, dict[str, HarnessFactory]] = {}

    def add(self, benchmark: str, harness: str, factory: HarnessFactory) -> None:
        """Register ``factory`` as ``benchmark``'s harness ``harness``, a name it has once."""
        harnesses = self._harnesses.setdefault(benchmark, {})
        if harness in harnesses:
            raise ValueError(f"{benchmark!r} already has a harness {harness!r}")
        harnesses[harness] = factory

    def benchmarks(self) -> list[str]:
        """The benchmarks registered, in name order."""
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
