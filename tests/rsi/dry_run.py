"""The lane's dry-run wiring: the whole flow on a temporary repository, with no network.

``build`` is a ``wiring`` for goaly.rsi.cli (``wiring = "tests.rsi.dry_run:build"``). It runs:

- T20's real Implementer, on mini-swe-agent's DeterministicToolcallModel and a local shell
  instead of the sandbox. Each run writes the hypothesis's behavior to ``kind.txt`` in the pack,
  runs ``goaly-smoke`` once, through the lane's real smoke runner, and submits;
- a FakeLLM: GLM behind core's real client on an httpx MockTransport, for the analyst and the
  lessons;
- the toy benchmark of tests.rsi.round_world, which scores a commit by its ``kind.txt``;
- T18's real allowed-paths guard;
- hooks that append what they were given to ``hooks.jsonl`` in the repository.

The analyst proposes good, meh and bad in round 1, and only bad ones afterwards, so the lane
accepts once and then stops after 4 rounds without an accepted change.

Config's ``evaluator`` picks a variant: "hang" makes good's goaly-smoke wait forever and meh's
implementer run on T20's real GlmModel, whose GLM call never returns, so the deadline comes
while both are in flight; "smoke-error" makes every goaly-smoke evaluation fail.
"""

import asyncio
import functools
import json
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import SecretStr

from goaly.bench.effisql.data import EffiSqlInstance
from goaly.bench.effisql.grading_pool import GRADING_INSTANCES, GradingInstance, Priority
from goaly.contracts.gen.experiment_pb2 import LANE_EFFISQL, GuardResult, LaneEventKind
from goaly.harness.core.llm import LlmClient
from goaly.harness.core.settings import Settings
from goaly.rsi.git import GitError, run_git
from goaly.rsi.guards.paths import check_allowed_paths
from goaly.rsi.hypothesis_queue import HypothesisQueue
from goaly.rsi.implementer.config import load_config
from goaly.rsi.implementer.environment import HostCommand
from goaly.rsi.implementer.model import GlmModel
from goaly.rsi.implementer.run import Implementer
from goaly.rsi.lane import LaneHooks, StopInfo
from goaly.rsi.ledger import Ledger
from goaly.rsi.lessons import LessonStore
from goaly.rsi.probe import probe
from goaly.rsi.round import RoundDeps
from goaly.rsi.smoke import EvaluatorSmokeRunner
from goaly.rsi.usage import CallTally
from goaly.rsi.wiring import ImplementerPool
from goaly.rsi.workspace import Workspace, Worktree
from tests.rsi.gitrepo import git, make_repo
from tests.rsi.implementer.fakes import FakeEnvironment, NoLlm, ScriptedModel, shell, submit
from tests.rsi.round_world import TRAIN, VAL, ScriptedGlm, ToyEvaluator, analyst_input, behavior

KIND_FILE = "src/goaly/harness/packs/effisql/kind.txt"
ROUNDS = 12  # More analyses than the lane can use before it stops.


@dataclass
class DryRun:
    """The dry run's parts, for tests that look inside."""

    deps: RoundDeps
    hooks: LaneHooks
    evaluator: ToyEvaluator
    glm: ScriptedGlm
    agents: "Agents"


class Agents:
    """Counts the implementer runs in progress, to check the implementer pool's size."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.most = 0

    def enter(self) -> None:
        with self._lock:
            self.active += 1
            self.most = max(self.most, self.active)

    def leave(self) -> None:
        with self._lock:
            self.active -= 1


def analyst_rounds() -> list[list[str]]:
    first = ["good-1", "meh-1", "bad-1"]
    return [first, *([f"bad-{n}-{x}" for x in "abc"] for n in range(2, ROUNDS + 2))]


@asynccontextmanager
async def build(settings: Any) -> AsyncIterator[tuple[RoundDeps, LaneHooks]]:
    async with dry_run(settings) as parts:
        yield parts.deps, parts.hooks


@asynccontextmanager
async def dry_run(settings: Any) -> AsyncIterator[DryRun]:
    repo = Path(settings.repo).resolve()
    workspace = Workspace(repo, root=settings.worktree_root)
    ledger = Ledger(settings.ledger())
    queue = HypothesisQueue(settings.ledger(), LANE_EFFISQL)
    runs = repo / settings.runs_dir
    tally = CallTally(runs / "rsi" / "effisql" / "llm_calls.jsonl")
    (runs / "rsi" / "effisql").mkdir(parents=True, exist_ok=True)
    glm = ScriptedGlm(analyst_ids=analyst_rounds())
    gateway = Settings(api_key=SecretStr("sk-dry-run-never-sent"), base_url="https://gw.test/v1")
    evaluator = ToyEvaluator(
        kind_of=lambda commit: _kind_of(repo, commit),
        concurrency=settings.lane.task_concurrency,
        smoke_gate=asyncio.Event() if settings.evaluator == "hang" else None,
        smoke_error=settings.evaluator == "smoke-error",
    )
    glm.hang_tools = settings.evaluator == "hang"
    v0 = run_git(["rev-parse", f"{settings.v0}^{{commit}}"], cwd=repo).strip()

    def guards(experiment_id: str, parent: str, candidate: str) -> list[GuardResult]:
        del experiment_id
        return [check_allowed_paths(repo, LANE_EFFISQL, parent=parent, candidate=candidate)]

    def model(experiment_id: str, stop: threading.Event) -> Any:
        kind = behavior(
            ledger.latest_experiments(LANE_EFFISQL)[experiment_id].hypothesis.hypothesis_id
        )
        if settings.evaluator == "hang" and kind == "meh":
            config = load_config()
            return GlmModel(
                llm_holder[0],
                experiment_id=experiment_id,
                limits=config.limits,
                templates=config.model,
                stop=stop,
            )
        return ScriptedModel(
            f"printf '%s\\n' {kind} > {KIND_FILE}",
            f"goaly-smoke {TRAIN[0]}",
            submit(f"Wrote {kind} to kind.txt"),
        )

    agents = Agents()

    @contextmanager
    def environment(worktree: Worktree, host: HostCommand | None) -> Iterator[FakeEnvironment]:
        local = shell(worktree.path)

        def handle(command: str) -> dict[str, Any]:
            time.sleep(0.05)  # Long enough for the runs to overlap.
            return local(command)

        sandbox = FakeEnvironment(handle)
        sandbox.smoke = host  # goaly-smoke's host side, as the sandbox hands it on.
        agents.enter()
        try:
            yield sandbox
        finally:
            agents.leave()

    loop = asyncio.get_running_loop()
    llm_holder: list[LlmClient] = []
    async with (
        LlmClient(gateway, tally, transport=httpx.MockTransport(glm.handle)) as llm,
        ImplementerPool(
            settings.lane.implementer_threads, timeout_s=settings.lane.shutdown_timeout_s
        ) as pool,
    ):
        llm_holder.append(llm)
        smoke = EvaluatorSmokeRunner(evaluator, guards, repo, loop, train_ids=TRAIN)
        pool.on_stop(smoke.cancel_all)
        implementer = Implementer(
            NoLlm(),
            workspace,
            image="dry-run",
            runs_dir=runs,
            pool=pool.executor,
            make_environment=environment,
            make_model=model,
            smoke=smoke,
            train_ids=TRAIN,
        )
        deps = RoundDeps(
            lane=LANE_EFFISQL,
            ledger=ledger,
            lessons=LessonStore(ledger),
            queue=queue,
            workspace=workspace,
            llm=llm,
            implementer=implementer,
            guards=guards,
            evaluator=evaluator,
            analyst_input=analyst_input,
            train_ids=TRAIN,
            val_ids=VAL,
            v0_commit=v0,
            usage=tally.usage,
        )
        hooks = replace(
            stop_hooks(settings),
            identity_probe=functools.partial(probe, FakeGrader(), PROBE_TASK, GRADING_INSTANCES),
        )
        yield DryRun(deps, hooks, evaluator, glm, agents)


PROBE_TASK = EffiSqlInstance.model_validate(
    {"id": TRAIN[5], "db": "tpch", "base_sql": "SELECT 1", "base_time": 10_000}
)


class FakeGrader:
    """T10's identity probe, on instances that time as at setup, except grade-3, 20% off."""

    async def identity_probe(
        self, inst: EffiSqlInstance, *, instance: GradingInstance, priority: Priority = Priority.RSI
    ) -> float:
        return 1.2 if instance.name == "grade-3" else 1.0

    def identity_reference(self, inst: EffiSqlInstance) -> float | None:
        return 1.0


def write_config(tmp_path: Path, deadline: datetime, variant: str = "") -> tuple[Path, Path]:
    """A repository tagged baseline-v0 and best/effisql, and a dry-run config for it."""
    repo = make_repo(tmp_path / "repo")
    git(repo, "tag", "baseline-v0")
    config = tmp_path / "lane.toml"
    config.write_text(
        f"""
repo = "{repo}"
ledger_dir = "{tmp_path / "ledger"}"
worktree_root = "{tmp_path / "worktrees"}"
wiring = "tests.rsi.dry_run:build"
hooks = "tests.rsi.dry_run:stop_hooks"
evaluator = "{variant}"
log_level = "WARNING"

[lane]
deadline = "{deadline.isoformat()}"
infrastructure_backoff_s = 0
shutdown_timeout_s = 2
""",
        encoding="utf-8",
    )
    return repo, config


def stop_hooks(settings: Any) -> LaneHooks:
    """The dry run's stop hooks, also its config's ``hooks`` for a lane past its deadline."""
    repo = Path(settings.repo).resolve()
    ledger = Ledger(settings.ledger())
    return LaneHooks(
        results_pr=_hook(repo, "results_pr", ledger), notify=_hook(repo, "notify", ledger)
    )


def _kind_of(repo: Path, commit: str) -> str:
    try:
        return run_git(["show", f"{commit}:{KIND_FILE}"], cwd=repo).strip() or "base"
    except GitError:
        return "base"


def _hook(repo: Path, name: str, ledger: Ledger) -> Callable[[StopInfo], Awaitable[None]]:
    async def hook(info: StopInfo) -> None:
        events = ledger.lane_events(LANE_EFFISQL)
        line = {
            "hook": name,
            "rule": info.rule,
            "best": info.best,
            "aborted": list(info.aborted),
            "last_event": LaneEventKind.Name(events[-1].kind) if events else None,
        }
        with (repo / "hooks.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(line) + "\n")

    return hook
