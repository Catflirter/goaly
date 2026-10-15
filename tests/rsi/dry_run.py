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
"""

import asyncio
import concurrent.futures
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import SecretStr

from goaly.contracts.gen.experiment_pb2 import LANE_EFFISQL, GuardResult, LaneEventKind
from goaly.harness.core.llm import LlmClient
from goaly.harness.core.settings import Settings
from goaly.rsi.git import GitError, run_git
from goaly.rsi.guards.paths import check_allowed_paths
from goaly.rsi.hypothesis_queue import HypothesisQueue
from goaly.rsi.implementer.environment import HostCommand
from goaly.rsi.implementer.run import Implementer
from goaly.rsi.implementer.smoke import mentions_smoke
from goaly.rsi.lane import LaneHooks, StopInfo
from goaly.rsi.ledger import Ledger
from goaly.rsi.lessons import LessonStore
from goaly.rsi.round import RoundDeps
from goaly.rsi.smoke import EvaluatorSmokeRunner
from goaly.rsi.usage import CallTally
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
    )
    v0 = run_git(["rev-parse", f"{settings.v0}^{{commit}}"], cwd=repo).strip()

    def guards(experiment_id: str, parent: str, candidate: str) -> list[GuardResult]:
        del experiment_id
        return [check_allowed_paths(repo, LANE_EFFISQL, parent=parent, candidate=candidate)]

    def model(experiment_id: str, stop: object) -> ScriptedModel:
        del stop
        kind = behavior(
            ledger.latest_experiments(LANE_EFFISQL)[experiment_id].hypothesis.hypothesis_id
        )
        return ScriptedModel(
            f"printf '%s\\n' {kind} > {KIND_FILE}",
            f"goaly-smoke {TRAIN[0]}",
            submit(f"Wrote {kind} to kind.txt"),
        )

    def environment(worktree: Worktree, host: HostCommand | None) -> FakeEnvironment:
        local = shell(worktree.path)

        def handle(command: str) -> dict[str, Any]:
            if host is not None and mentions_smoke(command):
                return host(command)
            return local(command)

        return FakeEnvironment(handle)

    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=settings.lane.implementer_threads, thread_name_prefix="implementer"
    ) as pool:
        async with LlmClient(gateway, tally, transport=httpx.MockTransport(glm.handle)) as llm:
            implementer = Implementer(
                NoLlm(),
                workspace,
                image="dry-run",
                runs_dir=runs,
                pool=pool,
                make_environment=environment,
                make_model=model,
                smoke=EvaluatorSmokeRunner(evaluator, guards, repo, loop),
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
            hooks = LaneHooks(
                results_pr=_hook(repo, "results_pr", ledger), notify=_hook(repo, "notify", ledger)
            )
            yield DryRun(deps, hooks, evaluator, glm)


def write_config(tmp_path: Path, deadline: datetime) -> tuple[Path, Path]:
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
log_level = "WARNING"

[lane]
deadline = "{deadline.isoformat()}"
infrastructure_backoff_s = 0
""",
        encoding="utf-8",
    )
    return repo, config


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
