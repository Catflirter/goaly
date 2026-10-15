"""The real Effi-SQL lane's wiring: the ``wiring`` goaly.rsi.cli uses unless config names another.

:func:`build` assembles the lane from its parts:

- core's LLM client, with the key from ``~/.config/goaly/.env``, recording to the lane's call log
  through a CallTally (goaly.rsi.usage);
- the ledger, the lessons and the hypothesis queue in the ledger directory, and the worktrees;
- T18's guards, against the frozen layers' checksums at v0, with the gold windows and the sandbox
  image; leftover sandbox containers are removed when the lane starts;
- T20's implementer on a pool of ``implementer_threads`` threads, with goaly-smoke on train tasks
  through the lane's evaluator at SMOKE priority (goaly.rsi.smoke);
- T19's analyst input, from the splits less the exclusions and the full dataset;
- the evaluator, from config's ``evaluator``: ``module:function``, called with plain values,
  ``repo``, ``runs_dir`` and ``task_concurrency``, and the ``evaluation_type`` and
  ``cost_stats_type`` to answer with, since bench cannot import rsi; it returns an Evaluator.
  T11's two-phase pipeline provides it, driving a host run of ``goaly run --phase all`` for
  the tasks, seeds and split each request names (V27), with at most ``task_concurrency``
  tasks at once (spec §6.2);
- the stop hooks, from config's ``hooks``: ``module:function``, called with the settings and
  returning LaneHooks. T22 provides them; without them the lane stops with no results PR and
  no notification, which is logged.
"""

import asyncio
import concurrent.futures
import functools
import logging
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import cast

from goaly.bench.effisql.data import default_cache_dir, fetch_dataset, load_instances
from goaly.bench.effisql.splits import load_exclusions, load_splits
from goaly.contracts.gen.experiment_pb2 import LANE_EFFISQL, GuardResult, Hypothesis
from goaly.contracts.gen.result_pb2 import TaskResult
from goaly.contracts.gen.task_pb2 import SPLIT_TRAIN, SPLIT_VAL
from goaly.contracts.gen.trajectory_pb2 import Trajectory
from goaly.harness.core.llm import LlmClient
from goaly.harness.core.settings import SettingsError, load_settings
from goaly.rsi.analyst import AnalystInput
from goaly.rsi.analyst_input import build_analyst_input
from goaly.rsi.cli import ConfigError, LaneSettings, import_object
from goaly.rsi.decision import CostStats
from goaly.rsi.git import run_git
from goaly.rsi.guards import GuardsConfig, run_guards
from goaly.rsi.guards.checksums import frozen_checksums
from goaly.rsi.guards.gold_windows import DEFAULT_STORE, GoldWindows
from goaly.rsi.hypothesis_queue import HypothesisQueue
from goaly.rsi.implementer.run import Implementer
from goaly.rsi.lane import LaneHooks
from goaly.rsi.ledger import Ledger
from goaly.rsi.lessons import LessonStore
from goaly.rsi.round import Evaluation, Evaluator, RoundDeps
from goaly.rsi.sandbox import ensure_image, remove_containers
from goaly.rsi.smoke import EvaluatorSmokeRunner
from goaly.rsi.usage import CallTally
from goaly.rsi.workspace import Workspace

__all__ = ["build"]

_logger = logging.getLogger(__name__)


@asynccontextmanager
async def build(settings: LaneSettings) -> AsyncIterator[tuple[RoundDeps, LaneHooks]]:
    """Build the real lane; raise goaly.rsi.cli.ConfigError for an unusable configuration."""
    if not settings.evaluator:
        raise ConfigError("evaluator must name T11's evaluator factory, as 'module:function'")
    make_evaluator = cast(Callable[..., Evaluator], import_object(settings.evaluator, "evaluator"))
    hooks = LaneHooks()
    if settings.hooks:
        hooks = cast(Callable[[LaneSettings], LaneHooks], import_object(settings.hooks, "hooks"))(
            settings
        )
    else:
        _logger.warning("no stop hooks: the lane will open no results PR and send no notification")
    try:
        gateway = load_settings()
    except SettingsError as error:
        raise ConfigError(str(error)) from error

    repo = settings.repo.resolve()
    lane = LANE_EFFISQL
    workspace = Workspace(repo, root=settings.worktree_root)
    ledger = Ledger(settings.ledger())
    runs = repo / settings.runs_dir
    tally = CallTally(runs / "rsi" / "effisql" / "llm_calls.jsonl")
    tally.path.parent.mkdir(parents=True, exist_ok=True)
    splits = load_splits().without(load_exclusions())
    instances = load_instances(fetch_dataset(default_cache_dir()))
    v0 = run_git(["rev-parse", f"{settings.v0}^{{commit}}"], cwd=repo).strip()
    image = await asyncio.to_thread(ensure_image, repo)
    gold = GoldWindows.load(DEFAULT_STORE.expanduser())
    with workspace.snapshot(v0) as snapshot:
        frozen = frozen_checksums(snapshot)
    guard_config = GuardsConfig()

    def guards(experiment_id: str, parent: str, candidate: str) -> list[GuardResult]:
        return run_guards(
            workspace,
            lane,
            experiment_id=experiment_id,
            parent=parent,
            candidate=candidate,
            gold=gold,
            frozen=frozen,
            image=image,
            config=guard_config,
        )

    def analyst_input(
        results: Sequence[TaskResult],
        trajectories: Sequence[Trajectory],
        queued: Sequence[Hypothesis],
    ) -> AnalystInput:
        return build_analyst_input(
            splits=splits,
            ledger=ledger,
            best_results=results,
            trajectories=trajectories,
            instances=instances,
            queued=queued,
        )

    train_ids = splits.ids(SPLIT_TRAIN)
    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=settings.lane.implementer_threads, thread_name_prefix="implementer"
    ) as pool:
        async with LlmClient(gateway, tally) as llm:
            evaluator = make_evaluator(
                repo=repo,
                runs_dir=runs,
                task_concurrency=settings.lane.task_concurrency,
                evaluation_type=Evaluation,
                cost_stats_type=CostStats,
            )
            implementer = Implementer(
                llm,
                workspace,
                image=image,
                runs_dir=runs,
                pool=pool,
                sandbox=guard_config.sandbox,
                smoke=EvaluatorSmokeRunner(evaluator, guards, repo, loop),
                train_ids=train_ids,
            )
            deps = RoundDeps(
                lane=lane,
                ledger=ledger,
                lessons=LessonStore(ledger),
                queue=HypothesisQueue(settings.ledger(), lane),
                workspace=workspace,
                llm=llm,
                implementer=implementer,
                guards=guards,
                evaluator=evaluator,
                analyst_input=analyst_input,
                train_ids=train_ids,
                val_ids=splits.ids(SPLIT_VAL),
                v0_commit=v0,
                usage=tally.usage,
            )
            cleanup = functools.partial(remove_containers, lane=lane, config=guard_config.sandbox)
            yield deps, replace(hooks, remove_containers=cleanup)
