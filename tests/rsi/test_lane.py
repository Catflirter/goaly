"""Tests for goaly.rsi.lane: rounds back to back, resuming, stopping and the deadline.

The lane runs in-process on the dry run's wiring (tests.rsi.dry_run).
"""

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from goaly.contracts.gen.experiment_pb2 import (
    DECISION_ABORTED,
    DECISION_ACCEPTED,
    DECISION_PENDING,
    IMPLEMENTER_EXIT_INFRASTRUCTURE,
    LANE_EFFISQL,
    LANE_EVENT_KIND_STARTED,
    LANE_EVENT_KIND_STOPPED,
)
from goaly.rsi import lane as lane_module
from goaly.rsi.cli import LaneSettings, load_settings
from goaly.rsi.hypothesis_queue import HypothesisQueue
from goaly.rsi.lane import (
    DEADLINE,
    NO_ACCEPTED_CHANGE,
    LaneConfig,
    LaneRunner,
    RoundEntry,
    RoundLog,
    StopInfo,
)
from goaly.rsi.ledger import Ledger
from goaly.rsi.round import RoundOutcome, RoundRunner
from tests.rsi.dry_run import DryRun, dry_run, write_config
from tests.rsi.gitrepo import git

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 10, 15, 16, 0, tzinfo=UTC)


def settings_for(tmp_path: Path, seconds: float) -> tuple[Path, LaneSettings]:
    repo, config = write_config(tmp_path, datetime.now(UTC) + timedelta(seconds=seconds))
    return repo, load_settings(config)


def lane(parts: DryRun, settings: LaneSettings, **config: object) -> LaneRunner:
    lane_config = settings.lane.model_copy(update=config) if config else settings.lane
    return LaneRunner(parts.deps, lane_config, hooks=parts.hooks)


def hook_calls(repo: Path) -> list[dict[str, object]]:
    path = repo / "hooks.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


async def test_a_deadline_mid_round_aborts_the_round_keeps_the_best_and_stops(
    tmp_path: Path,
) -> None:
    repo, settings = settings_for(tmp_path, 4)
    ledger = Ledger(settings.ledger())
    async with dry_run(settings) as parts:
        parts.evaluator.gate = asyncio.Event()  # Validations wait: the deadline comes first.
        assert await lane(parts, settings).run() == 0
    latest = ledger.latest_experiments(LANE_EFFISQL)
    decisions = {r.hypothesis.hypothesis_id: r.decision for r in latest.values()}
    assert decisions == {"good-1": DECISION_ABORTED, "meh-1": DECISION_ABORTED, "bad-1": 3}
    for name in ("good-1", "meh-1"):
        record = next(r for r in latest.values() if r.hypothesis.hypothesis_id == name)
        assert "deadline" in record.decision_reason
        assert record.HasField("screening")
    assert ledger.pending_experiments(LANE_EFFISQL) == []
    v0 = git(repo, "rev-parse", "baseline-v0").strip()
    assert git(repo, "rev-parse", "best/effisql").strip() == v0  # The best is kept.
    events = ledger.lane_events(LANE_EFFISQL)
    assert [e.kind for e in events] == [LANE_EVENT_KIND_STARTED, LANE_EVENT_KIND_STOPPED]
    assert events[-1].detail == "deadline: the deadline came during round 1"
    calls = hook_calls(repo)
    assert [c["hook"] for c in calls] == ["results_pr", "notify"]
    assert calls[0]["rule"] == DEADLINE
    assert len(calls[0]["aborted"]) == 2  # type: ignore[arg-type]  # A list in the JSON.
    assert calls[0]["last_event"] == "LANE_EVENT_KIND_STOPPED"

    # A restart after the deadline stops at once: no round, no start, the stop steps again.
    async with dry_run(settings) as parts:
        assert await lane(parts, settings).run() == 0
        assert parts.evaluator.requests == []
    assert ledger.latest_experiments(LANE_EFFISQL).keys() == latest.keys()
    events = ledger.lane_events(LANE_EFFISQL)
    assert [e.kind for e in events][-1] == LANE_EVENT_KIND_STOPPED
    assert LANE_EVENT_KIND_STARTED not in [e.kind for e in events[2:]]
    assert events[-1].detail.startswith("deadline: the deadline had passed")
    assert [c["hook"] for c in hook_calls(repo)] == ["results_pr", "notify"] * 2


async def test_a_crashed_lane_resumes_its_round_without_duplicating_an_experiment(
    tmp_path: Path,
) -> None:
    _, settings = settings_for(tmp_path, 3600)
    ledger = Ledger(settings.ledger())
    async with dry_run(settings) as parts:
        parts.evaluator.gate = asyncio.Event()
        task = asyncio.create_task(lane(parts, settings).run())
        for _ in range(1000):
            await asyncio.sleep(0.02)
            records = ledger.latest_experiments(LANE_EFFISQL).values()
            screened = [r for r in records if r.decision == DECISION_PENDING]
            # Wait until the two that pass the screen wait in validation and bad-1 is rejected.
            ready = len(records) == 3 and len(screened) == 2
            if ready and all(r.HasField("screening") for r in screened):
                break
        task.cancel()  # The process dies mid-round.
        with pytest.raises(asyncio.CancelledError):
            await task
    before = ledger.latest_experiments(LANE_EFFISQL)
    assert sum(r.decision == DECISION_PENDING for r in before.values()) == 2
    async with dry_run(settings) as parts:
        assert await lane(parts, settings).run() == 0
        built = [r for r in parts.evaluator.requests if r.phase == "screen"]
    after = ledger.latest_experiments(LANE_EFFISQL)
    round_1 = {name for name, r in after.items() if r.round == 1}
    assert round_1 == set(before)  # Round 1 resumed with its own experiments, and no others.
    assert not [r for r in built if r.run_id.split("-screen")[0] in before]  # Not screened again.
    accepted = [r for r in after.values() if r.decision == DECISION_ACCEPTED]
    assert [r.experiment_id for r in accepted] == [
        next(n for n, r in before.items() if r.hypothesis.hypothesis_id == "good-1")
    ]
    kinds = [e.kind for e in ledger.lane_events(LANE_EFFISQL)]
    assert kinds == [LANE_EVENT_KIND_STARTED, LANE_EVENT_KIND_STARTED, LANE_EVENT_KIND_STOPPED]


async def test_rounds_whose_analysis_stays_invalid_count_toward_the_stop(tmp_path: Path) -> None:
    repo, settings = settings_for(tmp_path, 3600)
    async with dry_run(settings) as parts:
        parts.glm.invalid = True
        assert await lane(parts, settings, max_rounds_without_accept=2).run() == 0
    entries = RoundLog(settings.ledger(), LANE_EFFISQL).entries()
    finished = [e for e in entries if e.status == "finished"]
    assert [e.round for e in finished] == [1, 2]
    assert all(e.skipped.startswith("3 invalid replies") for e in finished)
    (call, _) = hook_calls(repo)
    assert call["rule"] == NO_ACCEPTED_CHANGE


async def test_infrastructure_rounds_do_not_count_and_back_off_until_the_deadline(
    tmp_path: Path,
) -> None:
    repo, settings = settings_for(tmp_path, 1.5)
    slept: list[float] = []

    async def sleep(seconds: float) -> None:
        slept.append(seconds)
        await asyncio.sleep(min(seconds, 0.05))

    async with dry_run(settings) as parts:
        parts.glm.fail = True  # Every analyst call fails: the round aborts as infrastructure.
        runner = LaneRunner(
            parts.deps, settings.lane.model_copy(update={"infrastructure_backoff_s": 0.2}),
            hooks=parts.hooks, sleep=sleep,
        )  # fmt: skip
        assert await runner.run() == 0
    assert slept
    assert all(0 <= s <= 0.2 for s in slept)
    entries = [
        e for e in RoundLog(settings.ledger(), LANE_EFFISQL).entries() if e.status == "finished"
    ]
    assert entries
    assert all(e.infrastructure for e in entries)
    assert hook_calls(repo)[0]["rule"] == DEADLINE


def test_the_round_log_counts_only_rounds_that_learned_something(tmp_path: Path) -> None:
    log = RoundLog(tmp_path, LANE_EFFISQL)
    assert log.state([]).next_round == 1

    def entry(number: int, status: str = "finished", **fields: object) -> RoundEntry:
        return RoundEntry(round=number, status=status, at=NOW, **fields)  # type: ignore[arg-type]

    log.append(entry(1, "started"))
    log.append(entry(1, accepted="e1", decided=3))
    log.append(entry(2, "started"))
    log.append(entry(2, decided=2))
    log.append(entry(3, "started"))
    log.append(entry(3, skipped="3 invalid replies"))
    log.append(entry(4, "started"))
    log.append(entry(4, skipped="the analyst's call failed", infrastructure=True))
    log.append(entry(5, "started"))
    log.append(entry(5, aborted=3))  # Every experiment aborted: nothing learned.
    log.append(entry(6, "started"))
    with log.path.open("a") as file:
        file.write('{"round": 7, "sta')  # A torn line, as a crash leaves.
    state = log.state([6, 7])
    assert (state.next_round, state.unfinished, state.without_accept) == (8, 6, 2)
    log.append(entry(6, decided=1))
    assert log.entries()[-1].round == 6
    assert log.state([]).unfinished is None


def test_the_lane_config_has_the_spec_defaults() -> None:
    config = LaneConfig(deadline=NOW)
    assert config.max_rounds_without_accept == 4
    assert config.task_concurrency == 16
    assert config.implementer_threads == 3
    assert config.round.k == 3
    with pytest.raises(ValueError, match="timezone"):
        LaneConfig(deadline=datetime(2026, 10, 15, 17, 0))  # noqa: DTZ001  (naive, refused)


async def test_a_failing_hook_never_stops_the_stop(tmp_path: Path) -> None:
    _, settings = settings_for(tmp_path, -1)
    seen: list[str] = []

    async def broken(info: StopInfo) -> None:
        raise RuntimeError("GitHub is down")

    async def notify(info: StopInfo) -> None:
        seen.append(info.rule)

    async with dry_run(settings) as parts:
        hooks = replace(parts.hooks, results_pr=broken, notify=notify)
        assert await LaneRunner(parts.deps, settings.lane, hooks=hooks).run() == 0
    assert seen == [DEADLINE]


async def test_the_lane_runs_inside_its_running_hook_and_reports_each_round(
    tmp_path: Path,
) -> None:
    _, settings = settings_for(tmp_path, 3600)
    seen: list[str] = []

    @contextlib.asynccontextmanager
    async def running() -> AsyncIterator[None]:
        seen.append("start")
        yield
        seen.append("end")

    async def after_round(outcome: RoundOutcome) -> None:
        seen.append(f"round {outcome.round}")
        if outcome.round == 1:
            raise RuntimeError("the results PR failed")  # Logged, never raised.

    async with dry_run(settings) as parts:
        parts.glm.invalid = True
        hooks = replace(parts.hooks, running=running, after_round=after_round)
        config = settings.lane.model_copy(update={"max_rounds_without_accept": 2})
        assert await LaneRunner(parts.deps, config, hooks=hooks).run() == 0
    assert seen == ["start", "round 1", "round 2", "end"]


async def test_the_lane_wires_its_task_and_implementer_limits(tmp_path: Path) -> None:
    _, settings = settings_for(tmp_path, 3600)
    limits = {"task_concurrency": 4, "implementer_threads": 2, "max_rounds_without_accept": 1}
    settings = settings.model_copy(update={"lane": settings.lane.model_copy(update=limits)})
    async with dry_run(settings) as parts:
        assert await lane(parts, settings).run() == 0
        # Three hypotheses on two implementer threads: two at once, never three.
        assert parts.agents.most == 2
        # 8 val tasks x 2 seeds per reference, on at most 4 at once: the cap is reached.
        assert parts.evaluator.most_in_flight == 4


async def test_smoke_infrastructure_failures_abort_and_requeue(tmp_path: Path) -> None:
    _, config = write_config(tmp_path, datetime.now(UTC) + timedelta(hours=1), "smoke-error")
    settings = load_settings(config)
    async with dry_run(settings) as parts:
        outcome = await RoundRunner(parts.deps, settings.lane.round).run(1)
    assert (outcome.decided, outcome.aborted) == (0, 3)  # Nothing counts toward the stop.
    latest = Ledger(settings.ledger()).latest_experiments(LANE_EFFISQL).values()
    assert {r.decision for r in latest} == {DECISION_ABORTED}
    assert {r.implementer.exit for r in latest} == {IMPLEMENTER_EXIT_INFRASTRUCTURE}
    queued = HypothesisQueue(settings.ledger(), LANE_EFFISQL).pending()
    assert sorted(h.hypothesis_id for h in queued) == ["bad-1", "good-1", "meh-1"]


async def test_a_hung_hook_times_out_and_a_failing_settle_still_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, settings = settings_for(tmp_path, -1)
    seen: list[str] = []

    async def hung(info: StopInfo) -> None:
        await asyncio.sleep(3600)  # GitHub stalls.

    async def notify(info: StopInfo) -> None:
        seen.append(info.rule)

    def broken(*args: object, **kwargs: object) -> object:
        raise RuntimeError("git is gone")

    monkeypatch.setattr(lane_module, "reconcile_best", broken)
    async with dry_run(settings) as parts:
        hooks = replace(parts.hooks, results_pr=hung, notify=notify)
        config = settings.lane.model_copy(update={"hook_timeout_s": 0.2})
        runner = LaneRunner(parts.deps, config, hooks=hooks)
        runner._resume = _no_resume  # type: ignore[method-assign]  # Skip the start's reconcile.
        assert await runner.run() == 0
    assert seen == [DEADLINE]
    events = Ledger(settings.ledger()).lane_events(LANE_EFFISQL)
    assert events[-1].kind == LANE_EVENT_KIND_STOPPED


async def _no_resume() -> None:
    return None
