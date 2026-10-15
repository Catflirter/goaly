"""Tests for goaly.rsi.hooks: the lane's results-PR, notification and heartbeat hooks."""

import logging
from datetime import UTC, datetime
from pathlib import Path

import pytest

from goaly.contracts.gen.experiment_pb2 import LANE_EFFISQL, Lane
from goaly.rsi.cli import LaneSettings
from goaly.rsi.hooks import BreakerWatch, lane_hooks, state_dir
from goaly.rsi.hooks import build_hooks as build
from goaly.rsi.lane import StopInfo
from goaly.rsi.ledger import Ledger
from goaly.rsi.notify import FakeChannel, NotificationKind, QueuedNotifier
from goaly.rsi.results_pr import PublishedPr, ResultsPublisherImpl
from goaly.rsi.round import RoundOutcome

pytestmark = pytest.mark.anyio


class Publisher:
    def __init__(self, due: bool) -> None:
        self.is_due = due
        self.calls: list[tuple[str, bool]] = []

    def due(self, lane: Lane.ValueType, now: datetime) -> bool:
        self.calls.append(("due", False))
        return self.is_due

    def publish(self, lane: Lane.ValueType, *, final: bool, now: datetime) -> PublishedPr | None:
        self.calls.append(("publish", final))
        return None


def lane_settings(tmp_path: Path) -> LaneSettings:
    return LaneSettings.model_validate(
        {
            "repo": str(tmp_path),
            "ledger_dir": str(tmp_path / "ledger"),
            "lane": {"deadline": "2026-10-15T20:00:00Z"},
        }
    )


def stop() -> StopInfo:
    return StopInfo(
        lane=LANE_EFFISQL,
        rule="deadline",
        detail="the deadline passed",
        best="abc",
        stopped_at=datetime.now(UTC),
        aborted=("e1", "e2"),
    )


def outcome() -> RoundOutcome:
    return RoundOutcome.__new__(RoundOutcome)  # The hook ignores its fields.


@pytest.mark.parametrize("due", [True, False])
async def test_after_round_publishes_only_when_due(tmp_path: Path, due: bool) -> None:
    publisher = Publisher(due)
    notifier = QueuedNotifier(FakeChannel(), Ledger(tmp_path / "ledger"))
    hooks = build(lane_settings(tmp_path), publisher, notifier)
    await hooks.after_round(outcome())
    expected = [("due", False), ("publish", False)] if due else [("due", False)]
    assert publisher.calls == expected
    notifier.close()


async def test_stop_opens_the_final_pr_then_notifies_and_drains(tmp_path: Path) -> None:
    publisher = Publisher(False)
    channel = FakeChannel()
    notifier = QueuedNotifier(channel, Ledger(tmp_path / "ledger"))
    hooks = build(lane_settings(tmp_path), publisher, notifier)
    await hooks.results_pr(stop())
    await hooks.notify(stop())
    assert publisher.calls == [("publish", True)]
    (sent,) = channel.delivered  # Delivered before notify() returned.
    assert sent.kind is NotificationKind.STOPPED
    assert sent.text == "Effi-SQL stopped: deadline, 2 experiment(s) aborted."


async def test_running_beats_the_heartbeat_and_resends_failures(tmp_path: Path) -> None:
    settings = lane_settings(tmp_path)
    ledger = Ledger(tmp_path / "ledger")
    first = QueuedNotifier(FakeChannel(failures=99), ledger, close_retry_delays_s=(0.01,))
    first.notify(LANE_EFFISQL, NotificationKind.RESULTS_PR, "Results PR #1 is open.")
    first.close(timeout_s=0.05)
    channel = FakeChannel()
    notifier = QueuedNotifier(channel, ledger)
    hooks = build(settings, Publisher(False), notifier)
    async with hooks.running():
        assert (state_dir(settings) / "effisql.heartbeat").is_file()
    notifier.close()
    assert [sent.text for sent in channel.delivered] == ["Results PR #1 is open."]


def test_lane_hooks_builds_the_real_parts_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GOALY_API_KEY", "k")
    hooks = lane_hooks(lane_settings(tmp_path))
    publisher = hooks.after_round.__closure__  # The wiring inside.
    assert publisher is not None
    cells = [cell.cell_contents for cell in publisher]
    assert any(isinstance(cell, ResultsPublisherImpl) for cell in cells)


def test_the_lane_config_names_these_hooks() -> None:
    from goaly.rsi.cli import import_object
    from goaly.rsi.cli import load_settings as load_lane_settings

    root = Path(__file__).resolve().parents[2]
    settings = load_lane_settings(root / "rsi" / "lane-effisql.toml")
    assert import_object(settings.hooks, "hooks") is lane_hooks


def test_breaker_watch_notifies_on_the_third_opening_within_the_window(tmp_path: Path) -> None:
    clock = [0.0]
    channel = FakeChannel()
    notifier = QueuedNotifier(channel, Ledger(tmp_path / "ledger"), monotonic=lambda: clock[0])
    watch = BreakerWatch(notifier, LANE_EFFISQL, clock=lambda: clock[0])
    throttle = logging.getLogger("goaly.harness.core.llm_throttle")
    throttle.addHandler(watch)
    try:
        for minute in (0, 10, 45, 50):  # Only two within 30 minutes so far.
            clock[0] = minute * 60.0
            throttle.warning("opened", extra={"event": "llm_breaker_open"})
        throttle.warning("other", extra={"event": "llm_breaker_closed"})
        clock[0] = 55 * 60.0
        throttle.warning("opened", extra={"event": "llm_breaker_open"})
    finally:
        throttle.removeHandler(watch)
    notifier.close()
    (sent,) = channel.delivered
    assert sent.kind is NotificationKind.BREAKER_TRIPPED
    assert sent.text == "Effi-SQL: the gateway circuit breaker opened 3 times within 30 minutes."


async def test_running_removes_the_breaker_watch(tmp_path: Path) -> None:
    notifier = QueuedNotifier(FakeChannel(), Ledger(tmp_path / "ledger"))
    hooks = build(lane_settings(tmp_path), Publisher(False), notifier)
    throttle = logging.getLogger("goaly.harness.core.llm_throttle")
    async with hooks.running():
        assert any(isinstance(h, BreakerWatch) for h in throttle.handlers)
    assert not any(isinstance(h, BreakerWatch) for h in throttle.handlers)
    notifier.close()


def test_lane_hooks_without_settings_turn_notifications_off(tmp_path: Path) -> None:
    hooks = lane_hooks(lane_settings(tmp_path))  # No GOALY_API_KEY in the test environment.
    assert hooks.notify is not None
