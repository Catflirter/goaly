"""Tests for goaly.rsi.supervisor, with fake lane processes that follow a plan."""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from google.protobuf.timestamp_pb2 import Timestamp

from goaly.contracts.gen.experiment_pb2 import (
    LANE_EFFISQL,
    LANE_EVENT_KIND_CRASHED,
    LANE_EVENT_KIND_NOTIFIED,
    LANE_EVENT_KIND_RECOVERY_FAILED,
    LANE_EVENT_KIND_RESTARTED,
    LANE_EVENT_KIND_STARTED,
    LANE_EVENT_KIND_STOPPED,
    LaneEvent,
)
from goaly.rsi.cli import LaneSettings
from goaly.rsi.cli import main as cli_main
from goaly.rsi.ledger import Ledger
from goaly.rsi.notify import FakeChannel, NotificationKind, QueuedNotifier
from goaly.rsi.supervisor import (
    EXIT_LOCKED,
    Supervisor,
    SupervisorConfig,
    _kill_orphan,
    crash_streak,
    main,
)

# A lane that does what plan.json says for its n-th run, n counted in runs.txt: exit with a
# code, hang without a heartbeat, or beat and hang.
FAKE_LANE = """
import json, pathlib, sys, time
plan, beat = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
runs = plan.with_name("runs.txt")
n = int(runs.read_text()) if runs.exists() else 0
runs.write_text(str(n + 1))
assert sys.stdin.read() == ""  # Standard input is /dev/null.
action = json.loads(plan.read_text())[n]
if action == "hang":
    time.sleep(120)
while action == "beat":
    beat.with_suffix(".tmp").write_text(str(time.time()))
    beat.with_suffix(".tmp").replace(beat)  # Atomic, as goaly.rsi.heartbeat.beat writes it.
    time.sleep(0.02)
sys.exit(int(action))
"""
FAST = SupervisorConfig(
    poll_s=0.01,
    heartbeat_stale_s=1.0,
    backoffs_s=(0.05, 0.1, 0.15, 0.2, 0.25),
    kill_wait_s=5,
)


class Lab:
    def __init__(self, tmp_path: Path, plan: list[str]) -> None:
        self.root = tmp_path
        self.plan = tmp_path / "plan.json"
        self.plan.write_text(json.dumps(plan), encoding="utf-8")
        self.script = tmp_path / "lane.py"
        self.script.write_text(FAKE_LANE, encoding="utf-8")
        self.settings = LaneSettings.model_validate(
            {
                "repo": str(tmp_path),
                "ledger_dir": str(tmp_path / "ledger"),
                "lane": {"deadline": "2026-10-15T20:00:00Z"},
            }
        )
        self.ledger = Ledger(tmp_path / "ledger")
        self.channel = FakeChannel()
        self.notifier = QueuedNotifier(self.channel, self.ledger, retry_delays_s=(0.0,))
        self.preflight_problems: list[list[str]] = []
        self.state = tmp_path / "runs" / "rsi" / "supervisor"

    def supervisor(self, config: SupervisorConfig = FAST) -> Supervisor:
        def command(config_path: Path, settings: LaneSettings) -> list[str]:
            beat = self.state / "effisql.heartbeat"
            return [sys.executable, str(self.script), str(self.plan), str(beat)]

        def preflight(settings: LaneSettings) -> list[str]:
            return self.preflight_problems.pop(0) if self.preflight_problems else []

        return Supervisor(
            [(self.root / "lane.toml", self.settings)],
            config=config,
            preflight=preflight,
            command=command,
            notifiers=lambda _: self.notifier,
        )

    def runs(self) -> int:
        path = self.root / "runs.txt"
        return int(path.read_text()) if path.exists() else 0

    def events(self) -> list[tuple[str, str]]:
        names = {
            LANE_EVENT_KIND_CRASHED: "crashed",
            LANE_EVENT_KIND_RESTARTED: "restarted",
            LANE_EVENT_KIND_RECOVERY_FAILED: "recovery-failed",
        }
        return [
            (names[event.kind], event.detail)
            for event in self.ledger.lane_events(LANE_EFFISQL)
            if event.kind in names
        ]


@pytest.fixture
def lab(tmp_path: Path) -> Iterator[Lab]:
    made = Lab(tmp_path, [])
    yield made
    made.notifier.close(timeout_s=5)


def plan(lab: Lab, actions: list[str]) -> Lab:
    lab.plan.write_text(json.dumps(actions), encoding="utf-8")
    return lab


def test_a_lane_that_exits_0_is_not_restarted(lab: Lab) -> None:
    assert plan(lab, ["0"]).supervisor().run() == 0
    assert lab.runs() == 1
    assert lab.events() == []
    assert not (lab.state / "effisql.pid").exists()


def test_a_crashed_lane_is_restarted_after_its_backoff(lab: Lab) -> None:
    assert plan(lab, ["1", "1", "0"]).supervisor().run() == 0
    assert lab.runs() == 3
    assert lab.events() == [
        ("crashed", "the lane process exited with 1"),
        ("restarted", "after a backoff of 0.05 s"),
        ("crashed", "the lane process exited with 1"),
        ("restarted", "after a backoff of 0.1 s"),
    ]


def test_five_crashes_within_an_hour_give_the_lane_up_and_notify(lab: Lab) -> None:
    assert plan(lab, ["2"] * 9).supervisor().run() == 0
    assert lab.runs() == 5
    events = lab.events()
    assert [kind for kind, _ in events].count("crashed") == 5
    assert [detail for kind, detail in events if kind == "restarted"] == [
        "after a backoff of 0.05 s",
        "after a backoff of 0.1 s",
        "after a backoff of 0.15 s",
        "after a backoff of 0.2 s",
    ]
    assert events[-1] == (
        "recovery-failed",
        "5 crashes within 1 h; the last: the lane process exited with 2",
    )
    (sent,) = lab.channel.delivered
    assert sent.kind is NotificationKind.RECOVERY_FAILED
    assert sent.text.startswith("Effi-SQL: automatic recovery failed and the lane is stopped")
    notified = [e for e in lab.ledger.lane_events() if e.kind == LANE_EVENT_KIND_NOTIFIED]
    assert len(notified) == 1


def test_crashes_older_than_the_window_do_not_count(lab: Lab) -> None:
    old = Timestamp()
    old.FromDatetime(datetime.now(UTC) - timedelta(hours=2))
    for _ in range(4):
        lab.ledger.record_lane_event(
            LaneEvent(lane=LANE_EFFISQL, kind=LANE_EVENT_KIND_CRASHED, occurred_at=old)
        )
    assert plan(lab, ["1", "0"]).supervisor().run() == 0
    assert lab.runs() == 2


def test_a_stale_heartbeat_gets_the_lane_killed_and_counted(lab: Lab) -> None:
    assert plan(lab, ["hang", "0"]).supervisor().run() == 0
    kind, detail = lab.events()[0]
    assert kind == "crashed"
    assert detail.startswith("its heartbeat was ")
    assert detail.endswith(" s old, so it was killed")
    assert lab.runs() == 2


def test_a_beating_lane_is_left_alone(lab: Lab) -> None:
    supervisor = plan(lab, ["beat"]).supervisor()
    thread = threading.Thread(target=supervisor.run)
    thread.start()
    time.sleep(2.0)  # Twice the stale limit.
    assert lab.events() == []
    supervisor.stop()
    thread.join(10)
    assert not thread.is_alive()
    assert lab.events() == []  # Stopping the supervisor is no crash.


def test_a_lane_killed_with_sigkill_is_restarted(lab: Lab) -> None:
    supervisor = plan(lab, ["beat", "0"]).supervisor()
    thread = threading.Thread(target=supervisor.run)
    thread.start()
    pid_file = lab.state / "effisql.pid"
    deadline = time.monotonic() + 10
    while lab.runs() < 1 and time.monotonic() < deadline:  # The lane has begun its first run.
        time.sleep(0.01)
    os.kill(int(pid_file.read_text()), signal.SIGKILL)
    thread.join(20)
    if thread.is_alive():  # pragma: no cover  (keep a stuck supervisor from outliving the test)
        supervisor.stop()
        thread.join(10)
        raise AssertionError("the supervisor did not finish")
    assert lab.events()[:2] == [
        ("crashed", "the lane process killed by signal 9"),
        ("restarted", "after a backoff of 0.05 s"),
    ]
    assert lab.runs() == 2


def test_a_failed_preflight_is_a_crash(lab: Lab) -> None:
    lab.preflight_problems = [["docker does not answer: exit 1", "only 3.0 GB free"]]
    assert plan(lab, ["0"]).supervisor().run() == 0
    assert lab.events()[0] == (
        "crashed",
        "the preflight checks failed: docker does not answer: exit 1; only 3.0 GB free",
    )
    assert lab.runs() == 1  # Started once, after the failed check.


def test_a_configuration_error_is_not_restarted(lab: Lab) -> None:
    assert plan(lab, ["3", "0"]).supervisor().run() == 0
    assert lab.runs() == 1
    assert lab.events() == [
        ("crashed", "a configuration error (exit 3)"),
        ("recovery-failed", "its configuration is unusable (exit 3), so it is not restarted"),
    ]
    assert lab.channel.delivered[0].kind is NotificationKind.RECOVERY_FAILED


def test_a_ledger_failure_never_stops_the_supervisor(lab: Lab) -> None:
    def broken(event: LaneEvent) -> None:
        raise OSError("disk full")

    supervisor = plan(lab, ["1", "0"]).supervisor()
    supervisor._slots[0].ledger.record_lane_event = broken  # type: ignore[method-assign]
    assert supervisor.run() == 0
    assert lab.runs() == 2


def test_an_orphan_lane_is_killed(tmp_path: Path) -> None:
    orphan = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)", "goaly.rsi", "lane"],
        start_new_session=True,
    )
    pid_file = tmp_path / "effisql.pid"
    pid_file.write_text(f"{orphan.pid}\n", encoding="utf-8")
    _kill_orphan(pid_file)
    assert orphan.wait(10) == -signal.SIGKILL
    assert not pid_file.exists()
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    pid_file.write_text(f"{other.pid}\n", encoding="utf-8")
    _kill_orphan(pid_file)  # Not a lane: left alone.
    assert other.poll() is None
    other.kill()
    other.wait()
    pid_file.write_text("garbage", encoding="utf-8")
    _kill_orphan(pid_file)
    assert not pid_file.exists()


def event(kind: int, minutes: float) -> LaneEvent:
    moment = Timestamp()
    moment.FromDatetime(datetime(2026, 10, 15, 16, tzinfo=UTC) + timedelta(minutes=minutes))
    return LaneEvent(lane=LANE_EFFISQL, kind=kind, occurred_at=moment)  # type: ignore[arg-type]  # A kind's number.


def test_crash_streak_counts_crashes_in_a_row() -> None:
    crashed, started, stopped = (
        LANE_EVENT_KIND_CRASHED,
        LANE_EVENT_KIND_STARTED,
        LANE_EVENT_KIND_STOPPED,
    )
    assert crash_streak([], 3600) == 0
    quick = [event(started, 0), event(crashed, 1), event(crashed, 2), event(started, 3)]
    assert crash_streak([*quick, event(crashed, 4)], 3600) == 3
    long_run = [*quick, event(crashed, 3 + 61)]
    assert crash_streak(long_run, 3600) == 1  # It ran for an hour first.
    assert crash_streak([*quick, event(stopped, 5), event(crashed, 6)], 3600) == 1


# The command line.


def write_lane_config(tmp_path: Path) -> Path:
    config = tmp_path / "lane.toml"
    config.write_text(
        f'repo = "{tmp_path}"\nledger_dir = "{tmp_path / "ledger"}"\n\n'
        '[lane]\ndeadline = "2026-10-15T20:00:00Z"\n',
        encoding="utf-8",
    )
    return config


def test_run_refuses_a_bad_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "lane.toml"
    bad.write_text('repo = "."\n', encoding="utf-8")
    assert cli_main(["run", "--config", str(bad)]) == 3
    assert "configuration error" in capsys.readouterr().err
    good = write_lane_config(tmp_path)
    supervisor_config = tmp_path / "supervisor.toml"
    supervisor_config.write_text('preflight_function = "tests.nowhere:x"\n', encoding="utf-8")
    assert main(["--config", str(good), "--supervisor-config", str(supervisor_config)]) == 3


def test_run_exits_4_while_another_supervisor_holds_the_lock(tmp_path: Path) -> None:
    from goaly.rsi.locks import file_lock

    config = write_lane_config(tmp_path)
    state = tmp_path / "runs" / "rsi" / "supervisor"
    state.mkdir(parents=True)
    with file_lock(state / "supervisor.lock"):
        assert main(["--config", str(config)]) == EXIT_LOCKED
        assert main(["--config", str(config), "--detach"]) == 0  # Says it runs; starts none.
    assert not (state / "loop.pid").exists()
