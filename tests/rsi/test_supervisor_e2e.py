"""The supervisor end to end (T22's acceptance): ``goaly rsi run --detach`` with standard input
closed, running T21's dry-run lane until it stops, through a kill -9 mid-round."""

import os
import signal
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

from goaly.contracts.gen.experiment_pb2 import (
    DECISION_ABORTED,
    DECISION_ACCEPTED,
    LANE_EFFISQL,
    LANE_EVENT_KIND_CRASHED,
    LANE_EVENT_KIND_RESTARTED,
    LANE_EVENT_KIND_STARTED,
    LANE_EVENT_KIND_STOPPED,
)
from goaly.rsi.cli import LaneSettings
from goaly.rsi.ledger import Ledger
from tests.rsi.dry_run import write_config

ROOT = Path(__file__).resolve().parents[2]


def no_problems(settings: LaneSettings) -> list[str]:
    """The dry run's preflight: no Docker, PostgreSQL or gateway to check."""
    return []


def detach(tmp_path: Path) -> tuple[Path, Path]:
    """Start the supervisor detached, with stdin closed; return the repo and the state dir."""
    repo, config = write_config(tmp_path, datetime.now(UTC) + timedelta(hours=1))
    supervisor = tmp_path / "supervisor.toml"
    supervisor.write_text(
        "poll_s = 0.05\nbackoffs_s = [0.2, 0.4]\nrestart_s = 0\n"
        'preflight_function = "tests.rsi.test_supervisor_e2e:no_problems"\n',
        encoding="utf-8",
    )
    arguments = ["--config", str(config), "--supervisor-config", str(supervisor)]
    command = " ".join([sys.executable, "-m", "goaly.rsi", "run", "--detach", *arguments])
    done = subprocess.run(  # noqa: S603  (bash, with the test's own command)
        ["bash", "-c", f"exec 0<&- && exec {command}"],  # noqa: S607
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert '"running": true' in done.stdout
    return repo, repo / "runs" / "rsi" / "supervisor"


def wait_for_loop(state: Path, timeout_s: float = 300) -> None:
    pid = int((state / "loop.pid").read_text())
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        # Reap it if this process is its parent (it is not: it runs in its own session).
        time.sleep(0.1)
    os.killpg(pid, signal.SIGKILL)
    raise AssertionError((state / "supervisor.log").read_text()[-3000:])


def test_detached_supervisor_runs_the_lane_until_it_stops(tmp_path: Path) -> None:
    repo, state = detach(tmp_path)
    wait_for_loop(state)
    events = Ledger(tmp_path / "ledger").lane_events(LANE_EFFISQL)
    assert [e.kind for e in events] == [LANE_EVENT_KIND_STARTED, LANE_EVENT_KIND_STOPPED]
    hooks = (repo / "hooks.jsonl").read_text().splitlines()
    assert len(hooks) == 2  # The final results PR, then the notification.
    assert not (state / "effisql.pid").exists()
    assert (state / "effisql.log").exists()


def test_a_lane_killed_mid_round_resumes_from_the_ledger(tmp_path: Path) -> None:
    repo, state = detach(tmp_path)
    ledger = Ledger(tmp_path / "ledger")
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:  # Wait for round 2 to begin.
        if any(r.round >= 2 for r in ledger.experiments()):
            break
        time.sleep(0.02)
    else:
        raise AssertionError("round 2 never started")
    os.kill(int((state / "effisql.pid").read_text()), signal.SIGKILL)
    wait_for_loop(state)
    kinds = [e.kind for e in ledger.lane_events(LANE_EFFISQL)]
    assert kinds == [
        LANE_EVENT_KIND_STARTED,
        LANE_EVENT_KIND_CRASHED,
        LANE_EVENT_KIND_RESTARTED,
        LANE_EVENT_KIND_STARTED,
        LANE_EVENT_KIND_STOPPED,
    ]
    crashed = ledger.lane_events(LANE_EFFISQL)[1]
    assert crashed.detail == "the lane process killed by signal 9"
    latest = ledger.latest_experiments(LANE_EFFISQL).values()
    accepted = [r.hypothesis.hypothesis_id for r in latest if r.decision == DECISION_ACCEPTED]
    assert accepted == ["good-1"]
    # No experiment ran twice: an aborted one comes back only as a new experiment.
    decided = Counter(
        (r.round, r.hypothesis.hypothesis_id) for r in latest if r.decision != DECISION_ABORTED
    )
    assert max(decided.values()) == 1
    assert len((repo / "hooks.jsonl").read_text().splitlines()) == 2  # One final PR.
