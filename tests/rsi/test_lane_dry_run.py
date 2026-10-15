"""The lane's end-to-end dry run (T21's acceptance): the CLI, unattended, with stdin closed."""

import json
import os
import shlex
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from goaly.contracts.gen.experiment_pb2 import (
    DECISION_ABORTED,
    DECISION_ACCEPTED,
    LANE_EFFISQL,
    LANE_EVENT_KIND_STARTED,
    LANE_EVENT_KIND_STOPPED,
)
from goaly.rsi.lane import RoundLog
from goaly.rsi.ledger import Ledger
from tests.rsi.dry_run import write_config
from tests.rsi.gitrepo import git

ROOT = Path(__file__).resolve().parents[2]


def run_cli(config: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run the lane CLI in a fresh process whose standard input is closed, not just empty."""
    command = shlex.join([sys.executable, "-m", "goaly.rsi", *args, "--config", str(config)])
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    return subprocess.run(  # noqa: S603  (bash, with the test's own command)
        ["bash", "-c", f"exec 0<&- && exec {command}"],  # noqa: S607
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


def test_the_dry_run_runs_unattended_until_its_stopping_rule(tmp_path: Path) -> None:
    repo, config = write_config(tmp_path, datetime.now(UTC) + timedelta(hours=1))
    completed = run_cli(config, "lane", "--lane", "effisql")
    assert completed.returncode == 0, completed.stderr[-3000:]
    ledger = Ledger(tmp_path / "ledger")
    events = ledger.lane_events(LANE_EFFISQL)
    assert [e.kind for e in events] == [LANE_EVENT_KIND_STARTED, LANE_EVENT_KIND_STOPPED]
    assert events[0].deadline_at.ToDatetime(UTC) > datetime.now(UTC)
    assert events[1].detail.startswith("no accepted change: 4 rounds in a row")
    # The stop record came first, then the results-PR hook, then the notifier.
    hooks = [json.loads(line) for line in (repo / "hooks.jsonl").read_text().splitlines()]
    assert [h["hook"] for h in hooks] == ["results_pr", "notify"]
    assert {h["last_event"] for h in hooks} == {"LANE_EVENT_KIND_STOPPED"}
    latest = ledger.latest_experiments(LANE_EFFISQL)
    accepted = [r for r in latest.values() if r.decision == DECISION_ACCEPTED]
    assert [r.hypothesis.hypothesis_id for r in accepted] == ["good-1"]
    best = git(repo, "rev-parse", "best/effisql").strip()
    assert best == accepted[0].candidate_commit == hooks[0]["best"]
    assert max(r.round for r in latest.values()) == 5
    # The implementer ran on mini-swe-agent's scripted model and smoke-tested once.
    (smoke,) = accepted[0].implementer.smoke_tests
    assert smoke.instance_id == "1"
    assert smoke.HasField("score")
    assert accepted[0].implementer.summary.strip() == "Wrote good to kind.txt"
    assert (
        repo / "runs/rsi/effisql" / accepted[0].experiment_id / "implementer.traj.json"
    ).exists()
    # Each round probed the three grading instances first; grade-3's drift is recorded.
    started = [
        e for e in RoundLog(tmp_path / "ledger", LANE_EFFISQL).entries() if e.status == "started"
    ]
    assert len(started) == 5
    for entry in started:
        assert [p.instance for p in entry.probes] == ["grade-1", "grade-2", "grade-3"]
        assert entry.probes[2].drift == pytest.approx(0.2)
    # Status reads the same state, changing nothing.
    status = run_cli(config, "status", "--lane", "effisql")
    assert status.returncode == 0, status.stderr
    state = json.loads(status.stdout)
    assert state["best_tag"] == best
    assert state["rounds_without_accept"] == 4
    assert state["last_event"]["kind"] == "LANE_EVENT_KIND_STOPPED"


def test_a_configuration_error_exits_with_3(tmp_path: Path) -> None:
    config = tmp_path / "lane.toml"
    config.write_text('repo = "."\n', encoding="utf-8")  # No [lane] and no deadline.
    completed = run_cli(config, "lane", "--lane", "effisql")
    assert completed.returncode == 3
    assert "configuration error" in completed.stderr


@pytest.mark.parametrize("missing", ["module", "attribute"])
def test_a_wiring_that_cannot_be_loaded_is_a_configuration_error(
    tmp_path: Path, missing: str
) -> None:
    _, config = write_config(tmp_path, datetime.now(UTC) + timedelta(hours=1))
    wiring = "tests.rsi.nowhere:build" if missing == "module" else "tests.rsi.dry_run:nothing"
    text = config.read_text().replace("tests.rsi.dry_run:build", wiring)
    config.write_text(text, encoding="utf-8")
    completed = run_cli(config, "lane", "--lane", "effisql")
    assert completed.returncode == 3, completed.stderr


def test_a_deadline_during_implementation_still_exits_0(tmp_path: Path) -> None:
    # Every goaly-smoke evaluation waits forever, and meh's GLM call never returns: at the
    # deadline the implementers are mid-step, and the process must exit all the same.
    repo, config = write_config(tmp_path, datetime.now(UTC) + timedelta(seconds=8), "hang")
    started = time.monotonic()
    completed = run_cli(config, "lane", "--lane", "effisql")
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert time.monotonic() - started < 60
    ledger = Ledger(tmp_path / "ledger")
    events = ledger.lane_events(LANE_EFFISQL)
    assert events[-1].kind == LANE_EVENT_KIND_STOPPED
    assert events[-1].detail.startswith("deadline: ")
    latest = ledger.latest_experiments(LANE_EFFISQL)
    assert len(latest) == 3
    assert {r.decision for r in latest.values()} == {DECISION_ABORTED}
    hooks = [json.loads(line) for line in (repo / "hooks.jsonl").read_text().splitlines()]
    assert [h["hook"] for h in hooks] == ["results_pr", "notify"]
    assert len(hooks[0]["aborted"]) == 3


def test_a_restart_after_the_deadline_needs_nothing_but_the_ledger(tmp_path: Path) -> None:
    # The wiring cannot even load, as if Docker or the evaluator were gone after the freeze:
    # the lane still stops, writes its stop record and calls the hooks.
    repo, config = write_config(tmp_path, datetime.now(UTC) - timedelta(minutes=1))
    text = config.read_text().replace("tests.rsi.dry_run:build", "tests.rsi.nowhere:build")
    config.write_text(text, encoding="utf-8")
    completed = run_cli(config, "lane", "--lane", "effisql")
    assert completed.returncode == 0, completed.stderr[-3000:]
    (event,) = Ledger(tmp_path / "ledger").lane_events(LANE_EFFISQL)
    assert event.kind == LANE_EVENT_KIND_STOPPED
    hooks = [json.loads(line) for line in (repo / "hooks.jsonl").read_text().splitlines()]
    assert [h["hook"] for h in hooks] == ["results_pr", "notify"]


def test_an_unknown_log_level_is_a_configuration_error(tmp_path: Path) -> None:
    _, config = write_config(tmp_path, datetime.now(UTC) + timedelta(hours=1))
    config.write_text(config.read_text().replace('"WARNING"', '"LOUD"'), encoding="utf-8")
    assert run_cli(config, "lane", "--lane", "effisql").returncode == 3
