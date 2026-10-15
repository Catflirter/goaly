"""Tests for scripts/effisql/final_runs.py, with a fake goaly command and a throwaway repository."""

import asyncio
import itertools
import json
import os
import sys
from pathlib import Path

import pytest

from scripts.effisql import final_runs
from scripts.effisql.final_runs import (
    ALL_INSTANCES,
    PRESETS,
    PlanError,
    Stage,
    check_checkout,
    execute,
    plan,
)
from tests.bench.effisql.rundirs import write_run
from tests.scripts.gitrepo import git, repo_with_origin

# A goaly that logs each call, with its start and end, sleeps a little, and fails on request.
FAKE_GOALY = """
import json, sys, time
log, fail, args = sys.argv[1], sys.argv[2], sys.argv[3:]
phase = args[args.index("--phase") + 1]
out = args[args.index("--out") + 1].rsplit("/", 1)[1]
start = time.time()
time.sleep(0.3)
with open(log, "a") as file:
    file.write(json.dumps({"stage": f"{out} {phase}", "start": start, "end": time.time()}) + "\\n")
sys.exit(3 if f"{out} {phase}" == fail else 0)
"""


def fake_goaly(tmp_path: Path, fail: str = "") -> tuple[list[str], Path]:
    script, log = tmp_path / "goaly.py", tmp_path / "calls.jsonl"
    script.write_text(FAKE_GOALY, encoding="utf-8")
    return [sys.executable, str(script), str(log), fail], log


def calls(log: Path) -> dict[str, dict[str, float]]:
    lines = log.read_text(encoding="utf-8").splitlines()
    return {record["stage"]: record for record in map(json.loads, lines)}


def test_the_final_runs_held_out_then_val_in_two_phases(tmp_path: Path) -> None:
    stages = plan(PRESETS["final"], root=tmp_path, harness="v0", goaly=["goaly"])

    assert [(s.run, s.phase) for s in stages] == [
        ("heldout", "solve"),
        ("heldout", "grade"),
        ("val", "solve"),
        ("val", "grade"),
    ]
    assert stages[0].argv == (
        *("goaly", "run", "--bench", "effisql", "--phase", "solve", "--split", "heldout"),
        *("--harness", "v0", "--seeds", "3", "--budget-phase", "final"),
        *("--out", str(tmp_path / "heldout")),
    )
    assert stages[1].argv == (
        *("goaly", "run", "--bench", "effisql", "--phase", "grade"),
        *("--out", str(tmp_path / "heldout")),
    )
    assert "--seeds 3" in " ".join(stages[2].argv)


def test_the_baseline_reruns_v0_as_t12_ran_it(tmp_path: Path) -> None:
    stages = plan(PRESETS["baseline"], root=tmp_path, harness="v0", goaly=["goaly"])

    solves = [s.argv for s in stages if s.phase == "solve"]
    assert [(a[a.index("--split") + 1], a[a.index("--seeds") + 1]) for a in solves] == [
        ("train", "1"),
        ("val", "2"),
        ("heldout", "1"),
    ]
    assert all("baseline" in argv for argv in solves)


def test_a_rerun_skips_graded_runs_and_resumes_the_rest(tmp_path: Path) -> None:
    write_run(tmp_path / "heldout", {"1": [2.0, 2.0, 2.0]})  # Graded in full.
    val = tmp_path / "val"
    val.mkdir()
    (val / "manifest.json").write_text("{}", encoding="utf-8")

    stages = plan(PRESETS["final"], root=tmp_path, harness="v0", goaly=["goaly"])

    assert [(s.run, s.phase) for s in stages] == [("val", "solve"), ("val", "grade")]
    assert stages[0].argv[-1] == "--resume"


def test_a_solved_run_goes_straight_to_grading(tmp_path: Path) -> None:
    path = write_run(tmp_path / "heldout", {"1": [2.0, 2.0, 2.0]})
    (path / "results.jsonl").unlink()  # Solved, with its official records, but not graded.

    stages = plan(PRESETS["final"], root=tmp_path, harness="v0", goaly=["goaly"])

    assert [(s.run, s.phase) for s in stages][:1] == [("heldout", "grade")]


def test_solving_overlaps_grading_but_never_itself(tmp_path: Path) -> None:
    goaly, log = fake_goaly(tmp_path)
    stages = plan(PRESETS["baseline"], root=tmp_path / "runs", harness="v0", goaly=goaly)

    status = asyncio.run(execute(stages, tmp_path / "runs"))

    assert status == 0
    ran = calls(log)
    order = ["train solve", "val solve", "heldout solve"]
    for earlier, later in itertools.pairwise(order):
        assert ran[earlier]["end"] <= ran[later]["start"]
    for name in ("train", "val", "heldout"):
        assert ran[f"{name} solve"]["end"] <= ran[f"{name} grade"]["start"]
    assert ran["train grade"]["end"] <= ran["val grade"]["start"]
    assert ran["val solve"]["start"] < ran["train grade"]["end"]  # They overlap.
    stages_log = (tmp_path / "runs" / "stages.jsonl").read_text(encoding="utf-8").splitlines()
    assert sorted(json.loads(line)["exit_status"] for line in stages_log) == [0] * 6
    assert (tmp_path / "runs" / "logs" / "val-grade.log").exists()


def test_a_failure_starts_no_new_stage(tmp_path: Path) -> None:
    goaly, log = fake_goaly(tmp_path, fail="val solve")
    stages = plan(PRESETS["baseline"], root=tmp_path / "runs", harness="v0", goaly=goaly)

    status = asyncio.run(execute(stages, tmp_path / "runs"))

    assert status == 3
    assert set(calls(log)) == {"train solve", "train grade", "val solve"}


def test_a_grade_left_from_an_earlier_invocation_runs_at_once(tmp_path: Path) -> None:
    goaly, log = fake_goaly(tmp_path)
    grade = Stage("val", "grade", (*goaly, "run", "--phase", "grade", "--out", "x/val"))

    assert asyncio.run(execute([grade], tmp_path / "runs")) == 0
    assert set(calls(log)) == {"val grade"}


# The checkout.


def test_checks_the_checkout_is_clean_and_at_the_ref(tmp_path: Path) -> None:
    repo, _ = repo_with_origin(tmp_path)
    git(repo, "tag", "final")

    assert check_checkout("final", repo) == git(repo, "rev-parse", "HEAD")

    (repo / "src" / "goaly" / "harness" / "packs" / "effisql" / "prompt.md").write_text("x")
    with pytest.raises(PlanError, match="uncommitted changes"):
        check_checkout("final", repo)
    with pytest.raises(PlanError, match="names no commit"):
        check_checkout("v0", repo)


def test_refuses_a_checkout_at_another_commit(tmp_path: Path) -> None:
    repo, _ = repo_with_origin(tmp_path)
    git(repo, "tag", "final")
    git(repo, "commit", "--quiet", "--allow-empty", "--message", "later")

    with pytest.raises(PlanError, match="not final"):
        check_checkout("final", repo)


def test_main_runs_the_plan_only_at_the_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo, _ = repo_with_origin(tmp_path)
    monkeypatch.setattr(final_runs, "REPO", repo)
    goaly, log = fake_goaly(tmp_path)
    args = ["final", "--root", str(tmp_path / "runs"), "--goaly", *goaly]

    assert final_runs.main(args) == 2
    assert "'final' names no commit" in capsys.readouterr().err
    assert final_runs.main([*args, "--dry-run"]) == 0
    assert "heldout solve:" in capsys.readouterr().err
    assert not log.exists()

    git(repo, "tag", "final")
    assert final_runs.main(args) == 0
    assert set(calls(log)) == {"heldout solve", "heldout grade", "val solve", "val grade"}


# Grading on all 5 instances.

WIDE_GOALY = """
import sys
if sys.argv[1:] == ["run", "--help"]:
    print("usage: goaly run [--grading-instances {3,all}]")
"""


def test_detects_whether_goaly_can_grade_on_all_instances(tmp_path: Path) -> None:
    script = tmp_path / "wide.py"
    script.write_text(WIDE_GOALY, encoding="utf-8")

    assert final_runs.grades_wide([sys.executable, str(script)])
    assert not final_runs.grades_wide([sys.executable, "-c", "print('usage: goaly run')"])
    assert not final_runs.grades_wide([str(tmp_path / "missing")])


def test_grading_wide_solves_everything_first(tmp_path: Path) -> None:
    goaly, log = fake_goaly(tmp_path)
    stages = plan(PRESETS["final"], root=tmp_path / "runs", harness="v0", goaly=goaly, wide=True)

    assert [s.argv[-2:] for s in stages if s.phase == "grade"] == [ALL_INSTANCES] * 2
    assert asyncio.run(execute(stages, tmp_path / "runs")) == 0
    ran = calls(log)
    assert ran["val solve"]["end"] <= ran["heldout grade"]["start"]
    assert ran["heldout grade"]["end"] <= ran["val grade"]["start"]


def test_the_baseline_never_grades_wide(tmp_path: Path) -> None:
    stages = plan(PRESETS["baseline"], root=tmp_path, harness="v0", goaly=["goaly"], wide=True)

    assert not any(s.after_solves or "--grading-instances" in s.argv for s in stages)


def test_refuses_untracked_files_in_the_code(tmp_path: Path) -> None:
    repo, _ = repo_with_origin(tmp_path)
    git(repo, "tag", "final")
    (repo / "src" / "goaly" / "harness" / "packs" / "effisql" / "extra.py").write_text("x = 1")
    (repo / "runs").mkdir()
    (repo / "runs" / "elsewhere.txt").write_text("run artifacts are fine")

    with pytest.raises(PlanError, match=r"untracked files under src/, such as .*extra\.py"):
        check_checkout("final", repo)


def test_a_cancelled_stage_stops_its_goaly_run(tmp_path: Path) -> None:
    pid_file = tmp_path / "pid"
    script = "import os, sys, time; open(sys.argv[1], 'w').write(str(os.getpid())); time.sleep(60)"
    stage = Stage("val", "solve", (sys.executable, "-c", script, str(pid_file)))

    async def cancel_it() -> None:
        task = asyncio.ensure_future(execute([stage], tmp_path / "runs"))
        while not pid_file.exists():
            await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_it())

    pid = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)  # Terminated and reaped.
