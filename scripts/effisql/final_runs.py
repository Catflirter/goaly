"""Run the final evaluation, or rerun the baseline, through the isolated pipeline (spec §6.1, §10).

    uv run --locked python scripts/effisql/final_runs.py final [--dry-run] [--root runs/final]
    uv run --locked python scripts/effisql/final_runs.py baseline --ref v0 [--dry-run]

``final`` runs the frozen harness on held-out, 3 runs graded in official mode, then on val, 3
runs graded in dev mode. ``baseline`` reruns v0 as T12 ran it: train with 1 seed and val with 2
in dev mode, then held-out with 1 seed in official mode. Every run is ``goaly run --bench
effisql``, split into its solve phase (the proxy and the solve container) and its grade phase on
the host (ruling V27), so that solving and grading overlap without sharing an instance:

    solve run 1 | solve run 2 | solve run 3
                | grade run 1 | grade run 2 | grade run 3

One run solves at a time, and one grades at a time, since a grading process holds the grading
instances alone (spec §6.2). Run i+1 solves while run i grades. Once goaly can grade on all 5
instances (T11's ``--grading-instances all``, allowed only while nothing solves), the final
solves both runs first and then grades each on all 5; until then it grades on the 3.

The checkout must be clean and at ``--ref``: the tag ``final`` for the final, v0's commit for
the baseline. Each run's manifest then records that commit. Rerunning the command continues: a
graded run is skipped, a solve phase resumes with --resume, and a grade phase grades what is
left. Each phase's output goes to ``<root>/logs/``, and ``<root>/stages.jsonl`` records the
host's clock around every phase. --dry-run prints the plan and runs nothing.
"""

import argparse
import asyncio
import json
import subprocess
import sys
import sysconfig
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from goaly.bench.effisql.report_inputs import RunDirError, load_run

REPO = Path(__file__).resolve().parents[2]
# T11's host option to grade on all 5 instances, the 2 solving ones included (spec §6.2, §10).
ALL_INSTANCES = ("--grading-instances", "all")


@dataclass(frozen=True)
class Run:
    """One run of the plan: a split, its seeds and its run directory's name under the root."""

    name: str
    split: str
    seeds: int


@dataclass(frozen=True)
class Preset:
    """A plan's runs, where they go, the commit they evaluate and their LLM budget phase."""

    root: Path
    ref: str
    budget_phase: str
    runs: tuple[Run, ...]
    wide_grading: bool = False  # Grade on all 5 instances once every solve is done (spec §6.2).


PRESETS = {
    "final": Preset(
        Path("runs/final"),
        "final",
        "final",
        (Run("heldout", "heldout", 3), Run("val", "val", 3)),
        wide_grading=True,
    ),
    "baseline": Preset(
        Path("runs/baseline"),
        "v0",
        "baseline",
        (Run("train", "train", 1), Run("val", "val", 2), Run("heldout", "heldout", 1)),
    ),
}


@dataclass(frozen=True)
class Stage:
    """One phase of one run, as the goaly command line that runs it."""

    run: str
    phase: str  # "solve" or "grade".
    argv: tuple[str, ...]
    after_solves: bool = False  # A grade that waits for every solve, as it uses their instances.


class PlanError(Exception):
    """The plan cannot run here: the checkout is not the one to evaluate."""


def plan(
    preset: Preset, *, root: Path, harness: str, goaly: Sequence[str], wide: bool = False
) -> list[Stage]:
    """The stages still to run for ``preset``, solve before grade for each run.

    With ``wide``, goaly can grade on all 5 instances (:data:`ALL_INSTANCES`), which it allows
    only while no solve phase holds the solving instances. A preset that grades wide then
    grades every run after every solve, with that option: for the final, solving val first and
    then grading held-out on 5 instances beats grading it on 3 while val solves, since
    held-out's official grading takes the longest. Without it, run i+1 solves while run i grades.
    """
    wide = wide and preset.wide_grading
    stages = []
    for run in preset.runs:
        out = root / run.name
        if _graded(out):
            continue
        if not _solved(out, run.seeds):
            solve = [*goaly, "run", "--bench", "effisql", "--phase", "solve"]
            solve += ["--split", run.split, "--harness", harness, "--seeds", str(run.seeds)]
            solve += ["--budget-phase", preset.budget_phase, "--out", str(out)]
            if (out / "manifest.json").exists():
                solve.append("--resume")
            stages.append(Stage(run.name, "solve", tuple(solve)))
        grade = (*goaly, "run", "--bench", "effisql", "--phase", "grade", "--out", str(out))
        if wide:
            grade = (*grade, *ALL_INSTANCES)
        stages.append(Stage(run.name, "grade", grade, after_solves=wide))
    return stages


def check_checkout(ref: str, repo: Path) -> str:
    """Return the commit ``ref`` names, if the clean checkout at ``repo`` is at it."""
    head = _git(repo, "rev-parse", "HEAD")
    try:
        commit = _git(repo, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
    except subprocess.CalledProcessError:
        raise PlanError(f"{ref!r} names no commit here; fetch its tag, or pass --ref") from None
    if head != commit:
        raise PlanError(f"the checkout is at {head[:12]}, not {ref} ({commit[:12]}): check it out")
    if _git(repo, "status", "--porcelain", "--untracked-files=no"):
        raise PlanError("the checkout has uncommitted changes, which no manifest would record")
    return commit


async def execute(stages: Sequence[Stage], root: Path) -> int:
    """Run ``stages``: solves in order, grades in order, each grade after its run's solve.

    Returns 0 once every stage exits 0. After a failure no new stage starts; the one running
    finishes, and the status is that of the first failure.
    """
    (root / "logs").mkdir(parents=True, exist_ok=True)
    solved: dict[str, asyncio.Event] = {}
    for stage in stages:
        solved.setdefault(stage.run, asyncio.Event())
    for stage in stages:
        if stage.phase == "grade" and not any(
            s.run == stage.run and s.phase == "solve" for s in stages
        ):
            solved[stage.run].set()  # Solved by an earlier invocation.
    failures: list[int] = []

    async def lane(phase: str) -> None:
        for stage in (s for s in stages if s.phase == phase):
            if phase == "grade":
                runs = solved.values() if stage.after_solves else [solved[stage.run]]
                await asyncio.gather(*(event.wait() for event in runs))
            if failures:
                return
            status = await _run_stage(stage, root)
            if status:
                failures.append(status)
                for event in solved.values():  # Wake the grade lane, which then stops.
                    event.set()
                return
            if phase == "solve":
                solved[stage.run].set()

    await asyncio.gather(lane("solve"), lane("grade"))
    return failures[0] if failures else 0


async def _run_stage(stage: Stage, root: Path) -> int:
    log = root / "logs" / f"{stage.run}-{stage.phase}.log"
    started = time.time()
    _say(f"{stage.run} {stage.phase}: {' '.join(stage.argv)}")
    with log.open("ab") as output:
        process = await asyncio.create_subprocess_exec(
            *stage.argv, cwd=REPO, stdout=output, stderr=subprocess.STDOUT
        )
        status = await process.wait()
    record: dict[str, object] = {"run": stage.run, "phase": stage.phase, "argv": list(stage.argv)}
    record |= {"started_at": started, "finished_at": time.time(), "exit_status": status}
    with (root / "stages.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")
    _say(f"{stage.run} {stage.phase}: exit {status} after {time.time() - started:.0f} s; {log}")
    return status


def grades_wide(goaly: Sequence[str]) -> bool:
    """Whether this goaly has T11's option to grade on all 5 instances."""
    try:
        completed = subprocess.run(  # noqa: S603 - the goaly command, as configured.
            [*goaly, "run", "--help"], capture_output=True, text=True, timeout=120, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return ALL_INSTANCES[0] in completed.stdout


def _graded(out: Path) -> bool:
    try:
        run = load_run(out)
    except (OSError, RunDirError, ValueError):
        return False
    return not run.missing


def _solved(out: Path, seeds: int) -> bool:
    """True once the host has written every seed's official records: the solve phase is done."""
    return all((out / "official" / f"seed-{seed}.jsonl").exists() for seed in range(seeds))


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(  # noqa: S603 - git, with arguments built here.
        ["git", "-C", str(repo), *args],  # noqa: S607 - git from PATH, as the developer runs it.
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return completed.stdout.strip()


def _say(text: str) -> None:
    sys.stderr.write(text + "\n")
    sys.stderr.flush()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the final evaluation or the baseline.")
    parser.add_argument("preset", choices=sorted(PRESETS))
    parser.add_argument("--root", type=Path, help="where the run directories go")
    parser.add_argument("--ref", help="the commit to evaluate (default: final, or v0)")
    parser.add_argument("--harness", default="v0", help="the registered harness (default v0)")
    parser.add_argument("--goaly", nargs="+", help="the goaly command (default: the venv's)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    args = parser.parse_args(argv)
    preset = PRESETS[args.preset]
    root = (args.root or REPO / preset.root).absolute()
    goaly = args.goaly or [str(Path(sysconfig.get_path("scripts")) / "goaly")]
    ref = args.ref or preset.ref
    try:
        commit = check_checkout(ref, REPO)
    except PlanError as error:
        if not args.dry_run:
            _say(f"error: {error}")
            return 2
        _say(f"warning: {error}; the plan cannot run until then")
        commit = "?"
    wide = preset.wide_grading and grades_wide(goaly)
    stages = plan(preset, root=root, harness=args.harness, goaly=goaly, wide=wide)
    grading = "all 5 instances, after every solve" if wide else "the 3 grading instances"
    _say(f"{args.preset} at {ref} ({commit[:12]}): {len(stages)} stages in {root}")
    _say(f"grading on {grading}")
    for stage in stages:
        _say(f"  {stage.run} {stage.phase}: {' '.join(stage.argv)}")
    if args.dry_run or not stages:
        return 0
    return asyncio.run(execute(stages, root))


if __name__ == "__main__":
    sys.exit(main())
