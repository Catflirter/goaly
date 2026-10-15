"""`goaly run`: run a registered harness on a benchmark's tasks, and record the run (spec §6.4).

For example, a validation run during RSI, and a run on the organizers' own task file:

    goaly run --bench effisql --split val --budget-phase rsi --budget-share background --out r/v
    goaly run --bench effisql --instances heldout.jsonl --budget-phase final --out runs/heldout

A run may also come in two phases (spec V24), so that candidate pack code runs only in the
first, which may run in a container: --phase solve runs the pack's policy and records every task
ungraded, and --phase grade, later, grades the run with the benchmark's grader alone, importing
no pack code. --phase all, the default, does both at once. Every path comes from the options;
nothing is read from the home directory.

The run directory, --out, holds the manifest, the results, the trajectories and, once every task
is recorded, official/seed-<k>.jsonl: each replicate's predictions in the official format. On
stdout the run's summary is one line of JSON; the logs go to stderr as JSON lines.

The exit status is 0 once the run is done, whatever errors its tasks met, as those are in its
records; 1 if the run failed; 2 for a usage error, such as an unknown benchmark or a run
directory that holds another run; and 130 if it was interrupted, after which --resume
continues it.
"""

import argparse
import asyncio
import json
import logging
import math
import sys
from collections.abc import Sequence
from pathlib import Path

from goaly.contracts.gen.manifest_pb2 import RunManifest
from goaly.contracts.gen.task_pb2 import (
    SPLIT_EXTERNAL,
    SPLIT_HELDOUT,
    SPLIT_TRAIN,
    SPLIT_VAL,
    Split,
)
from goaly.contracts.interfaces import JsonValue
from goaly.harness.core.eval_cache import EvalCache
from goaly.harness.core.llm_budgets import BudgetsError, load_budgets
from goaly.harness.core.logs import configure_logging
from goaly.harness.core.manifest import (
    CORE_DIR,
    config_hash,
    core_hash,
    harness_hash,
    new_run_id,
    provenance,
    sha256_file,
)
from goaly.harness.core.registry import (
    GradingOptions,
    HarnessOptions,
    Registry,
    RegistryError,
    RunnableHarness,
)
from goaly.harness.core.runner import (
    DEFAULT_MAX_TASKS,
    RunDirectoryError,
    RunFiles,
    RunLock,
    RunSummary,
    lock_run_dir,
)

__all__ = ["FAILED", "INTERRUPTED", "USAGE", "build_parser", "main"]

_logger = logging.getLogger(__name__)

# Exit statuses besides 0.
FAILED = 1
USAGE = 2  # As argparse exits.
INTERRUPTED = 130  # As a shell reports a process that SIGINT ended.

_SPLITS = {"train": SPLIT_TRAIN, "val": SPLIT_VAL, "heldout": SPLIT_HELDOUT}
# Errors that mean the command cannot run as asked; the message says what to change.
_USAGE_ERRORS = (BudgetsError, RegistryError, RunDirectoryError, ValueError)


def build_parser(budgets_file: Path) -> argparse.ArgumentParser:
    """The `goaly` command's arguments; ``budgets_file`` is the default of --budgets."""
    parser = argparse.ArgumentParser(prog="goaly", description="The goaly harness.")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    run = commands.add_parser(
        "run",
        help="run a harness on a benchmark's tasks",
        description="Run a harness on a benchmark's tasks, and record the run in --out.",
    )
    run.add_argument("--bench", required=True, metavar="NAME", help="the benchmark")
    run.add_argument(
        "--phase",
        choices=["solve", "grade", "all"],
        default="all",
        help="solve only, grade a solved run, or both (default all)",
    )
    run.add_argument("--harness", metavar="NAME", help="the harness, if the benchmark has several")
    tasks = run.add_mutually_exclusive_group()
    tasks.add_argument("--split", choices=sorted(_SPLITS), help="run one of our splits")
    tasks.add_argument(
        "--instances",
        type=Path,
        metavar="FILE",
        help="run the tasks of an official-format task file, such as the organizers': not graded",
    )
    run.add_argument("--only", nargs="+", metavar="ID", help="run only these instances")
    run.add_argument("--seeds", type=_count, metavar="N", help="how many replicates (default 1)")
    run.add_argument(
        "--first-seed",
        type=_index,
        metavar="K",
        help="the first replicate's index (default 0); a confirmation run takes a new one",
    )
    run.add_argument("--out", type=Path, required=True, metavar="DIR", help="the run directory")
    run.add_argument(
        "--resume", action="store_true", help="continue the run in --out, if there is one"
    )
    run.add_argument(
        "--budget-phase",
        metavar="NAME",
        help="the schedule's phase whose LLM concurrency shares apply: baseline, rsi or final",
    )
    run.add_argument(
        "--budget-share",
        metavar="NAME",
        help="this process's share in that phase (default: --bench)",
    )
    run.add_argument(
        "--budgets", type=Path, default=budgets_file, metavar="FILE", help="the budgets file"
    )
    run.add_argument(
        "--max-tasks",
        type=_count,
        metavar="N",
        help=f"how many tasks solve at once (default {DEFAULT_MAX_TASKS})",
    )
    run.add_argument(
        "--task-timeout",
        type=_seconds,
        metavar="SECONDS",
        help="cancel a task that runs longer (default: no limit)",
    )
    run.add_argument(
        "--cache",
        type=Path,
        metavar="DIR",
        help="the evaluation cache; runs that grade differently use different ones",
    )
    run.add_argument("--run-id", metavar="ID", help="the run's ID (default: a new one)")
    run.add_argument(
        "--in-container",
        action="store_true",
        help="this is the solve phase inside its container (with --phase solve and --inputs)",
    )
    run.add_argument(
        "--inputs",
        type=Path,
        metavar="DIR",
        help="the container's read-only inputs, which its launcher staged (with --in-container)",
    )
    run.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main(argv: Sequence[str], registry: Registry, *, budgets_file: Path) -> int:
    """Run the `goaly` command with ``argv``, its arguments, and return the exit status.

    ``registry`` holds the harnesses it can run. argparse exits with status 2 by itself for
    arguments it refuses, and with 0 for --help.
    """
    args = build_parser(budgets_file).parse_args(argv)
    configure_logging(args.log_level)
    # PYDANTIC_SETTINGS_DEBUG makes pydantic-settings log raw settings, the API key among them,
    # at DEBUG, so its logger never logs below WARNING.
    logging.getLogger("pydantic_settings").setLevel(logging.WARNING)
    try:
        summary = asyncio.run(_run(args, ["goaly", *argv], registry))
    except KeyboardInterrupt:
        again = "again" if args.phase == "grade" else "with --resume"
        _say(f"interrupted; to continue the run, run the same command {again}")
        return INTERRUPTED
    except _USAGE_ERRORS as error:
        _say(f"error: {error}")
        return USAGE
    except Exception as error:
        _logger.exception("the run failed")
        _say(f"the run failed: {type(error).__name__}: {error}")
        return FAILED
    fields = {
        "run_id": summary.run_id,
        "phase": args.phase,
        "run_dir": str(args.out),
        "tasks": summary.tasks,
        "resumed": summary.resumed,
        "cached": summary.cached,
        "ran": summary.ran,
        "errors": dict(summary.errors),
    }
    sys.stdout.write(json.dumps(fields) + "\n")
    return 0


async def _run(args: argparse.Namespace, command_line: list[str], registry: Registry) -> RunSummary:
    if args.phase == "grade":
        return await _grade(args, registry)
    if args.split is None and args.instances is None:
        raise ValueError(f"--phase {args.phase} needs --split or --instances")
    if args.instances is not None and not args.instances.is_file():
        raise ValueError(f"no such task file: {args.instances}")
    if args.in_container and (args.phase != "solve" or args.inputs is None):
        raise ValueError("--in-container runs --phase solve only, with --inputs")
    if args.inputs is not None and not args.in_container:
        raise ValueError("--inputs is for --in-container")
    factory = registry.factory(args.bench, args.harness)  # First: the benchmark names the share.
    harness_name = args.harness or registry.harnesses(args.bench)[0]
    if args.budget_phase is None:
        raise ValueError(f"--phase {args.phase} needs --budget-phase, as it calls the LLM")
    share = args.budget_share or args.bench
    budgets = await asyncio.to_thread(load_budgets, args.budgets)
    llm_shares = budgets.share(args.budget_phase, share)
    files = RunFiles(args.out)
    # Locked before the harness opens, so that nothing it writes there, such as the LLM call log,
    # can meet another process's run.
    with await asyncio.to_thread(lock_run_dir, files) as lock:
        options = HarnessOptions(
            files=files,
            instances_file=args.instances,
            llm_shares=llm_shares,
            split=_split(args),
            budget_phase=args.budget_phase,
            in_container=args.in_container,
            inputs_dir=args.inputs,
        )
        async with factory(options) as harness:
            return await _run_harness(args, command_line, harness, harness_name, llm_shares, lock)


# The options a grade phase takes from the run's manifest instead.
_SOLVING_OPTIONS = (
    "harness",
    "split",
    "instances",
    "only",
    "seeds",
    "first_seed",
    "resume",
    "budget_phase",
    "budget_share",
    "max_tasks",
    "task_timeout",
    "run_id",
    "in_container",
    "inputs",
)


async def _grade(args: argparse.Namespace, registry: Registry) -> RunSummary:
    given = [name for name in _SOLVING_OPTIONS if getattr(args, name) not in (None, False)]
    if given:
        options = ", ".join("--" + name.replace("_", "-") for name in given)
        raise ValueError(f"--phase grade takes these from the run it grades: {options}")
    factory = registry.grading(args.bench)
    files = RunFiles(args.out)
    if not await asyncio.to_thread(files.manifest.exists):  # Before the lock creates --out.
        raise RunDirectoryError(f"{args.out} holds no run to grade")
    with await asyncio.to_thread(lock_run_dir, files) as lock:
        async with factory(GradingOptions(files=files)) as grading:
            if grading.benchmark_name != args.bench:
                raise RuntimeError(
                    f"the grading registered for {args.bench!r} is of {grading.benchmark_name!r}"
                )
            cache = _cache(args, grading.cache_namespace, f"the benchmark {args.bench!r}")
            return await grading.grader(cache=cache).grade(files.root, lock=lock)


def _cache(args: argparse.Namespace, namespace: str, owner: str) -> EvalCache | None:
    if args.cache is None:
        return None
    if not namespace:
        raise ValueError(f"{owner} names no grading namespace for --cache")
    return EvalCache(args.cache, namespace)


async def _run_harness(
    args: argparse.Namespace,
    command_line: list[str],
    harness: RunnableHarness,
    harness_name: str,
    llm_shares: dict[str, int],
    lock: RunLock,
) -> RunSummary:
    if harness.benchmark_name != args.bench:
        raise RuntimeError(
            f"the harness registered for {args.bench!r} runs the benchmark"
            f" {harness.benchmark_name!r}"
        )
    cache = _cache(args, harness.cache_namespace, f"the harness {harness_name!r}")
    first_seed = args.first_seed or 0
    seeds = list(range(first_seed, first_seed + (args.seeds or 1)))
    max_tasks = args.max_tasks or DEFAULT_MAX_TASKS
    # Every option that shapes the run: not where it is written, cached or logged.
    settings: dict[str, JsonValue] = {
        "benchmark": args.bench,
        "harness": harness_name,
        "split": args.split,
        "instances_sha256": None if args.instances is None else sha256_file(args.instances),
        "only": None if args.only is None else sorted(args.only),
        "seeds": [*seeds],
        "max_tasks": max_tasks,
        "task_timeout_s": args.task_timeout,
        "budget_phase": args.budget_phase,
        "budget_share": args.budget_share or args.bench,
        "llm_shares": dict(llm_shares),
    }
    manifest = await asyncio.to_thread(_manifest, args, command_line, harness, settings)
    runner = harness.runner(cache=cache, max_tasks=max_tasks, task_timeout_s=args.task_timeout)
    return await runner.run(
        lock.files.root,
        manifest,
        split=_split(args),
        seeds=seeds,
        instance_ids=args.only,
        resume=args.resume,
        lock=lock,
        grading=args.phase == "all",
    )


def _manifest(
    args: argparse.Namespace,
    command_line: list[str],
    harness: RunnableHarness,
    settings: dict[str, JsonValue],
) -> RunManifest:
    """The caller's part of the run's manifest; the runner adds the rest."""
    manifest = provenance(CORE_DIR)
    manifest.run_id = args.run_id or new_run_id()
    pack = harness.pack_dir
    manifest.harness_hash = core_hash() if pack is None else harness_hash(pack)
    manifest.config_hash = config_hash(settings)
    manifest.dataset_revision = harness.dataset.revision
    manifest.dataset_notes[:] = harness.dataset.notes
    manifest.gold_sha256 = harness.dataset.gold_sha256
    # The benchmark's own fields (Benchmark.manifest_fields): it knows its task file and grader.
    fields = harness.manifest_fields()
    if fields.dataset_revision:
        manifest.dataset_revision = fields.dataset_revision
    if fields.dataset_notes:
        manifest.dataset_notes[:] = fields.dataset_notes
    manifest.dataset_sha256 = fields.dataset_sha256
    manifest.environment_settings.update(fields.environment_settings)
    manifest.models.extend(harness.models)
    manifest.command_line[:] = command_line
    return manifest


def _split(args: argparse.Namespace) -> Split.ValueType:
    """The split the run's tasks come from: SPLIT_EXTERNAL for --instances."""
    return SPLIT_EXTERNAL if args.instances is not None else _SPLITS[args.split]


def _say(message: str) -> None:
    sys.stderr.write(f"goaly run: {message}\n")


def _count(text: str) -> int:
    """An argument that is a whole number, 1 or more."""
    value = _parsed(int, text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or more: {text}")
    return value


def _index(text: str) -> int:
    """An argument that is a whole number, 0 or more."""
    value = _parsed(int, text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be 0 or more: {text}")
    return value


def _seconds(text: str) -> float:
    """An argument that is a positive, finite number of seconds."""
    value = _parsed(float, text)
    if not 0 < value < math.inf:  # Also refuses NaN, which compares false.
        raise argparse.ArgumentTypeError(f"must be a positive number of seconds: {text}")
    return value


def _parsed[T: (int, float)](kind: type[T], text: str) -> T:
    try:
        return kind(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text}") from None
