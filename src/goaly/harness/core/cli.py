"""`goaly run`: run a registered harness on a benchmark's tasks, and record the run (spec §6.4).

For example, a validation run during RSI, and a run on the organizers' own task file:

    goaly run --bench effisql --split val --phase rsi --share background --out runs/val-2
    goaly run --bench effisql --instances heldout.jsonl --phase final --out runs/heldout

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
from goaly.contracts.gen.task_pb2 import SPLIT_EXTERNAL, SPLIT_HELDOUT, SPLIT_TRAIN, SPLIT_VAL
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
from goaly.harness.core.registry import HarnessOptions, Registry, RegistryError, RunnableHarness
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
    run.add_argument("--harness", metavar="NAME", help="the harness, if the benchmark has several")
    tasks = run.add_mutually_exclusive_group(required=True)
    tasks.add_argument("--split", choices=sorted(_SPLITS), help="run one of our splits")
    tasks.add_argument(
        "--instances",
        type=Path,
        metavar="FILE",
        help="run the tasks of an official-format task file, such as the organizers': not graded",
    )
    run.add_argument("--only", nargs="+", metavar="ID", help="run only these instances")
    run.add_argument(
        "--seeds", type=_count, default=1, metavar="N", help="how many replicates (default 1)"
    )
    run.add_argument(
        "--first-seed",
        type=_index,
        default=0,
        metavar="K",
        help="the first replicate's index (default 0); a confirmation run takes a new one",
    )
    run.add_argument("--out", type=Path, required=True, metavar="DIR", help="the run directory")
    run.add_argument(
        "--resume", action="store_true", help="continue the run in --out, if there is one"
    )
    run.add_argument(
        "--phase",
        required=True,
        metavar="NAME",
        help="the phase whose LLM concurrency shares apply, such as baseline, rsi or final",
    )
    run.add_argument(
        "--share", metavar="NAME", help="this process's share in the phase (default: --bench)"
    )
    run.add_argument(
        "--budgets", type=Path, default=budgets_file, metavar="FILE", help="the budgets file"
    )
    run.add_argument(
        "--max-tasks",
        type=_count,
        default=DEFAULT_MAX_TASKS,
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
        _say("interrupted; to continue the run, run the same command with --resume")
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
    if args.instances is not None and not args.instances.is_file():
        raise ValueError(f"no such task file: {args.instances}")
    factory = registry.factory(args.bench, args.harness)  # First: the benchmark names the share.
    harness_name = args.harness or registry.harnesses(args.bench)[0]
    share = args.share or args.bench
    llm_shares = (await asyncio.to_thread(load_budgets, args.budgets)).share(args.phase, share)
    files = RunFiles(args.out)
    # Locked before the harness opens, so that nothing it writes there, such as the LLM call log,
    # can meet another process's run.
    with await asyncio.to_thread(lock_run_dir, files) as lock:
        options = HarnessOptions(files=files, instances_file=args.instances, llm_shares=llm_shares)
        async with factory(options) as harness:
            return await _run_harness(args, command_line, harness, harness_name, llm_shares, lock)


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
    cache = None
    if args.cache is not None:
        if not harness.cache_namespace:
            raise ValueError(f"the harness {harness_name!r} names no grading namespace for --cache")
        cache = EvalCache(args.cache, harness.cache_namespace)
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    # Every option that shapes the run: not where it is written, cached or logged.
    settings: dict[str, JsonValue] = {
        "benchmark": args.bench,
        "harness": harness_name,
        "split": args.split,
        "instances_sha256": None if args.instances is None else sha256_file(args.instances),
        "only": None if args.only is None else sorted(args.only),
        "seeds": [*seeds],
        "max_tasks": args.max_tasks,
        "task_timeout_s": args.task_timeout,
        "phase": args.phase,
        "share": args.share or args.bench,
        "llm_shares": dict(llm_shares),
    }
    manifest = await asyncio.to_thread(_manifest, args, command_line, harness, settings)
    runner = harness.runner(cache=cache, max_tasks=args.max_tasks, task_timeout_s=args.task_timeout)
    return await runner.run(
        lock.files.root,
        manifest,
        split=SPLIT_EXTERNAL if args.instances is not None else _SPLITS[args.split],
        seeds=seeds,
        instance_ids=args.only,
        resume=args.resume,
        lock=lock,
    )


def _manifest(
    args: argparse.Namespace,
    command_line: list[str],
    harness: RunnableHarness,
    settings: dict[str, JsonValue],
) -> RunManifest:
    """The caller's part of the run's manifest; the runner adds the rest, the dataset's too."""
    manifest = provenance(CORE_DIR)
    manifest.run_id = args.run_id or new_run_id()
    pack = harness.pack_dir
    manifest.harness_hash = core_hash() if pack is None else harness_hash(pack)
    manifest.config_hash = config_hash(settings)
    manifest.gold_sha256 = harness.gold_sha256
    manifest.models.extend(harness.models)
    manifest.command_line[:] = command_line
    return manifest


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
