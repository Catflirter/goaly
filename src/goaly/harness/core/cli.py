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
stdout the run's summary is one line of JSON. The logs go to stderr as JSON lines, each line of
a task's work with its run_id, instance_id and seed; a line that starts with "goaly run:",
which is not JSON, says why the command stopped, for a person to read.

The exit status is 0 once the run is done, whatever errors its tasks met, as those are in its
records; 1 if the run failed; 2 for a usage error, such as an unknown benchmark or a run
directory that holds another run; 130 if it was interrupted (SIGINT) and 143 if it was
terminated (SIGTERM). After either, --resume continues it, and --retry-errors with it also runs
again the tasks that ended with an error, such as an LLM outage.
"""

import argparse
import asyncio
import json
import logging
import math
import re
import signal
import sys
from collections.abc import Callable, Sequence
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
    read_manifest,
    sha256_file,
)
from goaly.harness.core.registry import (
    HarnessOptions,
    HostOptions,
    RegisteredHarness,
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
    UsageError,
    lock_run_dir,
)

__all__ = ["FAILED", "INTERRUPTED", "TERMINATED", "USAGE", "build_parser", "main"]

_logger = logging.getLogger(__name__)

# Exit statuses besides 0.
FAILED = 1
USAGE = 2  # As argparse exits.
INTERRUPTED = 130  # As a shell reports a process that SIGINT ended.
TERMINATED = 143  # As a shell reports a process that SIGTERM ended.

_SPLITS = {"train": SPLIT_TRAIN, "val": SPLIT_VAL, "heldout": SPLIT_HELDOUT}
# Errors that mean the command cannot run as asked; the message says what to change. Any other
# error, a ValueError included, fails the run, with its traceback logged.
_USAGE_ERRORS = (BudgetsError, RegistryError, RunDirectoryError, UsageError)
# A run ID names the run's records and files, so it is a plain name.
_RUN_ID = re.compile(r"[0-9A-Za-z_-]+")


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
        "--retry-errors",
        action="store_true",
        help="with --resume, run again the tasks whose last result is an error",
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
        help=(
            "the evaluation cache, kept by the harness's grading namespace; a task's evaluation"
            " is reused whatever its --task-timeout was"
        ),
    )
    run.add_argument(
        "--run-id", type=_run_id, metavar="ID", help="the run's ID (default: a new one)"
    )
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


def main(
    argv: Sequence[str],
    registry: Registry,
    *,
    budgets_file: Path,
    confine: Callable[[], None] = lambda: None,
) -> int:
    """Run the `goaly` command with ``argv``, its arguments, and return the exit status.

    ``registry`` holds the harnesses it can run. ``confine`` keeps the process to its CPUs,
    away from the database instances', before the harness opens; the composition root passes
    the benchmark's, and the default does nothing. argparse exits with status 2 by itself for
    arguments it refuses, and with 0 for --help.
    """
    args = build_parser(budgets_file).parse_args(argv)
    configure_logging(args.log_level)
    # PYDANTIC_SETTINGS_DEBUG makes pydantic-settings log raw settings, the API key among them,
    # at DEBUG, so its logger never logs below WARNING.
    logging.getLogger("pydantic_settings").setLevel(logging.WARNING)
    try:
        confine()
        summary = asyncio.run(_main(args, ["goaly", *argv], registry))
    except KeyboardInterrupt:
        again = "again" if args.phase == "grade" else "with --resume"
        _say(f"interrupted; to continue the run, run the same command {again}")
        return INTERRUPTED
    except _TerminatedError:
        again = "again" if args.phase == "grade" else "with --resume"
        _say(f"terminated; to continue the run, run the same command {again}")
        return TERMINATED
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


class _TerminatedError(Exception):
    """SIGTERM stopped the run, which cleaned up as for an interrupt."""


async def _main(
    args: argparse.Namespace, command_line: list[str], registry: Registry
) -> RunSummary:
    """Run the command, which SIGTERM cancels: its harness and its lock are then released as for
    Ctrl-C, and _TerminatedError is raised once it has stopped."""
    task = asyncio.current_task()
    if task is None:  # pragma: no cover - asyncio.run runs it in a task.
        raise RuntimeError("goaly run must run in an asyncio task")
    loop = asyncio.get_running_loop()
    terminated = False

    def terminate() -> None:
        nonlocal terminated
        if not terminated:  # A second SIGTERM must not cut the cleanup short.
            terminated = True
            task.cancel()

    loop.add_signal_handler(signal.SIGTERM, terminate)
    try:
        return await _run(args, command_line, registry)
    except asyncio.CancelledError:
        if terminated:
            raise _TerminatedError from None
        raise
    finally:
        loop.remove_signal_handler(signal.SIGTERM)


async def _run(args: argparse.Namespace, command_line: list[str], registry: Registry) -> RunSummary:
    if args.phase == "grade":
        return await _grade(args, registry)
    if args.retry_errors and not args.resume:
        raise UsageError("--retry-errors needs --resume: a new run has no errors to retry")
    if args.split is None and args.instances is None:
        raise UsageError(f"--phase {args.phase} needs --split or --instances")
    instances_sha256 = None
    if args.instances is not None:
        if not await asyncio.to_thread(args.instances.is_file):
            raise UsageError(f"no such task file: {args.instances}")
        # Hashed once, before the harness reads it.
        instances_sha256 = await asyncio.to_thread(sha256_file, args.instances)
    if args.in_container and (args.phase != "solve" or args.inputs is None):
        raise UsageError("--in-container runs --phase solve only, with --inputs")
    if args.inputs is not None and not args.in_container:
        raise UsageError("--inputs is for --in-container")
    registered = registry.harness(args.bench, args.harness)  # First: the share names the bench.
    if args.budget_phase is None:
        raise UsageError(f"--phase {args.phase} needs --budget-phase, as it calls the LLM")
    share = args.budget_share or args.bench
    budgets = await asyncio.to_thread(load_budgets, args.budgets)
    llm_shares = budgets.share(args.budget_phase, share)
    _logger.info("LLM shares of %s in the phase %s: %s", share, args.budget_phase, llm_shares)
    files = RunFiles(args.out)
    # Locked before the harness opens, so that nothing it writes there, such as the LLM call log,
    # can meet another process's run.
    with await asyncio.to_thread(lock_run_dir, files) as lock:
        await asyncio.to_thread(_preflight, args, files)
        options = HarnessOptions(
            files=files,
            instances_file=args.instances,
            llm_shares=llm_shares,
            split=_split(args),
            budget_phase=args.budget_phase,
            in_container=args.in_container,
            inputs_dir=args.inputs,
        )
        async with registered.factory(options) as harness:
            return await _run_harness(
                args, command_line, registered, harness, instances_sha256, lock
            )


# The options a grade phase takes from the run's manifest instead: flags, and options with a
# value, which may be 0.
_SOLVING_FLAGS = ("resume", "retry_errors", "in_container")
_SOLVING_OPTIONS = (
    "harness",
    "split",
    "instances",
    "only",
    "seeds",
    "first_seed",
    "budget_share",
    "max_tasks",
    "task_timeout",
    "run_id",
    "inputs",
)


async def _grade(args: argparse.Namespace, registry: Registry) -> RunSummary:
    given = [name for name in _SOLVING_OPTIONS if getattr(args, name) is not None]
    given += [name for name in _SOLVING_FLAGS if getattr(args, name)]
    if given:
        options = ", ".join("--" + name.replace("_", "-") for name in given)
        raise UsageError(f"--phase grade takes these from the run it grades: {options}")
    factory = registry.host(args.bench)
    files = RunFiles(args.out)
    if not await asyncio.to_thread(files.manifest.exists):  # Before the lock creates --out.
        raise RunDirectoryError(f"{args.out} holds no run to grade")
    with await asyncio.to_thread(lock_run_dir, files) as lock:
        stored = await asyncio.to_thread(read_manifest, files.manifest)
        opening = HostOptions(
            files=files, instances_file=None, split=stored.split, budget_phase=args.budget_phase
        )
        async with factory(opening) as host:
            if host.benchmark_name != args.bench:
                raise RuntimeError(
                    f"the host side registered for {args.bench!r} is of {host.benchmark_name!r}"
                )
            cache = _cache(args, host.cache_namespace, f"the benchmark {args.bench!r}")
            return await host.grader(cache=cache).grade(files.root, lock=lock)


def _cache(args: argparse.Namespace, namespace: str, owner: str) -> EvalCache | None:
    if args.cache is None:
        return None
    if not namespace:
        raise UsageError(f"{owner} names no grading namespace for --cache")
    return EvalCache(args.cache, namespace)


async def _run_harness(
    args: argparse.Namespace,
    command_line: list[str],
    registered: RegisteredHarness,
    harness: RunnableHarness,
    instances_sha256: str | None,
    lock: RunLock,
) -> RunSummary:
    if harness.benchmark_name != args.bench:
        raise RuntimeError(
            f"the harness registered for {args.bench!r} runs the benchmark"
            f" {harness.benchmark_name!r}"
        )
    cache = _cache(args, harness.cache_namespace, f"the harness {registered.name!r}")
    first_seed = args.first_seed or 0
    seeds = list(range(first_seed, first_seed + (args.seeds or 1)))
    max_tasks = args.max_tasks or DEFAULT_MAX_TASKS
    # Every option that shapes the run's results: the config hash, which a resume must match.
    # Not where it is written, cached or logged, nor how much of the gateway it may use at once,
    # such as --max-tasks and the LLM shares: a run may resume under other limits.
    settings: dict[str, JsonValue] = {
        "benchmark": args.bench,
        "harness": registered.name,
        "split": args.split,
        "instances_sha256": instances_sha256,
        "only": None if args.only is None else sorted(set(args.only)),
        "seeds": [*seeds],
        "task_timeout_s": args.task_timeout,
        "grading": harness.cache_namespace,  # A run never mixes two grading identities.
    }
    manifest = await asyncio.to_thread(_manifest, args, command_line, registered, harness, settings)
    runner = harness.runner(cache=cache, max_tasks=max_tasks, task_timeout_s=args.task_timeout)
    return await runner.run(
        lock.files.root,
        manifest,
        split=_split(args),
        seeds=seeds,
        instance_ids=args.only,
        resume=args.resume,
        retry_errors=args.retry_errors,
        lock=lock,
        grading=args.phase == "all",
    )


def _manifest(
    args: argparse.Namespace,
    command_line: list[str],
    registered: RegisteredHarness,
    harness: RunnableHarness,
    settings: dict[str, JsonValue],
) -> RunManifest:
    """The caller's part of the run's manifest; the runner adds the rest, the dataset's too."""
    manifest = provenance(CORE_DIR)
    manifest.run_id = args.run_id or new_run_id()
    pack = registered.pack_dir
    manifest.harness_hash = core_hash() if pack is None else harness_hash(pack)
    manifest.config_hash = config_hash(settings)
    manifest.gold_sha256 = harness.gold_sha256
    manifest.models.extend(registered.models)
    manifest.command_line[:] = command_line
    return manifest


def _split(args: argparse.Namespace) -> Split.ValueType:
    """The split of the run's tasks: SPLIT_EXTERNAL for --instances."""
    return SPLIT_EXTERNAL if args.instances is not None else _SPLITS[args.split]


def _preflight(args: argparse.Namespace, files: RunFiles) -> None:
    """Refuse, before the harness opens, a run directory the runner would refuse.

    The runner checks the rest, once the harness can tell it what the run is.
    """
    if not files.manifest.exists():
        return
    if not args.resume:
        stored = read_manifest(files.manifest)
        raise RunDirectoryError(
            f"{files.root} already holds the run {stored.run_id}: resume it with --resume,"
            " or choose another directory"
        )
    if args.run_id is not None and (stored := read_manifest(files.manifest)).run_id != args.run_id:
        raise UsageError(
            f"{files.root} holds the run {stored.run_id}, not {args.run_id}: a resumed run"
            " keeps its run ID"
        )


def _run_id(text: str) -> str:
    """An argument that is a plain name: letters, digits, '-' and '_'."""
    if not _RUN_ID.fullmatch(text):
        raise argparse.ArgumentTypeError(f"must be letters, digits, '-' and '_' only: {text!r}")
    return text


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
