"""`goaly run`: run a registered harness on a benchmark's tasks, and record the run (spec §6.4).

For example, a validation run during RSI, and the organizers' run of their own task file:

    goaly run --bench effisql --split val --budget-phase rsi --budget-share background --out r/v
    goaly run --bench effisql --instances heldout.jsonl --phase solve --out runs/heldout

A run has a solve phase, which runs the harness's pack, candidate code, and a grade phase,
which runs the benchmark's grader alone (spec §6.1, V27). By default, --phase all, both run
isolated: this process, the trusted host, plans the run, takes what the evaluation cache holds,
and has the benchmark's launcher run the solve phase of the rest in the solve container. That
runs `goaly run --phase solve --in-container`, which writes only the run's solve directory.
Then the host grades every task, trusting nothing the container wrote but its predictions.
--phase solve stops after the container, and --phase grade, later, grades on the host. The host
never imports the pack. --harness-checkout runs the pack of another checkout, such as a
snapshot of a candidate's commit, which the host hashes and the container runs.

--in-process runs the whole run in this process, the pack included, as tests and local
debugging do; no real experiment does. It is refused with a real gateway key unless
--unsafe-in-process is given too. Every path comes from the options; nothing but the LLM
settings is read from the home directory.

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
import ipaddress
import json
import logging
import math
import re
import signal
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from urllib.parse import urlsplit

from goaly.contracts.gen.manifest_pb2 import ModelSettings, RunManifest
from goaly.contracts.gen.task_pb2 import (
    SPLIT_EXTERNAL,
    SPLIT_HELDOUT,
    SPLIT_TRAIN,
    SPLIT_VAL,
    Split,
)
from goaly.contracts.interfaces import JsonValue
from goaly.harness.core.eval_cache import EvalCache
from goaly.harness.core.grading import SolveOptions
from goaly.harness.core.isolation import read_solve_tasks
from goaly.harness.core.llm_budgets import BudgetsError, load_budgets
from goaly.harness.core.logs import configure_logging
from goaly.harness.core.manifest import (
    CORE_DIR,
    CORE_IN_CHECKOUT,
    REPO_ROOT,
    checkout_harness_hash,
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
)
from goaly.harness.core.runner import (
    DEFAULT_MAX_TASKS,
    RunDirectoryError,
    RunFiles,
    RunSummary,
    UsageError,
    lock_run_dir,
)
from goaly.harness.core.settings import SettingsError, load_settings

__all__ = ["FAILED", "INTERRUPTED", "TERMINATED", "USAGE", "build_parser", "main", "run"]

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
# The defaults of the options a grade phase refuses, which argparse leaves None.
_DEFAULT_SEEDS = 1
_DEFAULT_FIRST_SEED = 0
# The budget phase of an organizers' run (--instances) that names none: the final evaluation's.
_ORGANIZERS_BUDGET_PHASE = "final"


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
    run.add_argument(
        "--harness-checkout",
        type=Path,
        metavar="DIR",
        help="run the harness's pack from this checkout, such as a candidate's snapshot",
    )
    tasks = run.add_mutually_exclusive_group()
    tasks.add_argument("--split", choices=sorted(_SPLITS), help="run one of our splits")
    tasks.add_argument(
        "--instances",
        type=Path,
        metavar="FILE",
        help="run the tasks of an official-format task file, such as the organizers': not graded",
    )
    run.add_argument("--only", nargs="+", metavar="ID", help="run only these instances")
    run.add_argument(
        "--seeds",
        type=_count,
        metavar="N",
        help=f"how many replicates (default {_DEFAULT_SEEDS})",
    )
    run.add_argument(
        "--first-seed",
        type=_index,
        metavar="K",
        help=(
            f"the first replicate's index (default {_DEFAULT_FIRST_SEED}); a confirmation run"
            " takes a new one"
        ),
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
        help=(
            "the schedule's phase whose LLM concurrency shares apply: baseline, rsi or final"
            f" (default {_ORGANIZERS_BUDGET_PHASE} with --instances)"
        ),
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
            "the evaluation cache, kept by the benchmark's grading namespace; a task's evaluation"
            " is reused whatever its --task-timeout was"
        ),
    )
    run.add_argument(
        "--grading-priority",
        metavar="NAME",
        help="the priority of this run's grading, such as smoke (default: by phase and split)",
    )
    run.add_argument(
        "--grading-instances",
        choices=["grading", "all"],
        default="grading",
        help="grade on the grading instances only, or on all, while nothing solves (default)",
    )
    run.add_argument(
        "--run-id", type=_run_id, metavar="ID", help="the run's ID (default: a new one)"
    )
    run.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    modes = run.add_argument_group("how the solve phase runs")
    modes.add_argument(
        "--in-process",
        action="store_true",
        help="run the pack in this process, for tests and local debugging only",
    )
    modes.add_argument(
        "--unsafe-in-process",
        action="store_true",
        help="with --in-process, run it even though this process holds a real gateway key",
    )
    modes.add_argument(
        "--in-container",
        action="store_true",
        help="be the solve phase in the solve container, as the benchmark's launcher runs it",
    )
    modes.add_argument(
        "--tasks", type=Path, metavar="FILE", help="in the container: the host's tasks file"
    )
    modes.add_argument(
        "--inputs", type=Path, metavar="DIR", help="in the container: the staged inputs"
    )
    modes.add_argument(
        "--llm-share",
        type=_share,
        action="append",
        metavar="MODEL=N",
        help="in the container: this run's share of a model's concurrency, from the host",
    )
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
    again = "again" if args.phase == "grade" else "with --resume"
    try:
        confine()
        summary = asyncio.run(_main(args, ["goaly", *argv], registry))
    except KeyboardInterrupt:
        _say(f"interrupted; to continue the run, run the same command {again}")
        return INTERRUPTED
    except _TerminatedError:
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


async def run(argv: Sequence[str], registry: Registry, *, budgets_file: Path) -> RunSummary:
    """Run `goaly` with ``argv`` in the running event loop, and return the run's summary.

    It is what main() runs, without the logging setup, the signals and the exit statuses: a
    usage error raises UsageError or another of the errors main() exits 2 for.
    """
    args = build_parser(budgets_file).parse_args(argv)
    return await _run(args, ["goaly", *argv], registry)


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

    try:
        loop.add_signal_handler(signal.SIGTERM, terminate)
    except (ValueError, RuntimeError):  # Not the main thread, as when a test is the container.
        handled = False
    else:
        handled = True
    try:
        return await _run(args, command_line, registry)
    except asyncio.CancelledError:
        if terminated:
            raise _TerminatedError from None
        raise
    finally:
        if handled:
            loop.remove_signal_handler(signal.SIGTERM)


async def _run(args: argparse.Namespace, command_line: list[str], registry: Registry) -> RunSummary:
    _check_mode(args)
    if args.in_container:
        return await _solve_in_container(args, command_line, registry)
    if args.phase == "grade":
        return await _grade(args, registry)
    if args.retry_errors and not args.resume:
        raise UsageError("--retry-errors needs --resume: a new run has no errors to retry")
    if args.split is None and args.instances is None:
        raise UsageError(f"--phase {args.phase} needs --split or --instances")
    if args.in_process:
        return await _run_in_process(args, command_line, registry)
    return await _run_isolated(args, command_line, registry)


# The options each mode refuses, by their names in args: they make no sense there.
_HOST_ONLY = ("cache", "harness_checkout", "grading_priority", "only", "seeds", "first_seed")
_CONTAINER_ONLY = ("tasks", "inputs", "llm_share")
_SOLVING_FLAGS = ("resume", "in_process", "unsafe_in_process")
_SOLVING_OPTIONS = (
    "harness",
    "harness_checkout",
    "split",
    "instances",
    "only",
    "seeds",
    "first_seed",
    "budget_share",
    "max_tasks",
    "task_timeout",
    "run_id",
)


def _check_mode(args: argparse.Namespace) -> None:
    """Refuse the options the mode the command asks for cannot take."""
    if args.unsafe_in_process and not args.in_process:
        raise UsageError("--unsafe-in-process goes with --in-process")
    if args.in_container:
        if args.phase != "solve" or args.in_process:
            raise UsageError("--in-container runs only the solve phase: --phase solve")
        given = _given(args, (*_HOST_ONLY, "budget_phase", "budget_share"))
        if args.grading_instances != "grading":
            given.append("grading_instances")
        if given:
            raise UsageError(f"the solve container takes none of: {_options(given)}")
        missing = [name for name in ("tasks", "inputs", "run_id") if getattr(args, name) is None]
        if missing:
            raise UsageError(f"the solve container needs: {_options(missing)}")
        return
    if _given(args, _CONTAINER_ONLY):
        raise UsageError(
            f"only the solve container takes: {_options(_given(args, _CONTAINER_ONLY))}"
        )
    if args.in_process:
        if args.phase != "all":
            raise UsageError("--in-process runs both phases at once: --phase all")
        if args.harness_checkout is not None:
            raise UsageError("--in-process imports the pack here: it cannot run another checkout's")
        _refuse_the_real_key(args)
    if args.phase == "grade":
        given = _given(args, _SOLVING_OPTIONS) + [
            name for name in _SOLVING_FLAGS if getattr(args, name)
        ]
        if given:
            raise UsageError(f"--phase grade takes these from the run it grades: {_options(given)}")


def _given(args: argparse.Namespace, names: Sequence[str]) -> list[str]:
    return [name for name in names if getattr(args, name) is not None]


def _options(names: Sequence[str]) -> str:
    return ", ".join("--" + name.replace("_", "-") for name in names)


def _refuse_the_real_key(args: argparse.Namespace) -> None:
    """Refuse --in-process with the settings of a real gateway, unless --unsafe-in-process:
    pack code in this process could read its key (spec §6.1, V27)."""
    if args.unsafe_in_process:
        return
    try:
        settings = load_settings()
    except SettingsError:
        return  # No key at all, as with the fake LLM.
    if not _is_loopback(urlsplit(settings.base_url).hostname or ""):
        raise UsageError(
            "--in-process runs the pack in the process that holds the gateway key: run the"
            " isolated pipeline, or add --unsafe-in-process to debug locally"
        )


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


# The isolated pipeline, on the host.


async def _run_isolated(
    args: argparse.Namespace, command_line: list[str], registry: Registry
) -> RunSummary:
    instances_sha256 = await _task_file_hash(args)
    registered = registry.harness(args.bench, args.harness)
    llm_shares = await _shares(args)
    checkout, pack_dir, hashed = await asyncio.to_thread(
        _harness_identity, registered, args.harness_checkout
    )
    files = RunFiles(args.out)
    # Locked before anything opens, so that nothing written there meets another process's run.
    with await asyncio.to_thread(lock_run_dir, files) as lock:
        await asyncio.to_thread(_preflight, args, files)
        opening = HostOptions(
            files=files,
            instances_file=args.instances,
            split=_split(args),
            budget_phase=args.budget_phase,
            grading_priority=args.grading_priority,
            grading_instances=args.grading_instances,
        )
        async with registry.host(args.bench)(opening) as host:
            _check_benchmark(args, host.benchmark_name, "host side")
            cache = _cache(args, host.cache_namespace, f"the benchmark {args.bench!r}")
            settings = _settings(args, registered, instances_sha256, host.cache_namespace)
            manifest = await asyncio.to_thread(
                _manifest, args, command_line, registered.models, hashed, host.gold_sha256, settings
            )
            solve = SolveOptions(
                harness=registered.name,
                checkout=checkout,
                pack_dir=pack_dir,
                instances_file=args.instances,
                llm_shares=llm_shares,
                max_tasks=args.max_tasks or DEFAULT_MAX_TASKS,
                task_timeout_s=args.task_timeout,
                log_level=args.log_level,
            )
            return await host.host_runner(cache=cache).run(
                files.root,
                manifest,
                split=_split(args),
                seeds=_seeds(args),
                instance_ids=args.only,
                solve=solve,
                resume=args.resume,
                retry_errors=args.retry_errors,
                lock=lock,
                grading=args.phase == "all",
            )


def _harness_identity(
    registered: RegisteredHarness, checkout: Path | None
) -> tuple[Path, Path | None, str]:
    """The checkout the container runs, the pack's directory in it, and their harness hash."""
    if checkout is None:
        pack = registered.pack_dir
        return REPO_ROOT, pack, core_hash() if pack is None else harness_hash(pack)
    if not checkout.is_dir():
        raise UsageError(f"no such checkout: {checkout}")
    if registered.pack_dir is None:
        raise UsageError(f"the harness {registered.name!r} has no pack to take from a checkout")
    try:
        pack = registered.pack_dir.resolve().relative_to(REPO_ROOT)
    except ValueError:
        raise UsageError(
            f"the pack of {registered.name!r} is not in this checkout, so another cannot hold it"
        ) from None
    checkout = checkout.resolve()
    for needed in (checkout / pack, checkout / CORE_IN_CHECKOUT):
        if not needed.is_dir():
            raise UsageError(f"the checkout {checkout} has no {needed.relative_to(checkout)}")
    try:
        hashed = checkout_harness_hash(checkout, pack)
    except ValueError as error:  # Such as a symbolic link in it.
        raise UsageError(f"the checkout {checkout} cannot be hashed: {error}") from error
    return checkout, checkout / pack, hashed


async def _grade(args: argparse.Namespace, registry: Registry) -> RunSummary:
    factory = registry.host(args.bench)
    files = RunFiles(args.out)
    if not await asyncio.to_thread(files.manifest.exists):  # Before the lock creates --out.
        raise RunDirectoryError(f"{args.out} holds no run to grade")
    with await asyncio.to_thread(lock_run_dir, files) as lock:
        stored = await asyncio.to_thread(read_manifest, files.manifest)
        opening = HostOptions(
            files=files,
            instances_file=None,
            split=stored.split,
            budget_phase=args.budget_phase,
            grading_priority=args.grading_priority,
            grading_instances=args.grading_instances,
        )
        async with factory(opening) as host:
            _check_benchmark(args, host.benchmark_name, "host side")
            cache = _cache(args, host.cache_namespace, f"the benchmark {args.bench!r}")
            return await host.host_runner(cache=cache).grade(
                files.root, lock=lock, retry_errors=args.retry_errors
            )


# The one-process mode, and the solve container.


async def _run_in_process(
    args: argparse.Namespace, command_line: list[str], registry: Registry
) -> RunSummary:
    instances_sha256 = await _task_file_hash(args)
    registered = registry.harness(args.bench, args.harness)
    llm_shares = await _shares(args)
    files = RunFiles(args.out)
    with await asyncio.to_thread(lock_run_dir, files) as lock:
        await asyncio.to_thread(_preflight, args, files)
        options = HarnessOptions(
            files=files,
            instances_file=args.instances,
            llm_shares=llm_shares,
            split=_split(args),
            budget_phase=args.budget_phase,
            grading_priority=args.grading_priority,
            grading_instances=args.grading_instances,
        )
        async with registered.factory(options) as harness:
            _check_benchmark(args, harness.benchmark_name, "harness")
            cache = _cache(args, harness.cache_namespace, f"the harness {registered.name!r}")
            settings = _settings(args, registered, instances_sha256, harness.cache_namespace)
            settings["isolated"] = False
            hashed = await asyncio.to_thread(_identity_hash, registered)
            manifest = await asyncio.to_thread(
                _manifest,
                args,
                command_line,
                registered.models,
                hashed,
                harness.gold_sha256,
                settings,
            )
            runner = harness.runner(
                cache=cache,
                max_tasks=args.max_tasks or DEFAULT_MAX_TASKS,
                task_timeout_s=args.task_timeout,
            )
            return await runner.run(
                files.root,
                manifest,
                split=_split(args),
                seeds=_seeds(args),
                instance_ids=args.only,
                resume=args.resume,
                retry_errors=args.retry_errors,
                lock=lock,
            )


async def _solve_in_container(
    args: argparse.Namespace, command_line: list[str], registry: Registry
) -> RunSummary:
    if args.split is None and args.instances is None:
        raise UsageError("the solve container needs --split or --instances")
    if args.retry_errors and not args.resume:
        raise UsageError("--retry-errors needs --resume: a new run has no errors to retry")
    solve = await asyncio.to_thread(read_solve_tasks, args.tasks)
    instances_sha256 = await _task_file_hash(args)
    registered = registry.harness(args.bench, args.harness)
    files = RunFiles(args.out)
    with await asyncio.to_thread(lock_run_dir, files) as lock:
        await asyncio.to_thread(_preflight, args, files)
        options = HarnessOptions(
            files=files,
            instances_file=args.instances,
            llm_shares=dict(args.llm_share or ()),
            split=_split(args),
            budget_phase=None,
            in_container=True,
            inputs_dir=args.inputs,
        )
        async with registered.factory(options) as harness:
            _check_benchmark(args, harness.benchmark_name, "harness")
            settings: dict[str, JsonValue] = {
                "benchmark": args.bench,
                "harness": registered.name,
                "split": args.split,
                "instances_sha256": instances_sha256,
                "instance_ids": list(solve.instance_ids),
                "seeds": list(solve.seeds),
                "task_timeout_s": args.task_timeout,
                "solve_phase": True,
            }
            hashed = await asyncio.to_thread(_identity_hash, registered)
            manifest = await asyncio.to_thread(
                _manifest,
                args,
                command_line,
                registered.models,
                hashed,
                harness.gold_sha256,
                settings,
            )
            runner = harness.runner(
                cache=None,
                max_tasks=args.max_tasks or DEFAULT_MAX_TASKS,
                task_timeout_s=args.task_timeout,
            )
            return await runner.run(
                files.root,
                manifest,
                split=_split(args),
                seeds=list(solve.seeds),
                instance_ids=list(solve.instance_ids),
                resume=args.resume,
                retry_errors=args.retry_errors,
                lock=lock,
                grading=False,
                tasks=solve.tasks,
            )


# Helpers.


async def _task_file_hash(args: argparse.Namespace) -> str | None:
    """The SHA-256 of --instances, hashed once, before anything reads it; None without one."""
    if args.instances is None:
        return None
    if not await asyncio.to_thread(args.instances.is_file):
        raise UsageError(f"no such task file: {args.instances}")
    return await asyncio.to_thread(sha256_file, args.instances)


async def _shares(args: argparse.Namespace) -> dict[str, int]:
    """This process's LLM shares, from its budget phase and share; logged, as no hash has them."""
    if args.budget_phase is None and args.instances is not None:
        args.budget_phase = _ORGANIZERS_BUDGET_PHASE
    if args.budget_phase is None:
        raise UsageError(f"--phase {args.phase} needs --budget-phase, as it calls the LLM")
    share = args.budget_share or args.bench
    budgets = await asyncio.to_thread(load_budgets, args.budgets)
    llm_shares = budgets.share(args.budget_phase, share)
    _logger.info("LLM shares of %s in the phase %s: %s", share, args.budget_phase, llm_shares)
    return llm_shares


def _settings(
    args: argparse.Namespace,
    registered: RegisteredHarness,
    instances_sha256: str | None,
    grading: str,
) -> dict[str, JsonValue]:
    """Every option that shapes the run's results: the config hash, which a resume must match.

    Not where the run is written, cached or logged, nor how much of the gateway it may use at
    once, such as --max-tasks and the LLM shares: a run may resume under other limits.
    """
    return {
        "benchmark": args.bench,
        "harness": registered.name,
        "split": args.split,
        "instances_sha256": instances_sha256,
        "only": None if args.only is None else sorted(set(args.only)),
        "seeds": _seeds(args),
        "task_timeout_s": args.task_timeout,
        "grading": grading,  # A run never mixes two grading identities.
        "isolated": True,  # Nor the records of the two modes.
    }


def _identity_hash(registered: RegisteredHarness) -> str:
    return core_hash() if registered.pack_dir is None else harness_hash(registered.pack_dir)


def _manifest(
    args: argparse.Namespace,
    command_line: list[str],
    models: Sequence[ModelSettings],
    hashed: str,
    gold_sha256: str,
    settings: dict[str, JsonValue],
) -> RunManifest:
    """The caller's part of the run's manifest; the runner adds the rest, the dataset's too."""
    manifest = provenance(CORE_DIR)
    manifest.run_id = args.run_id or new_run_id()
    manifest.harness_hash = hashed
    manifest.config_hash = config_hash(settings)
    manifest.gold_sha256 = gold_sha256
    manifest.models.extend(models)
    manifest.command_line[:] = command_line
    return manifest


def _check_benchmark(args: argparse.Namespace, name: str, what: str) -> None:
    if name != args.bench:
        raise RuntimeError(f"the {what} registered for {args.bench!r} is of {name!r}")


def _cache(args: argparse.Namespace, namespace: str, owner: str) -> EvalCache | None:
    if args.cache is None:
        return None
    if not namespace:
        raise UsageError(f"{owner} names no grading namespace for --cache")
    return EvalCache(args.cache, namespace)


def _seeds(args: argparse.Namespace) -> list[int]:
    first = _DEFAULT_FIRST_SEED if args.first_seed is None else args.first_seed
    return list(range(first, first + (_DEFAULT_SEEDS if args.seeds is None else args.seeds)))


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


def _say(message: str) -> None:
    sys.stderr.write(f"goaly run: {message}\n")


def _run_id(text: str) -> str:
    """An argument that is a plain name: letters, digits, '-' and '_'."""
    if not _RUN_ID.fullmatch(text):
        raise argparse.ArgumentTypeError(f"must be letters, digits, '-' and '_' only: {text!r}")
    return text


def _share(text: str) -> tuple[str, int]:
    """An argument MODEL=N: a model's share, a whole number, 1 or more."""
    model, equals, count = text.partition("=")
    if not equals or not model:
        raise argparse.ArgumentTypeError(f"must be MODEL=N: {text}")
    return model, _count(count)


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
