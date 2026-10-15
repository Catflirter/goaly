"""Run the Effi-SQL grader's one-time setup on this machine (spec §4.1, §6.2).

    uv run python scripts/effisql/setup_run.py [--phase PHASE] [--ids 1,2,...] [--write-exclusions]

Phases: ``measure`` on all 5 instances (base cache, identity τ under load, data and gold
checks; it also retries a base the server ran out of resources on, once, and re-measures a base
timeout under the server's statement_timeout), ``recheck`` on the 3 idle grading instances (the
marginal golds and the identity outliers again, in official mode), ``official`` (the kept val
golds and identity in official mode), ``negatives`` (the fixed sample of 10), and ``summary``;
``all`` runs them in that order. Each phase resumes from the records under ``--records``, and
retries any task a grader failure left unrecorded. The summary is written next to the records as
``summary.json`` and ``summary.md``.

``--write-exclusions`` writes the exclusion list, the measured exclusions plus the reviewed ones
in ``effisql_exclusions_manual.json``, to the committed file. It refuses a run over a subset of
the tasks, a run with tasks still unrecorded or with grader failures to rerun, and a run whose
stop rule triggered; it also refuses one whose self-checks failed, unless
``--accept-failed-checks`` is given after review.

The script reads the full dataset, gold included, so only the setup runs it (spec §8.5). Its
records hold grades and measurements, never gold SQL. It confines itself and its workers to the
CPUs our Python processes use, away from the PostgreSQL instances' cores.
"""

import argparse
import asyncio
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from goaly.bench.effisql.data import DatasetError, default_cache_dir, fetch_dataset, load_instances
from goaly.bench.effisql.grading import BaseCache, Mode
from goaly.bench.effisql.grading_pool import (
    GRADING_INSTANCES,
    SETUP_INSTANCES,
    GradingInstance,
    GradingPool,
)
from goaly.bench.effisql.grading_service import ProcessRunner, SlotRunner, WorkerSpec
from goaly.bench.effisql.pg_settings import PgEnvConfig, write_private
from goaly.bench.effisql.setup_run import (
    IDENTITY_TOLERANCE,
    MANUAL_EXCLUSIONS_PATH,
    Phase,
    RecordStore,
    SetupSummary,
    dump_exclusions,
    load_manual_exclusions,
    run_phase,
    summarize,
    to_markdown,
)
from goaly.bench.effisql.splits import EXCLUSIONS_PATH, SPLIT_DB, load_splits

PYTHON_CPUS = {0, 1, 12, 13}  # Physical cores 0-1 (spec §6.2).
PHASES: tuple[Phase, ...] = ("measure", "recheck", "official", "negatives")
# Each phase's instances: the measure phase on all 5, the others with the solving ones idle.
PHASE_INSTANCES: dict[Phase, tuple[GradingInstance, ...]] = {
    "measure": SETUP_INSTANCES,
    "recheck": GRADING_INSTANCES,
    "official": GRADING_INSTANCES,
    "negatives": GRADING_INSTANCES,
}
EXIT_STOP = 2  # More than 10% of tasks excluded.
EXIT_FAILED_CHECKS = 3  # A self-check failed.
EXIT_REFUSED = 4  # --write-exclusions refused to write.

_log = logging.getLogger("setup_run")


def _run(phase: Phase, args: argparse.Namespace, val: Sequence[str]) -> int:
    runners: list[SlotRunner] = [ProcessRunner(WorkerSpec(i)) for i in PHASE_INSTANCES[phase]]
    try:
        return asyncio.run(
            run_phase(
                phase,
                GradingPool(runners),
                args.tasks,
                RecordStore(args.records),
                BaseCache(args.cache_root / Mode.DEV.value),
                val=val,
                manual=[e.instance_id for e in load_manual_exclusions(args.manual_exclusions)],
            )
        )
    finally:
        for runner in runners:
            runner.close()


def _refusal(args: argparse.Namespace, summary: SetupSummary) -> str | None:
    if args.ids:
        return "a run over --ids covers only some tasks"
    if summary.missing:
        return f"{len(summary.missing)} tasks have no record yet"
    if summary.grader_errors:
        return f"{len(summary.grader_errors)} tasks have grader failures to rerun"
    if summary.stop:
        return "the stop rule triggered"
    if not summary.passed and not args.accept_failed_checks:
        return "a self-check failed; review it, then pass --accept-failed-checks"
    return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Effi-SQL grader's setup.")
    data_root = PgEnvConfig().data_root
    parser.add_argument("--phase", choices=[*PHASES, "summary", "all"], default="all")
    parser.add_argument("--ids", help="only these task IDs, comma-separated")
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    parser.add_argument("--cache-root", type=Path, default=data_root / "grader")
    parser.add_argument("--records", type=Path, default=data_root / "setup" / "records")
    parser.add_argument("--identity-tolerance", type=float, default=IDENTITY_TOLERANCE)
    parser.add_argument("--manual-exclusions", type=Path, default=MANUAL_EXCLUSIONS_PATH)
    parser.add_argument("--write-exclusions", action="store_true")
    parser.add_argument("--accept-failed-checks", action="store_true")
    parser.add_argument("--exclusions-out", type=Path, default=EXCLUSIONS_PATH)
    args = parser.parse_args(argv)
    _configure_logging()
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, PYTHON_CPUS & os.sched_getaffinity(0) or os.sched_getaffinity(0))

    try:
        tasks = [t for t in load_instances(fetch_dataset(args.cache_dir)) if t.db == SPLIT_DB]
    except (DatasetError, OSError) as error:
        print(f"setup_run: {error}", file=sys.stderr)  # noqa: T201
        return 1
    if args.ids:
        wanted = set(args.ids.split(","))
        tasks = [t for t in tasks if t.instance_id in wanted]
    args.tasks = tasks
    splits = load_splits()
    val = splits.val
    split_of = {
        task: name
        for name, ids in (("train", splits.train), ("val", splits.val), ("heldout", splits.heldout))
        for task in ids
    }

    phases = PHASES if args.phase == "all" else () if args.phase == "summary" else (args.phase,)
    for phase in phases:
        ran = _run(phase, args, val)
        _log.info("phase %s done: %d tasks", phase, ran)

    summary = summarize(
        [t.instance_id for t in tasks],
        RecordStore(args.records).load(),
        val=val,
        tasks={t.instance_id: t for t in tasks},
        manual=load_manual_exclusions(args.manual_exclusions),
        identity_tolerance=args.identity_tolerance,
        splits=split_of,
    )
    out = args.records.parent
    write_private(out / "summary.json", summary.model_dump_json(indent=2) + "\n")
    markdown = to_markdown(summary)
    write_private(out / "summary.md", markdown)
    print(markdown)  # noqa: T201
    if args.write_exclusions:
        refusal = _refusal(args, summary)
        if refusal is not None:
            print(f"setup_run: not writing the exclusions: {refusal}", file=sys.stderr)  # noqa: T201
            return EXIT_REFUSED
        args.exclusions_out.write_text(dump_exclusions(summary.exclusions), encoding="utf-8")
    if summary.stop:
        print("setup_run: more than 10% of tasks excluded: stop and investigate", file=sys.stderr)  # noqa: T201
        return EXIT_STOP
    return 0 if summary.passed else EXIT_FAILED_CHECKS


def _configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s %(extra)s"))
    handler.addFilter(_ExtraFields())
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


class _ExtraFields(logging.Filter):
    """Show a record's extra fields, such as the task, after its message."""

    _STANDARD = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
        "message",
        "asctime",
        "extra",
    }

    def filter(self, record: logging.LogRecord) -> bool:
        extras = {k: v for k, v in record.__dict__.items() if k not in self._STANDARD}
        record.extra = extras or ""
        return True


if __name__ == "__main__":
    sys.exit(main())
