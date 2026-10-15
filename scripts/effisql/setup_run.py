"""Run the Effi-SQL grader's one-time setup on this machine (spec §4.1, §6.2).

    uv run python scripts/effisql/setup_run.py [--phase PHASE] [--ids 1,2,...] [--write-exclusions]

Phases: ``measure`` on all 5 instances (base cache, identity τ under load, data and gold
checks), ``official`` on the 3 grading instances (val golds and identity in official mode, with
the solving instances idle), ``negatives`` (the fixed sample of 10), and ``summary``; ``all``
runs them in that order. Each phase resumes from the records under ``--records``. The summary is
written next to the records as ``summary.json`` and ``summary.md``; with ``--write-exclusions``
the exclusion list is written to the committed file, unless the stop rule triggered.

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
from goaly.bench.effisql.grading_pool import GRADING_INSTANCES, SETUP_INSTANCES, GradingPool
from goaly.bench.effisql.grading_service import ProcessRunner, SlotRunner, WorkerSpec
from goaly.bench.effisql.pg_settings import PgEnvConfig, write_private
from goaly.bench.effisql.setup_run import (
    Phase,
    RecordStore,
    dump_exclusions,
    run_phase,
    summarize,
    to_markdown,
)
from goaly.bench.effisql.splits import EXCLUSIONS_PATH, SPLIT_DB, load_splits

PYTHON_CPUS = {0, 1, 12, 13}  # Physical cores 0-1 (spec §6.2).
PHASES: tuple[Phase, ...] = ("measure", "official", "negatives")

_log = logging.getLogger("setup_run")


def _run(phase: Phase, instances_for: str, args: argparse.Namespace) -> int:
    instances = SETUP_INSTANCES if instances_for == "all" else GRADING_INSTANCES
    runners: list[SlotRunner] = [ProcessRunner(WorkerSpec(i)) for i in instances]
    try:
        return asyncio.run(
            run_phase(
                phase,
                GradingPool(runners),
                args.tasks,
                RecordStore(args.records),
                BaseCache(args.cache_root / Mode.DEV.value),
                val=load_splits().val,
            )
        )
    finally:
        for runner in runners:
            runner.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Effi-SQL grader's setup.")
    data_root = PgEnvConfig().data_root
    parser.add_argument("--phase", choices=[*PHASES, "summary", "all"], default="all")
    parser.add_argument("--ids", help="only these task IDs, comma-separated")
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    parser.add_argument("--cache-root", type=Path, default=data_root / "grader")
    parser.add_argument("--records", type=Path, default=data_root / "setup" / "records")
    parser.add_argument("--write-exclusions", action="store_true")
    parser.add_argument("--exclusions-out", type=Path, default=EXCLUSIONS_PATH)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s %(extra)s"
    )
    logging.getLogger().handlers[0].addFilter(_ExtraFields())
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

    phases = PHASES if args.phase == "all" else () if args.phase == "summary" else (args.phase,)
    for phase in phases:
        ran = _run(phase, "all" if phase == "measure" else "grading", args)
        _log.info("phase %s done: %d tasks", phase, ran)

    summary = summarize(
        [t.instance_id for t in tasks],
        RecordStore(args.records).load(),
        val=load_splits().val,
        tasks={t.instance_id: t for t in tasks},
    )
    out = args.records.parent
    write_private(out / "summary.json", summary.model_dump_json(indent=2) + "\n")
    markdown = to_markdown(summary)
    write_private(out / "summary.md", markdown)
    print(markdown)  # noqa: T201
    if summary.stop:
        print("setup_run: more than 10% of tasks excluded: stop and investigate", file=sys.stderr)  # noqa: T201
        return 2
    if args.write_exclusions:
        args.exclusions_out.write_text(dump_exclusions(summary.exclusions), encoding="utf-8")
    return 0 if summary.passed else 3


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
