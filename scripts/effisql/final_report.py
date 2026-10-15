"""Write the final report from run directories and the RSI ledger (spec §10).

    uv run --locked python scripts/effisql/final_report.py \\
        --v0-heldout runs/baseline/heldout --final-heldout runs/final/heldout \\
        --v0-val runs/baseline/val --final-val runs/final/val \\
        [--ledger rsi/ledger] [--tasks FILE] [--template FILE] [--out FILE] [--allow-partial]

Each run is optional: a missing one renders as pending, so the report can be drafted before
the final runs exist. --allow-partial reports a run that is still grading, from the tasks graded
in every seed, and the Runs section marks it partial. --tasks is the task file the difficulty
labels come from, by default the cached dataset (scripts/effisql/prepare_data.py). With
--template, the narrative template's markers are filled; without it the sections are written in
order. The report goes to --out, or to stdout.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from goaly.bench.effisql.data import DATASET_FILE, default_cache_dir
from goaly.bench.effisql.report import BOOTSTRAP_SEED, ROLES, ReportInputs, render
from goaly.bench.effisql.report_inputs import (
    RunDirError,
    load_difficulty,
    load_experiments,
    load_lessons,
    load_run,
)

DEFAULT_LEDGER = Path("rsi/ledger")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the final report.")
    for role in ROLES:
        parser.add_argument(f"--{role}", type=Path, metavar="DIR", help=f"the {role} run")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER, metavar="DIR")
    parser.add_argument("--tasks", type=Path, default=default_cache_dir() / DATASET_FILE)
    parser.add_argument("--template", type=Path, metavar="FILE")
    parser.add_argument("--out", type=Path, metavar="FILE")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args(argv)
    try:
        runs = {
            role: load_run(path, allow_partial=args.allow_partial)
            for role in ROLES
            if (path := getattr(args, role.replace("-", "_"))) is not None
        }
        inputs = ReportInputs(
            runs=runs,
            difficulty=load_difficulty(args.tasks),
            experiments=load_experiments(args.ledger),
            lessons=load_lessons(args.ledger),
            bootstrap_seed=args.bootstrap_seed,
        )
        template = None if args.template is None else args.template.read_text(encoding="utf-8")
        report = render(inputs, template)
    except (OSError, RunDirError, ValueError) as error:
        sys.stderr.write(f"error: {error}\n")
        return 1
    if args.out is None:
        sys.stdout.write(report)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
