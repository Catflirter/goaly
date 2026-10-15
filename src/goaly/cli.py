"""The `goaly` command: the composition root, the one module that may import every layer (spec
§3.1).

It wires the layers together. It registers each benchmark with the harnesses that solve it, which
`goaly run` can then run: Effi-SQL, with its v0 pack as the harness "v0", and its grading for the
grade phase. Neither factory imports pack code until a harness opens, so a grade phase loads
none (spec V24).
"""

import sys
from pathlib import Path

from goaly.bench.effisql.benchmark import BENCHMARK_NAME as EFFISQL
from goaly.bench.effisql.harness import open_grading, open_harness
from goaly.bench.effisql.pg_instances import confine_to_python_cpus
from goaly.harness.core import cli
from goaly.harness.core.registry import Registry

__all__ = ["BUDGETS_FILE", "main", "registry"]

# The LLM concurrency budgets (spec §6.2); `goaly run --budgets` reads another file instead.
BUDGETS_FILE = Path(__file__).resolve().parent / "config" / "llm_budgets.toml"


def registry() -> Registry:
    """Every benchmark `goaly run` can run, with the harnesses that solve it."""
    registry = Registry()
    registry.add(EFFISQL, "v0", open_harness)
    registry.add_benchmark(EFFISQL, open_grading)
    return registry


def main() -> int:
    """The console script: run the `goaly` command with this process's arguments.

    The process keeps to the CPUs of our Python processes, away from the PostgreSQL instances'
    (spec §6.2), and so does every thread and process it starts.
    """
    confine_to_python_cpus()
    return cli.main(sys.argv[1:], registry(), budgets_file=BUDGETS_FILE)
