"""The `goaly` command: the composition root, the one module that may import every layer (spec
§3.1).

It wires the layers together. It registers each benchmark with the harnesses that solve it, which
`goaly run` can then run: Effi-SQL, with its v0 pack as the harness "v0", and its trusted host
side. Neither imports pack code until a harness opens, so the host loads none (spec V24, V27).
`goaly rsi ...` runs the RSI lane's command line, goaly.rsi.cli.
"""

import logging
import sys
from pathlib import Path

from goaly.bench.effisql.benchmark import BENCHMARK_NAME as EFFISQL
from goaly.bench.effisql.harness import PACK_DIR, open_harness, open_host, pack_models
from goaly.bench.effisql.pg_instances import confine_to_python_cpus
from goaly.harness.core import cli
from goaly.harness.core.registry import Registry

__all__ = ["BUDGETS_FILE", "main", "registry"]

# The LLM concurrency budgets (spec §6.2); `goaly run --budgets` reads another file instead.
BUDGETS_FILE = Path(__file__).resolve().parent / "config" / "llm_budgets.toml"


def registry() -> Registry:
    """Every benchmark `goaly run` can run, with the harnesses that solve it."""
    registry = Registry()
    registry.add(EFFISQL, "v0", open_harness, pack_dir=PACK_DIR, models=pack_models())
    registry.add_benchmark(EFFISQL, open_host)
    return registry


def main() -> int:
    """The console script: run the `goaly` command with this process's arguments.

    On the host, the process keeps to the CPUs of our Python processes, away from the
    PostgreSQL instances' (spec §6.2), and so does every thread and process it starts; the solve
    container's cpuset confines it there.
    """
    argv = sys.argv[1:]
    if "--in-container" not in argv:  # The container's own cpuset already confines it.
        try:
            confine_to_python_cpus()
        except OSError as error:  # A cpuset without those CPUs.
            logging.getLogger(__name__).warning("cannot confine to the Python CPUs: %s", error)
    if argv[:1] == ["rsi"]:  # Imported here, so `goaly run` never loads goaly.rsi (spec §3.1).
        from goaly.rsi.cli import main as rsi_main

        return rsi_main(argv[1:])
    return cli.main(argv, registry(), budgets_file=BUDGETS_FILE)
