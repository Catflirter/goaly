"""The `goaly` command: the composition root, the one module that may import every layer (spec
§3.1).

It wires the layers together. It registers each benchmark with the harnesses that solve it, which
`goaly run` can then run: T11 registers Effi-SQL and its v0 pack here.
"""

import sys
from pathlib import Path

from goaly.harness.core import cli
from goaly.harness.core.registry import Registry

__all__ = ["BUDGETS_FILE", "main", "registry"]

# The LLM concurrency budgets (spec §6.2); `goaly run --budgets` reads another file instead.
BUDGETS_FILE = Path(__file__).resolve().parent / "config" / "llm_budgets.toml"


def registry() -> Registry:
    """Every benchmark `goaly run` can run, with the harnesses that solve it."""
    return Registry()


def main() -> int:
    """The console script: run the `goaly` command with this process's arguments."""
    return cli.main(sys.argv[1:], registry(), budgets_file=BUDGETS_FILE)
