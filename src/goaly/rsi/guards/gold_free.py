"""The gold-free runtime data check (spec §8.5): a candidate's evaluation loads no gold answer.

The data files a candidate is evaluated with hold only what the agent may see and what grading
needs; T08 writes the runtime task file that way. Effi-SQL grades against base_sql, so it needs
no gold at all. This check names every gold field it finds in a JSON or JSONL file, wherever it
is nested, by file, line and field, and never repeats the gold itself.
"""

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

__all__ = ["GOLD_FIELDS", "GoldDataError", "check_gold_free"]

# The fields of an Effi-SQL task that hold or give away its gold answer (spec §4.1).
GOLD_FIELDS = frozenset({"optimized_sql", "fast_time", "optimized_explain_analyze"})


class GoldDataError(Exception):
    """A data file holds a gold field, or cannot be checked."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("gold in the evaluation's data: " + "; ".join(problems))
        self.problems = problems


def check_gold_free(paths: Iterable[Path]) -> None:
    """Raise GoldDataError naming every gold field in the JSON or JSONL files ``paths``.

    A file of another kind, or one that does not parse, raises it too: what cannot be checked
    is not trusted.
    """
    problems: list[str] = []
    for path in paths:
        try:
            for line, value in _documents(path):
                problems.extend(f"{path}:{line}: {field}" for field in sorted(_gold_in(value)))
        except (OSError, ValueError) as error:
            problems.append(f"{path}: cannot be checked: {error}")
    if problems:
        raise GoldDataError(problems)


def _documents(path: Path) -> Iterator[tuple[int, object]]:
    """Each JSON document of the file with its line: one per line of a JSONL file."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        for number, line in enumerate(text.splitlines(), start=1):
            if line.strip():
                yield number, json.loads(line)
    elif path.suffix == ".json":
        yield 1, json.loads(text)
    else:
        raise ValueError("not a .json or .jsonl file")


def _gold_in(value: object) -> set[str]:
    """The gold fields that appear as keys anywhere in ``value``."""
    found: set[str] = set()
    if isinstance(value, dict):
        found.update(GOLD_FIELDS.intersection(value))
        for item in value.values():
            found |= _gold_in(item)
    elif isinstance(value, list):
        for item in value:
            found |= _gold_in(item)
    return found
