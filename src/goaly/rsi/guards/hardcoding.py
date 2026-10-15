"""GUARD_HARD_CODING_SCAN (spec §8.5): a candidate embeds no gold answer and no data table.

Two scans of what the candidate adds, file by file:

- The lines it adds are normalized and joined (goaly.rsi.guards.gold_windows), and every window
  of them is looked up among the gold-only windows. In a Python file, each string constant that
  reaches an added line is scanned the same way, as Python joins it, so that splitting a gold
  answer over adjacent string literals hides nothing. A hit names the file and the line, never the
  text, so the report repeats no gold.
- A large literal collection fails, in a file of any type (spec §8.5):
  - In Python, a list, tuple, set or dict display on an added line counts its constants, through
    nested displays, ``*`` unpacking and ``+`` concatenation.
  - In JSON, JSONL and TOML, an array or mapping counts its constants the same way, and a JSONL
    file counts its records. In YAML, the longest run of sequence items or mapping keys at one
    indentation counts, and so does the largest flow collection on a line.
  - In CSV, the rows count; in SQL, the items of the largest list, such as IN (...) or VALUES;
    and in Markdown and plain text, the longest run of list items or table rows.

  Only a collection larger than config allows, and larger than the parent's largest in the same
  file, fails: a table the best version already had is not the candidate's. A long single string
  is fine, since prompts are long strings. A file that cannot be parsed fails, since what cannot
  be checked is not trusted.

Effi-SQL instance IDs are not scanned: they are small integers, so a match would mean nothing
(spec §8.5).
"""

import ast
import csv
import io
import json
import re
import tomllib
from bisect import bisect_right
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from goaly.contracts.gen.experiment_pb2 import GUARD_HARD_CODING_SCAN, GuardResult
from goaly.rsi.git import run_git
from goaly.rsi.guards.gold_windows import GoldWindows, normalize
from goaly.rsi.guards.paths import changes, shown

__all__ = ["HardCodingConfig", "check_hard_coding"]

_DELETED = "D"
_ADDED = "A"
_DISPLAYS = (ast.List, ast.Tuple, ast.Set, ast.Dict)
_DATA = frozenset({".json", ".jsonl", ".yaml", ".yml", ".toml", ".csv", ".sql"})
_TEXT = frozenset({".md", ".txt"})
# A Markdown or text line that is one item of a list or one row of a table.
_ROW = re.compile(r"^\s*(?:[-*+]\s|\d+[.)]\s|\|)")
# A YAML line that is one item of a block sequence, or one key of a block mapping.
_YAML_ITEM = re.compile(r"^(\s*)(?:-(?:\s|$)|[^\s#][^:#]*:(?:\s|$))")
_UNPARSABLE = -1  # The size of a collection in a file that cannot be parsed: always refused.


class HardCodingConfig(BaseModel):
    """The scan's limit, from config."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # The most constant elements a literal collection may hold.
    max_literal_elements: int = Field(default=32, gt=0)


def check_hard_coding(
    repo: Path,
    *,
    parent: str,
    candidate: str,
    snapshot: Path,
    gold: GoldWindows,
    config: HardCodingConfig,
) -> GuardResult:
    """Scan what ``candidate`` adds since ``parent``; ``snapshot`` holds the candidate's files.

    Raises ValueError for ``gold`` with no window or another width, since the scan would then
    pass every candidate.
    """
    gold.check()
    problems = []
    for change in changes(repo, parent, candidate):
        if change.status == _DELETED:
            continue
        added = _added_lines(repo, parent, candidate, change.path)
        name = shown(change.path)
        problems += [f"{name}:{n}: a fragment of a gold answer" for n in _gold_lines(added, gold)]
        source = (snapshot / change.path).read_text(encoding="utf-8", errors="replace")
        suffix = Path(change.path).suffix
        if suffix == ".py":
            for problem in _python_problems(name, source, added, gold, config):
                if problem not in problems:
                    problems.append(problem)
        elif suffix in _DATA | _TEXT:
            before = "" if change.status == _ADDED else _parent_text(repo, parent, change.path)
            problems += _data_problems(name, suffix, source, before, config)
    if problems:
        return GuardResult(guard=GUARD_HARD_CODING_SCAN, passed=False, detail="\n".join(problems))
    return GuardResult(
        guard=GUARD_HARD_CODING_SCAN,
        passed=True,
        detail="no gold-only fragment, no large literal",
    )


def _added_lines(repo: Path, parent: str, candidate: str, path: str) -> list[tuple[int, str]]:
    """The lines the diff adds to ``path``, each with its number in the candidate's file."""
    output = run_git(
        [
            "--literal-pathspecs",
            "diff",
            "--unified=0",
            "--no-color",
            "--no-ext-diff",
            "--no-renames",
            "--text",
            parent,
            candidate,
            "--",
            path,
        ],
        cwd=repo,
    )
    added: list[tuple[int, str]] = []
    number = 0
    in_hunk = False
    for line in output.split("\n"):
        if line.startswith("@@ "):  # "@@ -<old start>[,<count>] +<new start>[,<count>] @@"
            new = line.split(" ")[2]
            number = int(new.lstrip("+").split(",")[0])
            in_hunk = True
        elif in_hunk and line.startswith("+"):
            added.append((number, line[1:]))
            number += 1
    return added


def _parent_text(repo: Path, parent: str, path: str) -> str:
    return run_git(["--literal-pathspecs", "cat-file", "blob", f"{parent}:{path}"], cwd=repo)


def _gold_lines(added: list[tuple[int, str]], gold: GoldWindows) -> list[int]:
    """The lines on which a gold-only window of the added text starts."""
    starts: list[int] = []  # Where each line's text starts in the joined, normalized text.
    lines: list[int] = []
    pieces: list[str] = []
    offset = 0
    for number, text in added:
        piece = normalize(text)
        if piece:
            starts.append(offset)
            lines.append(number)
            pieces.append(piece)
            offset += len(piece) + 1  # The space that joins it to the next.
    joined = " ".join(pieces)
    # A window that starts on the space joining two lines starts, in effect, on the second.
    hits = {
        lines[bisect_right(starts, start + 1) - 1]
        for start in range(len(joined) - gold.width + 1)
        if joined[start : start + gold.width] in gold
    }
    return sorted(hits)


# Python.


def _python_problems(
    name: str,
    source: str,
    added: list[tuple[int, str]],
    gold: GoldWindows,
    config: HardCodingConfig,
) -> list[str]:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return [f"{name}:1: cannot be parsed for literal collections"]
    numbers = {number for number, _ in added}
    found: list[tuple[int, str]] = []
    try:
        for node in ast.walk(tree):
            first = getattr(node, "lineno", None)
            if first is None:
                continue
            last = getattr(node, "end_lineno", None) or first
            if not numbers.intersection(range(first, last + 1)):
                continue
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                # Python joins adjacent literals into this one constant.
                if _holds_gold(node.value, gold):
                    found.append((first, "a fragment of a gold answer"))
            elif _is_display(node):
                count = _constants(node)
                if count > config.max_literal_elements:
                    found.append((first, f"a literal collection of {count} constants"))
    except RecursionError:
        return [f"{name}:1: cannot be checked for literal collections: it nests too deeply"]
    return [f"{name}:{line}: {reason}" for line, reason in sorted(dict.fromkeys(found))]


def _holds_gold(text: str, gold: GoldWindows) -> bool:
    normalized = normalize(text)
    return any(
        normalized[start : start + gold.width] in gold
        for start in range(len(normalized) - gold.width + 1)
    )


def _is_display(node: ast.AST) -> bool:
    """A literal collection: a display, or displays joined with ``+``."""
    if isinstance(node, _DISPLAYS):
        return True
    return (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Add)
        and _is_display(node.left)
        and _is_display(node.right)
    )


def _constants(node: ast.AST) -> int:
    """The constants a literal collection holds, counted through nested displays, ``*``
    unpacking and ``+``.
    """
    if isinstance(node, ast.Constant):
        return 1
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
        return 1 if isinstance(node.operand, ast.Constant) else 0
    if isinstance(node, ast.Starred):
        return _constants(node.value)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _constants(node.left) + _constants(node.right)
    if isinstance(node, ast.List | ast.Tuple | ast.Set):
        return sum(_constants(element) for element in node.elts)
    if isinstance(node, ast.Dict):  # A pair counts once, a key being a label for its value.
        return sum(
            _constants(value) if isinstance(value, _DISPLAYS) else min(1, _constants(value))
            for key, value in zip(node.keys, node.values, strict=True)
            if key is not None
        )
    return 0


# Data and text files.


def _data_problems(
    name: str, suffix: str, source: str, before: str, config: HardCodingConfig
) -> list[str]:
    largest = _largest(suffix, source)
    if largest == _UNPARSABLE:
        return [f"{name}:1: cannot be parsed for literal collections"]
    if largest <= config.max_literal_elements:
        return []
    if before and largest <= _largest(suffix, before):  # The parent already had it.
        return []
    return [f"{name}:1: a literal collection of {largest} constants"]


def _largest(suffix: str, source: str) -> int:
    """The largest collection in a data or text file, or _UNPARSABLE."""
    try:
        if suffix == ".json":
            return _largest_value(json.loads(source))
        if suffix == ".jsonl":
            records = [json.loads(line) for line in source.split("\n") if line.strip()]
            return max([len(records), *map(_largest_value, records)])
        if suffix in (".yaml", ".yml"):
            return _yaml_largest(source)
        if suffix == ".toml":
            return _largest_value(tomllib.loads(source))
        if suffix == ".csv":
            return sum(1 for row in csv.reader(io.StringIO(source)) if row)
        if suffix == ".sql":
            return _sql_largest(source)
    except (ValueError, csv.Error, RecursionError, MemoryError):
        return _UNPARSABLE
    return _longest_run(source)


def _largest_value(value: object) -> int:
    """The constants of the largest array or mapping in ``value``, counted through nesting."""
    if isinstance(value, dict):
        return max([_leaves(value), *map(_largest_value, value.values())])
    if isinstance(value, list):
        return max([_leaves(value), *map(_largest_value, value)])
    return 0


def _leaves(value: object) -> int:
    """The constants under an array or mapping: a mapping's keys and every leaf value."""
    if isinstance(value, dict):
        return sum(1 + _leaves(item) for item in value.values())
    if isinstance(value, list):
        return sum(_leaves(item) if isinstance(item, dict | list) else 1 for item in value)
    return 0


def _yaml_largest(text: str) -> int:
    """The longest run of YAML items at one indentation, or the largest flow collection."""
    longest = 0
    runs: dict[int, int] = {}  # Indentation: the items in a row at it.
    for line in text.split("\n"):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        item = _YAML_ITEM.match(line)
        indent = len(line) - len(line.lstrip())
        runs = {depth: count for depth, count in runs.items() if depth <= indent}
        if item is not None:
            runs[indent] = runs.get(indent, 0) + 1
            longest = max(longest, runs[indent])
        else:
            runs.pop(indent, None)
        if "[" in line or "{" in line:
            longest = max(longest, _flow_elements(line))
    return longest


def _flow_elements(line: str) -> int:
    """The elements of the largest flow collection on a YAML line, by its commas."""
    largest = 0
    counts: list[int] = []
    for char in line:
        if char in "[{":
            counts.append(1)
        elif char in "]}" and counts:
            largest = max(largest, counts.pop())
        elif char == "," and counts:
            counts[-1] += 1
    return max([largest, *counts])


def _sql_largest(text: str) -> int:
    """The largest list in SQL, counted by its commas outside strings: the items of a
    parenthesized list, as in IN (...) or one row of VALUES, or the rows of a VALUES list.
    """
    largest = 0
    counts: list[int] = [1]  # The items so far at each depth of parentheses; 0 is the top.
    statement: list[str] = []
    quote = ""
    for char in [*text, ";"]:
        statement.append(char)
        if quote:
            quote = "" if char == quote else quote
        elif char in "'\"":
            quote = char
        elif char == "(":
            counts.append(1)
        elif char == ")" and len(counts) > 1:
            largest = max(largest, counts.pop())
        elif char == ",":
            counts[-1] += 1
        elif char == ";":  # Rows count only in a VALUES list, not a SELECT's columns.
            if "values" in "".join(statement).lower():
                largest = max(largest, counts[0])
            counts, statement = [1], []
    return largest


def _longest_run(text: str) -> int:
    """The most list items or table rows in a row, in Markdown or plain text."""
    longest = run = 0
    for line in text.split("\n"):
        run = run + 1 if _ROW.match(line) else 0
        longest = max(longest, run)
    return longest
