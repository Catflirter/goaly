"""GUARD_HARD_CODING_SCAN (spec §8.5): a candidate embeds no gold answer and no data table.

Two scans of what the candidate adds, file by file:

- The lines it adds are normalized and joined (goaly.rsi.guards.gold_windows), and every window
  of them is looked up among the gold-only windows. A hit names the file and the line, never the
  text, so the report repeats no gold.
- In a Python file it adds or changes, each list, tuple, set or dict display on an added line
  counts its constant elements, a constant tuple in a list counting as one. A display with more
  than config allows fails. A long single string is fine, since prompts are long strings.

Effi-SQL instance IDs are not scanned: they are small integers, so a match would mean nothing
(spec §8.5).
"""

import ast
from bisect import bisect_right
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from goaly.contracts.gen.experiment_pb2 import GUARD_HARD_CODING_SCAN, GuardResult, Lane
from goaly.rsi.git import run_git
from goaly.rsi.guards.gold_windows import GoldWindows, normalize
from goaly.rsi.guards.paths import changes

__all__ = ["HardCodingConfig", "check_hard_coding"]

_DELETED = "D"
_DISPLAYS = (ast.List, ast.Tuple, ast.Set, ast.Dict)


class HardCodingConfig(BaseModel):
    """The scan's limit, from config."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # The most constant elements a literal display may hold.
    max_literal_elements: int = Field(default=32, gt=0)


def check_hard_coding(
    repo: Path,
    lane: Lane.ValueType,
    *,
    parent: str,
    candidate: str,
    snapshot: Path,
    gold: GoldWindows,
    config: HardCodingConfig,
) -> GuardResult:
    """Scan what ``candidate`` adds since ``parent``; ``snapshot`` holds the candidate's files.

    ``lane`` is the candidate's; the scan reads every file the diff adds or changes.
    """
    problems = []
    for change in changes(repo, parent, candidate):
        if change.status == _DELETED:
            continue
        added = _added_lines(repo, parent, candidate, change.path)
        problems += [
            f"{change.path}:{line}: a fragment of a gold answer"
            for line in _gold_lines(added, gold)
        ]
        if change.path.endswith(".py"):
            source = (snapshot / change.path).read_text(encoding="utf-8", errors="replace")
            problems += [
                f"{change.path}:{line}: a literal collection of {count} constants"
                for line, count in _large_literals(
                    source, {number for number, _ in added}, config.max_literal_elements
                )
            ]
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


def _large_literals(source: str, added: set[int], limit: int) -> list[tuple[int, int]]:
    """The displays on an added line with more than ``limit`` constant elements."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []  # Its text was scanned for gold all the same.
    found = []
    for node in ast.walk(tree):
        if isinstance(node, _DISPLAYS):
            end = node.end_lineno or node.lineno
            count = _constant_elements(node)
            if count > limit and added.intersection(range(node.lineno, end + 1)):
                found.append((node.lineno, count))
    return sorted(found)


def _constant_elements(node: ast.List | ast.Tuple | ast.Set | ast.Dict) -> int:
    if isinstance(node, ast.Dict):
        return sum(
            1
            for key, value in zip(node.keys, node.values, strict=True)
            if key is not None and _constant(key) and _constant(value)
        )
    return sum(1 for element in node.elts if _constant(element))


def _constant(node: ast.expr) -> bool:
    """Whether ``node`` is data: a constant, a signed number, or a display of them."""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
        return isinstance(node.operand, ast.Constant)
    if isinstance(node, ast.List | ast.Tuple | ast.Set):
        return all(_constant(element) for element in node.elts)
    if isinstance(node, ast.Dict):
        return all(
            key is not None and _constant(key) and _constant(value)
            for key, value in zip(node.keys, node.values, strict=True)
        )
    return False
