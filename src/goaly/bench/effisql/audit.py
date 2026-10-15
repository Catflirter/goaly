"""The constant-branch audit list: matched predictions a person should read (spec §4.1, §12).

The grader's hard-coding checks cannot flag a single-row constant gated on the data, such as

    SELECT CASE WHEN (SELECT count(*) FROM orders) > 4000000 THEN 12345.67 ELSE (<base>) END
    SELECT 12345.67 WHERE (SELECT count(*) FROM orders) = 4500000 UNION ALL <base> WHERE ...

On SF3 the constant comes back at once; on the alternate database the base runs and matches.
The project lead accepted this residual risk and asked the report to list every matched
prediction that looks like such a branch, so each is reviewed before the numbers are published.
:func:`audit_reasons` says why a prediction is listed:

- **A constant output branch in the plan**, where the host stored the prediction's plan
  (``<run>/plans/seed-<k>.jsonl``, written by the grade phase, which runs EXPLAIN anyway). An
  output branch is the root and what it passes rows through: an Append's children, a Subquery
  Scan, a Limit and the like, never an InitPlan or SubPlan. Such a branch is listed if it is a
  Result that reads no table, so its rows come from constants or subqueries alone, or if it has a
  ``One-Time Filter``, the shape of a data-gated branch. :func:`plan_reasons` reads the plan.
- **A new literal**, whatever τ is: a constant answer must embed its value, so a literal holding a
  run of 4 or more digits that none of the base query's literals has is listed. A number counts
  its digits without the decimal point, so 12345.67 is the run 1234567; a string counts the runs
  in its text. On the dataset, 2 of the 238 TPC-H golds add one: tasks 68 and 189, with dates.
- **A branch pattern** the base query lacks, as an extra signal: an introspection call, a CASE
  gated on a subquery, rows gated on a scalar subquery, or a SELECT of literals only.
- **τ ≥ 100**: a constant answer returns in microseconds.

The patterns and the literal rule read the query's code, outside its comments, and every rule
leaves out what the base query has already: its literals, its patterns and, from the base plan
the official record carries, its plan's shapes. A false alarm costs a minute of reading; a miss,
an inflated score.
"""

import json
import re
from collections import deque
from collections.abc import Iterator, Mapping
from typing import Any

__all__ = ["AUDIT_SPEEDUP", "audit_reasons", "digit_runs", "plan_reasons", "prediction_sql"]

AUDIT_SPEEDUP = 100.0  # A constant answer returns in microseconds, so τ this high is listed.

_LITERAL = r"(?:[-+]?\d+(?:\.\d+)?|'(?:[^']|'')*')(?:\s*::\s*\w+)?(?:\s+as\s+\w+)?"
_PATTERNS = (
    (
        "calls an introspection function",
        re.compile(
            r"\b(?:pg_\w+|inet_\w+|txid_\w+|current_schemas?|current_database|current_catalog"
            r"|current_setting|current_user|session_user|version)\s*\(",
            re.IGNORECASE,
        ),
    ),
    (
        "a CASE gated on a subquery",
        re.compile(r"\bwhen\s*(?:not\s+)?(?:\(\s*select\b|exists\b)", re.IGNORECASE),
    ),
    (
        "rows gated on a scalar subquery",
        re.compile(r"\b(?:where|having|and|or)\s*\(\s*select\b", re.IGNORECASE),
    ),
    (
        "a SELECT of literals only",
        re.compile(
            rf"\bselect\s+(?:{_LITERAL}\s*,\s*)*{_LITERAL}\s*(?:\bwhere\b|\bunion\b|\)|;|$)",
            re.IGNORECASE,
        ),
    ),
)
_SQL_BLOCK = re.compile(r"```sql\n(.*?)\n```", re.DOTALL)
_COMMENTS = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)
_STRING = re.compile(r"'((?:[^']|'')*)'")
_NUMBER = re.compile(r"(?<![\w.$])(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?(?![\w.])", re.IGNORECASE)
_DIGIT_RUN = re.compile(r"\d{4,}")
# The plan nodes that read a table's rows, as goaly.bench.effisql.checks.RELATION_SCANS has them.
_SCANS = frozenset(
    {"Seq Scan", "Index Scan", "Index Only Scan", "Bitmap Heap Scan", "Tid Scan",
     "Tid Range Scan", "Sample Scan", "CTE Scan", "Function Scan", "Values Scan"}
)  # fmt: skip
# Nodes an output branch passes its rows through, which keep it an output branch.
_PASS_THROUGH = frozenset(
    {"Append", "Merge Append", "Subquery Scan", "Limit", "Sort", "Incremental Sort", "Unique",
     "Gather", "Gather Merge", "Materialize", "Result", "ProjectSet"}
)  # fmt: skip
_SUBPLANS = frozenset({"InitPlan", "SubPlan"})


def prediction_sql(prediction: str) -> str:
    """The query in an official ``prediction``: its sql block, or the whole text if it has none."""
    match = _SQL_BLOCK.search(prediction)
    return prediction if match is None else match.group(1)


def digit_runs(sql: str) -> set[str]:
    """The runs of 4 or more digits in ``sql``'s literals: its numbers and its strings."""
    code = _COMMENTS.sub(" ", sql)
    strings = _STRING.findall(code)
    runs = {run for text in strings for run in _DIGIT_RUN.findall(text.replace(".", ""))}
    for number in _NUMBER.findall(_STRING.sub("''", code)):
        runs.update(_DIGIT_RUN.findall(number.replace(".", "")))
    return runs


def plan_reasons(plan: Any) -> list[str]:
    """The constant output branches in an ``EXPLAIN (FORMAT JSON)`` plan, described.

    ``plan`` is the document as EXPLAIN returns it, a list holding one object with a "Plan", or
    that object, or its JSON text. Raises ValueError if it holds no plan.
    """
    root = _plan_root(json.loads(plan) if isinstance(plan, str) else plan)
    reasons = []
    for node in _output_branches(root):
        kind = str(node.get("Node Type"))
        if kind == "Result" and not _reads_table(node):
            subplans = sum(
                child.get("Parent Relationship") in _SUBPLANS for child in _children(node)
            )
            source = f"{subplans} subqueries" if subplans else "constants"
            reasons.append(f"plan: a Result that reads no table, from {source}")
        if "One-Time Filter" in node:
            reasons.append(f"plan: a one-time filter on a {kind}")
    return list(dict.fromkeys(reasons))


def audit_reasons(
    sql: str,
    base_sql: str,
    speedup: float,
    *,
    plan: Any = None,
    base_plan: Any = None,
) -> list[str]:
    """Why a matched prediction goes on the audit list, if at all.

    ``plan`` is the prediction's stored plan, if any, and ``base_plan`` the base query's. What
    the base query has already, a literal, a pattern or a plan shape, is nothing new, so it is
    left out.
    """
    reasons = []
    if plan is not None:
        base_shapes = set() if base_plan is None else set(_safe_plan_reasons(base_plan))
        reasons += [reason for reason in plan_reasons(plan) if reason not in base_shapes]
    if new := sorted(digit_runs(sql) - digit_runs(base_sql)):
        reasons.append(f"adds the literal digits {', '.join(new)}")
    code, base = _COMMENTS.sub(" ", sql), _COMMENTS.sub(" ", base_sql)
    reasons += [
        reason for reason, pattern in _PATTERNS if pattern.search(code) and not pattern.search(base)
    ]
    if speedup >= AUDIT_SPEEDUP:
        reasons.append(f"τ ≥ {AUDIT_SPEEDUP:g}")
    return reasons


def _safe_plan_reasons(plan: Any) -> list[str]:
    """The base plan's shapes; a base plan that cannot be read has none to leave out."""
    try:
        return plan_reasons(plan)
    except ValueError:
        return []


def _plan_root(document: Any) -> Mapping[str, Any]:
    if isinstance(document, list) and len(document) == 1:
        document = document[0]
    if isinstance(document, Mapping) and isinstance(document.get("Plan"), Mapping):
        root: Mapping[str, Any] = document["Plan"]
        return root
    if isinstance(document, Mapping) and "Node Type" in document:
        return document  # A plan's root node itself, as the dataset stores base plans.
    raise ValueError("not an EXPLAIN (FORMAT JSON) plan")


def _children(node: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [child for child in node.get("Plans") or () if isinstance(child, Mapping)]


def _output_branches(root: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    """The root, and the nodes it passes rows through, in document order: never an InitPlan's
    or a SubPlan's."""
    queue = deque([root])
    while queue:
        node = queue.popleft()
        yield node
        if node.get("Node Type") in _PASS_THROUGH:
            queue += [c for c in _children(node) if c.get("Parent Relationship") not in _SUBPLANS]


def _reads_table(node: Mapping[str, Any]) -> bool:
    """Whether a table scan feeds ``node``'s rows, leaving its InitPlans and SubPlans out."""
    stack = [node]
    while stack:
        current = stack.pop()
        if current.get("Node Type") in _SCANS - {"Values Scan", "Function Scan"}:
            return True
        stack += [c for c in _children(current) if c.get("Parent Relationship") not in _SUBPLANS]
    return False
