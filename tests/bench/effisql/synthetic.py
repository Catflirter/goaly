"""Synthetic Effi-SQL tasks for the tests: none of them comes from the real dataset.

The gold answer here is made up and easy to search for, so a test can show that it never leaks.
"""

import copy
import json
from collections.abc import Iterable
from pathlib import Path

from goaly.bench.effisql.data import EffiSqlInstance
from goaly.contracts.interfaces import JsonValue

GOLD_SQL = "SELECT 'synthetic gold answer' AS gold"
GOLD_PLAN: dict[str, JsonValue] = {"Node Type": "Result", "Note": "synthetic gold plan"}
BASE_PLAN: dict[str, JsonValue] = {
    "Node Type": "Aggregate",
    "Actual Rows": 1,
    "Plans": [{"Node Type": "Seq Scan", "Relation Name": "orders", "Actual Rows": 4500000}],
}
BASE_SQL = "SELECT count(*)\nFROM orders\nWHERE o_orderkey IN (SELECT o_orderkey FROM orders)"


def task(**fields: JsonValue) -> dict[str, JsonValue]:
    """Return a synthetic task record, in the dataset's field order, with ``fields`` replaced."""
    record: dict[str, JsonValue] = {
        "id": 7,
        "db": "tpch",
        "base_sql": BASE_SQL,
        "optimized_sql": GOLD_SQL,
        "base_time": 1234.5,
        "fast_time": 12.25,
        "base_explain_analyze": copy.deepcopy(BASE_PLAN),
        "optimized_explain_analyze": copy.deepcopy(GOLD_PLAN),
        "difficulty": "medium",
    }
    record.update(fields)
    return record


def without(record: dict[str, JsonValue], *names: str) -> dict[str, JsonValue]:
    """Return ``record`` without the fields ``names``."""
    return {name: value for name, value in record.items() if name not in names}


def write_tasks(path: Path, *records: dict[str, JsonValue]) -> Path:
    """Write ``records`` to ``path`` as a task file, one JSON object per line, and return it."""
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def ids(instances: Iterable[EffiSqlInstance]) -> list[str]:
    """Return the instance IDs of ``instances``, in order."""
    return [inst.instance_id for inst in instances]
