"""foreign_scale_plan over the real dataset's 238 TPC-H plans.

Run with ``pytest --integration``: it needs the dataset in T08's cache,
``~/.cache/goaly/effisql/effi-sql-pg.jsonl``, and skips without it. It reads each task's id, db
and base plan only, never a gold field, and the dataset is never committed.
"""

import hashlib
import json
from pathlib import Path

import pytest

from goaly.bench.effisql.pg_snapshot import foreign_scale_plan

pytestmark = pytest.mark.integration

DATASET = Path.home() / ".cache" / "goaly" / "effisql" / "effi-sql-pg.jsonl"
DATASET_SHA256 = "2af3bdc9fe6c1197f23e31089660e7ce10fca6956b32e5970a3113f0c87e9d97"
# The tasks whose recorded plans read SF10-sized tables: their scans read lineitem's 59,986,052
# rows, orders' 15,000,000 or customer's 1,500,000, or (task 40) ran a filter subplan that often.
SF10_RECORDED = {2, 24, 40, 57, 72, 80, 87, 94, 99, 134, 137, 146}


def test_exactly_the_sf10_recorded_plans_are_foreign() -> None:
    if not DATASET.is_file():
        pytest.skip(f"the dataset is not in {DATASET}")
    data = DATASET.read_bytes()
    assert hashlib.sha256(data).hexdigest() == DATASET_SHA256

    plans = {}
    for line in data.decode("utf-8").splitlines():
        task = json.loads(line)
        if task["db"] == "tpch":
            plans[task["id"]] = task["base_explain_analyze"]

    assert len(plans) == 238
    assert {task for task, plan in plans.items() if foreign_scale_plan(plan)} == SF10_RECORDED
