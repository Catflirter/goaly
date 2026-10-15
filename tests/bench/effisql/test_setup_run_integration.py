"""The setup's measurement and negative checks on a real grading instance.

Run with ``GOALY_EFFISQL_ITEST_REAL=1 pytest --integration``; without the variable it skips.
"""

import os
from pathlib import Path

import pytest

from goaly.bench.effisql.data import EffiSqlInstance
from goaly.bench.effisql.grading import BaseCache, GradingSlot
from goaly.bench.effisql.grading_pool import GRADING_INSTANCES
from goaly.bench.effisql.grading_service import WorkerSpec
from goaly.bench.effisql.setup_run import measure_task, negative_task

pytestmark = pytest.mark.integration

BASE = """SELECT n_name, count(*) AS suppliers, sum(s_acctbal) AS balance, min(s_name)
FROM supplier JOIN nation ON s_nationkey = n_nationkey GROUP BY n_name;"""
GOLD = (
    "SELECT n_name, count(*), sum(s_acctbal), min(s_name) FROM nation JOIN supplier "
    "ON s_nationkey = n_nationkey GROUP BY n_name"
)
INST = EffiSqlInstance.model_validate(
    {"id": "itest-2", "db": "tpch", "base_sql": BASE, "base_explain_analyze": {"Actual Rows": 25}}
)


@pytest.fixture
def slot() -> GradingSlot:
    if os.environ.get("GOALY_EFFISQL_ITEST_REAL") != "1":
        pytest.skip("needs the real instances: set GOALY_EFFISQL_ITEST_REAL=1")
    spec = WorkerSpec(GRADING_INSTANCES[0])
    connect = spec.connect()
    return GradingSlot(spec.instance, lambda _: connect)


def test_measure_and_negatives(slot: GradingSlot, tmp_path: Path) -> None:
    try:
        record = measure_task(slot, INST, GOLD, BaseCache(tmp_path))
        assert record.base is not None
        assert record.base.row_count == 25
        assert record.base.relations == ("nation", "supplier")
        assert record.base.identity_speedup is not None
        assert record.data is not None
        assert record.data.outcome == "pass"
        assert record.gold is not None
        assert record.gold.matched
        assert record.gold.alt_matched
        assert record.gold.literal_result is False
        matched = negative_task(slot, INST, record.base)
    finally:
        slot.reset()
    assert matched == {
        "control": True,
        "drop": False,
        "duplicate": False,
        "reorder": False,
        "perturb": False,
    }
