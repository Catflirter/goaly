"""The grader end to end on a real grading instance, in both modes.

Run with ``GOALY_EFFISQL_ITEST_REAL=1 pytest --integration``; without the variable it skips.
"""

import asyncio
import os
from pathlib import Path

import pytest

from goaly.bench.effisql.data import EffiSqlInstance
from goaly.bench.effisql.grading import EffiSqlGrader, Mode
from goaly.bench.effisql.grading_pool import GRADING_INSTANCES, GradingInstance, Priority
from goaly.bench.effisql.pg_build import Connect, connector
from goaly.bench.effisql.pg_settings import GRADER_ROLE, load_credentials
from goaly.contracts.gen.task_pb2 import (
    GRADE_STATUS_GRADED,
    GRADE_STATUS_PREDICTION_FAILED,
    EffiSqlPrediction,
    Grade,
    Prediction,
)

pytestmark = pytest.mark.integration

BASE = """SELECT l_returnflag, l_linestatus, count(*), sum(l_extendedprice)
FROM lineitem WHERE l_orderkey < 200000 GROUP BY l_returnflag, l_linestatus"""
INST = EffiSqlInstance.model_validate({"id": "itest-1", "db": "tpch", "base_sql": BASE})


def grader(tmp_path: Path) -> EffiSqlGrader:
    if os.environ.get("GOALY_EFFISQL_ITEST_REAL") != "1":
        pytest.skip("needs the real instances: set GOALY_EFFISQL_ITEST_REAL=1")
    password = load_credentials().grader_password

    def connect(instance: GradingInstance) -> Connect:
        return connector("127.0.0.1", instance.port, password, user=GRADER_ROLE)

    return EffiSqlGrader.on_instances(GRADING_INSTANCES[:1], connect, cache_root=tmp_path)


def answer(sql: str) -> Prediction:
    return Prediction(instance_id=INST.instance_id, effisql=EffiSqlPrediction(sql=sql))


def test_grading_in_both_modes(tmp_path: Path) -> None:
    effisql = grader(tmp_path)

    async def grade(sql: str | None, mode: Mode) -> Grade:
        prediction = Prediction() if sql is None else answer(sql)
        return await effisql.grade(INST, prediction, mode, priority=Priority.SMOKE)

    async def scenario() -> dict[str, Grade]:
        first = await grade(BASE, Mode.DEV)  # Measures the base and caches it.
        cache = effisql.cache(Mode.DEV)
        stored = cache.get(INST)
        assert stored is not None
        cache.put(stored.model_copy(update={"identity_speedup": 1.0}))
        grades = {
            "dev identity": await grade(BASE.lower() + ";", Mode.DEV),
            "dev no branch": await grade(None, Mode.DEV),
            "dev rewrite": await grade(
                BASE.replace("l_orderkey < 200000", "l_orderkey <= 199999"), Mode.DEV
            ),
            "dev wrong": await grade(BASE.replace("200000", "100000"), Mode.DEV),
            "dev delete": await grade(
                "WITH d AS (DELETE FROM region RETURNING *) SELECT 1", Mode.DEV
            ),
            "official identity": await grade(BASE, Mode.OFFICIAL),
        }
        grades["first"] = first
        return grades

    grades = asyncio.run(scenario())
    first = grades["first"]
    assert first.status == GRADE_STATUS_GRADED
    assert first.effisql.matched
    assert first.effisql.alt_matched
    assert not first.effisql.literal_result
    assert not first.effisql.base_cached
    for name in ("dev identity", "dev no branch"):
        assert grades[name].effisql.identity_cached, name
        assert grades[name].effisql.credit == 0.5
    rewrite = grades["dev rewrite"].effisql
    assert rewrite.matched
    assert rewrite.base_cached
    assert rewrite.timing_runs == 3
    assert 0.3 < rewrite.speedup < 3
    assert not grades["dev wrong"].effisql.matched
    assert not grades["dev wrong"].effisql.HasField("alt_matched")
    assert grades["dev delete"].status == GRADE_STATUS_PREDICTION_FAILED
    official = grades["official identity"].effisql
    assert official.matched
    assert not official.identity_cached
    assert official.timing_runs == 3
    assert 0.5 < official.speedup < 2


def test_hard_coded_answers_get_no_credit(tmp_path: Path) -> None:
    effisql = grader(tmp_path)
    password = load_credentials().grader_password
    with connector("127.0.0.1", GRADING_INSTANCES[0].port, password, user=GRADER_ROLE)(
        "tpch"
    ) as conn:
        rows = conn.execute(BASE).fetchall()
    values = ", ".join(f"('{a}', '{b}', {c}::bigint, {d}::numeric)" for a, b, c, d in rows)
    constant = f"SELECT * FROM (VALUES {values}) v"  # noqa: S608  # Test data from the base.

    async def scenario() -> tuple[Grade, Grade]:
        return (
            await effisql.grade(INST, answer(constant), Mode.DEV),
            await effisql.grade(INST, answer(constant), Mode.OFFICIAL),
        )

    for grade in asyncio.run(scenario()):
        assert grade.status == GRADE_STATUS_GRADED
        assert grade.effisql.matched
        assert grade.effisql.literal_result
        assert not grade.effisql.alt_matched
        assert grade.effisql.credit == 0
        assert grade.effisql.speedup > 10, "timed even though flagged"
        assert not grade.correct
