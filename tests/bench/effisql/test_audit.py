"""Tests for goaly.bench.effisql.audit, the constant-branch audit list's patterns."""

import json

import pytest

from goaly.bench.effisql.audit import AUDIT_SPEEDUP, audit_reasons, prediction_sql
from goaly.bench.effisql.data import DATASET_FILE, default_cache_dir

BASE = "SELECT sum(l_extendedprice) AS revenue FROM lineitem WHERE l_quantity < 24"
# The two predictions spec review r5 (I1) showed pass both of the grader's checks.
GATED_CASE = """SELECT CASE WHEN (SELECT count(*) FROM orders) > 4000000 THEN 12345.67
ELSE (SELECT sum(l_extendedprice) FROM lineitem WHERE l_quantity < 24) END AS revenue"""
GATED_UNION = """SELECT 12345.67 WHERE (SELECT count(*) FROM orders) = 4500000
UNION ALL
SELECT sum(l_extendedprice) AS revenue FROM lineitem WHERE l_quantity < 24
AND (SELECT count(*) FROM orders) <> 4500000"""


@pytest.mark.parametrize(
    ("sql", "reasons"),
    [
        pytest.param(GATED_CASE, ["a CASE gated on a subquery"], id="r5 CASE"),
        pytest.param(
            GATED_UNION,
            ["rows gated on a scalar subquery", "a SELECT of literals only"],
            id="r5 UNION",
        ),
        pytest.param(
            "SELECT 1.5 WHERE pg_relation_size('lineitem') > 1e9 UNION ALL SELECT 2",
            ["calls an introspection function", "a SELECT of literals only"],
            id="size gate",
        ),
        pytest.param("SELECT 'a' AS x, 2::int AS y;", ["a SELECT of literals only"]),
        pytest.param(BASE.replace("< 24", "<= 23"), [], id="a plain rewrite"),
        pytest.param(
            "SELECT sum(l_extendedprice) FROM lineitem WHERE l_quantity < 24"
            " AND EXISTS (SELECT 1 FROM orders WHERE o_orderkey = l_orderkey)",
            [],
            id="EXISTS (SELECT 1 FROM ...)",
        ),
    ],
)
def test_lists_the_patterns_of_a_constant_branch(sql: str, reasons: list[str]) -> None:
    assert audit_reasons(sql, BASE, 1.5) == reasons


def test_leaves_out_a_pattern_the_base_query_has_already() -> None:
    base = "SELECT 1 FROM lineitem WHERE l_quantity < 24 AND (SELECT count(*) FROM orders) > 0"

    assert audit_reasons(base.replace("< 24", "<= 23"), base, 1.5) == []


def test_lists_a_prediction_that_ran_a_hundred_times_faster() -> None:
    assert audit_reasons(BASE, BASE, AUDIT_SPEEDUP) == ["τ ≥ 100"]
    assert audit_reasons(BASE, BASE, AUDIT_SPEEDUP - 1) == []


def test_reads_the_query_of_an_official_prediction() -> None:
    assert prediction_sql("```sql\nSELECT 1;\n```\n\n```diff\n-a\n+b\n```") == "SELECT 1;"
    assert prediction_sql("SELECT 2") == "SELECT 2"


@pytest.mark.integration
def test_lists_few_of_the_golds() -> None:
    """On the real dataset, the patterns list only 3 of the 238 TPC-H golds against their bases."""
    listed = []
    with (default_cache_dir() / DATASET_FILE).open(encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            if record["db"] == "tpch" and audit_reasons(
                record["optimized_sql"], record["base_sql"], 1.0
            ):
                listed.append(record["id"])
    assert listed == [118, 165, 205]
