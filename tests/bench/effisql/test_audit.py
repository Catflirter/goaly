"""Tests for goaly.bench.effisql.audit, the constant-branch audit list's rules."""

import json

import pytest

from goaly.bench.effisql.audit import (
    AUDIT_SPEEDUP,
    audit_reasons,
    digit_runs,
    plan_reasons,
    prediction_sql,
)
from goaly.bench.effisql.data import DATASET_FILE, default_cache_dir

BASE = "SELECT sum(l_extendedprice) AS revenue FROM lineitem WHERE l_quantity < 24"
LITERAL = "adds the literal digits 1234567"  # 12345.67; any gate's threshold is listed too.
# The two predictions spec review r5 (I1) showed pass both of the grader's checks.
GATED_CASE = """SELECT CASE WHEN (SELECT count(*) FROM orders) > 4000000 THEN 12345.67
ELSE (SELECT sum(l_extendedprice) FROM lineitem WHERE l_quantity < 24) END AS revenue"""
GATED_UNION = """SELECT 12345.67 WHERE (SELECT count(*) FROM orders) = 4500000
UNION ALL
SELECT sum(l_extendedprice) AS revenue FROM lineitem WHERE l_quantity < 24
AND (SELECT count(*) FROM orders) <> 4500000"""
# Variants of them, from the review of #56, that no branch pattern catches.
VARIANTS = [
    "SELECT CASE WHEN 4000000 < (SELECT count(*) FROM orders) THEN 12345.67 ELSE (x) END",
    "SELECT CASE (SELECT count(*) FROM orders) WHEN 4500000 THEN 12345.67 ELSE (x) END",
    "SELECT CASE WHEN ((SELECT count(*) FROM orders)) > 4000000 THEN 12345.67 ELSE (x) END",
    "WITH g AS (SELECT count(*) n FROM orders) SELECT CASE WHEN g.n > 4000000 THEN 12345.67"
    " ELSE (x) END FROM g",
    "SELECT coalesce(nullif((SELECT count(*) FROM orders), 4500000) * 0 + (x), 12345.67)",
    "SELECT numeric '12345.67' WHERE 4500000 = (SELECT count(*) FROM orders) UNION ALL x",
    "SELECT 12345.67 FROM orders WHERE o_orderkey = 1"
    " AND 4500000 = (SELECT count(*) FROM orders) UNION ALL x",
]


@pytest.mark.parametrize(
    ("sql", "reasons"),
    [
        pytest.param(
            GATED_CASE, [f"{LITERAL}, 4000000", "a CASE gated on a subquery"], id="r5 CASE"
        ),
        pytest.param(
            GATED_UNION,
            [f"{LITERAL}, 4500000", "rows gated on a scalar subquery", "a SELECT of literals only"],
            id="r5 UNION",
        ),
        pytest.param(
            "SELECT 1.5 WHERE pg_relation_size('lineitem') > 1e9 UNION ALL SELECT 2",
            ["calls an introspection function", "a SELECT of literals only"],
            id="a size gate, with no long literal",
        ),
        pytest.param("SELECT 'a' AS x, 2::int AS y;", ["a SELECT of literals only"]),
        pytest.param(BASE.replace("< 24", "<= 23"), [], id="a plain rewrite"),
        pytest.param(
            "SELECT sum(l_extendedprice) FROM lineitem WHERE l_quantity < 24"
            " AND EXISTS (SELECT 1 FROM orders WHERE o_orderkey = l_orderkey)",
            [],
            id="EXISTS (SELECT 1 FROM ...)",
        ),
        pytest.param(f"{BASE} -- 12345.67, a comment", [], id="digits in a comment"),
    ],
)
def test_lists_the_signs_of_a_constant_branch(sql: str, reasons: list[str]) -> None:
    assert audit_reasons(sql, BASE, 1.5) == reasons


@pytest.mark.parametrize("sql", VARIANTS)
def test_lists_a_new_literal_whatever_the_branch_looks_like(sql: str) -> None:
    reasons = audit_reasons(sql, BASE, 5.0)

    assert reasons[0].startswith(LITERAL)


def test_reads_digit_runs_from_numbers_and_strings() -> None:
    assert digit_runs("SELECT 12345.67, '1995-01-01', 42, 1e9, x1234") == {"1234567", "1995"}


def test_leaves_out_what_the_base_query_has_already() -> None:
    base = "SELECT 1 FROM lineitem WHERE l_shipdate < '1998-12-01' AND (SELECT count(*) FROM t) > 0"

    assert audit_reasons(base.replace("< ", "<= "), base, 1.5) == []


def test_lists_a_prediction_that_ran_a_hundred_times_faster() -> None:
    assert audit_reasons(BASE, BASE, AUDIT_SPEEDUP) == ["τ ≥ 100"]
    assert audit_reasons(BASE, BASE, AUDIT_SPEEDUP - 1) == []


def test_reads_the_query_of_an_official_prediction() -> None:
    assert prediction_sql("```sql\nSELECT 1;\n```\n\n```diff\n-a\n+b\n```") == "SELECT 1;"
    assert prediction_sql("SELECT 2") == "SELECT 2"


@pytest.mark.integration
def test_lists_few_of_the_golds() -> None:
    """On the real dataset, the rules list 5 of the 238 TPC-H golds against their bases."""
    listed = []
    with (default_cache_dir() / DATASET_FILE).open(encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            if record["db"] == "tpch" and audit_reasons(
                record["optimized_sql"], record["base_sql"], 1.0
            ):
                listed.append(record["id"])
    assert listed == [68, 118, 165, 189, 205]


# The plan rules, on plans shaped as PostgreSQL's EXPLAIN (FORMAT JSON) writes them.


def scan(table: str, relationship: str = "Outer") -> dict[str, object]:
    return {"Node Type": "Seq Scan", "Relation Name": table, "Parent Relationship": relationship}


def initplan(*plans: dict[str, object]) -> dict[str, object]:
    return {"Node Type": "Aggregate", "Parent Relationship": "InitPlan", "Plans": list(plans)}


GATED_UNION_PLAN = {
    "Plan": {
        "Node Type": "Append",
        "Plans": [
            {
                "Node Type": "Result",
                "Parent Relationship": "Member",
                "One-Time Filter": "($0 = 4500000)",
                "Plans": [initplan(scan("orders"))],
            },
            {
                "Node Type": "Result",
                "Parent Relationship": "Member",
                "One-Time Filter": "($1 <> 4500000)",
                "Plans": [
                    initplan(scan("orders")),
                    {"Node Type": "Aggregate", "Parent Relationship": "Outer",
                     "Plans": [scan("lineitem")]},
                ],
            },
        ],
    }
}  # fmt: skip
PLAIN_PLAN = {"Plan": {"Node Type": "Aggregate", "Plans": [scan("lineitem")]}}


def test_reads_a_constant_output_branch_from_the_plan() -> None:
    assert plan_reasons(GATED_UNION_PLAN) == [
        "plan: a Result that reads no table, from 1 subqueries",
        "plan: a one-time filter on a Result",
    ]
    assert plan_reasons([GATED_UNION_PLAN]) == plan_reasons(json.dumps(GATED_UNION_PLAN))
    assert plan_reasons(PLAIN_PLAN) == []
    assert plan_reasons({"Node Type": "Result"}) == [  # A root node, as the dataset stores plans.
        "plan: a Result that reads no table, from constants"
    ]


def test_a_subquery_inside_the_query_is_not_an_output_branch() -> None:
    nested = {"Node Type": "Result", "Parent Relationship": "SubPlan", "One-Time Filter": "x"}
    plan = {"Plan": {"Node Type": "Seq Scan", "Relation Name": "orders", "Plans": [nested]}}

    assert plan_reasons(plan) == []


def test_refuses_a_document_with_no_plan() -> None:
    with pytest.raises(ValueError, match="not an EXPLAIN"):
        plan_reasons({"rows": 1})


def test_lists_the_plan_shapes_the_base_plan_lacks() -> None:
    reasons = audit_reasons(BASE, BASE, 1.5, plan=GATED_UNION_PLAN, base_plan=PLAIN_PLAN)
    same = audit_reasons(BASE, BASE, 1.5, plan=GATED_UNION_PLAN, base_plan=GATED_UNION_PLAN)
    unreadable_base = audit_reasons(BASE, BASE, 1.5, plan=PLAIN_PLAN, base_plan="oops")

    assert reasons == plan_reasons(GATED_UNION_PLAN)
    assert same == []
    assert unreadable_base == []
