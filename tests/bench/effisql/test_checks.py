"""Unit tests of the grader's checks: the plan check, the robustness check and the role guard."""

from typing import Any

import pytest

from goaly.bench.effisql.checks import (
    alt_matches,
    inspect_plan,
    plan_nodes,
    read_only_problems,
    scanned_relations,
)
from goaly.bench.effisql.grader import (
    DigestBuilder,
    GraderError,
    QueryError,
    QueryTimeoutError,
    ResultDigest,
)

type Node = dict[str, Any]


def node(kind: str, *children: Node, relation: str | None = None, link: str | None = None) -> Node:
    built: Node = {"Node Type": kind}
    if relation is not None:
        built["Relation Name"] = relation
    if link is not None:
        built["Parent Relationship"] = link
    if children:
        built["Plans"] = list(children)
    return built


def scan(relation: str, *, link: str = "Outer", kind: str = "Seq Scan") -> Node:
    return node(kind, relation=relation, link=link)


def doc(root: Node) -> dict[str, Any]:
    return {"Plan": root}


BASE = frozenset({"lineitem", "supplier", "nation"})


def test_scanned_relations_reads_every_scan_subplans_included() -> None:
    root = node(
        "Hash Join",
        scan("lineitem", kind="Index Only Scan"),
        node("Hash", scan("nation", kind="Bitmap Heap Scan"), link="Inner"),
        scan("orders", link="SubPlan"),
        node("Function Scan", link="InitPlan"),
    )
    assert scanned_relations(doc(root)) == {"lineitem", "nation", "orders"}
    assert len(list(plan_nodes(root))) == 6


def test_a_document_without_a_plan_is_rejected() -> None:
    with pytest.raises(ValueError, match="no Plan"):
        scanned_relations({"Planning Time": 1.0})


class TestPlanCheck:
    @pytest.mark.parametrize(
        "root",
        [
            pytest.param(node("Result"), id="SELECT 12345.67"),
            pytest.param(node("Values Scan"), id="SELECT * FROM (VALUES ...)"),
            pytest.param(node("Sort", node("Values Scan", link="Outer")), id="sorted VALUES"),
            pytest.param(
                node("Aggregate", node("Values Scan", link="Outer")), id="aggregate over VALUES"
            ),
            pytest.param(
                node(
                    "Result",
                    node("Values Scan", link="Outer"),
                    scan("lineitem", link="InitPlan"),
                ),
                id="VALUES behind an InitPlan over lineitem",
            ),
            pytest.param(node("Function Scan"), id="generate_series"),
            pytest.param(scan("nation", link="Outer"), id="constant answer over nation"),
            pytest.param(
                node("Hash Join", scan("nation"), node("Values Scan", link="Inner")),
                id="VALUES joined to nation",
            ),
            pytest.param(scan("orders"), id="none of the base's tables"),
        ],
    )
    def test_flags_a_literal_result(self, root: Node) -> None:
        verdict = inspect_plan(doc(root), BASE)
        assert verdict.literal_result
        assert not verdict.modifies_data

    @pytest.mark.parametrize(
        "root",
        [
            pytest.param(
                node(
                    "HashAggregate",
                    node(
                        "Nested Loop",
                        node("Hash Join", scan("lineitem"), scan("supplier", link="Inner")),
                        node("Values Scan", link="Inner"),
                        link="Outer",
                    ),
                ),
                id="task 1's unpivot with VALUES",
            ),
            pytest.param(node("Result", scan("lineitem", link="InitPlan")), id="scalar subquery"),
            pytest.param(
                node("Result", node("Seq Scan", relation="lineitem", link="Outer")),
                id="Result over a scan",
            ),
            pytest.param(
                node("Append", scan("lineitem", link="Member"), node("Values Scan", link="Member")),
                id="a scan appended to VALUES",
            ),
            pytest.param(scan("lineitem"), id="one of the base's tables"),
        ],
    )
    def test_passes_a_query_that_reads_the_data(self, root: Node) -> None:
        assert not inspect_plan(doc(root), BASE).literal_result

    def test_a_base_on_only_scale_invariant_tables_needs_one_of_them(self) -> None:
        small = frozenset({"nation", "region"})
        assert not inspect_plan(doc(scan("region")), small).literal_result
        assert inspect_plan(doc(scan("lineitem")), small).literal_result

    def test_detects_a_modify_table_anywhere(self) -> None:
        root = node(
            "CTE Scan",
            node("ModifyTable", scan("region"), relation="region", link="InitPlan"),
        )
        verdict = inspect_plan(doc(root), frozenset({"region"}))
        assert verdict.modifies_data
        assert verdict.relations == {"region"}


def digest(rows: list[tuple[Any, ...]], *, complete: bool = True) -> ResultDigest:
    builder = DigestBuilder()
    builder.add(rows)
    return builder.digest(complete=complete)


class FakeAlt:
    def __init__(self, outcome: ResultDigest | Exception) -> None:
        self.outcome = outcome
        self.calls: list[tuple[str, int | None]] = []

    def digest(self, sql: str, *, row_limit: int | None = None) -> ResultDigest:
        self.calls.append((sql, row_limit))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class TestRobustnessCheck:
    BASE_ALT = digest([(1,), (2,)])

    def test_the_same_result_on_the_alternate_database_passes(self) -> None:
        alt = FakeAlt(digest([(2,), (1,)]))
        assert alt_matches(alt, "SELECT x", self.BASE_ALT)  # type: ignore[arg-type]  # A fake.
        assert alt.calls == [("SELECT x", 2)]

    @pytest.mark.parametrize(
        "outcome",
        [
            pytest.param(digest([(1,)]), id="different rows"),
            pytest.param(digest([(1,), (2,), (3,)], complete=False), id="too many rows"),
            pytest.param(QueryError("division by zero"), id="an error"),
            pytest.param(QueryTimeoutError("late"), id="a timeout"),
        ],
    )
    def test_anything_else_fails(self, outcome: ResultDigest | Exception) -> None:
        assert not alt_matches(FakeAlt(outcome), "SELECT x", self.BASE_ALT)  # type: ignore[arg-type]  # A fake.

    def test_a_grader_error_propagates(self) -> None:
        with pytest.raises(GraderError):
            alt_matches(FakeAlt(GraderError("lost")), "SELECT x", self.BASE_ALT)  # type: ignore[arg-type]  # A fake.


class FakeRoleConn:
    def __init__(self, row: tuple[Any, ...]) -> None:
        self.row = row

    def execute(self, _query: str) -> "FakeRoleConn":
        return self

    def fetchone(self) -> tuple[Any, ...]:
        return self.row


class TestReadOnlyGuard:
    def test_the_grader_role_as_set_up_has_no_problem(self) -> None:
        conn = FakeRoleConn(("effisql_grader", "on", False, False, False, []))
        assert read_only_problems(conn) == []  # type: ignore[arg-type]  # A fake.

    def test_reports_every_problem(self) -> None:
        conn = FakeRoleConn(("postgres", "off", True, True, False, ["lineitem", "orders"]))
        assert read_only_problems(conn) == [  # type: ignore[arg-type]  # A fake.
            "connected as postgres, not effisql_grader",
            "default_transaction_read_only is not on",
            "postgres has a superuser, CREATEROLE, CREATEDB or BYPASSRLS attribute",
            "postgres may CREATE in the database or the public schema",
            "postgres may write to lineitem, orders",
        ]
