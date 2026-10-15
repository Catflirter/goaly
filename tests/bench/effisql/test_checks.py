"""Unit tests of the grader's checks: the plan check, the robustness check and the role guard."""

from typing import Any

import pytest

from goaly.bench.effisql.checks import (
    AltOutcome,
    alt_matches,
    environment_probes,
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


def node(
    kind: str,
    *children: Node,
    relation: str | None = None,
    link: str | None = None,
    rows: float | None = None,
    fields: dict[str, Any] | None = None,
) -> Node:
    built: Node = {"Node Type": kind, **(fields or {})}
    if relation is not None:
        built["Relation Name"] = relation
    if link is not None:
        built["Parent Relationship"] = link
    if rows is not None:
        built["Plan Rows"] = rows
    if children:
        built["Plans"] = list(children)
    return built


def scan(relation: str, *, link: str = "Outer", kind: str = "Seq Scan") -> Node:
    return node(kind, relation=relation, link=link)


def doc(root: Node) -> dict[str, Any]:
    return {"Plan": root}


BASE = frozenset({"lineitem", "supplier", "nation"})


def check(root: Node, *, base: frozenset[str] = BASE, sql: str = "SELECT 1") -> Any:
    return inspect_plan(doc(root), base, sql=sql)


def test_scanned_relations_reads_every_scan_subplans_included() -> None:
    root = node(
        "Hash Join",
        scan("lineitem", kind="Index Only Scan"),
        node("Hash", scan("nation", kind="Bitmap Heap Scan"), link="Inner"),
        scan("orders", link="SubPlan"),
        node("Function Scan", link="InitPlan"),
        scan("part", kind="Tid Scan"),
        scan("partsupp", kind="Tid Range Scan"),
        scan("customer", kind="Sample Scan"),
    )
    assert scanned_relations(doc(root)) == {
        "lineitem", "nation", "orders", "part", "partsupp", "customer"
    }  # fmt: skip
    assert len(list(plan_nodes(root))) == 9


def test_plan_nodes_walks_a_deep_plan_without_recursion() -> None:
    root = scan("lineitem")
    for _ in range(5000):
        root = node("Materialize", root)
    assert len(list(plan_nodes(root))) == 5001
    assert not check(root).literal_result


def test_a_document_without_a_plan_is_rejected() -> None:
    with pytest.raises(ValueError, match="no Plan"):
        scanned_relations({"Planning Time": 1.0})


def test_a_base_without_tables_is_refused() -> None:
    with pytest.raises(ValueError, match="scans no table"):
        check(scan("lineitem"), base=frozenset())


class TestPlanCheck:
    @pytest.mark.parametrize(
        "root",
        [
            pytest.param(node("Result"), id="SELECT 12345.67"),
            pytest.param(node("Values Scan", rows=2), id="SELECT * FROM (VALUES ...)"),
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
            pytest.param(
                node(
                    "Result",
                    node("CTE Scan", link="Outer", fields={"CTE Name": "c"}),
                    node("Values Scan", link="InitPlan", fields={"Subplan Name": "CTE c"}),
                    scan("lineitem", link="InitPlan"),
                ),
                id="a VALUES CTE behind EXISTS over lineitem",
            ),
            pytest.param(
                node(
                    "Append",
                    node("Values Scan", link="Member", rows=4),
                    scan("lineitem", link="Member"),
                ),
                id="VALUES rows in a UNION ALL with the real query",
            ),
            pytest.param(
                node(
                    "HashAggregate",
                    node(
                        "Append",
                        scan("lineitem", link="Member"),
                        node("Subquery Scan", node("Result", rows=3), link="Member", rows=3),
                        link="Outer",
                    ),
                ),
                id="constant rows in a UNION",
            ),
        ],
    )
    def test_flags_a_literal_result(self, root: Node) -> None:
        verdict = check(root)
        assert verdict.literal_result
        assert verdict.reasons
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
                node("Limit", node("Result", link="Outer"), scan("lineitem", link="InitPlan")),
                id="SELECT max(...) LIMIT 1: the InitPlan hangs on the Limit",
            ),
            pytest.param(
                node("ProjectSet", node("Result", link="Outer"), scan("lineitem", link="InitPlan")),
                id="a set-returning function over an InitPlan",
            ),
            pytest.param(
                node(
                    "Sort",
                    node("Result", node("CTE Scan", link="Outer", fields={"CTE Name": "c"})),
                    node(
                        "Aggregate",
                        scan("lineitem"),
                        link="InitPlan",
                        fields={"Subplan Name": "CTE c"},
                    ),
                ),
                id="a Result over a CTE that scans lineitem",
            ),
            pytest.param(
                node(
                    "Append",
                    node("Values Scan", link="Member", rows=1),
                    scan("lineitem", link="Member"),
                ),
                id="one constant row beside the real query, VALUES first",
            ),
            pytest.param(
                node(
                    "Append",
                    scan("lineitem", link="Member"),
                    node("Result", link="Member", rows=1),
                ),
                id="a one-row total line in a UNION ALL",
            ),
            pytest.param(scan("lineitem"), id="one of the base's tables"),
        ],
    )
    def test_passes_a_query_that_reads_the_data(self, root: Node) -> None:
        verdict = check(root)
        assert not verdict.literal_result, verdict.reasons

    def test_a_base_on_only_scale_invariant_tables_needs_one_of_them(self) -> None:
        small = frozenset({"nation", "region"})
        assert not check(scan("region"), base=small).literal_result
        assert check(scan("lineitem"), base=small).literal_result

    def test_detects_a_modify_table_anywhere(self) -> None:
        root = node(
            "CTE Scan",
            node("ModifyTable", scan("region"), relation="region", link="InitPlan"),
        )
        verdict = check(root, base=frozenset({"region"}))
        assert verdict.modifies_data
        assert verdict.relations == {"region"}

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT * FROM v WHERE current_database() = 'tpch'",
            "SELECT * FROM v WHERE pg_catalog.CURRENT_DATABASE() = 'tpch'",
            "SELECT * FROM v WHERE \"current_database\"() = 'tpch'",
            "SELECT CASE WHEN current_setting('cluster_name') = '' THEN 1 END",
            "SELECT 1 WHERE current_user = 'x' OR session_user = 'y'",
            "SELECT inet_server_port(), pg_backend_pid(), version()",
            "SELECT reltuples FROM pg_class WHERE relname = 'lineitem'",
            "SELECT pg_total_relation_size('lineitem'), pg_stat_get_live_tuples(1)",
            "SELECT 'lineitem'::regclass::oid, tableoid FROM lineitem",
        ],
    )
    def test_an_environment_probe_is_flagged(self, sql: str) -> None:
        verdict = check(scan("lineitem"), sql=sql)
        assert verdict.literal_result
        assert any("probes the environment" in reason for reason in verdict.reasons)

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT 'current_database()' FROM lineitem",
            "SELECT 1 -- current_database()\nFROM lineitem",
            "SELECT 1 /* version() /* nested */ */ FROM lineitem",
            "SELECT $$ current_setting('x') $$, E'it\\'s version()' FROM lineitem",
            "SELECT 'it''s version()' FROM lineitem",
            "SELECT l_conversion, versioned FROM lineitem",
        ],
    )
    def test_names_in_strings_comments_or_other_words_are_not_probes(self, sql: str) -> None:
        assert environment_probes(sql) == []


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
        assert alt_matches(alt, "SELECT x", self.BASE_ALT) == AltOutcome(True)
        assert alt.calls == [("SELECT x", 2)]

    @pytest.mark.parametrize(
        ("outcome", "reason"),
        [
            pytest.param(digest([(1,)]), "differs from the base's", id="different rows"),
            pytest.param(digest([(1,), (2,), (3,)], complete=False), "differs", id="too many rows"),
            pytest.param(
                QueryError("division by zero"),
                "failed on the alternate database: division by zero",
                id="an error",
            ),
            pytest.param(QueryTimeoutError("late"), "timed out on the alternate", id="a timeout"),
        ],
    )
    def test_anything_else_fails_and_says_why(
        self, outcome: ResultDigest | Exception, reason: str
    ) -> None:
        result = alt_matches(FakeAlt(outcome), "SELECT x", self.BASE_ALT)
        assert not result.matched
        assert reason in str(result.reason)

    def test_a_grader_error_propagates(self) -> None:
        with pytest.raises(GraderError):
            alt_matches(FakeAlt(GraderError("lost")), "SELECT x", self.BASE_ALT)

    def test_an_incomplete_base_digest_is_refused(self) -> None:
        with pytest.raises(ValueError, match="incomplete"):
            alt_matches(FakeAlt(digest([])), "SELECT x", digest([(1,)], complete=False))


class FakeRoleConn:
    def __init__(self, row: tuple[Any, ...]) -> None:
        self.row = row

    def execute(self, _query: str) -> "FakeRoleConn":
        return self

    def fetchone(self) -> tuple[Any, ...]:
        return self.row


CLEAN: tuple[Any, ...] = ("effisql_grader", "on", False, False, False, [], [], False, [], [])


class TestReadOnlyGuard:
    def test_the_grader_role_as_set_up_has_no_problem(self) -> None:
        assert read_only_problems(FakeRoleConn(CLEAN)) == []

    @pytest.mark.parametrize(
        ("index", "value", "problem"),
        [
            (0, "postgres", "connected as postgres, not effisql_grader"),
            (1, "off", "default_transaction_read_only is not on"),
            (2, True, "superuser, CREATEROLE, CREATEDB, BYPASSRLS or REPLICATION"),
            (3, True, "may CREATE in the database"),
            (4, True, "may create temporary tables"),
            (5, ["public"], "may CREATE in the schemas public"),
            (6, ["pg_signal_backend"], "is a member of pg_signal_backend"),
            (7, True, "may cancel or end other sessions"),
            (8, ["public.lineitem", "public.v"], "may write to public.lineitem, public.v"),
            (9, ["public.s"], "may advance the sequences public.s"),
        ],
    )
    def test_reports_each_problem_on_its_own(self, index: int, value: Any, problem: str) -> None:
        row = list(CLEAN)
        row[index] = value
        problems = read_only_problems(FakeRoleConn(tuple(row)))
        assert len(problems) == 1
        assert problem in problems[0]
