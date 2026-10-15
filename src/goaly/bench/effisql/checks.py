"""The grader's checks against hard-coded results and data changes (spec §4.1, §8.5).

A prediction can return the base query's result as constants, such as ``SELECT 12345.67`` or
``SELECT * FROM (VALUES …)``: it matches and scores τ = 1000. Two checks give it no credit:

- **The robustness check** (:func:`alt_matches`): the prediction must also return the base
  query's result on the alternate TPC-H database, whose data differs.
- **The plan check** (:func:`inspect_plan`), on the prediction's ``EXPLAIN (FORMAT JSON)``. It
  flags ``literal_result`` if the plan scans none of the base query's relations, or if the
  output comes from a Values Scan or a Result subtree with no relation scan beneath it.
  ``VALUES`` itself stays allowed: task 1's gold unpivots lineitem's columns with it.

region and nation hold the same rows at every TPC-H scale, so a constant answer over them
passes the robustness check. The plan check closes that gap: when the base query reads a table
whose data depends on the scale, the prediction must scan one of those tables too.

Both checks run only on a matched prediction (the controller's ruling): an unmatched one fails
on the alternate database too, and flagging it would say nothing. The plan check's other job
runs first, on every prediction: a plan with a ``ModifyTable`` node is refused before it runs.

:func:`read_only_problems` is the guard on the grading role itself: a grading connection must be
the grader's role, read-only by default, with no privilege to change data or the schema.
"""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import psycopg

from goaly.bench.effisql.grader import PgSession, QueryError, QueryTimeoutError, ResultDigest
from goaly.bench.effisql.pg_settings import GRADER_ROLE

__all__ = [
    "RELATION_SCANS",
    "SCALE_INVARIANT_TABLES",
    "PlanVerdict",
    "alt_matches",
    "inspect_plan",
    "plan_nodes",
    "read_only_problems",
    "scanned_relations",
]

# The plan nodes that read a table's rows.
RELATION_SCANS = frozenset(
    {"Seq Scan", "Index Scan", "Index Only Scan", "Bitmap Heap Scan", "Tid Scan",
     "Tid Range Scan", "Sample Scan"}
)  # fmt: skip
# TPC-H tables whose rows are the same at every scale factor.
SCALE_INVARIANT_TABLES = frozenset({"region", "nation"})
# Children that compute a value for their parent rather than feed it rows.
_SUBPLAN_LINKS = frozenset({"InitPlan", "SubPlan"})
_LITERAL_SOURCES = frozenset({"Values Scan", "Result"})


@dataclass(frozen=True, slots=True)
class PlanVerdict:
    """What the plan check found in one plan."""

    modifies_data: bool  # The plan has a ModifyTable node: refuse it before it runs.
    literal_result: bool  # The literal_result flag (see the module docstring).
    relations: frozenset[str]  # The tables the plan scans.


def plan_nodes(node: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    """Yield ``node`` and every node beneath it, subplans included, depth first."""
    yield node
    for child in node.get("Plans") or ():
        if isinstance(child, Mapping):
            yield from plan_nodes(child)


def _root(document: Mapping[str, Any]) -> Mapping[str, Any]:
    root = document.get("Plan")
    if not isinstance(root, Mapping):
        raise ValueError("not an EXPLAIN (FORMAT JSON) document: it has no Plan")
    return root


def scanned_relations(document: Mapping[str, Any]) -> frozenset[str]:
    """Return the tables an ``EXPLAIN (FORMAT JSON)`` document's plan scans."""
    return frozenset(
        str(node["Relation Name"])
        for node in plan_nodes(_root(document))
        if node.get("Node Type") in RELATION_SCANS and "Relation Name" in node
    )


def inspect_plan(document: Mapping[str, Any], base_relations: frozenset[str]) -> PlanVerdict:
    """Run the plan check on a prediction's plan, given the tables the base query's plan scans."""
    root = _root(document)
    relations = scanned_relations(document)
    required = base_relations - SCALE_INVARIANT_TABLES or base_relations
    literal = not (relations & required) or _output_is_literal(root)
    modifies = any(node.get("Node Type") == "ModifyTable" for node in plan_nodes(root))
    return PlanVerdict(modifies_data=modifies, literal_result=literal, relations=relations)


def _output_is_literal(root: Mapping[str, Any]) -> bool:
    """Say whether the output comes from a Values Scan, or a Result with no scan beneath it.

    The output's source is followed down from the root through each node that has one input of
    rows; subplans, which only compute values, are not inputs. A join or an Append has several
    inputs, so a Values Scan joined to a table's rows, as in task 1's gold, is not the output.
    """
    node: Mapping[str, Any] | None = root
    while node is not None:
        kind = node.get("Node Type")
        if kind in _LITERAL_SOURCES and not any(
            child.get("Node Type") in RELATION_SCANS for child in plan_nodes(node)
        ):
            return True
        inputs = [
            child
            for child in node.get("Plans") or ()
            if isinstance(child, Mapping) and child.get("Parent Relationship") not in _SUBPLAN_LINKS
        ]
        node = inputs[0] if len(inputs) == 1 else None
    return False


def alt_matches(alt: PgSession, sql: str, base_alt: ResultDigest) -> bool:
    """Run the robustness check: does ``sql`` return the base query's result on the alternate
    database? A prediction that fails or times out there does not.

    Raises GraderError if the grader's own connection fails.
    """
    try:
        digest = alt.digest(sql, row_limit=base_alt.row_count)
    except (QueryError, QueryTimeoutError):
        return False
    return digest.matches(base_alt)


_ROLE_SQL = """
SELECT current_user,
       current_setting('default_transaction_read_only'),
       (SELECT rolsuper OR rolcreaterole OR rolcreatedb OR rolbypassrls
          FROM pg_roles WHERE rolname = current_user),
       has_database_privilege(current_database(), 'CREATE'),
       has_schema_privilege('public', 'CREATE'),
       ARRAY(SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
             WHERE n.nspname = 'public' AND c.relkind IN ('r', 'p')
               AND has_table_privilege(c.oid, 'INSERT, UPDATE, DELETE, TRUNCATE')
             ORDER BY c.relname)
"""


def read_only_problems(conn: psycopg.Connection[Any]) -> list[str]:
    """Return what is wrong with ``conn`` as a grading connection; empty if nothing is.

    It must be the grader's role, read-only by default, with no role attribute that grants more,
    no CREATE on the database or the public schema, and no table it may write to.
    """
    row = conn.execute(_ROLE_SQL).fetchone()
    if row is None:  # pragma: no cover - a SELECT without FROM always returns its row.
        return ["the role check returned no row"]
    user, read_only, elevated, create_db, create_schema, writable = row
    problems = []
    if user != GRADER_ROLE:
        problems.append(f"connected as {user}, not {GRADER_ROLE}")
    if read_only != "on":
        problems.append("default_transaction_read_only is not on")
    if elevated:
        problems.append(f"{user} has a superuser, CREATEROLE, CREATEDB or BYPASSRLS attribute")
    if create_db or create_schema:
        problems.append(f"{user} may CREATE in the database or the public schema")
    if writable:
        problems.append(f"{user} may write to {', '.join(writable)}")
    return problems
