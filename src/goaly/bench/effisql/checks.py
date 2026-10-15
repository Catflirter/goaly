"""The grader's checks against hard-coded results and data changes (spec §4.1, §8.5).

A prediction can return the base query's result as constants, such as ``SELECT 12345.67`` or
``SELECT * FROM (VALUES …)``: it matches and scores τ = 1000. Two checks give it no credit:

- **The robustness check** (:func:`alt_matches`): the prediction must also return the base
  query's result on the alternate TPC-H database, whose data differs.
- **The plan check** (:func:`inspect_plan`). It flags ``literal_result`` when any of these holds:

  - the plan scans none of the base query's tables. When the base reads a table whose data
    depends on the scale, the prediction must scan one of those: region and nation hold the
    same rows at every scale, so a constant answer over them would pass the robustness check;
  - the output comes from a Values Scan, or from a Result with no table scan beneath it,
    counting the InitPlans and CTEs that PostgreSQL attaches to the nodes above it;
  - a scan-free branch of more than one row reaches the output through an Append (a UNION);
  - the SQL calls a function that tells the databases apart, such as ``current_database()``:
    a prediction could return precomputed rows on one database and compute them on the other.

  ``VALUES`` joined to real rows stays allowed: task 1's gold unpivots lineitem's columns with
  it. The plan check also finds ``ModifyTable`` nodes, so a data change is refused before it runs.

goaly.bench.effisql.grading.evaluate decides when each check runs. :func:`read_only_problems` is
the guard on the grading role itself.
"""

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from goaly.bench.effisql.grader import QueryError, QueryTimeoutError, ResultDigest
from goaly.bench.effisql.pg_settings import GRADER_ROLE

__all__ = [
    "ENVIRONMENT_PROBES",
    "RELATION_SCANS",
    "SCALE_INVARIANT_TABLES",
    "AltOutcome",
    "DigestSession",
    "PlanVerdict",
    "RoleConnection",
    "alt_matches",
    "environment_probes",
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
# Functions, views and columns that differ between the main and the alternate database, or
# between sessions, so a query can branch on them (the controller's ruling).
ENVIRONMENT_PROBES = re.compile(
    r"\b(current_database|current_catalog|current_setting|current_user|current_role|"
    r"session_user|inet_server_\w+|inet_client_\w+|pg_backend_pid|version|"
    r"pg_postmaster_start_time|pg_conf_load_time|pg_\w+_size|pg_relation_filenode|"
    r"pg_stat\w*|pg_class|pg_database|pg_settings|regclass|to_regclass|tableoid)\b"
)
# Children that compute a value for their parent rather than feed it rows.
_SUBPLAN_LINKS = frozenset({"InitPlan", "SubPlan"})
_APPENDS = frozenset({"Append", "Merge Append"})
_DOLLAR_TAG = re.compile(r"\$(?:[A-Za-z_][A-Za-z0-9_]*)?\$")
# Roles whose membership would let a prediction reach the server's files or other sessions.
_DANGEROUS_ROLES = (
    "pg_signal_backend",
    "pg_read_server_files",
    "pg_write_server_files",
    "pg_execute_server_program",
    "pg_write_all_data",
)


@dataclass(frozen=True, slots=True)
class PlanVerdict:
    """What the plan check found in one plan."""

    modifies_data: bool  # The plan has a ModifyTable node: refuse it before it runs.
    literal_result: bool  # The literal_result flag (see the module docstring).
    relations: frozenset[str]  # The tables the plan scans.
    reasons: tuple[str, ...] = ()  # Why literal_result is set, for the feedback.


@dataclass(frozen=True, slots=True)
class AltOutcome:
    """The robustness check's outcome, and why it failed when it did."""

    matched: bool
    reason: str | None = None


def plan_nodes(node: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    """Yield ``node`` and every node beneath it, subplans included, depth first."""
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        children = current.get("Plans") or ()
        stack.extend(child for child in reversed(children) if isinstance(child, Mapping))


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


def environment_probes(sql: str) -> list[str]:
    """Return the names in ``sql``'s code, outside strings and comments, that probe the
    environment (:data:`ENVIRONMENT_PROBES`), in order of appearance."""
    return [match.group(1) for match in ENVIRONMENT_PROBES.finditer(_code(sql))]


def inspect_plan(
    document: Mapping[str, Any], base_relations: frozenset[str], *, sql: str
) -> PlanVerdict:
    """Run the plan check on a prediction's plan and SQL, given the base query's tables."""
    if not base_relations:
        raise ValueError("the base query's plan scans no table, so nothing can be checked")
    root = _root(document)
    relations = scanned_relations(document)
    required = base_relations - SCALE_INVARIANT_TABLES or base_relations
    reasons = []
    if not relations & required:
        reasons.append(f"the plan scans none of the base query's tables {sorted(required)}")
    literal_output = _literal_output(root)
    if literal_output is not None:
        reasons.append(literal_output)
    probes = environment_probes(sql)
    if probes:
        reasons.append(f"the query probes the environment: {', '.join(sorted(set(probes)))}")
    modifies = any(node.get("Node Type") == "ModifyTable" for node in plan_nodes(root))
    return PlanVerdict(
        modifies_data=modifies,
        literal_result=bool(reasons),
        relations=relations,
        reasons=tuple(reasons),
    )


def _literal_output(root: Mapping[str, Any]) -> str | None:
    """Say how the output comes from no table data, or return None if it does not.

    The output's source is followed down from the root through each node with one input of
    rows, and through a CTE Scan into its CTE. PostgreSQL attaches a query level's InitPlans and
    CTEs to the topmost node of that level, so a scan in any of them counts as beneath a Result
    further down. A Values Scan on the path is always literal. An Append is literal when any of
    its branches is a scan-free subtree of more than one row.
    """
    ctes = _ctes(root)
    above: list[Mapping[str, Any]] = []  # The subplans of the nodes on the path so far.
    node: Mapping[str, Any] | None = root
    while node is not None:
        kind = node.get("Node Type")
        children = [c for c in node.get("Plans") or () if isinstance(c, Mapping)]
        inputs = [c for c in children if c.get("Parent Relationship") not in _SUBPLAN_LINKS]
        above.extend(c for c in children if c.get("Parent Relationship") in _SUBPLAN_LINKS)
        if kind == "Values Scan":
            return "the output comes from a Values Scan"
        if kind == "Result" and not any(_reads_table(n, ctes) for n in (node, *above)):
            return "the output comes from a Result with no table scan beneath it"
        if kind in _APPENDS:
            if any(_literal_branch(branch, ctes) for branch in inputs):
                return "a branch of constant rows reaches the output through a union"
            return None
        if kind == "CTE Scan":
            node = ctes.get(str(node.get("CTE Name")))
            continue
        node = inputs[0] if len(inputs) == 1 else None
    return None


def _ctes(root: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    prefix = "CTE "
    return {
        str(node["Subplan Name"])[len(prefix) :]: node
        for node in plan_nodes(root)
        if str(node.get("Subplan Name", "")).startswith(prefix)
    }


def _reads_table(node: Mapping[str, Any], ctes: Mapping[str, Mapping[str, Any]]) -> bool:
    """Say whether a table scan is beneath ``node``, following CTE Scans into their CTEs."""
    seen: set[str] = set()
    pending = [node]
    while pending:
        for current in plan_nodes(pending.pop()):
            kind = current.get("Node Type")
            if kind in RELATION_SCANS:
                return True
            name = str(current.get("CTE Name"))
            if kind == "CTE Scan" and name in ctes and name not in seen:
                seen.add(name)
                pending.append(ctes[name])
    return False


def _literal_branch(branch: Mapping[str, Any], ctes: Mapping[str, Mapping[str, Any]]) -> bool:
    rows = branch.get("Plan Rows")
    many = isinstance(rows, int | float) and rows > 1
    return many and not _reads_table(branch, ctes)


def _code(sql: str) -> str:
    """Return ``sql`` with comments and string literals blanked and unquoted text lowercased.

    A quoted identifier keeps its text, without the quotes, as it names the same thing.
    """
    out: list[str] = []
    i, n = 0, len(sql)
    while i < n:
        if sql.startswith("--", i):
            end = sql.find("\n", i)
            i = n if end < 0 else end
            out.append(" ")
            continue
        if sql.startswith("/*", i):
            depth = 0
            while i < n:
                if sql.startswith("/*", i):
                    depth, i = depth + 1, i + 2
                elif sql.startswith("*/", i):
                    depth, i = depth - 1, i + 2
                    if depth == 0:
                        break
                else:
                    i += 1
            out.append(" ")
            continue
        char = sql[i]
        word_before = i > 0 and (sql[i - 1].isalnum() or sql[i - 1] in "_$")
        if char == "'" or (char in "eE" and sql.startswith("'", i + 1) and not word_before):
            i = _string_end(sql, i + (char != "'"), escapes=char != "'")
            out.append(" ")
        elif char == '"':
            end = i + 1
            while end < n and not (sql[end] == '"' and not sql.startswith('""', end)):
                end += 2 if sql.startswith('""', end) else 1
            out.append(sql[i + 1 : end].replace('""', '"'))
            i = end + 1
        elif char == "$" and not word_before and (tag := _DOLLAR_TAG.match(sql, i)):
            close = sql.find(tag.group(), tag.end())
            i = n if close < 0 else close + len(tag.group())
            out.append(" ")
        else:
            out.append(char.lower())
            i += 1
    return "".join(out)


def _string_end(sql: str, quote: int, *, escapes: bool) -> int:
    i = quote + 1
    while i < len(sql):
        if (escapes and sql[i] == "\\") or sql.startswith("''", i):
            i += 2
        elif sql[i] == "'":
            return i + 1
        else:
            i += 1
    return len(sql)


class DigestSession(Protocol):
    """What the robustness check needs of a session: goaly.bench.effisql.grader.PgSession."""

    def digest(self, sql: str, *, row_limit: int | None = None) -> ResultDigest: ...


def alt_matches(alt: DigestSession, sql: str, base_alt: ResultDigest) -> AltOutcome:
    """Run the robustness check: does ``sql`` return the base query's result on the alternate
    database? A prediction that fails or times out there does not, and the outcome says why.

    Raises GraderError if the grader's own connection fails.
    """
    if not base_alt.complete:
        raise ValueError("the base query's result on the alternate database is incomplete")
    try:
        digest = alt.digest(sql, row_limit=base_alt.row_count)
    except QueryError as error:
        return AltOutcome(False, f"the prediction failed on the alternate database: {error}")
    except QueryTimeoutError:
        return AltOutcome(False, "the prediction timed out on the alternate database")
    if digest.matches(base_alt):
        return AltOutcome(True)
    return AltOutcome(False, "the prediction's result differs from the base's there")


class _Row(Protocol):
    def fetchone(self) -> Any: ...


class RoleConnection(Protocol):
    """What the role guard needs of a connection: a psycopg Connection."""

    def execute(self, query: str) -> _Row: ...


_ROLE_SQL = f"""
SELECT current_user,
       current_setting('default_transaction_read_only'),
       (SELECT rolsuper OR rolcreaterole OR rolcreatedb OR rolbypassrls OR rolreplication
          FROM pg_roles WHERE rolname = current_user),
       has_database_privilege(current_database(), 'CREATE'),
       has_database_privilege(current_database(), 'TEMP'),
       ARRAY(SELECT nspname FROM pg_namespace WHERE has_schema_privilege(oid, 'CREATE')
             ORDER BY nspname),
       ARRAY(SELECT rolname FROM pg_roles
             WHERE rolname IN ({", ".join(f"'{role}'" for role in _DANGEROUS_ROLES)})
               AND pg_has_role(current_user, oid, 'MEMBER') ORDER BY rolname),
       has_function_privilege('pg_cancel_backend(integer)', 'EXECUTE')
         OR has_function_privilege('pg_terminate_backend(integer, bigint)', 'EXECUTE'),
       ARRAY(SELECT n.nspname || '.' || c.relname FROM pg_class c
             JOIN pg_namespace n ON n.oid = c.relnamespace
             WHERE n.nspname NOT IN ('pg_catalog', 'information_schema')
               AND n.nspname NOT LIKE 'pg_toast%'
               AND c.relkind IN ('r', 'p', 'v', 'f', 'm')
               AND has_table_privilege(c.oid, 'INSERT, UPDATE, DELETE, TRUNCATE, TRIGGER')
             ORDER BY 1),
       ARRAY(SELECT n.nspname || '.' || c.relname FROM pg_class c
             JOIN pg_namespace n ON n.oid = c.relnamespace
             WHERE c.relkind = 'S'
               AND CASE WHEN c.relkind = 'S' THEN has_sequence_privilege(c.oid, 'UPDATE') END
             ORDER BY 1)
"""  # noqa: S608  # Only the constant role names are interpolated.


def read_only_problems(conn: RoleConnection) -> list[str]:
    """Return what is wrong with ``conn`` as a grading connection; empty if nothing is.

    It must be the grader's role, read-only by default, with no role attribute that grants
    more, no CREATE or TEMP on the database, no CREATE on any schema, no membership in a role
    that reaches files or other sessions, no right to cancel or end other sessions, and no
    table, view or sequence it may change, in any schema.
    """
    row = conn.execute(_ROLE_SQL).fetchone()
    if row is None:  # pragma: no cover - a SELECT without FROM always returns its row.
        return ["the role check returned no row"]
    user, read_only, elevated, create_db, temp, schemas, roles, signal, writable, sequences = row
    checks = [
        (user != GRADER_ROLE, f"connected as {user}, not {GRADER_ROLE}"),
        (read_only != "on", "default_transaction_read_only is not on"),
        (elevated, f"{user} has a superuser, CREATEROLE, CREATEDB, BYPASSRLS or REPLICATION role"),
        (create_db, f"{user} may CREATE in the database"),
        (temp, f"{user} may create temporary tables"),
        (schemas, f"{user} may CREATE in the schemas {', '.join(schemas or ())}"),
        (roles, f"{user} is a member of {', '.join(roles or ())}"),
        (signal, f"{user} may cancel or end other sessions"),
        (writable, f"{user} may write to {', '.join(writable or ())}"),
        (sequences, f"{user} may advance the sequences {', '.join(sequences or ())}"),
    ]
    return [message for failed, message in checks if failed]
