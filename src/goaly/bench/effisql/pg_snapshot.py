"""The schema snapshot and the plan cache: what T08 puts in the agent's input (spec §4.1).

``schema_snapshot.json`` describes one database: its server version, each table's columns, keys,
indexes and exact row count, and the DDL text. ``plans.json`` maps instance IDs to our own
``EXPLAIN (FORMAT JSON)`` plans, for the tasks whose task file gives no plan or a plan recorded
on another TPC-H scale. Both are plain JSON with sorted keys; T08 reads them with its own models,
and neither ticket imports the other.
"""

import json
import os
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import psycopg
from psycopg import sql
from psycopg.types.string import TextLoader
from pydantic import BaseModel, ConfigDict, Field

from goaly.bench.effisql.pg_build import TPCH, TPCH_ALT, Connect
from goaly.bench.effisql.pg_settings import PgEnvConfig
from goaly.contracts.interfaces import JsonValue

__all__ = [
    "ColumnInfo",
    "ForeignKeyInfo",
    "IndexInfo",
    "PlanRequest",
    "TableInfo",
    "explain_plans",
    "foreign_scale_plan",
    "plan_requests",
    "render_ddl",
    "schema_snapshot",
    "write_json",
    "write_snapshots",
]

# A scan that returns more rows than the table holds at SF3, by this margin, ran on another scale.
# The margin absorbs estimates a little above the true count; SF10 is 3.3 times SF3.
_SCALE_MARGIN = 1.1
# region and nation are the same at every scale, so they cannot reveal one: a large estimate for
# them means stale statistics, not another scale.
_SF3_ROWS = {
    name: entry.rows for name, entry in TPCH.files.items() if entry != TPCH_ALT.files[name]
}
_SCANS = frozenset(
    {"Seq Scan", "Index Scan", "Index Only Scan", "Bitmap Heap Scan", "Tid Scan", "Sample Scan"}
)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ColumnInfo(_Frozen):
    name: str
    type: str
    nullable: bool


class ForeignKeyInfo(_Frozen):
    columns: tuple[str, ...]
    ref_table: str
    ref_columns: tuple[str, ...]


class IndexInfo(_Frozen):
    name: str
    definition: str


class TableInfo(_Frozen):
    """One table of the snapshot. ``constraints`` (name, definition) only feed the DDL text."""

    name: str
    row_count: int
    columns: tuple[ColumnInfo, ...]
    primary_key: tuple[str, ...]
    foreign_keys: tuple[ForeignKeyInfo, ...]
    indexes: tuple[IndexInfo, ...]
    constraints: tuple[tuple[str, str], ...] = Field(default=(), exclude=True)

    def to_json(self) -> dict[str, JsonValue]:
        dumped: dict[str, JsonValue] = self.model_dump(mode="json")
        return dumped


def render_ddl(tables: Sequence[TableInfo]) -> str:
    """Return the DDL text: each table with its constraints, then any index no constraint owns."""
    statements = []  # One per table, each with its indexes, separated by blank lines.
    for table in tables:
        lines = [
            f"{column.name} {column.type}{'' if column.nullable else ' NOT NULL'}"
            for column in table.columns
        ]
        lines += [f"CONSTRAINT {name} {definition}" for name, definition in table.constraints]
        body = ",\n".join(f"    {line}" for line in lines)
        # A primary key's or unique constraint's index shares the constraint's name.
        owned = {name for name, _ in table.constraints}
        extra = "".join(
            f"{index.definition};\n" for index in table.indexes if index.name not in owned
        )
        statements.append(f"CREATE TABLE {table.name} (\n{body}\n);\n{extra}")
    return "\n".join(statements)


def schema_snapshot(conn: psycopg.Connection[Any]) -> dict[str, JsonValue]:
    """Describe the connected database's tables in the public schema, with exact row counts."""
    tables = [_table_info(conn, oid, name) for oid, name in _rows(conn, _TABLES_SQL)]
    return {
        "db": conn.info.dbname,
        "server_version": str(_scalar(conn, "SHOW server_version")),
        "tables": [table.to_json() for table in tables],
        "ddl": render_ddl(tables),
    }


def write_snapshots(connect: Connect, config: PgEnvConfig, dbs: Sequence[str]) -> None:
    """Write the schema snapshot of each database in ``dbs`` to its path under the data root."""
    for db in dbs:
        with connect(db) as conn:
            write_json(config.snapshot_path(db), schema_snapshot(conn))


def foreign_scale_plan(plan_json: object) -> bool:
    """Say whether a task file's plan was recorded on another TPC-H scale than SF3.

    It was if any pass of a scan over a TPC-H table read, or was estimated to return, more
    rows than the table holds at SF3: counting the rows it returned, the rows its filters
    removed, and the rows a subplan in its filter was run for. Some of the dataset's plans were
    recorded on SF10, where lineitem holds 59,986,052 rows; the agent should see our plan for
    those tasks instead. Accepts a plan's root node, EXPLAIN's object or list, or their JSON
    text; anything else is not one.
    """
    root = _plan_root(plan_json)
    return root is not None and any(_beyond_sf3(node) for node in _nodes(root))


@dataclass(frozen=True)
class PlanRequest:
    instance_id: str
    base_sql: str
    reason: Literal["no plan", "foreign scale"]


def plan_requests(tasks: Path, *, db: str) -> list[PlanRequest]:
    """Return the tasks on ``db`` in the JSONL task file ``tasks`` that need our own plan."""
    requests = []
    for line in tasks.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("db") != db:
            continue
        plan = record.get("base_explain_analyze")
        reason: Literal["no plan", "foreign scale"]
        if plan is None or plan in ({}, []) or (isinstance(plan, str) and not plan.strip()):
            reason = "no plan"
        elif foreign_scale_plan(plan):
            reason = "foreign scale"
        else:
            continue
        instance_id, base_sql = str(record["id"]), str(record["base_sql"])
        requests.append(PlanRequest(instance_id=instance_id, base_sql=base_sql, reason=reason))
    return requests


def explain_plans(
    conn: psycopg.Connection[Any], requests: Sequence[PlanRequest]
) -> tuple[dict[str, str], dict[str, str]]:
    """Plan each request's query with ``EXPLAIN (FORMAT JSON)``; return the plans and errors.

    Both map instance IDs, to the plan as the server printed it, or to the error. Each query
    is only planned, never run, in a read-only transaction, and as one prepared statement, so a
    task's SQL can never smuggle in a second statement.
    """
    plans: dict[str, str] = {}
    errors: dict[str, str] = {}
    with conn.cursor() as cursor:
        cursor.adapters.register_loader("json", TextLoader)  # The server's text, as printed.
        for request in requests:
            statement = sql.SQL("EXPLAIN (FORMAT JSON) {}").format(
                sql.SQL(request.base_sql.strip().rstrip(";").rstrip())
            )
            try:
                with conn.transaction():
                    cursor.execute("SET TRANSACTION READ ONLY")
                    row = cursor.execute(statement, prepare=True).fetchone()
            except psycopg.Error as error:
                errors[request.instance_id] = error.diag.message_primary or type(error).__name__
                continue
            if row is None:
                errors[request.instance_id] = "EXPLAIN returned no row"
            else:
                plans[request.instance_id] = str(row[0])
    return plans, errors


def write_json(path: Path, value: Mapping[str, JsonValue]) -> None:
    """Write ``value`` to ``path`` as JSON with sorted keys, atomically: readers never see half."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        Path(temporary).chmod(0o644)
        Path(temporary).replace(path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


_TABLES_SQL = """
SELECT c.oid, c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public' AND c.relkind IN ('r', 'p') ORDER BY c.relname
"""
_COLUMNS_SQL = """
SELECT attname, format_type(atttypid, atttypmod), NOT attnotnull FROM pg_attribute
WHERE attrelid = %s AND attnum > 0 AND NOT attisdropped ORDER BY attnum
"""
_KEY_NAMES = """
ARRAY(SELECT a.attname FROM unnest(con.{key}) WITH ORDINALITY AS k(attnum, ord)
      JOIN pg_attribute a ON a.attrelid = con.{rel} AND a.attnum = k.attnum ORDER BY k.ord)
"""
_CONSTRAINTS_SQL = f"""
SELECT con.conname, con.contype, pg_get_constraintdef(con.oid),
       {_KEY_NAMES.format(key="conkey", rel="conrelid")}, ref.relname,
       {_KEY_NAMES.format(key="confkey", rel="confrelid")}
FROM pg_constraint con LEFT JOIN pg_class ref ON ref.oid = con.confrelid
WHERE con.conrelid = %s ORDER BY con.contype = 'p' DESC, con.conname
"""  # noqa: S608  # Only constants are interpolated.
_INDEXES_SQL = """
SELECT i.relname, pg_get_indexdef(i.oid) FROM pg_index x JOIN pg_class i ON i.oid = x.indexrelid
WHERE x.indrelid = %s ORDER BY i.relname
"""


def _table_info(conn: psycopg.Connection[Any], oid: int, name: str) -> TableInfo:
    columns = [
        ColumnInfo(name=n, type=t, nullable=null) for n, t, null in _rows(conn, _COLUMNS_SQL, oid)
    ]
    constraints = _rows(conn, _CONSTRAINTS_SQL, oid)
    count_sql = sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier("public", name))
    return TableInfo(
        name=name,
        row_count=int(_scalar(conn, count_sql)),
        columns=tuple(columns),
        primary_key=next(
            (tuple(keys) for _, kind, _, keys, _, _ in constraints if kind == "p"), ()
        ),
        foreign_keys=tuple(
            ForeignKeyInfo(columns=tuple(keys), ref_table=ref, ref_columns=tuple(ref_keys))
            for _, kind, _, keys, ref, ref_keys in constraints
            if kind == "f"
        ),
        indexes=tuple(IndexInfo(name=n, definition=d) for n, d in _rows(conn, _INDEXES_SQL, oid)),
        constraints=tuple((n, definition) for n, _, definition, _, _, _ in constraints),
    )


def _rows(conn: psycopg.Connection[Any], query: str, *params: object) -> list[Any]:
    return conn.execute(query, params or None).fetchall()


def _scalar(
    conn: psycopg.Connection[Any], query: str | sql.Composed, *, prepare: bool | None = None
) -> Any:
    row = conn.execute(query, prepare=prepare).fetchone()
    if row is None:
        raise psycopg.ProgrammingError("the query returned no row")
    return row[0]


def _plan_root(value: object) -> Mapping[str, object] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    if isinstance(value, list):
        value = value[0] if value else None
    if isinstance(value, Mapping) and isinstance(value.get("Plan"), Mapping):
        value = value["Plan"]
    return value if isinstance(value, Mapping) else None


def _nodes(node: Mapping[str, object]) -> Iterator[Mapping[str, object]]:
    yield node
    for child in _children(node):
        yield from _nodes(child)


def _beyond_sf3(node: Mapping[str, object]) -> bool:
    """Whether one pass of this scan read more rows than its table holds at SF3.

    The rows a pass read are those it returned plus those its filters removed. A subplan in
    the scan's filter or output runs at most once per row read, so its loops count them too:
    task 40's lineitem scan returned 15M rows, but its filter's subplan ran 59,986,052 times.
    The loops of an inner scan reveal nothing: it runs once per outer row.
    """
    rows = _SF3_ROWS.get(str(node.get("Relation Name")))
    if rows is None or node.get("Node Type") not in _SCANS:
        return False
    # A parallel scan shares one pass among its processes, and EXPLAIN reports each count per
    # process; any other scan reports counts per loop, each loop one pass.
    loops = _count(node.get("Actual Loops")) or 1
    passes = 1 if node.get("Parallel Aware") is True else loops
    per_process = loops if node.get("Parallel Aware") is True else 1
    read = [_count(node.get("Plan Rows"))]
    actual = _count(node.get("Actual Rows"))
    if actual is not None:
        removed = sum(
            _count(node.get(key)) or 0
            for key in ("Rows Removed by Filter", "Rows Removed by Index Recheck")
        )
        read.append(actual * per_process)
        read.append((actual + removed) * per_process)
    for child in _children(node):
        child_loops = _count(child.get("Actual Loops"))
        if child.get("Parent Relationship") == "SubPlan" and child_loops is not None:
            read.append(child_loops / passes)  # A subplan's loops are summed over passes.
    return any(value is not None and value > rows * _SCALE_MARGIN for value in read)


def _count(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _children(node: Mapping[str, object]) -> list[Mapping[str, object]]:
    children = node.get("Plans")
    return (
        [child for child in children if isinstance(child, Mapping)]
        if isinstance(children, list)
        else []
    )
