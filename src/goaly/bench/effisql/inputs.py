"""What the Effi-SQL agent sees of a task (spec §4.1): its database's schema and its base plan.

T09's environment setup writes two files, and the adapter loads each of them once:

- ``schema_snapshot.json``, one per database: its tables, with their row counts, columns, keys
  and indexes, and its full DDL. load_schema_snapshot reads it into a SchemaSnapshot.
- ``plans.json``, which maps instance IDs to base plans, each the text of an
  ``EXPLAIN (FORMAT JSON)``. load_plans reads it.

agent_input builds a task's EffiSqlTaskInput from the task and those two, and does no I/O. The
agent sees a plan in one shape whatever its source: its root node, as indented JSON.
"""

import json
import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from goaly.bench.effisql.data import EffiSqlInstance, NonBlankStr
from goaly.contracts.gen.task_pb2 import EffiSqlTaskInput, TaskInput
from goaly.contracts.interfaces import JsonValue

__all__ = [
    "Column",
    "ForeignKey",
    "Index",
    "SchemaSnapshot",
    "Table",
    "agent_input",
    "load_plans",
    "load_schema_snapshot",
    "render_schema",
]

# The snapshot is read strictly: no unknown key, and no value of another type.
_STRICT = ConfigDict(frozen=True, extra="forbid", strict=True)
# Timings that EXPLAIN ANALYZE reports beside the plan, which the agent's plan leaves out.
_TIMINGS = ("Planning Time", "Execution Time")


class Column(BaseModel):
    """A column of a table."""

    model_config = _STRICT

    name: NonBlankStr
    type: NonBlankStr
    nullable: bool


class ForeignKey(BaseModel):
    """A foreign key: ``columns`` reference ``ref_columns`` of ``ref_table``, pair by pair."""

    model_config = _STRICT

    columns: tuple[NonBlankStr, ...] = Field(min_length=1)
    ref_table: NonBlankStr
    ref_columns: tuple[NonBlankStr, ...]

    @model_validator(mode="after")
    def _check_the_columns_pair_up(self) -> Self:
        if len(self.columns) != len(self.ref_columns):
            raise ValueError("columns and ref_columns must have the same length")
        return self


class Index(BaseModel):
    """An index, with its ``CREATE INDEX`` statement as pg_indexes gives it."""

    model_config = _STRICT

    name: NonBlankStr
    definition: NonBlankStr


class Table(BaseModel):
    """A table of the database, with its row count."""

    model_config = _STRICT

    name: NonBlankStr
    row_count: int = Field(ge=0)
    columns: tuple[Column, ...] = Field(min_length=1)
    primary_key: tuple[NonBlankStr, ...]
    foreign_keys: tuple[ForeignKey, ...]
    indexes: tuple[Index, ...]


class SchemaSnapshot(BaseModel):
    """A database's schema, as T09's setup records it in ``schema_snapshot.json``."""

    model_config = _STRICT

    db: NonBlankStr
    server_version: NonBlankStr
    tables: tuple[Table, ...] = Field(min_length=1)
    ddl: NonBlankStr

    @model_validator(mode="after")
    def _check_each_table_is_listed_once(self) -> Self:
        names = [table.name for table in self.tables]
        if len(set(names)) != len(names):
            raise ValueError("a table is listed more than once")
        return self


def load_schema_snapshot(path: Path) -> SchemaSnapshot:
    """Read the schema snapshot in the file at ``path``.

    Raises pydantic's ValidationError, a ValueError, if the file does not hold a valid snapshot.
    """
    return SchemaSnapshot.model_validate_json(path.read_bytes())


def _explain_output(text: str) -> str:
    """Check that ``text`` is one statement's EXPLAIN (FORMAT JSON) output, with a plan."""
    try:
        output = json.loads(text)
    except ValueError:
        output = None
    statement = output[0] if isinstance(output, list) and len(output) == 1 else None
    root = statement.get("Plan") if isinstance(statement, dict) else None
    if not isinstance(root, dict) or not root:
        raise ValueError('must be EXPLAIN (FORMAT JSON) output, [{"Plan": {...}}], with a plan')
    return text


_PLANS = TypeAdapter(
    dict[NonBlankStr, Annotated[str, AfterValidator(_explain_output)]],
    config=ConfigDict(strict=True),
)


def load_plans(path: Path) -> Mapping[str, str]:
    """Read the plan cache in the file at ``path``: base plans by instance ID, as JSON text.

    Each plan is one statement's ``EXPLAIN (FORMAT JSON)`` output, ``[{"Plan": {...}, ...}]``.
    The mapping is read-only. Raises pydantic's ValidationError, a ValueError, if the file does
    not map instance IDs to such plans.
    """
    return MappingProxyType(_PLANS.validate_json(path.read_bytes()))


def render_schema(snapshot: SchemaSnapshot) -> str:
    """Return the schema text the agent reads (spec §4.1): the DDL, the row counts and indexes.

    Tables and each table's indexes are listed by name, so the text depends only on what the
    snapshot holds, not on the order the setup wrote it in. An index the DDL already creates is
    not listed again: one whose statement the DDL holds, whatever its spacing or case, and one
    the DDL declares as a constraint of its name, as pg_dump declares a primary key.
    """
    tables = sorted(snapshot.tables, key=lambda table: table.name)
    indexes = [
        (index.name, index.definition.strip().removesuffix(";").rstrip())
        for table in tables
        for index in sorted(table.indexes, key=lambda index: index.name)
    ]
    ddl = _normalized(snapshot.ddl)
    statements = [
        f"{sql};"
        for name, sql in indexes
        if _normalized(sql) not in ddl and not _declares_constraint(snapshot.ddl, name)
    ]
    if statements:
        listed = ["-- Indexes:", *statements]
    else:
        listed = ["-- Indexes: all in the DDL above" if indexes else "-- Indexes: none"]
    lines = [
        f"-- Database {snapshot.db} on PostgreSQL {snapshot.server_version}.",
        "",
        snapshot.ddl.strip(),
        "",
        "-- Rows per table:",
        *(f"--   {table.name}: {table.row_count:,}" for table in tables),
        "",
        *listed,
    ]
    return "\n".join(lines) + "\n"


def _declares_constraint(ddl: str, name: str) -> bool:
    """Say whether ``ddl`` declares a constraint named ``name``, quoted or not, in any case."""
    pattern = rf'\bconstraint\s+"?{re.escape(name)}"?(?![\w$])'
    return re.search(pattern, ddl, flags=re.IGNORECASE) is not None


def _normalized(sql: str) -> str:
    """Return ``sql`` with its whitespace collapsed and in lower case, to compare statements."""
    return " ".join(sql.split()).lower()


def agent_input(
    inst: EffiSqlInstance, snapshot: SchemaSnapshot, plans: Mapping[str, str]
) -> TaskInput:
    """Copy out what the agent may see of ``inst`` (spec §4.1), never its gold answer.

    The input holds the task's ``db`` and ``base_sql``, the schema text of ``snapshot``, the base
    plan, and the difficulty, empty if the task file has none. The base plan comes from:

    1. ``plans``, the plan cache, when it has the task. T09's setup writes an entry for each
       task whose file has no plan, and for each whose file plan was recorded at another scale:
       some recorded plans come from TPC-H SF10, not SF3;
    2. otherwise the task file's ``base_explain_analyze``. An empty plan counts as none
       (EffiSqlInstance);
    3. otherwise nowhere: ValueError, naming the task.

    Either way the agent sees the plan's root node, as indented JSON (see _root_node), so the
    same plan gives the same text from either source.

    ``snapshot`` must be the schema of the task's database; otherwise ValueError.
    """
    if inst.db != snapshot.db:
        raise ValueError(
            f"task {inst.instance_id} runs on {inst.db}, but the schema snapshot is of "
            f"{snapshot.db}"
        )
    return TaskInput(
        effisql=EffiSqlTaskInput(
            db=inst.db,
            base_sql=inst.base_sql,
            schema=render_schema(snapshot),
            base_plan=_base_plan(inst, plans),
            difficulty=inst.difficulty or "",
        )
    )


def _base_plan(inst: EffiSqlInstance, plans: Mapping[str, str]) -> str:
    cached = plans.get(inst.instance_id)
    plan: JsonValue
    if cached is not None:
        try:
            plan = json.loads(cached)
        except ValueError:
            plan = None  # Not JSON: _root_node rejects it, naming the task.
    elif inst.base_explain_analyze is not None:
        plan = inst.base_explain_analyze
    else:
        raise ValueError(
            f"task {inst.instance_id} has no base plan: plans.json has no entry for it, and its "
            "task file has no base_explain_analyze"
        )
    return json.dumps(_root_node(plan, inst.instance_id), indent=2, ensure_ascii=False)


def _root_node(plan: JsonValue, instance_id: str) -> dict[str, JsonValue]:
    """Return ``plan``'s root node, without the timings EXPLAIN ANALYZE reports beside it.

    ``EXPLAIN (FORMAT JSON)`` writes ``[{"Plan": root, "Planning Time": ..., ...}]``, whose
    root is the "Plan" object. The dataset's plans are root nodes already.
    """
    if isinstance(plan, list) and len(plan) == 1:
        plan = plan[0]
    if isinstance(plan, dict) and "Plan" in plan:  # A root node has no "Plan", only "Plans".
        plan = plan["Plan"]
    if not isinstance(plan, dict) or not plan:
        raise ValueError(f"task {instance_id} has a base plan that is not a query plan")
    return {key: value for key, value in plan.items() if key not in _TIMINGS}
