"""Tests for goaly.bench.effisql.inputs: what the agent sees of an Effi-SQL task.

The schema snapshot and the plans are synthetic, in the hand-off format T09 writes.
"""

import json
import random
from pathlib import Path

import pytest
from pydantic import ValidationError

from goaly.bench.effisql.data import EffiSqlInstance
from goaly.bench.effisql.inputs import (
    SchemaSnapshot,
    agent_input,
    load_plans,
    load_schema_snapshot,
    render_schema,
)
from goaly.contracts.gen.task_pb2 import EffiSqlTaskInput, TaskInput
from goaly.contracts.interfaces import JsonValue
from goaly.contracts.jsonl import to_json_line
from tests.bench.effisql.synthetic import BASE_PLAN, BASE_SQL, GOLD_SQL, task, without

ORDERS_DDL = "CREATE TABLE orders (\n    o_orderkey integer NOT NULL,\n    o_comment text\n);"
LINEITEM_DDL = "CREATE TABLE lineitem (\n    l_orderkey integer NOT NULL\n);"

SNAPSHOT: dict[str, JsonValue] = {
    "db": "tpch",
    "server_version": "16.4",
    "tables": [
        {
            "name": "orders",
            "row_count": 4500000,
            "columns": [
                {"name": "o_orderkey", "type": "integer", "nullable": False},
                {"name": "o_comment", "type": "text", "nullable": True},
            ],
            "primary_key": ["o_orderkey"],
            "foreign_keys": [],
            "indexes": [
                {
                    "name": "orders_pkey",
                    "definition": "CREATE UNIQUE INDEX orders_pkey ON public.orders USING btree "
                    "(o_orderkey)",
                }
            ],
        },
        {
            "name": "lineitem",
            "row_count": 17996609,
            "columns": [
                {"name": "l_orderkey", "type": "integer", "nullable": False},
                {"name": "l_linenumber", "type": "integer", "nullable": False},
            ],
            "primary_key": ["l_orderkey", "l_linenumber"],
            "foreign_keys": [
                {"columns": ["l_orderkey"], "ref_table": "orders", "ref_columns": ["o_orderkey"]}
            ],
            "indexes": [
                {
                    "name": "lineitem_pkey",
                    "definition": "CREATE UNIQUE INDEX lineitem_pkey ON public.lineitem USING "
                    "btree (l_orderkey, l_linenumber);",
                },
                {
                    "name": "lineitem_l_orderkey_idx",
                    "definition": "CREATE INDEX lineitem_l_orderkey_idx ON public.lineitem USING "
                    "btree (l_orderkey)",
                },
            ],
        },
    ],
    "ddl": f"{ORDERS_DDL}\n\n{LINEITEM_DDL}\n",
}

SCHEMA_TEXT = f"""\
-- Database tpch on PostgreSQL 16.4.

{ORDERS_DDL}

{LINEITEM_DDL}

-- Rows per table:
--   lineitem: 17,996,609
--   orders: 4,500,000

-- Indexes:
CREATE INDEX lineitem_l_orderkey_idx ON public.lineitem USING btree (l_orderkey);
CREATE UNIQUE INDEX lineitem_pkey ON public.lineitem USING btree (l_orderkey, l_linenumber);
CREATE UNIQUE INDEX orders_pkey ON public.orders USING btree (o_orderkey);
"""

# EXPLAIN (FORMAT JSON) output, as the plan cache holds it.
CACHED_PLAN = json.dumps([{"Plan": {"Node Type": "Seq Scan", "Relation Name": "orders"}}], indent=2)

# The forms of an empty plan, which counts as no plan (controller's ruling).
EMPTY_PLANS: list[JsonValue] = [None, {}, [], "", " \n\t"]
EMPTY_PLAN_IDS = ["null", "empty-object", "empty-array", "empty-string", "blank-string"]


def snapshot(**fields: JsonValue) -> SchemaSnapshot:
    return SchemaSnapshot.model_validate_json(json.dumps({**SNAPSHOT, **fields}))


def instance(**fields: JsonValue) -> EffiSqlInstance:
    return EffiSqlInstance.model_validate(task(**fields))


# The schema snapshot.


def test_loads_a_schema_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "schema_snapshot.json"
    path.write_text(json.dumps(SNAPSHOT), encoding="utf-8")

    loaded = load_schema_snapshot(path)

    assert loaded == snapshot()
    assert (loaded.db, loaded.server_version) == ("tpch", "16.4")
    lineitem = loaded.tables[1]
    assert (lineitem.name, lineitem.row_count) == ("lineitem", 17996609)
    assert lineitem.primary_key == ("l_orderkey", "l_linenumber")
    assert lineitem.foreign_keys[0].ref_table == "orders"
    assert lineitem.columns[1].nullable is False


def table(**fields: JsonValue) -> dict[str, JsonValue]:
    """Return the snapshot's orders table, with ``fields`` replaced."""
    tables = SNAPSHOT["tables"]
    assert isinstance(tables, list)
    orders = tables[0]
    assert isinstance(orders, dict)
    return {**orders, **fields}


@pytest.mark.parametrize(
    "fields",
    [
        {"extra": True},
        {"ddl": " \n"},
        {"db": ""},
        {"tables": []},
        {"tables": [table(), table()]},
        {"tables": [table(row_count=-1)]},
        {"tables": [table(row_count="4500000")]},
        {"tables": [table(row_count=True)]},
        {"tables": [table(columns=[])]},
        {"tables": [table(columns=[{"name": "o_orderkey", "type": "integer"}])]},
        {"tables": [table(columns=[{"name": "x", "type": "integer", "nullable": "no"}])]},
        {"tables": [table(indexes=[{"name": "orders_pkey"}])]},
        {"tables": [table(comment="hand-written")]},
        {
            "tables": [
                table(
                    foreign_keys=[{"columns": ["a", "b"], "ref_table": "t", "ref_columns": ["a"]}]
                )
            ]
        },
    ],
    ids=[
        "unknown-key",
        "blank-ddl",
        "blank-db",
        "no-tables",
        "table-twice",
        "negative-rows",
        "rows-as-text",
        "rows-as-bool",
        "no-columns",
        "column-without-nullable",
        "nullable-as-text",
        "index-without-definition",
        "unknown-table-key",
        "foreign-key-lengths-differ",
    ],
)
def test_a_schema_snapshot_is_validated_strictly(fields: dict[str, JsonValue]) -> None:
    with pytest.raises(ValidationError):
        snapshot(**fields)


def test_a_schema_snapshot_needs_every_field(tmp_path: Path) -> None:
    path = tmp_path / "schema_snapshot.json"
    path.write_text(json.dumps({key: SNAPSHOT[key] for key in ("db", "tables", "ddl")}), "utf-8")

    with pytest.raises(ValidationError, match="server_version"):
        load_schema_snapshot(path)


# The schema text.


def test_renders_the_ddl_the_row_counts_and_the_indexes() -> None:
    assert render_schema(snapshot()) == SCHEMA_TEXT


def test_the_schema_text_does_not_depend_on_the_snapshot_order() -> None:
    reordered = json.loads(json.dumps(SNAPSHOT))
    shuffle = random.Random(3)  # noqa: S311  # A reproducible order, not a secret.
    for _ in range(5):
        shuffle.shuffle(reordered["tables"])
        for each in reordered["tables"]:
            shuffle.shuffle(each["indexes"])

        assert render_schema(SchemaSnapshot.model_validate_json(json.dumps(reordered))) == (
            SCHEMA_TEXT
        )


def test_a_schema_without_indexes_says_so() -> None:
    text = render_schema(snapshot(tables=[table(indexes=[])]))

    assert text.endswith("--   orders: 4,500,000\n\n-- Indexes: none\n")


# The plan cache.


def write_plans(path: Path, plans: object) -> Path:
    path.write_text(json.dumps(plans), encoding="utf-8")
    return path


def test_loads_the_plan_cache(tmp_path: Path) -> None:
    path = write_plans(tmp_path / "plans.json", {"7": CACHED_PLAN, "8": '{"Plan": {}}'})

    plans = load_plans(path)

    assert dict(plans) == {"7": CACHED_PLAN, "8": '{"Plan": {}}'}
    with pytest.raises(TypeError):
        plans["9"] = CACHED_PLAN  # type: ignore[index]  # The cache is read-only.


@pytest.mark.parametrize(
    "plans",
    [
        {"7": {"Plan": {}}},
        {"7": ""},
        {"7": "Seq Scan on orders"},
        {"7": "42"},
        {"7": "[]"},
        {"7": " {} "},
        {"": CACHED_PLAN},
        [CACHED_PLAN],
    ],
    ids=[
        "plan-as-json",
        "blank-text",
        "not-json",
        "not-a-plan",
        "empty-array",
        "empty-object",
        "blank-id",
        "array",
    ],
)
def test_the_plan_cache_is_validated_strictly(tmp_path: Path, plans: object) -> None:
    path = write_plans(tmp_path / "plans.json", plans)

    with pytest.raises(ValidationError):
        load_plans(path)


# agent_input.


def test_copies_what_the_agent_may_see() -> None:
    inst = instance(difficulty="hard")

    task_input = agent_input(inst, snapshot(), {})

    assert task_input == TaskInput(
        effisql=EffiSqlTaskInput(
            db="tpch",
            base_sql=BASE_SQL,
            schema=SCHEMA_TEXT,
            base_plan=json.dumps(BASE_PLAN, indent=2),
            difficulty="hard",
        )
    )


def test_the_plan_cache_wins_over_the_task_files_plan() -> None:
    # Some recorded plans come from TPC-H SF10, so T09's cache replaces them.
    task_input = agent_input(instance(id=7), snapshot(), {"7": CACHED_PLAN, "8": '{"Plan": {}}'})

    assert task_input.effisql.base_plan == CACHED_PLAN


def test_the_task_files_plan_is_used_when_the_cache_has_no_entry() -> None:
    task_input = agent_input(instance(id=7), snapshot(), {"8": CACHED_PLAN})

    assert json.loads(task_input.effisql.base_plan) == BASE_PLAN


@pytest.mark.parametrize("plan", EMPTY_PLANS, ids=EMPTY_PLAN_IDS)
def test_an_empty_plan_falls_through_to_the_cache(plan: JsonValue) -> None:
    task_input = agent_input(
        instance(id=7, base_explain_analyze=plan), snapshot(), {"7": CACHED_PLAN}
    )

    assert task_input.effisql.base_plan == CACHED_PLAN


@pytest.mark.parametrize("plan", EMPTY_PLANS, ids=EMPTY_PLAN_IDS)
def test_a_task_with_no_plan_anywhere_is_an_error(plan: JsonValue) -> None:
    inst = instance(id=7, base_explain_analyze=plan)

    with pytest.raises(ValueError, match=r"^task 7 has no base plan: plans\.json has no entry"):
        agent_input(inst, snapshot(), {"8": CACHED_PLAN})


def test_a_task_missing_its_plan_is_an_error() -> None:
    inst = EffiSqlInstance.model_validate(without(task(id="tpch-9"), "base_explain_analyze"))

    with pytest.raises(ValueError, match=r"^task tpch-9 has no base plan"):
        agent_input(inst, snapshot(), {})


def test_never_shows_the_gold_answer() -> None:
    inst = instance(fast_time=98.765)

    line = to_json_line(agent_input(inst, snapshot(), {}))

    assert inst.optimized_sql == GOLD_SQL
    assert GOLD_SQL not in line
    assert "synthetic gold plan" not in line
    assert "98.765" not in line
    assert not [name for name in ("optimized", "fast_time") if name in line]


def test_a_task_without_gold_difficulty_base_time_or_plan_gives_a_valid_input() -> None:
    bare = without(task(id=7), "optimized_sql", "fast_time", "optimized_explain_analyze")
    bare = without(bare, "difficulty", "base_time", "base_explain_analyze")
    inst = EffiSqlInstance.model_validate(bare)

    task_input = agent_input(inst, snapshot(), {"7": CACHED_PLAN})

    assert task_input.effisql.base_plan == CACHED_PLAN
    assert task_input.effisql.difficulty == ""
    assert task_input.effisql.base_sql == BASE_SQL


def test_the_snapshot_must_be_of_the_tasks_database() -> None:
    inst = instance(id=7, db="solar_panel")

    with pytest.raises(
        ValueError, match=r"^task 7 runs on solar_panel, but the schema snapshot is of tpch$"
    ):
        agent_input(inst, snapshot(), {})
