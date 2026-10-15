import json
import stat
from pathlib import Path

import psycopg
import pytest

from goaly.bench.effisql.pg_settings import PgEnvConfig
from goaly.bench.effisql.pg_snapshot import (
    ColumnInfo,
    ForeignKeyInfo,
    IndexInfo,
    PlanRequest,
    TableInfo,
    explain_plans,
    foreign_scale_plan,
    plan_requests,
    render_ddl,
    schema_snapshot,
    write_json,
    write_snapshots,
)
from tests.bench.effisql.pg_fakes import Answer, FakeServer


def _scan(relation: str, rows: int, **extra: object) -> dict[str, object]:
    return {
        "Node Type": "Seq Scan",
        "Relation Name": relation,
        "Parallel Aware": False,
        "Plan Rows": rows,
        **extra,
    }


def _root(*children: dict[str, object]) -> dict[str, object]:
    return {"Node Type": "Hash Join", "Plan Rows": 10, "Plans": list(children)}


def test_a_plan_whose_scans_fit_sf3_is_native() -> None:
    plan = _root(_scan("lineitem", 17_996_609, **{"Actual Rows": 17_996_609, "Actual Loops": 1}))

    assert not foreign_scale_plan(plan)


def test_a_scan_with_more_rows_than_the_sf3_table_marks_another_scale() -> None:
    sf10_lineitem = _scan("lineitem", 59_986_052, **{"Actual Rows": 59_986_052, "Actual Loops": 1})

    assert foreign_scale_plan(_root(sf10_lineitem))


def test_estimates_alone_also_reveal_another_scale() -> None:
    assert foreign_scale_plan(_root(_scan("customer", 1_499_995)))
    assert not foreign_scale_plan(_root(_scan("customer", 450_000)))


def test_rows_repeated_over_the_loops_of_a_nested_loop_do_not_count() -> None:
    inner = _scan(
        "nation", 1, **{"Node Type": "Index Scan", "Actual Rows": 1, "Actual Loops": 450_000}
    )

    assert not foreign_scale_plan(_root(inner))


def test_a_parallel_scan_counts_the_rows_of_all_its_processes() -> None:
    def parallel(rows_per_process: int) -> dict[str, object]:
        return _scan(
            "orders",
            1_875_000,
            **{"Parallel Aware": True, "Actual Rows": rows_per_process, "Actual Loops": 3},
        )

    assert not foreign_scale_plan(_root(parallel(1_500_000)))  # 4.5M rows in all: SF3's orders.
    # 2M per process is within SF3's 4.5M rows, but the three processes read 6M in all.
    assert foreign_scale_plan(_root(parallel(2_000_000)))


def test_a_scan_just_above_the_sf3_size_is_still_native() -> None:
    estimate = int(17_996_609 * 1.05)  # Within the margin a fresh ANALYZE can overshoot by.

    assert not foreign_scale_plan(_root(_scan("lineitem", estimate)))
    assert foreign_scale_plan(_root(_scan("lineitem", int(17_996_609 * 1.15))))


def _task_40(subplan_loops: int) -> dict[str, object]:
    """Task 40's plan: a lineitem scan whose filter runs a subplan once per row it reads."""
    inner = _scan(
        "lineitem",
        149,
        **{"Node Type": "Index Scan", "Actual Rows": 5, "Actual Loops": subplan_loops},
    )
    subplan = {
        "Node Type": "Aggregate",
        "Parent Relationship": "SubPlan",
        "Actual Rows": 1,
        "Actual Loops": subplan_loops,
        "Plan Rows": 1,
        "Plans": [inner],
    }
    return _scan(
        "lineitem", 299_959, **{"Actual Rows": 15_240_492, "Actual Loops": 1, "Plans": [subplan]}
    )


def test_a_filter_subplan_counts_the_rows_its_scan_read() -> None:
    assert foreign_scale_plan(_task_40(subplan_loops=59_986_052))  # SF10's lineitem.
    assert not foreign_scale_plan(_task_40(subplan_loops=17_996_609))  # SF3's.


def test_rows_removed_by_the_filter_count_as_rows_read() -> None:
    def scan(removed: int) -> dict[str, object]:
        counts = {"Actual Rows": 1_000_000, "Actual Loops": 1, "Rows Removed by Filter": removed}
        return _scan("lineitem", 1_000_000, **counts)

    assert foreign_scale_plan(_root(scan(58_986_052)))  # 59,986,052 read: SF10.
    assert not foreign_scale_plan(_root(scan(16_996_609)))  # 17,996,609 read: SF3.


def test_a_parallel_scan_counts_rows_read_over_its_processes() -> None:
    counts = {"Parallel Aware": True, "Actual Rows": 500_000, "Actual Loops": 3}

    assert foreign_scale_plan(
        _root(_scan("orders", 1, **counts, **{"Rows Removed by Filter": 4_500_000}))
    )
    assert not foreign_scale_plan(
        _root(_scan("orders", 1, **counts, **{"Rows Removed by Filter": 1_000_000}))
    )


def _filtered(subplan_loops: int, **scan: object) -> dict[str, object]:
    subplan = {
        "Node Type": "Result",
        "Parent Relationship": "SubPlan",
        "Actual Loops": subplan_loops,
    }
    return _scan("orders", 1, **{**scan, "Plans": [subplan]})


def test_a_filter_subplan_is_counted_per_pass_of_its_scan() -> None:
    # SF3: an orders lookup for each of lineitem's rows, with a subplan in the lookup's filter.
    inner = _filtered(
        17_996_609, **{"Node Type": "Index Scan", "Actual Rows": 1, "Actual Loops": 17_996_609}
    )

    assert not foreign_scale_plan(_root(inner))  # One row read per pass.


def test_a_parallel_scans_subplan_loops_are_already_summed_over_its_processes() -> None:
    # SF3: orders' 4,500,000 rows read by 3 processes, the filter's subplan run once for each.
    parallel = _filtered(
        4_500_000, **{"Parallel Aware": True, "Actual Rows": 1_500_000, "Actual Loops": 3}
    )

    assert not foreign_scale_plan(_root(parallel))
    assert foreign_scale_plan(_root(_filtered(15_000_000, **{"Parallel Aware": True})))  # SF10.


def test_an_inner_scan_looping_once_per_outer_row_is_no_sign_of_scale() -> None:
    # Task 148 on SF3: a supplier lookup for each of lineitem's 17,996,609 rows.
    inner = _scan(
        "supplier", 1, **{"Node Type": "Index Scan", "Actual Rows": 1, "Actual Loops": 17_996_609}
    )

    assert not foreign_scale_plan(_root(inner))


def test_only_scan_nodes_count_even_if_they_name_a_relation() -> None:
    update = {"Node Type": "ModifyTable", "Relation Name": "orders", "Plan Rows": 15_000_000}

    assert not foreign_scale_plan(_root(update))


def test_tables_outside_tpch_are_ignored() -> None:
    assert not foreign_scale_plan(_root(_scan("observatories", 10**9)))


def test_region_and_nation_cannot_reveal_a_scale() -> None:
    # They hold the same rows at every scale; an estimate of 170 rows is stale statistics.
    assert not foreign_scale_plan(_root(_scan("region", 170), _scan("nation", 170)))


@pytest.mark.parametrize(
    "wrap",
    [
        lambda node: node,
        lambda node: {"Plan": node},
        lambda node: [{"Plan": node}],
        lambda node: json.dumps([{"Plan": node}]),
    ],
    ids=["root node", "explain object", "explain list", "json text"],
)
def test_every_form_of_a_plan_is_accepted(wrap: object) -> None:
    node = _root(_scan("orders", 15_000_000))

    assert foreign_scale_plan(wrap(node))  # type: ignore[operator]  # A lambda from parametrize.


def test_a_plan_that_is_not_a_plan_is_not_foreign() -> None:
    assert not foreign_scale_plan("not json")
    assert not foreign_scale_plan(None)


def _task(task_id: int, db: str = "tpch", plan: object = "absent") -> str:
    record: dict[str, object] = {"id": task_id, "db": db, "base_sql": f"SELECT {task_id}"}
    if plan != "absent":
        record["base_explain_analyze"] = plan
    return json.dumps(record)


def test_plan_requests_cover_tasks_without_a_plan_or_with_a_foreign_one(tmp_path: Path) -> None:
    native = _root(_scan("orders", 4_500_000))
    foreign = _root(_scan("orders", 15_000_000))
    path = tmp_path / "tasks.jsonl"
    lines = [
        _task(1, plan=native),
        _task(2),
        _task(3, plan=None),
        _task(4, plan={}),
        _task(5, plan=foreign),
        _task(6, db="solar_panel"),
        _task(7, plan="  \n"),
        "",
    ]
    path.write_text("\n".join(lines))

    requests = plan_requests(path, db="tpch")

    assert requests == [
        PlanRequest(instance_id="2", base_sql="SELECT 2", reason="no plan"),
        PlanRequest(instance_id="3", base_sql="SELECT 3", reason="no plan"),
        PlanRequest(instance_id="4", base_sql="SELECT 4", reason="no plan"),
        PlanRequest(instance_id="5", base_sql="SELECT 5", reason="foreign scale"),
        PlanRequest(instance_id="7", base_sql="SELECT 7", reason="no plan"),
    ]


def _table(name: str, *, indexes: tuple[IndexInfo, ...] = ()) -> TableInfo:
    return TableInfo(
        name=name,
        row_count=5,
        columns=(
            ColumnInfo(name="r_regionkey", type="integer", nullable=False),
            ColumnInfo(name="r_name", type="character(25)", nullable=True),
        ),
        primary_key=("r_regionkey",),
        foreign_keys=(ForeignKeyInfo(columns=("r_x",), ref_table="t", ref_columns=("x",)),),
        indexes=indexes,
        constraints=(
            ("region_pkey", "PRIMARY KEY (r_regionkey)"),
            ("region_r_x_fkey", "FOREIGN KEY (r_x) REFERENCES t(x)"),
        ),
    )


def test_the_ddl_shows_columns_constraints_and_extra_indexes() -> None:
    extra = IndexInfo(name="region_name_idx", definition="CREATE INDEX region_name_idx ON x")
    backing = IndexInfo(name="region_pkey", definition="CREATE UNIQUE INDEX region_pkey ON x")

    ddl = render_ddl([_table("region", indexes=(backing, extra))])

    assert ddl == (
        "CREATE TABLE region (\n"
        "    r_regionkey integer NOT NULL,\n"
        "    r_name character(25),\n"
        "    CONSTRAINT region_pkey PRIMARY KEY (r_regionkey),\n"
        "    CONSTRAINT region_r_x_fkey FOREIGN KEY (r_x) REFERENCES t(x)\n"
        ");\n"
        "CREATE INDEX region_name_idx ON x;\n"
    )


def test_the_snapshot_format_is_the_one_t08_reads() -> None:
    table = _table("region")

    assert table.to_json() == {
        "name": "region",
        "row_count": 5,
        "columns": [
            {"name": "r_regionkey", "type": "integer", "nullable": False},
            {"name": "r_name", "type": "character(25)", "nullable": True},
        ],
        "primary_key": ["r_regionkey"],
        "foreign_keys": [{"columns": ["r_x"], "ref_table": "t", "ref_columns": ["x"]}],
        "indexes": [],
    }


def test_json_files_are_written_atomically_with_sorted_keys(tmp_path: Path) -> None:
    path = tmp_path / "out" / "plans.json"

    write_json(path, {"b": "2", "a": "1"})

    assert path.read_text() == '{\n  "a": "1",\n  "b": "2"\n}\n'
    assert stat.S_IMODE(path.stat().st_mode) == 0o644
    assert list(path.parent.iterdir()) == [path]


CATALOG: list[tuple[str, Answer]] = [
    ("relkind IN", [(16384, "region")]),
    ("format_type(", [("r_regionkey", "integer", False), ("r_name", "character(25)", True)]),
    (
        "pg_get_constraintdef",
        [("region_pkey", "p", "PRIMARY KEY (r_regionkey)", ["r_regionkey"], None, [])],
    ),
    (
        "pg_get_indexdef",
        [("region_pkey", "CREATE UNIQUE INDEX region_pkey ON public.region (r_regionkey)")],
    ),
    ("SELECT count(*)", [(5,)]),
    ("SHOW server_version", [("16.15 (Debian 16.15-1.pgdg13+2)",)]),
]


def test_the_snapshot_describes_the_catalog_with_exact_row_counts() -> None:
    server = FakeServer(CATALOG)

    snapshot = schema_snapshot(server.connect("tpch"))

    assert snapshot == {
        "db": "tpch",
        "server_version": "16.15 (Debian 16.15-1.pgdg13+2)",
        "tables": [
            {
                "name": "region",
                "row_count": 5,
                "columns": [
                    {"name": "r_regionkey", "type": "integer", "nullable": False},
                    {"name": "r_name", "type": "character(25)", "nullable": True},
                ],
                "primary_key": ["r_regionkey"],
                "foreign_keys": [],
                "indexes": [
                    {
                        "name": "region_pkey",
                        "definition": "CREATE UNIQUE INDEX region_pkey ON public.region "
                        "(r_regionkey)",
                    }
                ],
            }
        ],
        "ddl": "CREATE TABLE region (\n"
        "    r_regionkey integer NOT NULL,\n"
        "    r_name character(25),\n"
        "    CONSTRAINT region_pkey PRIMARY KEY (r_regionkey)\n"
        ");\n",
    }
    assert 'SELECT count(*) FROM "public"."region"' in server.statements("tpch")


def test_write_snapshots_writes_one_file_per_database(tmp_path: Path) -> None:
    config = PgEnvConfig(data_root=tmp_path)

    write_snapshots(FakeServer(CATALOG).connect, config, ["tpch", "tpch_alt"])

    for db in ("tpch", "tpch_alt"):
        assert json.loads(config.snapshot_path(db).read_text())["db"] == db


def test_explain_plans_plans_each_query_alone_in_a_read_only_transaction() -> None:
    server = FakeServer(
        [
            ("EXPLAIN (FORMAT JSON) SELECT 1", [('[{"Plan": {"Node Type": "Result"}}]',)]),
            ("EXPLAIN (FORMAT JSON) SELECT bad", psycopg.errors.SyntaxError("syntax error")),
        ]
    )
    conn = server.connect("tpch")
    requests = [
        PlanRequest("1", "SELECT 1 ;\n", "no plan"),
        PlanRequest("2", "SELECT bad", "foreign scale"),
    ]

    plans, errors = explain_plans(conn, requests)

    assert plans == {"1": '[{"Plan": {"Node Type": "Result"}}]'}
    assert errors == {"2": "SyntaxError"}
    assert conn.statements == [
        "BEGIN",
        "SET TRANSACTION READ ONLY",
        "EXPLAIN (FORMAT JSON) SELECT 1",
        "COMMIT",
        "BEGIN",
        "SET TRANSACTION READ ONLY",
        "EXPLAIN (FORMAT JSON) SELECT bad",
        "ROLLBACK",
    ]
    assert conn.prepared == ["EXPLAIN (FORMAT JSON) SELECT 1", "EXPLAIN (FORMAT JSON) SELECT bad"]
    assert [name for name, _ in conn.loaders] == ["json"]


def test_a_query_that_returns_no_row_is_an_error() -> None:
    conn = FakeServer().connect("tpch")

    with pytest.raises(psycopg.ProgrammingError, match="no row"):
        schema_snapshot(conn)
