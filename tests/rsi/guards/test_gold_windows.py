"""Tests for goaly.rsi.guards.gold_windows: the gold-only windows and the setup that builds them.

Every task here is synthetic: no real gold answer is ever used in a test.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from goaly.rsi.guards.gold_windows import (
    WINDOW,
    GoldWindows,
    build_from_files,
    main,
    normalize,
    windows,
)
from tests.rsi.gitrepo import make_repo

# A base query and a gold that shares its first half with it: the standard part of the join.
BASE = "SELECT l_orderkey, sum(l_extendedprice) FROM lineitem GROUP BY l_orderkey"
GOLD = (
    "SELECT l_orderkey, sum(l_extendedprice) FROM lineitem "
    "WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.o_orderkey = l_orderkey "
    "AND o.o_orderstatus = 'F') GROUP BY l_orderkey"
)
DDL = "CREATE TABLE orders (o_orderkey integer PRIMARY KEY, o_orderstatus char(1));"


@pytest.fixture
def gold() -> GoldWindows:
    return GoldWindows.build([GOLD], [BASE], [DDL])


def test_normalize_lowercases_and_collapses_whitespace() -> None:
    assert normalize("  SELECT  *\n\tFROM   Orders \n") == "select * from orders"


def test_windows_are_every_run_of_40_characters() -> None:
    text = "x" * 38 + "ab"
    assert WINDOW == 40
    assert list(windows(text + "cd")) == [text, text[1:] + "c", text[2:] + "cd"]
    assert list(windows("too short")) == []


def test_a_window_the_base_query_shares_is_not_gold_only(gold: GoldWindows) -> None:
    shared = normalize(BASE)[:WINDOW]

    assert shared in normalize(GOLD)
    assert shared not in gold


def test_a_window_only_the_gold_has_is_gold_only(gold: GoldWindows) -> None:
    start = normalize(GOLD).index("where not exists")

    assert normalize(GOLD)[start : start + WINDOW] in gold


def test_a_window_the_schema_holds_is_not_gold_only() -> None:
    ddl = "o.o_orderkey = l_orderkey and o.o_orderstatus = 'f') group by"
    gold = GoldWindows.build([GOLD], [BASE], [ddl])
    start = normalize(GOLD).index("o.o_orderkey = l_orderkey")

    assert normalize(GOLD)[start : start + WINDOW] not in gold


def test_the_store_holds_hashes_never_the_gold(gold: GoldWindows, tmp_path: Path) -> None:
    path = tmp_path / "cache" / "gold_windows.json"

    gold.save(path)

    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["format"] == "goaly-gold-windows/1"
    assert stored["width"] == WINDOW
    assert stored["windows"] == sorted(stored["windows"])
    assert "exists" not in path.read_text(encoding="utf-8")
    first = normalize(GOLD).index("where not exists")
    window = normalize(GOLD)[first : first + WINDOW]
    assert hashlib.sha256(window.encode("utf-8")).hexdigest() in stored["windows"]
    assert GoldWindows.load(path) == gold


def test_the_store_is_never_written_inside_a_repository(gold: GoldWindows, tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo")

    with pytest.raises(ValueError, match="outside"):
        gold.save(repo / "rsi" / "gold_windows.json")


def test_a_store_of_another_format_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"format": "something-else"}), encoding="utf-8")

    with pytest.raises(ValueError, match="format"):
        GoldWindows.load(path)


def write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    dataset = tmp_path / "effi-sql-pg.jsonl"
    tasks = [
        {"id": 1, "db": "tpch", "base_sql": BASE, "optimized_sql": GOLD},
        {"id": 2, "db": "tpch", "base_sql": "SELECT 1", "optimized_sql": "SELECT 1"},
    ]
    lines = "".join(json.dumps(task) + "\n" for task in tasks)
    dataset.write_text(lines + "\n", encoding="utf-8")  # A blank last line is no task.
    schema = tmp_path / "schema_snapshot.json"
    schema.write_text(json.dumps({"db": "tpch", "tables": [], "ddl": DDL}), encoding="utf-8")
    return dataset, schema


def test_build_from_files_reads_the_dataset_and_the_schema_snapshots(tmp_path: Path) -> None:
    dataset, schema = write_inputs(tmp_path)

    built = build_from_files(dataset, [schema])

    assert built == GoldWindows.build(
        [GOLD, "SELECT 1"],
        [BASE, "SELECT 1"],
        [DDL],
        dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
    )


def test_the_cli_builds_and_stores_the_windows(tmp_path: Path) -> None:
    dataset, schema = write_inputs(tmp_path)
    out = tmp_path / "cache" / "gold_windows.json"

    status = main(["--dataset", str(dataset), "--schema", str(schema), "--out", str(out)])

    assert status == 0
    assert GoldWindows.load(out) == build_from_files(dataset, [schema])


def test_the_module_runs_as_a_command(tmp_path: Path) -> None:
    dataset, schema = write_inputs(tmp_path)
    out = tmp_path / "cache" / "gold_windows.json"

    completed = subprocess.run(  # noqa: S603  (this interpreter, with arguments built here)
        [
            sys.executable,
            "-m",
            "goaly.rsi.guards.gold_windows",
            *("--dataset", str(dataset), "--schema", str(schema), "--out", str(out)),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.endswith(f"gold-only windows stored in {out}\n")
    assert GoldWindows.load(out) == build_from_files(dataset, [schema])


def test_a_store_inside_a_git_directory_is_allowed(gold: GoldWindows, tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "repo")

    gold.save(repo / ".git" / "gold_windows.json")  # Never part of a commit.

    assert GoldWindows.load(repo / ".git" / "gold_windows.json") == gold
