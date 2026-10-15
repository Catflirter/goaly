"""Tests for goaly.rsi.guards.gold_free: a candidate's evaluation loads no gold answer."""

import json
from pathlib import Path

import pytest

from goaly.rsi.guards.gold_free import GoldDataError, check_gold_free

# A synthetic task in the T08 runtime file's shape: what the agent may see, and grading fields.
RUNTIME_TASK = {
    "id": 1,
    "db": "tpch",
    "base_sql": "SELECT count(*) FROM lineitem",
    "base_time": 1200.5,
    "base_explain_analyze": {"Plan": {"Node Type": "Aggregate"}},
    "difficulty": "easy",
}


def write_jsonl(path: Path, records: list[dict[str, object]]) -> Path:
    lines = [json.dumps(record) + "\n" for record in records]
    path.write_text("".join(lines) + "\n", encoding="utf-8")  # A blank last line is no record.
    return path


def test_runtime_task_files_pass(tmp_path: Path) -> None:
    tasks = write_jsonl(tmp_path / "effi-sql-pg.runtime.jsonl", [RUNTIME_TASK, RUNTIME_TASK])
    splits = tmp_path / "effisql.json"
    splits.write_text(json.dumps({"seed": 20261014, "train": ["1"]}), encoding="utf-8")

    check_gold_free([tasks, splits])


def test_every_gold_field_is_named_with_its_file_and_line(tmp_path: Path) -> None:
    gold = {**RUNTIME_TASK, "optimized_sql": "SELECT 1", "fast_time": 3.0}
    nested = {**RUNTIME_TASK, "extra": {"optimized_explain_analyze": {}}}
    tasks = write_jsonl(tmp_path / "full.jsonl", [RUNTIME_TASK, gold, nested])

    with pytest.raises(GoldDataError) as caught:
        check_gold_free([tasks])

    assert caught.value.problems == [
        f"{tasks}:2: fast_time",
        f"{tasks}:2: optimized_sql",
        f"{tasks}:3: optimized_explain_analyze",
    ]
    assert "SELECT 1" not in str(caught.value)  # The gold itself is never repeated.


def test_a_json_file_is_checked_whole(tmp_path: Path) -> None:
    cache = tmp_path / "plans.json"
    cache.write_text(json.dumps({"7": {"optimized_sql": "x"}}), encoding="utf-8")

    with pytest.raises(GoldDataError, match=r"plans\.json:1: optimized_sql"):
        check_gold_free([cache])


@pytest.mark.parametrize(("name", "text"), [("tasks.csv", "id\n1\n"), ("bad.jsonl", "{not json\n")])
def test_a_file_that_cannot_be_checked_fails(tmp_path: Path, name: str, text: str) -> None:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")

    with pytest.raises(GoldDataError, match=name):
        check_gold_free([path])
