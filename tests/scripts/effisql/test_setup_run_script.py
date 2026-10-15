"""Tests for scripts/effisql/setup_run.py: the summary step and its exit codes.

The phases themselves run on the real instances; tests/bench/effisql/test_setup_run.py covers
them with fakes.
"""

import json
import os
from pathlib import Path
from typing import Any

import pytest

from goaly.bench.effisql.setup_run import RecordStore
from scripts.effisql import setup_run as script
from tests.bench.effisql.synthetic import task, write_tasks
from tests.bench.effisql.test_setup_run import gold, record


@pytest.fixture
def dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = write_tasks(tmp_path / "tasks.jsonl", *(task(id=n) for n in range(1, 11)))
    monkeypatch.setattr(script, "fetch_dataset", lambda _cache: path)
    affinity: list[Any] = []
    monkeypatch.setattr(os, "sched_setaffinity", lambda _pid, cpus: affinity.append(cpus))
    return path


def run(tmp_path: Path, *extra: str) -> int:
    return script.main(
        ["--phase", "summary", "--records", str(tmp_path / "setup" / "records"), *extra]
    )


def test_writes_the_summary_and_the_exclusions(dataset: Path, tmp_path: Path) -> None:
    store = RecordStore(tmp_path / "setup" / "records")
    for n in range(1, 11):
        store.save(record(str(n)))
    store.save(record("3", gold=gold(speedup=1.1)))
    out = tmp_path / "exclusions.json"
    status = run(tmp_path, "--write-exclusions", "--exclusions-out", str(out))
    assert status == 3  # The negative checks and the agreement never ran.
    assert json.loads(out.read_text(encoding="utf-8")) == [
        {
            "instance_id": "3",
            "reason": "gold optimized_sql is only 1.10x faster than base_sql (under 1.2x)",
        }
    ]
    summary = json.loads((tmp_path / "setup" / "summary.json").read_text(encoding="utf-8"))
    assert summary["task_count"] == 10
    assert (tmp_path / "setup" / "summary.md").read_text(encoding="utf-8").startswith("- Tasks")


def test_the_stop_rule_writes_no_exclusions(dataset: Path, tmp_path: Path) -> None:
    store = RecordStore(tmp_path / "setup" / "records")
    for n in range(1, 11):
        store.save(record(str(n), gold=gold(matched=False) if n <= 2 else gold()))
    out = tmp_path / "exclusions.json"
    assert run(tmp_path, "--write-exclusions", "--exclusions-out", str(out)) == 2
    assert not out.exists()


def test_ids_select_tasks(dataset: Path, tmp_path: Path) -> None:
    RecordStore(tmp_path / "setup" / "records").save(record("1"))
    run(tmp_path, "--ids", "1")
    summary = json.loads((tmp_path / "setup" / "summary.json").read_text(encoding="utf-8"))
    assert summary["task_count"] == 1
    assert summary["missing"] == []


def test_a_missing_dataset_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(_cache: Path) -> Path:
        raise OSError("no such file")

    monkeypatch.setattr(script, "fetch_dataset", missing)
    assert run(tmp_path) == 1
