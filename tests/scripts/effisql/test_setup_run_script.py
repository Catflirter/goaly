"""Tests for scripts/effisql/setup_run.py: the summary step, its refusals and its exit codes.

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
        [
            "--phase",
            "summary",
            "--records",
            str(tmp_path / "setup" / "records"),
            "--manual-exclusions",
            str(tmp_path / "manual.json"),
            *extra,
        ]
    )


def save_all(tmp_path: Path, **overrides: Any) -> RecordStore:
    store = RecordStore(tmp_path / "setup" / "records")
    for n in range(1, 11):
        store.save(overrides.get(str(n), record(str(n))))
    return store


def write(tmp_path: Path, *extra: str) -> tuple[int, Path]:
    out = tmp_path / "exclusions.json"
    return run(tmp_path, "--write-exclusions", "--exclusions-out", str(out), *extra), out


def test_writes_the_summary_and_refuses_exclusions_from_a_failed_check(
    dataset: Path, tmp_path: Path
) -> None:
    save_all(tmp_path, **{"3": record("3", gold=gold(speedup=1.1))})
    status, out = write(tmp_path)
    assert status == script.EXIT_REFUSED  # The negative checks and the agreement never ran.
    assert not out.exists()
    summary = json.loads((tmp_path / "setup" / "summary.json").read_text(encoding="utf-8"))
    assert summary["task_count"] == 10
    assert (tmp_path / "setup" / "summary.md").read_text(encoding="utf-8").startswith("- Tasks")


def test_writes_exclusions_after_review_with_the_manual_ones(dataset: Path, tmp_path: Path) -> None:
    save_all(tmp_path)
    (tmp_path / "manual.json").write_text(
        '[{"instance_id": "4", "reason": "review: the gold is at fault"}]', encoding="utf-8"
    )
    status, out = write(tmp_path, "--accept-failed-checks")
    assert status == script.EXIT_FAILED_CHECKS
    assert json.loads(out.read_text(encoding="utf-8")) == [
        {"instance_id": "4", "reason": "review: the gold is at fault"}
    ]


def test_the_stop_rule_writes_no_exclusions(dataset: Path, tmp_path: Path) -> None:
    save_all(tmp_path, **{str(n): record(str(n), gold=gold(matched=False)) for n in (1, 2, 3)})
    status, out = write(tmp_path, "--accept-failed-checks")
    assert status == script.EXIT_REFUSED
    assert not out.exists()
    assert run(tmp_path) == script.EXIT_STOP


def test_a_subset_run_writes_no_exclusions(dataset: Path, tmp_path: Path) -> None:
    save_all(tmp_path)
    status, out = write(tmp_path, "--ids", "1", "--accept-failed-checks")
    assert status == script.EXIT_REFUSED
    assert not out.exists()


def test_missing_records_write_no_exclusions(dataset: Path, tmp_path: Path) -> None:
    RecordStore(tmp_path / "setup" / "records").save(record("1"))
    status, out = write(tmp_path, "--accept-failed-checks")
    assert status == script.EXIT_REFUSED
    assert not out.exists()


def test_grader_failures_write_no_exclusions(dataset: Path, tmp_path: Path) -> None:
    failed = record("3", gold=gold(status="GRADE_STATUS_GRADER_ERROR", matched=False))
    save_all(tmp_path, **{"3": failed})
    status, out = write(tmp_path, "--accept-failed-checks")
    assert status == script.EXIT_REFUSED
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
