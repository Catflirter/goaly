"""Tests for scripts/effisql/final_report.py, on toy run directories."""

from pathlib import Path

import pytest

from goaly.contracts.gen.task_pb2 import GRADE_STATUS_GRADER_ERROR
from scripts.effisql import final_report
from tests.bench.effisql.rundirs import effisql_grade, experiment, result, write_ledger, write_run
from tests.bench.effisql.synthetic import task, write_tasks

TEMPLATE = Path(__file__).resolve().parents[3] / "docs" / "results" / "final-report-template.md"


def tasks(tmp_path: Path) -> Path:
    return write_tasks(tmp_path / "tasks.jsonl", task(id=1, difficulty="easy"), task(id=2))


def test_writes_the_report_into_the_template(tmp_path: Path) -> None:
    v0 = write_run(tmp_path / "v0", {"1": [1.0], "2": [1.0]}, run_id="v0")
    final = write_run(tmp_path / "final", {"1": [2.0] * 3, "2": [1.0] * 3}, run_id="final")
    ledger = write_ledger(tmp_path / "ledger", [experiment("e1", 1)], [])
    out = tmp_path / "out" / "report.md"

    status = final_report.main(
        [
            *("--v0-heldout", str(v0), "--final-heldout", str(final)),
            *("--ledger", str(ledger), "--tasks", str(tasks(tmp_path))),
            *("--template", str(TEMPLATE), "--out", str(out)),
        ]
    )

    assert status == 0
    report = out.read_text(encoding="utf-8")
    assert report.startswith("# goaly: final report")
    assert "| checked | 2 | 50.0 | 75.0 | +25.0 |" in report
    assert "1 of 1 rounds finished" in report
    assert "| final-val | pending |" in report


def test_writes_the_sections_to_stdout_without_a_template(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status = final_report.main(["--ledger", str(tmp_path), "--tasks", str(tasks(tmp_path))])

    assert status == 0
    assert capsys.readouterr().out.startswith("## Runs")


def test_reports_a_run_still_grading_only_when_allowed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    error = result("run-1", "2", 0, effisql_grade(1.0), status=GRADE_STATUS_GRADER_ERROR)
    run = write_run(tmp_path / "final", {"1": [2.0], "2": [1.0]}, extra=[error])
    args = ["--final-heldout", str(run), "--ledger", str(tmp_path)]
    args += ["--tasks", str(tasks(tmp_path))]

    assert final_report.main(args) == 1
    assert "not graded yet" in capsys.readouterr().err

    assert final_report.main([*args, "--allow-partial"]) == 0
    assert "partial: 1 not yet" in capsys.readouterr().out
