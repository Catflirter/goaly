"""Tests for goaly.bench.effisql.report_inputs, on toy run directories and ledgers."""

import json
from pathlib import Path

import pytest

from goaly.bench.effisql.report_inputs import (
    DEFAULT_MAX_TASKS,
    RunDirError,
    load_difficulty,
    load_experiments,
    load_lessons,
    load_run,
)
from goaly.contracts.gen.experiment_pb2 import DECISION_PENDING, DECISION_REJECTED, Lesson
from goaly.contracts.gen.task_pb2 import GRADE_STATUS_GRADER_ERROR
from goaly.contracts.jsonl import append_jsonl
from tests.bench.effisql.rundirs import (
    effisql_grade,
    experiment,
    result,
    write_ledger,
    write_run,
)
from tests.bench.effisql.synthetic import task, write_tasks


def test_reads_a_graded_run(tmp_path: Path) -> None:
    run = load_run(write_run(tmp_path / "r", {"1": [2.0, 1.0], "2": [None, 0.5]}))

    assert run.run_id == "run-1"
    assert run.grading_mode == "official"
    assert run.missing == ()
    assert run.task_seeds == 4
    assert run.table is not None
    assert run.table.seeds == (0, 1)
    assert run.table.grades["1"][0].credit == 1.0  # The graded line won over the ungraded one.
    assert run.run_span_s == 3600


def test_takes_tokens_cost_and_the_solve_span_from_the_host_call_log(tmp_path: Path) -> None:
    run = load_run(write_run(tmp_path / "r", {"1": [2.0], "2": [1.0]}, calls=4, cost=0.25))

    assert run.usage is not None
    assert (run.usage.calls, run.usage.input_tokens, run.usage.output_tokens) == (4, 400, 40)
    assert run.usage.cost_usd == 1.0
    assert run.usage.unpriced_calls == 0
    assert run.usage.solve_span_s == 3 * 60 + 30  # First start to last end.
    assert run.cost_per_task_usd() == 0.5
    # 210 s over 2 tasks, at a concurrency of 2: spec §8.3.
    assert run.wall_per_task_s() == 210


@pytest.mark.parametrize(
    ("command_line", "max_tasks"),
    [
        (("goaly", "run", "--max-tasks", "8"), 8),
        (("goaly", "run", "--max-tasks=12"), 12),
        (("goaly", "run"), DEFAULT_MAX_TASKS),
    ],
)
def test_reads_the_task_concurrency_from_the_command_line(
    tmp_path: Path, command_line: tuple[str, ...], max_tasks: int
) -> None:
    run = load_run(write_run(tmp_path / "r", {"1": [2.0]}, command_line=command_line))

    assert run.max_tasks == max_tasks


def test_a_run_without_a_call_log_has_no_usage(tmp_path: Path) -> None:
    path = write_run(tmp_path / "r", {"1": [2.0]})
    (path / "llm_calls.jsonl").unlink()

    run = load_run(path)

    assert run.usage is None
    assert run.cost_per_task_usd() is None
    assert run.wall_per_task_s() is None


def test_refuses_a_directory_without_a_manifest(tmp_path: Path) -> None:
    with pytest.raises(RunDirError, match=r"no manifest\.json"):
        load_run(tmp_path)


def test_refuses_a_run_not_fully_graded(tmp_path: Path) -> None:
    error = result("run-1", "2", 0, effisql_grade(1.0), status=GRADE_STATUS_GRADER_ERROR)
    path = write_run(tmp_path / "r", {"1": [2.0], "2": [1.0]}, extra=[error])

    with pytest.raises(RunDirError, match=r"1 of 2 .* task '2' seed 0"):
        load_run(path)


def test_a_partial_run_keeps_the_tasks_graded_in_every_seed(tmp_path: Path) -> None:
    ungraded = result("run-1", "2", 1, None)
    path = write_run(tmp_path / "r", {"1": [2.0, 2.0], "2": [1.0, 1.0]}, extra=[ungraded])

    run = load_run(path, allow_partial=True)

    assert run.missing == (("2", 1),)
    assert run.table is not None
    assert run.table.tasks == ("1",)


def test_a_partial_run_with_nothing_graded_has_no_table(tmp_path: Path) -> None:
    path = write_run(tmp_path / "r", {"1": [2.0]}, extra=[result("run-1", "1", 0, None)])

    run = load_run(path, allow_partial=True)

    assert run.table is None
    assert run.missing == (("1", 0),)


def test_refuses_a_result_of_another_run(tmp_path: Path) -> None:
    path = write_run(tmp_path / "r", {"1": [2.0]}, extra=[result("other", "1", 0, None)])

    with pytest.raises(RunDirError, match="run 'other'"):
        load_run(path)


def test_refuses_a_result_for_a_task_the_run_lacks(tmp_path: Path) -> None:
    path = write_run(tmp_path / "r", {"1": [2.0]}, extra=[result("run-1", "9", 0, None)])

    with pytest.raises(RunDirError, match="task '9' seed 0"):
        load_run(path)


def test_reads_difficulty_labels(tmp_path: Path) -> None:
    tasks = write_tasks(
        tmp_path / "tasks.jsonl", task(id=1, difficulty="hard"), task(id=2, difficulty=None)
    )

    assert load_difficulty(tasks) == {"1": "hard", "2": "unlabelled"}


def test_reads_the_last_line_of_each_experiment_and_standing_lessons(tmp_path: Path) -> None:
    deleted = Lesson(lesson_id="l2", text="gone")
    deleted.deleted_at.FromSeconds(1)
    ledger = write_ledger(
        tmp_path / "ledger",
        [experiment("e1", 1, DECISION_PENDING), experiment("e2", 1)],
        [Lesson(lesson_id="l1", text="old"), deleted],
    )
    append_jsonl(ledger / "experiments.jsonl", experiment("e1", 1, DECISION_REJECTED))
    append_jsonl(ledger / "lessons.jsonl", Lesson(lesson_id="l1", text="new"))

    experiments = load_experiments(ledger)
    lessons = load_lessons(ledger)

    assert [(e.experiment_id, e.decision) for e in experiments] == [
        ("e1", DECISION_REJECTED),
        ("e2", experiments[1].decision),
    ]
    assert [lesson.text for lesson in lessons] == ["new"]


def test_a_missing_ledger_is_empty(tmp_path: Path) -> None:
    assert load_experiments(tmp_path) == []
    assert load_lessons(tmp_path) == []


def test_refuses_a_manifest_without_tasks(tmp_path: Path) -> None:
    path = write_run(tmp_path / "r", {"1": [2.0]})
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    del manifest["instance_ids"]
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RunDirError, match="names no tasks"):
        load_run(path)


def test_a_run_still_solving_has_no_results_and_no_finish(tmp_path: Path) -> None:
    path = write_run(tmp_path / "r", {"1": [2.0]})
    (path / "results.jsonl").unlink()
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    del manifest["finished_at"]
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    run = load_run(path, allow_partial=True)

    assert run.table is None
    assert run.missing == (("1", 0),)
    assert run.run_span_s is None
