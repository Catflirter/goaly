"""Tests for goaly.harness.core.isolation: what the host asks a launcher to run (spec V27)."""

import dataclasses
import json
from pathlib import Path

import pytest

from goaly.contracts.gen.task_pb2 import SPLIT_EXTERNAL, SPLIT_HELDOUT, SPLIT_VAL
from goaly.harness.core.isolation import (
    TASKS_FILE,
    SolveRequest,
    SolveTasks,
    read_solve_tasks,
    write_solve_tasks,
)

HOST = Path("/home/me/runs/r1")


def request(**change: object) -> SolveRequest:
    fields: dict[str, object] = {
        "run_id": "r1",
        "benchmark": "effisql",
        "harness": "v0",
        "pack_dir": Path("/home/me/goaly/packs/effisql"),
        "split": SPLIT_VAL,
        "instances_file": None,
        "solve_dir": HOST / "solve",
        "inputs_dir": HOST / "inputs",
        "llm_calls": HOST / "llm_calls.jsonl",
        "tasks": (("t1", 0), ("t2", 0)),
        "llm_shares": {"qwen3.6-35b-a3b": 4, "glm-5.1": 1},
        "max_tasks": 6,
        "task_timeout_s": None,
        "resume": False,
        "retry_errors": False,
        "log_level": "INFO",
    }
    fields.update(change)
    return SolveRequest(**fields)  # type: ignore[arg-type]  # The fields vary by test.


# The container's command.


def test_the_container_runs_the_solve_phase_with_its_own_paths() -> None:
    args = request().goaly_args(solve_dir=Path("/run/solve"), inputs_dir=Path("/run/inputs"))

    assert args == [
        "run",
        "--bench",
        "effisql",
        "--harness",
        "v0",
        "--phase",
        "solve",
        "--in-container",
        "--out",
        "/run/solve",
        "--tasks",
        f"/run/solve/{TASKS_FILE}",
        "--inputs",
        "/run/inputs",
        "--split",
        "val",
        "--run-id",
        "r1",
        "--max-tasks",
        "6",
        "--llm-share",
        "glm-5.1=1",
        "--llm-share",
        "qwen3.6-35b-a3b=4",
        "--log-level",
        "INFO",
    ]


def test_the_container_gets_the_time_limit_resume_and_retry_options() -> None:
    args = request(
        split=SPLIT_HELDOUT,
        task_timeout_s=0.1 + 0.2,
        resume=True,
        retry_errors=True,
        log_level="DEBUG",
    ).goaly_args(solve_dir=Path("/s"), inputs_dir=Path("/i"))

    assert args[args.index("--split") + 1] == "heldout"
    assert float(args[args.index("--task-timeout") + 1]) == 0.1 + 0.2  # Exactly, as repr() is.
    assert args[-4:] == ["--resume", "--retry-errors", "--log-level", "DEBUG"]


def test_an_organizers_run_gets_its_task_file_where_the_container_sees_it() -> None:
    organizers = request(split=SPLIT_EXTERNAL, instances_file=Path("/home/me/heldout.jsonl"))

    args = organizers.goaly_args(
        solve_dir=Path("/s"), inputs_dir=Path("/i"), instances_file=Path("/in/heldout.jsonl")
    )

    assert args[args.index("--instances") + 1] == "/in/heldout.jsonl"
    assert "--split" not in args


def test_the_task_file_is_given_exactly_when_the_run_has_one() -> None:
    with pytest.raises(ValueError, match="needs a task file"):
        request(split=SPLIT_EXTERNAL, instances_file=HOST / "t.jsonl").goaly_args(
            solve_dir=Path("/s"), inputs_dir=Path("/i")
        )
    with pytest.raises(ValueError, match="has no use for a task file"):
        request().goaly_args(
            solve_dir=Path("/s"), inputs_dir=Path("/i"), instances_file=Path("/t.jsonl")
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"split": SPLIT_EXTERNAL}, "task file exactly when its split is external"),
        ({"instances_file": HOST / "t.jsonl"}, "task file exactly when its split is external"),
        ({"tasks": ()}, "has no task to solve"),
        ({"max_tasks": 0}, "max_tasks must be at least 1"),
    ],
)
def test_a_request_that_cannot_be_run_is_refused(change: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        request(**change)


def test_a_request_is_a_value() -> None:
    assert dataclasses.replace(request(), harness="v1") != request()
    with pytest.raises(dataclasses.FrozenInstanceError):
        request().harness = "v1"  # type: ignore[misc]  # Frozen, as this checks.


# The tasks file.


SOLVE = SolveTasks(instance_ids=("t1", "t2"), seeds=(0, 1), tasks=(("t2", 1), ("t1", 0)))


def test_a_tasks_file_reads_back_as_written(tmp_path: Path) -> None:
    solve_dir = tmp_path / "solve"  # tmp_path also holds the isolated HOME of conftest.py.
    solve_dir.mkdir()
    path = solve_dir / TASKS_FILE

    write_solve_tasks(path, SOLVE)

    assert read_solve_tasks(path) == SOLVE
    assert list(solve_dir.iterdir()) == [path]  # Written atomically: no temporary file is left.


@pytest.mark.parametrize(
    "fields",
    [
        {"instance_ids": ["t1"], "seeds": [0]},
        {"instance_ids": "t1", "seeds": [0], "tasks": []},
        {"instance_ids": [""], "seeds": [0], "tasks": []},
        {"instance_ids": ["t1"], "seeds": [-1], "tasks": []},
        {"instance_ids": ["t1"], "seeds": [True], "tasks": []},
        {"instance_ids": ["t1"], "seeds": [0], "tasks": [["t1"]]},
        {"instance_ids": ["t1"], "seeds": [0], "tasks": [[1, 0]]},
        {"instance_ids": ["t1"], "seeds": [0], "tasks": [["", 0]]},
        {"instance_ids": ["t1"], "seeds": [0], "tasks": [["t1", "0"]]},
        {"instance_ids": ["t1"], "seeds": [0], "tasks": "t1"},
        [],
    ],
    ids=[
        "no tasks",
        "IDs not a list",
        "empty ID",
        "negative seed",
        "boolean seed",
        "short pair",
        "numeric ID",
        "empty task ID",
        "string seed",
        "tasks not a list",
        "not an object",
    ],
)
def test_a_tasks_file_that_is_not_one_names_the_file(tmp_path: Path, fields: object) -> None:
    path = tmp_path / TASKS_FILE
    path.write_text(json.dumps(fields), encoding="utf-8")

    with pytest.raises(ValueError, match=r"tasks\.json is not a tasks file"):
        read_solve_tasks(path)


@pytest.mark.parametrize("content", [b"", b"{", b"\xff"])
def test_a_tasks_file_that_is_not_json_names_the_file(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / TASKS_FILE
    path.write_bytes(content)

    with pytest.raises(ValueError, match=r"tasks\.json is not a tasks file"):
        read_solve_tasks(path)


@pytest.mark.parametrize(
    "tasks", [[["t3", 0]], [["t1", 2]], [["t1", 0], ["t1", 0]]], ids=["instance", "seed", "twice"]
)
def test_a_tasks_file_lists_only_the_runs_tasks_each_once(
    tmp_path: Path, tasks: list[list[object]]
) -> None:
    path = tmp_path / TASKS_FILE
    fields = {"instance_ids": ["t1", "t2"], "seeds": [0, 1], "tasks": tasks}
    path.write_text(json.dumps(fields), encoding="utf-8")

    with pytest.raises(ValueError, match="tasks the run does not have, or one twice"):
        read_solve_tasks(path)
