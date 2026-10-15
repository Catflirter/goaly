"""Toy run directories and ledgers, written as goaly run and the RSI lane write them."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from google.protobuf import json_format

from goaly.bench.effisql.metrics import tier_credit
from goaly.contracts.gen.experiment_pb2 import (
    CHANGE_KIND_PROMPT,
    DECISION_ACCEPTED,
    HYPOTHESIS_SOURCE_IDEA_BANK,
    LANE_EFFISQL,
    Decision,
    ExperimentRecord,
    Hypothesis,
    Lesson,
    PhaseResult,
    PurposeUsage,
)
from goaly.contracts.gen.llm_call_pb2 import CALL_PURPOSE_EXEC, LlmCallRecord, Usage
from goaly.contracts.gen.manifest_pb2 import RunManifest
from goaly.contracts.gen.result_pb2 import TaskResult
from goaly.contracts.gen.task_pb2 import (
    GRADE_STATUS_GRADED,
    SPLIT_HELDOUT,
    EffiSqlGrade,
    Grade,
    GradeStatus,
    Split,
)
from goaly.contracts.jsonl import write_jsonl
from tests.bench.effisql.synthetic import BASE_SQL

START = 1_760_515_200  # 2025-10-15 08:00 UTC, in seconds.


def effisql_grade(speedup: float | None, **flags: bool) -> EffiSqlGrade:
    """A grade timed at ``speedup``, or unmatched for None; ``flags`` sets the checks' fields."""
    graded = EffiSqlGrade(matched=speedup is not None)
    if speedup is not None:
        graded.speedup = speedup
        graded.credit = 0.0 if flags.get("literal_result") else tier_credit(speedup)
        if flags.get("alt_matched") is False:
            graded.credit = 0.0
    for name, value in flags.items():
        setattr(graded, name, value)
    return graded


def result(
    run_id: str,
    task: str,
    seed: int,
    graded: EffiSqlGrade | None,
    status: GradeStatus.ValueType = GRADE_STATUS_GRADED,
    wall_time_s: float = 0.0,
) -> TaskResult:
    """A result line: graded, or the solve phase's ungraded line for None."""
    line = TaskResult(run_id=run_id, instance_id=task, seed=seed, wall_time_s=wall_time_s)
    if graded is not None:
        line.grade.CopyFrom(
            Grade(instance_id=task, score=graded.credit, status=status, effisql=graded)
        )
    return line


def write_run(
    path: Path,
    speedups: Mapping[str, Sequence[float | None]],
    *,
    run_id: str = "run-1",
    split: Split.ValueType = SPLIT_HELDOUT,
    mode: str = "official",
    calls: int = 4,
    cost: float = 0.25,
    command_line: Sequence[str] = ("goaly", "run", "--max-tasks", "2"),
    extra: Sequence[TaskResult] = (),
    queries: Mapping[tuple[str, int], str] | None = None,
) -> Path:
    """A graded run of ``speedups[task][seed]``, each line after an ungraded solve line.

    Each graded line has a host-attributed wall_time_s of 30 s.
    The call log has ``calls`` calls, each of 100 input and 10 output tokens and ``cost``
    dollars, 60 s apart and 30 s long.
    """
    path.mkdir(parents=True)
    seeds = len(next(iter(speedups.values())))
    manifest = RunManifest(
        run_id=run_id,
        benchmark="effisql",
        split=split,
        git_commit="c0ffee" * 6,
        harness_hash="ab" * 32,
        seeds=range(seeds),
        instance_ids=list(speedups),
        command_line=command_line,
        environment_settings={"grading_mode": mode},
    )
    manifest.started_at.FromSeconds(START)
    manifest.finished_at.FromSeconds(START + 3600)
    manifest_json = json_format.MessageToDict(manifest, preserving_proto_field_name=True)
    manifest_json["a_field_from_later"] = 1  # Unknown fields are ignored.
    (path / "manifest.json").write_text(json.dumps(manifest_json, indent=2), encoding="utf-8")
    lines = []
    for task, values in speedups.items():
        for seed, speedup in enumerate(values):
            lines += [
                result(run_id, task, seed, None),
                result(run_id, task, seed, effisql_grade(speedup), wall_time_s=30.0),
            ]
    write_jsonl(path / "results.jsonl", [*lines, *extra])
    records = []
    for index in range(calls):
        call = LlmCallRecord(
            run_id=run_id,
            purpose=CALL_PURPOSE_EXEC,
            model="qwen3.6-35b-a3b",
            usage=Usage(input_tokens=100, output_tokens=10, cost_usd=cost),
            latency_s=30.0,
        )
        call.started_at.FromSeconds(START + 60 * index)
        records.append(call)
    write_jsonl(path / "llm_calls.jsonl", records)
    (path / "official").mkdir()
    for seed in range(seeds):  # Each task's official record, as the host writes it.
        records_text = []
        for task in speedups:
            sql = (queries or {}).get((task, seed), BASE_SQL)
            record = {"id": int(task), "base_sql": BASE_SQL, "prediction": official(sql)}
            records_text.append(json.dumps(record) + "\n")
        official_file = path / "official" / f"seed-{seed}.jsonl"
        official_file.write_text("".join(records_text), encoding="utf-8")
    return path


def official(sql: str) -> str:
    """An official prediction: its sql block, then its diff block."""
    return f"```sql\n{sql};\n```\n\n```diff\n-x\n+y\n```"


def experiment(
    experiment_id: str,
    round_number: int,
    decision: Decision.ValueType = DECISION_ACCEPTED,
    *,
    val_score: float = 0.5,
    val_delta: float = 0.1,
    cost: float = 1.5,
) -> ExperimentRecord:
    """An experiment of ``round_number`` with a validation, and one exec usage of ``cost``."""
    return ExperimentRecord(
        experiment_id=experiment_id,
        lane=LANE_EFFISQL,
        round=round_number,
        hypothesis=Hypothesis(
            change_kind=CHANGE_KIND_PROMPT,
            change="Ask for a plan | then\nrewrite",
            source=HYPOTHESIS_SOURCE_IDEA_BANK,
        ),
        screening=PhaseResult(delta=0.2),
        validation=PhaseResult(candidate_score=val_score, delta=val_delta),
        decision=decision,
        usage_by_purpose=[
            PurposeUsage(
                purpose=CALL_PURPOSE_EXEC,
                usage=Usage(input_tokens=1000, output_tokens=100, cost_usd=cost),
            )
        ],
    )


def write_ledger(
    path: Path, experiments: Sequence[ExperimentRecord], lessons: Sequence[Lesson]
) -> Path:
    path.mkdir(parents=True)
    write_jsonl(path / "experiments.jsonl", experiments)
    write_jsonl(path / "lessons.jsonl", lessons)
    return path


def write_stages(root: Path, run: Path, *durations: float) -> None:
    """Append a solve segment of each duration for ``run`` to ``root``'s stages.jsonl."""
    with (root / "stages.jsonl").open("a", encoding="utf-8") as file:
        for duration in durations:
            argv = ["goaly", "run", "--phase", "solve", "--out", str(run)]
            record: dict[str, object] = {"run": run.name, "phase": "solve", "argv": argv}
            record |= {"started_at": START, "finished_at": START + duration, "exit_status": 0}
            file.write(json.dumps(record) + "\n")


def write_plans(run: Path, plans: Mapping[tuple[str, int], object]) -> None:
    """Write the grade phase's stored plans, ``plans[(task, seed)]``."""
    (run / "plans").mkdir(exist_ok=True)
    for seed in sorted({seed for _, seed in plans}):
        lines = [
            json.dumps({"instance_id": task, "plan": plan}) + "\n"
            for (task, at), plan in plans.items()
            if at == seed
        ]
        (run / "plans" / f"seed-{seed}.jsonl").write_text("".join(lines), encoding="utf-8")
