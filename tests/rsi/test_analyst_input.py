"""Tests for goaly.rsi.analyst_input: what the analyst is shown, and what it is never shown.

Every task, SQL text and gold answer here is synthetic. Val and held-out tasks carry IDs and text
that appear nowhere else (777xx, 888xx, VAL-, HELD-), so a test can show none of them reaches the
prompt.
"""

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from goaly.bench.effisql.data import EffiSqlInstance
from goaly.bench.effisql.splits import EffiSqlSplits
from goaly.contracts.gen.experiment_pb2 import (
    CHANGE_KIND_PROMPT,
    CHANGE_KIND_SKILL,
    DECISION_ABORTED,
    DECISION_PENDING,
    DECISION_REJECTED,
    GUARD_ALLOWED_PATHS,
    GUARD_UNIT_TESTS,
    HYPOTHESIS_SOURCE_ANALYST,
    HYPOTHESIS_SOURCE_IDEA_BANK,
    IMPLEMENTER_EXIT_SUBMITTED,
    LANE_EFFISQL,
    REJECTION_REASON_HARD_CODED,
    ExperimentRecord,
    GuardResult,
    Hypothesis,
    ImplementerRun,
    PhaseResult,
)
from goaly.contracts.gen.result_pb2 import TaskResult
from goaly.contracts.gen.task_pb2 import (
    GRADE_STATUS_GRADED,
    GRADE_STATUS_GRADER_ERROR,
    GRADE_STATUS_PREDICTION_FAILED,
    SPLIT_HELDOUT,
    SPLIT_TRAIN,
    SPLIT_VAL,
    EffiSqlGrade,
    EffiSqlPrediction,
    EffiSqlTaskInput,
    Grade,
    GradeStatus,
    Prediction,
    TaskInput,
)
from goaly.contracts.gen.trajectory_pb2 import (
    STOP_REASON_STEP_BUDGET_EXHAUSTED,
    STOP_REASON_SUBMITTED,
    Step,
    ToolResult,
    Trajectory,
)
from goaly.rsi.analyst import Analysis, AnalystAnswer, AnalystInput, analyze
from goaly.rsi.analyst_input import CHARS_PER_TOKEN, build_analyst_input
from goaly.rsi.ledger import Ledger
from goaly.rsi.lessons import LessonStore
from goaly.rsi.structured import format_instructions
from tests.bench.effisql.synthetic import task
from tests.rsi.fake_llm import FakeGlm

pytestmark = pytest.mark.anyio

TRAIN = ("101", "102", "103", "104", "105")
VAL = ("77701", "77702")
HELDOUT = ("88801",)
SPLITS = EffiSqlSplits(seed=1, source_sha256="0" * 64, train=TRAIN, val=VAL, heldout=HELDOUT)
ROUND = "round-2"
# Text that only val and held-out data hold.
SECRETS = ("777", "888", "VAL-", "HELD-")


def instance(task_id: str, marker: str) -> EffiSqlInstance:
    return EffiSqlInstance.model_validate(
        task(
            id=int(task_id),
            base_sql=f"SELECT {marker}base_{task_id}",
            optimized_sql=f"SELECT {marker}GOLD_{task_id}",
            difficulty="hard" if task_id == "102" else "easy",
        )
    )


def result(
    task_id: str,
    credit: float,
    *,
    matched: bool = True,
    speedup: float | None = 1.0,
    seed: int = 0,
    status: GradeStatus.ValueType = GRADE_STATUS_GRADED,
    alt_matched: bool | None = None,
    feedback: str = "",
) -> TaskResult:
    details = EffiSqlGrade(matched=matched, credit=credit, speedup=speedup)
    if alt_matched is not None:
        details.alt_matched = alt_matched
    grade = Grade(
        instance_id=task_id,
        score=credit,
        status=status,
        feedback=feedback or f"feedback for {task_id}",
        effisql=details,
    )
    return TaskResult(
        run_id="best-run",
        instance_id=task_id,
        seed=seed,
        grade=grade,
        step_count=7,
        stop_reason=STOP_REASON_SUBMITTED,
    )


def trajectory(
    task_id: str, sql: str, *, seed: int = 0, errors: Sequence[str] = (), fallback: bool = False
) -> Trajectory:
    steps = [
        Step(index=index, tool_results=[ToolResult(name="execute_sql", content=content)])
        for index, content in enumerate(["12 rows in 3.2 s", *errors])
    ]
    return Trajectory(
        run_id="best-run",
        instance_id=task_id,
        seed=seed,
        task=TaskInput(effisql=EffiSqlTaskInput(base_sql=f"SELECT traj_base_{task_id}")),
        steps=steps,
        prediction=Prediction(
            instance_id=task_id, fallback=fallback, effisql=EffiSqlPrediction(sql=sql)
        ),
        stop_reason=STOP_REASON_STEP_BUDGET_EXHAUSTED if fallback else STOP_REASON_SUBMITTED,
    )


def train_data() -> tuple[list[TaskResult], list[Trajectory], list[EffiSqlInstance]]:
    results = [
        result("101", 0.5),  # The identity floor.
        result("102", 0.0, matched=False, speedup=None, status=GRADE_STATUS_PREDICTION_FAILED),
        result("103", 0.0, speedup=1000.0, alt_matched=False),  # Hard-coded.
        result("104", 0.25, speedup=0.7),
        result("105", 1.0, speedup=3.0),  # Solved.
    ]
    trajectories = [
        trajectory("101", "SELECT train_pred_101", fallback=True),
        trajectory("102", "SELECT train_pred_102", errors=["ERROR: column x does not exist"]),
        trajectory("103", "SELECT 42 AS train_pred_103"),
        trajectory("104", "SELECT train_pred_104"),
        trajectory("105", "SELECT train_pred_105"),
    ]
    return results, trajectories, [instance(task_id, "") for task_id in TRAIN]


def secret_data() -> tuple[list[TaskResult], list[Trajectory], list[EffiSqlInstance]]:
    """Results, trajectories and gold of val and held-out tasks, which must never be shown."""
    results = [result(task_id, 0.0, matched=False) for task_id in (*VAL, *HELDOUT)]
    trajectories = [trajectory(task_id, f"SELECT VAL-PRED-{task_id}") for task_id in VAL]
    trajectories.append(trajectory(HELDOUT[0], "SELECT HELD-PRED"))
    instances = [instance(task_id, "VAL-") for task_id in VAL]
    instances.append(instance(HELDOUT[0], "HELD-"))
    return results, trajectories, instances


def phase(split: Any, flagged: Sequence[str]) -> PhaseResult:
    return PhaseResult(
        split=split,
        task_count=20,
        seed_count=1,
        candidate_score=0.42,
        best_score=0.40,
        delta=0.02,
        wins=4,
        losses=2,
        sign_test_p=0.34,
        regressions=1,
        token_ratio_vs_best=1.2,
        flagged_instance_ids=flagged,
        candidate_run_ids=["VAL-RUN-1"],
        best_run_ids=["VAL-RUN-0"],
    )


def queued_hypothesis(hypothesis_id: str = "h-queued") -> Hypothesis:
    return Hypothesis(
        hypothesis_id=hypothesis_id,
        failure_pattern="slow joins",
        target_instance_ids=["104", "77701"],
        change_kind=CHANGE_KIND_SKILL,
        change="Add a skill guide on aggregating lineitem before the join.",
        expected_effect="Faster joins.",
        risk="Longer prompts.",
        source=HYPOTHESIS_SOURCE_IDEA_BANK,
    )


def leaky_queued() -> Hypothesis:
    leaky = queued_hypothesis("h-queued-2")
    leaky.change = "Copy what worked on IDs 104/77701."
    return leaky


def ledger_with_secrets(directory: Path) -> Ledger:
    """A ledger whose experiment holds val IDs, val runs and val-specific reasoning."""
    ledger = Ledger(directory)
    record = ExperimentRecord(
        experiment_id="exp-1",
        lane=LANE_EFFISQL,
        round=1,
        hypothesis=Hypothesis(
            hypothesis_id="h-old",
            failure_pattern="identity fallbacks",
            target_instance_ids=["101", "77702"],
            change_kind=CHANGE_KIND_PROMPT,
            change="Tell the executor to verify before submitting.",
            source=HYPOTHESIS_SOURCE_ANALYST,
        ),
        implementer=ImplementerRun(exit=IMPLEMENTER_EXIT_SUBMITTED, summary="Edited prompt.md."),
        guards=[
            GuardResult(guard=GUARD_ALLOWED_PATHS, passed=True),
            GuardResult(guard=GUARD_UNIT_TESTS, passed=False, detail="VAL-detail"),
        ],
        screening=phase(SPLIT_TRAIN, ["103"]),
        validation=phase(SPLIT_VAL, list(VAL)),
        confirmation=phase(SPLIT_VAL, ["77702"]),
        decision=DECISION_REJECTED,
        rejection_reasons=[REJECTION_REASON_HARD_CODED],
        decision_reason="VAL- task 77701 was hard-coded and 77702 regressed",
    )
    ledger.record_experiment(record)
    leaky = ExperimentRecord(
        experiment_id="exp-2",
        lane=LANE_EFFISQL,
        round=2,
        hypothesis=Hypothesis(
            hypothesis_id="h-leaky",
            failure_pattern="tasks 101 and 88801 time out",
            change_kind=CHANGE_KIND_PROMPT,
            change="Rewrite the way task 77701 does.",
            source=HYPOTHESIS_SOURCE_ANALYST,
        ),
        screening=phase(SPLIT_TRAIN, ["103", "77702"]),  # Mixed: counted, not named.
        confirmation=phase(SPLIT_HELDOUT, []),  # Never a held-out phase.
        decision=DECISION_PENDING,
    )
    leaky.confirmation.candidate_score = 0.61
    ledger.record_experiment(leaky)
    ledger.record_experiment(
        ExperimentRecord(
            experiment_id="exp-3",
            lane=LANE_EFFISQL,
            round=2,
            hypothesis=Hypothesis(hypothesis_id="h-3", change_kind=CHANGE_KIND_SKILL),
            implementer=ImplementerRun(
                exit=IMPLEMENTER_EXIT_SUBMITTED, summary="See instance #88801."
            ),
            decision=DECISION_ABORTED,
        )
    )
    lessons = LessonStore(ledger)
    lessons.add("L-1", LANE_EFFISQL, "Verify the rows before submitting.", ["exp-1"])
    lessons.mark_helped("L-1", "exp-1")
    lessons.add("L-2", LANE_EFFISQL, "Task 77702 regressed when the prompt grew.", ["exp-1"])
    lessons.add("L-3", LANE_EFFISQL, "A deleted lesson.", ["exp-1"])
    lessons.delete("L-3")
    return ledger


def build(tmp_path: Path, **options: Any) -> AnalystInput:
    results, trajectories, instances = train_data()
    secret_results, secret_trajectories, secret_instances = secret_data()
    return build_analyst_input(
        splits=SPLITS,
        ledger=ledger_with_secrets(tmp_path / "ledger"),
        best_results=[*results, *secret_results],
        trajectories=[*trajectories, *secret_trajectories],
        instances=[*instances, *secret_instances],
        queued=[queued_hypothesis(), leaky_queued()],
        **options,
    )


def prompt(analyst_input: AnalystInput) -> str:
    return "\n".join(message.content for message in analyst_input.messages())


def summary_of(analyst_input: AnalystInput, task_id: str) -> str:
    start = analyst_input.user.index(f"### Task {task_id} ")
    return analyst_input.user[start : analyst_input.user.index("\n\n", start)]


# Input hygiene (spec §7, §8.5).


def test_the_input_holds_nothing_of_val_or_heldout(tmp_path: Path) -> None:
    analyst_input = build(tmp_path)
    text = prompt(analyst_input)
    for secret in SECRETS:
        assert secret not in text
    assert analyst_input.failed_task_ids <= set(TRAIN)


def test_the_input_holds_train_failures_with_their_gold(tmp_path: Path) -> None:
    text = prompt(build(tmp_path))
    for task_id in ("101", "102", "103", "104"):
        assert f"### Task {task_id} " in text
        assert f"SELECT GOLD_{task_id}" in text
        assert f"train_pred_{task_id}" in text
        assert f"feedback for {task_id}" in text
    assert "### Task 105 " not in text  # Solved, so not a failure.
    assert "GOLD_105" not in text
    assert "Final SQL (the fallback):" in text
    assert "ERROR: column x does not exist" in text
    assert "flags=hard_coded" in text
    assert "tau=1e+03" in text


def test_the_ledger_summary_keeps_aggregates_and_train_ids_only(tmp_path: Path) -> None:
    text = prompt(build(tmp_path))
    assert "exp-1, round 1: hypothesis h-old [prompt, analyst]" in text
    assert "(train tasks 101)" in text
    assert "Decision: REJECTED (HARD_CODED)" in text
    assert "Implementer: SUBMITTED: Edited prompt.md." in text
    assert "Failed guards: GUARD_UNIT_TESTS" in text
    assert "flagged tasks: 103" in text  # The screen runs on train.
    assert "Validation: 20 tasks x 1 seeds; R-VES 42.0 vs best 40.0 (delta +2.0)" in text
    assert "2 flagged tasks" in text  # Validation's flags, counted, not named.
    assert "tokens x1.20 of best" in text
    assert "regressed" not in text  # The decision's free-text reasoning is left out.


def test_the_input_shows_the_queue_and_reserves_its_ids(tmp_path: Path) -> None:
    analyst_input = build(tmp_path)
    text = prompt(analyst_input)
    assert "hypothesis h-queued [skill, idea_bank]" in text
    assert "(train tasks 104)" in text
    assert analyst_input.taken_ids == {"h-old", "h-leaky", "h-3", "h-queued", "h-queued-2"}


def test_the_input_holds_the_rules_and_the_idea_bank(tmp_path: Path) -> None:
    analyst_input = build(tmp_path)
    assert "tau >= 2: 1\n  - tau >= 1.2: 0.75\n  - tau >= 1: 0.5\n  - tau >= 0.5: 0.25" in (
        analyst_input.system
    )
    assert "solved when its credit is at least 0.75;" in analyst_input.system
    assert "between 0.25 and 0.5" in analyst_input.system
    assert "Decorrelate subqueries" in analyst_input.user
    assert "$" not in prompt(analyst_input)  # Every placeholder was filled.


def test_train_results_list_every_train_task(tmp_path: Path) -> None:
    text = build(tmp_path).user
    assert "105, easy, GRADED, matched, tau=3, credit=1" in text
    assert "102, hard, PREDICTION_FAILED, unmatched, credit=0" in text


# What fails, and in which order failures are kept.


def test_failures_come_flagged_first_then_by_credit(tmp_path: Path) -> None:
    analyst_input = build(tmp_path)
    assert analyst_input.shown_task_ids == ("103", "102", "104", "101")
    assert analyst_input.dropped_task_ids == ()


def test_whole_failures_are_dropped_lowest_priority_first(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    full = build(tmp_path / "full", max_input_tokens=10**6)
    whole = len(prompt(full)) + len(format_instructions(AnalystAnswer))
    # Room for 103 and 102, and for half of 104; the budget also keeps room to name the dropped.
    room = whole + 220 - len(summary_of(full, "101")) - len(summary_of(full, "104")) // 2
    tokens = int(room / CHARS_PER_TOKEN)
    with caplog.at_level(logging.WARNING, logger="goaly.rsi.analyst_input"):
        analyst_input = build(tmp_path / "cut", max_input_tokens=tokens)
    assert analyst_input.shown_task_ids == ("103", "102")
    assert analyst_input.dropped_task_ids == ("104", "101")
    assert analyst_input.failed_task_ids == {"101", "102", "103", "104"}
    size = len(prompt(analyst_input)) + len(format_instructions(AnalystAnswer))
    assert size <= tokens * CHARS_PER_TOKEN
    assert "2 more failed tasks are not summarized, to keep the input short: 104, 101." in (
        analyst_input.user
    )
    assert "### Task 104 " not in analyst_input.user
    [log] = [record for record in caplog.records if record.msg.startswith("dropped")]
    assert vars(log)["dropped"] == ["104", "101"]


def test_long_sql_is_cut_in_the_middle(tmp_path: Path) -> None:
    sql = "SELECT " + "x, " * 2000 + "END_OF_QUERY"
    analyst_input = build_analyst_input(
        splits=SPLITS,
        ledger=Ledger(tmp_path / "ledger"),
        best_results=[result("101", 0.0)],
        trajectories=[trajectory("101", sql)],
        instances=[],
        queued=[],
    )
    assert "END_OF_QUERY" in analyst_input.user
    assert "characters cut ...]" in analyst_input.user
    assert "(no experiments yet)" in analyst_input.user
    assert "Base SQL:\n```sql\nSELECT traj_base_101" in analyst_input.user  # From the trajectory.


def test_a_task_fails_on_its_mean_credit_and_shows_its_worst_seed(tmp_path: Path) -> None:
    analyst_input = build_analyst_input(
        splits=SPLITS,
        ledger=Ledger(tmp_path / "ledger"),
        best_results=[
            result("101", 1.0, seed=0, speedup=2.5),
            result("101", 0.25, seed=1, speedup=0.6),
            result("102", 1.0, seed=0, speedup=2.5),
            result("102", 0.75, seed=1, speedup=1.5),
            result("103", 0.0, status=GRADE_STATUS_GRADER_ERROR),
        ],
        trajectories=[
            trajectory("101", "SELECT seed_zero", seed=0),
            trajectory("101", "SELECT seed_one", seed=1),
        ],
        instances=[instance("101", "")],
        queued=[],
    )
    assert analyst_input.failed_task_ids == {"101"}  # 102 averages 0.875; 103 is not scored.
    assert "credit 0.625" in analyst_input.user
    assert "SELECT seed_one" in analyst_input.user
    assert "SELECT seed_zero" not in analyst_input.user
    assert "101 seed 1, easy, GRADED, matched, tau=0.6, credit=0.25" in analyst_input.user
    assert "103 seed 0, unknown, GRADER_ERROR" in analyst_input.user


def test_a_failure_without_a_trajectory_is_still_summarized(tmp_path: Path) -> None:
    analyst_input = build_analyst_input(
        splits=SPLITS,
        ledger=Ledger(tmp_path / "ledger"),
        best_results=[result("101", 0.0), result("102", 0.0)],
        trajectories=[
            Trajectory(run_id="best-run", instance_id="102", error_message="the policy crashed")
        ],
        instances=[],
        queued=[queued_hypothesis(f"h-{n}") for n in range(25)],
    )
    assert "Final SQL: (no trajectory)" in analyst_input.user
    assert "Final SQL: (none: the policy returned no prediction)" in analyst_input.user
    assert "the run: the policy crashed" in analyst_input.user
    assert "(5 more are queued.)" in analyst_input.user


# The input, answered.


async def test_an_answer_to_a_built_input_becomes_hypotheses(tmp_path: Path) -> None:
    proposed = {
        "change_kind": "prompt",
        "change": "In prompt.md, ask for a verified rewrite.",
        "expected_effect": "Fewer unmatched tasks.",
        "risk": "More steps.",
        "source": "analysis",
    }
    reply = {
        "failure_patterns": [
            {"name": "wrong rewrites", "task_ids": ["102", "103"], "description": "Rows differ."}
        ],
        "hypotheses": [
            {"id": f"h-{n}", "target_pattern": "wrong rewrites", **proposed} for n in range(3)
        ],
    }
    gateway = FakeGlm(json.dumps(reply))
    async with gateway.client() as client:
        analysis = await analyze(client, build(tmp_path), round_id=ROUND)
    assert isinstance(analysis, Analysis)
    assert [h.hypothesis_id for h in analysis.hypotheses] == ["h-0", "h-1", "h-2"]
    sent = gateway.bodies[0]["messages"]
    assert sent[0]["content"].startswith("You are the failure analyst")
    assert "### Task 103 " in sent[1]["content"]


# Free text, lessons and phases (review M1, M2; the lessons follow-up).


def events(caplog: pytest.LogCaptureFixture, event: str) -> list[dict[str, Any]]:
    return [vars(record) for record in caplog.records if vars(record).get("event") == event]


def test_free_text_that_names_a_secret_task_is_withheld_and_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="goaly.rsi.analyst_input"):
        text = prompt(build(tmp_path))
    assert text.count("[withheld: it names a task outside train]") == 4
    assert "Implementer: SUBMITTED: [withheld" in text
    assert "hypothesis h-leaky [prompt, analyst] targeting '[withheld" in text
    where = {entry["where"] for entry in events(caplog, "analyst_input_withheld")}
    assert where == {
        "exp-2: hypothesis h-leaky, failure_pattern",
        "exp-2: hypothesis h-leaky, change",
        "experiment exp-3: implementer summary",
        "queued: hypothesis h-queued-2, change",
        "lesson L-2",
    }


def test_numbers_that_are_not_task_references_are_kept(tmp_path: Path) -> None:
    hypothesis = queued_hypothesis()
    hypothesis.change = "Allow 77701 tokens, and use 88801 as the row limit; see task 104."
    analyst_input = build_analyst_input(
        splits=SPLITS,
        ledger=Ledger(tmp_path / "ledger"),
        best_results=[result("101", 0.0)],
        trajectories=[],
        instances=[],
        queued=[hypothesis],
    )
    assert hypothesis.change in analyst_input.user


def test_the_lanes_latest_standing_lessons_are_shown(tmp_path: Path) -> None:
    ledger = ledger_with_secrets(tmp_path / "ledger")
    store = LessonStore(ledger)
    for n in range(12):
        store.add(f"L-n{n}", LANE_EFFISQL, f"Lesson number {n}.", ["exp-1"])
    text = build_analyst_input(
        splits=SPLITS,
        ledger=ledger,
        best_results=[result("101", 0.0)],
        trajectories=[],
        instances=[],
        queued=[],
    ).user
    shown = [f"Lesson number {n}." for n in range(12) if f"Lesson number {n}." in text]
    assert shown == [f"Lesson number {n}." for n in range(2, 12)]  # The latest 10.
    assert "Verify the rows" not in text  # Older than the latest 10.


def test_a_lesson_line_shows_its_counts_and_evidence(tmp_path: Path) -> None:
    text = build(tmp_path).user
    assert "- L-1 (helped 1, hurt 0; evidence: exp-1): Verify the rows before submitting." in text
    assert "A deleted lesson." not in text
    assert "L-2" not in text


def test_a_phase_on_the_wrong_split_is_left_out_and_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="goaly.rsi.analyst_input"):
        text = build(tmp_path).user
    exp_2 = text[text.index("- exp-2, round 2") : text.index("- exp-3, round 2")]
    assert "Confirmation" not in exp_2
    assert "R-VES 61.0" not in text
    assert "Screening: 20 tasks x 1 seeds" in exp_2
    assert "2 flagged tasks" in exp_2  # A val ID among them, so none is named.
    assert "Decision: PENDING\n" in exp_2
    [wrong] = events(caplog, "analyst_input_wrong_split")
    assert (wrong["experiment_id"], wrong["phase"], wrong["split"]) == (
        "exp-2",
        "confirmation",
        "SPLIT_HELDOUT",
    )


# Pairing trajectories with results (review I1).


def test_a_trajectory_relabelled_from_another_run_is_ignored(tmp_path: Path) -> None:
    forged = trajectory("101", "SELECT VAL-PRED-77701", errors=["ERROR: VAL- in 77701"])
    forged.run_id = "best-val-run"
    analyst_input = build_analyst_input(
        splits=SPLITS,
        ledger=Ledger(tmp_path / "ledger"),
        best_results=[result("101", 0.5)],
        trajectories=[forged, trajectory("101", "SELECT train_pred_101"), forged],
        instances=[],
        queued=[],
    )
    assert "SELECT train_pred_101" in analyst_input.user
    for secret in SECRETS:
        assert secret not in analyst_input.user


def test_a_trajectory_listed_twice_is_left_out(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="goaly.rsi.analyst_input"):
        analyst_input = build_analyst_input(
            splits=SPLITS,
            ledger=Ledger(tmp_path / "ledger"),
            best_results=[result("101", 0.5)],
            trajectories=[trajectory("101", "SELECT one"), trajectory("101", "SELECT two")],
            instances=[],
            queued=[],
        )
    assert "Final SQL: (no trajectory)" in analyst_input.user
    [duplicate] = events(caplog, "analyst_input_duplicate_trajectories")
    assert duplicate["tasks"] == ["101 seed 0 of best-run"]


def test_a_task_seed_graded_in_two_runs_is_left_out(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    other = result("101", 0.0)
    other.run_id = "best-run-2"
    regraded = result("102", 1.0, speedup=3.0)  # The same run graded again: the last one wins.
    with caplog.at_level(logging.WARNING, logger="goaly.rsi.analyst_input"):
        analyst_input = build_analyst_input(
            splits=SPLITS,
            ledger=Ledger(tmp_path / "ledger"),
            best_results=[result("101", 0.5), other, result("102", 0.0), regraded],
            trajectories=[],
            instances=[],
            queued=[],
        )
    assert analyst_input.failed_task_ids == frozenset()
    assert "101 seed" not in analyst_input.user
    assert "102, unknown, GRADED, matched, tau=3, credit=1" in analyst_input.user
    [ambiguous] = events(caplog, "analyst_input_ambiguous_results")
    assert ambiguous["tasks"] == ["101 seed 0"]


def test_each_summary_uses_its_own_runs_trajectory(tmp_path: Path) -> None:
    first, second = result("101", 0.5), result("102", 0.0)
    second.run_id = "best-run-2"
    own = trajectory("102", "SELECT from_run_2")
    own.run_id = "best-run-2"
    analyst_input = build_analyst_input(
        splits=SPLITS,
        ledger=Ledger(tmp_path / "ledger"),
        best_results=[first, second],
        trajectories=[
            trajectory("101", "SELECT from_run_1"),
            trajectory("102", "SELECT wrong"),
            own,
        ],
        instances=[],
        queued=[],
    )
    assert "SELECT from_run_1" in summary_of(analyst_input, "101")
    assert "SELECT from_run_2" in summary_of(analyst_input, "102")
    assert "SELECT wrong" not in analyst_input.user


# The budget (review M4) and the rest of the rendering (review M8).


def many_experiments(directory: Path, count: int) -> Ledger:
    ledger = Ledger(directory)
    for n in range(count):
        ledger.record_experiment(
            ExperimentRecord(
                experiment_id=f"exp-{n}",
                lane=LANE_EFFISQL,
                round=n + 1,
                hypothesis=Hypothesis(
                    hypothesis_id=f"h-{n}",
                    change_kind=CHANGE_KIND_PROMPT,
                    change=f"Change number {n}. " + "x" * 300,
                    target_instance_ids=list(TRAIN) * 3,
                ),
                decision=DECISION_PENDING,
            )
        )
    return ledger


def test_the_oldest_experiments_go_when_the_ledger_is_too_long(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    ledger = many_experiments(tmp_path / "ledger", 45)
    options: dict[str, Any] = {
        "splits": SPLITS,
        "ledger": ledger,
        "best_results": [result("101", 0.0)],
        "trajectories": [],
        "instances": [],
        "queued": [],
    }
    full = build_analyst_input(**options, max_input_tokens=10**6)
    whole = full.user
    assert "(5 earlier experiments are not shown.)" in whole  # At most 40 are shown.
    assert "(train tasks 101, 102, 103, 104, 105, 101, 102, 103, 104, 105 and 5 more)" in whole
    size = len(prompt(full)) + len(format_instructions(AnalystAnswer))
    tokens = int((size - 3000) / CHARS_PER_TOKEN)  # Room for several experiments fewer.
    with caplog.at_level(logging.WARNING, logger="goaly.rsi.analyst_input"):
        cut = build_analyst_input(**options, max_input_tokens=tokens)
    [event] = events(caplog, "analyst_input_sections_cut")
    hidden = 5 + event["cut"]["ledger"]
    assert event["cut"]["ledger"] > 0
    assert f"({hidden} earlier experiments are not shown.)" in cut.user
    assert f"Change number {hidden - 1}." not in cut.user
    assert f"Change number {hidden}." in cut.user
    size = len(prompt(cut)) + len(format_instructions(AnalystAnswer))
    assert size <= tokens * CHARS_PER_TOKEN
    with pytest.raises(ValueError, match="before any failure"):
        build_analyst_input(**options, max_input_tokens=500)


def test_flags_are_named(tmp_path: Path) -> None:
    literal = result("101", 0.0, speedup=1000.0)
    literal.grade.effisql.literal_result = True
    both = result("102", 0.0, speedup=1000.0, alt_matched=False)
    both.grade.effisql.literal_result = True
    text = build_analyst_input(
        splits=SPLITS,
        ledger=Ledger(tmp_path / "ledger"),
        best_results=[literal, both, result("103", 0.0, speedup=1000.0, alt_matched=True)],
        trajectories=[],
        instances=[],
        queued=[],
    ).user
    assert "101, unknown, GRADED, matched, tau=1e+03, credit=0, flags=literal_result" in text
    assert "flags=hard_coded+literal_result" in text
    assert "103, unknown, GRADED, matched, tau=1e+03, credit=0\n" in text
