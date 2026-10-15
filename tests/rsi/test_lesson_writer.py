"""Tests for goaly.rsi.lesson_writer: the lesson's one structured call, val-clean."""

import json

import pytest

from goaly.contracts.gen.experiment_pb2 import (
    DECISION_ABORTED,
    DECISION_PENDING,
    DECISION_REJECTED,
    GUARD_ALLOWED_PATHS,
    GUARD_UNIT_TESTS,
    IMPLEMENTER_EXIT_SUBMITTED,
    LANE_EFFISQL,
    REJECTION_REASON_HARD_CODED,
    ExperimentRecord,
    GuardResult,
    ImplementerRun,
    PhaseResult,
)
from goaly.contracts.gen.llm_call_pb2 import CallPurpose
from goaly.rsi import lesson_writer, structured
from goaly.rsi.analyst import ANALYST_ATTEMPT_TIMEOUT_S
from goaly.rsi.lesson_writer import (
    LESSON_ATTEMPT_TIMEOUT_S,
    LESSON_MAX_TOKENS,
    lesson_prompt,
    write_lesson,
)
from tests.rsi.fake_llm import FakeGlm
from tests.rsi.round_world import hypothesis

pytestmark = pytest.mark.anyio

VAL_IDS = ["777", "888"]


def flagged_record() -> ExperimentRecord:
    return ExperimentRecord(
        experiment_id="effisql-r2-1-abc123",
        lane=LANE_EFFISQL,
        round=2,
        hypothesis=hypothesis("flag-1"),
        implementer=ImplementerRun(exit=IMPLEMENTER_EXIT_SUBMITTED, summary="Added a skill."),
        guards=[
            GuardResult(guard=GUARD_ALLOWED_PATHS, passed=True),
            GuardResult(guard=GUARD_UNIT_TESTS, passed=True),
        ],
        screening=PhaseResult(task_count=20, seed_count=1, delta=0.1, wins=3),
        validation=PhaseResult(
            task_count=30, seed_count=1, delta=0.2, wins=8, flagged_instance_ids=VAL_IDS
        ),
        decision=DECISION_REJECTED,
        rejection_reasons=[REJECTION_REASON_HARD_CODED],
        decision_reason="validation: ΔR-VES +20.0 points; 2 flagged tasks; failed HARD_CODED",
    )


def test_the_prompt_gives_aggregates_and_counts_flags_without_naming_val_tasks() -> None:
    system, user = lesson_prompt(flagged_record())
    text = user.content
    assert "Experiment effisql-r2-1-abc123" in text
    assert "Tell the agent to try flag-1." in text
    assert "IMPLEMENTER_EXIT_SUBMITTED" in text
    assert "Added a skill." in text
    assert "Validation (val): 30 tasks x 1 seeds" in text
    assert "2 predictions flagged as hard-coded results" in text
    assert "REJECTION_REASON_HARD_CODED" in text
    assert "Confirmation" not in text
    for val_id in VAL_IDS:
        assert val_id not in text
        assert val_id not in system.content


def test_a_failed_guard_is_shown_with_its_detail() -> None:
    record = flagged_record()
    del record.guards[1:]
    record.guards.add(guard=GUARD_UNIT_TESTS, passed=False, detail="2 tests failed")
    text = lesson_prompt(record)[1].content
    assert "Guard GUARD_UNIT_TESTS: FAILED." in text
    assert "2 tests failed" in text


@pytest.mark.parametrize("decision", [DECISION_PENDING, DECISION_ABORTED])
def test_only_a_decided_experiment_gets_a_lesson(decision: int) -> None:
    record = flagged_record()
    record.decision = decision  # type: ignore[assignment]  # A Decision value.
    with pytest.raises(ValueError, match="not accepted or rejected"):
        lesson_prompt(record)


async def test_write_lesson_makes_one_rsi_lesson_call() -> None:
    gateway = FakeGlm(
        json.dumps({"lesson": "Flagged: embedding result rows fails the robustness check."})
    )
    async with gateway.client() as client:
        reply = await write_lesson(client, flagged_record())
    assert reply.value.lesson.startswith("Flagged")
    (record,) = gateway.recorder.records
    assert record.purpose == CallPurpose.CALL_PURPOSE_RSI_LESSON
    assert record.run_id == "effisql-r2-1-abc123"
    assert gateway.bodies[0]["max_tokens"] == LESSON_MAX_TOKENS == 24000


async def test_each_lesson_attempt_may_take_as_long_as_the_analysts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}
    original = structured.complete_structured

    async def spy(*args: object, **kwargs: object) -> object:
        seen.update(kwargs)
        return await original(*args, **kwargs)  # type: ignore[arg-type]  # Passed through.

    monkeypatch.setattr(lesson_writer, "complete_structured", spy)
    gateway = FakeGlm(json.dumps({"lesson": "Flagged: embedding result rows fails the check."}))
    async with gateway.client() as client:
        await write_lesson(client, flagged_record())
    assert seen["attempt_timeout_s"] == LESSON_ATTEMPT_TIMEOUT_S == ANALYST_ATTEMPT_TIMEOUT_S
