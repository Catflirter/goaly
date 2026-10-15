"""Tests for goaly.harness.core.trajectory: recording a task's chats and steps in memory."""

from pathlib import Path

import pytest

from goaly.contracts.gen.llm_call_pb2 import LlmCallRecord, Usage
from goaly.contracts.gen.task_pb2 import EffiSqlPrediction, Prediction
from goaly.contracts.gen.trajectory_pb2 import (
    ChatMessage,
    ChatRole,
    ContextFold,
    StopReason,
    ToolCall,
    ToolCallSource,
    ToolResult,
    Trajectory,
)
from goaly.contracts.jsonl import append_jsonl, read_jsonl
from goaly.harness.core.trajectory import TrajectoryRecorder, add_usage

SUBMITTED = StopReason.STOP_REASON_SUBMITTED
TOKENS = StopReason.STOP_REASON_TOKEN_BUDGET_EXHAUSTED
PREDICTION = Prediction(effisql=EffiSqlPrediction(sql="SELECT 1"))


def system(text: str) -> ChatMessage:
    return ChatMessage(role=ChatRole.CHAT_ROLE_SYSTEM, content=text)


def user(text: str) -> ChatMessage:
    return ChatMessage(role=ChatRole.CHAT_ROLE_USER, content=text)


def call_record(call_id: str, usage: Usage, latency_s: float = 1.5) -> LlmCallRecord:
    record = LlmCallRecord(call_id=call_id, usage=usage, latency_s=latency_s)
    record.started_at.FromSeconds(1_760_500_000)
    return record


def assistant(*calls: ToolCall) -> ChatMessage:
    return ChatMessage(role=ChatRole.CHAT_ROLE_ASSISTANT, content="", tool_calls=calls)


def test_add_usage_sums_tokens_and_the_reported_optionals() -> None:
    total = Usage()

    add_usage(total, Usage(input_tokens=10, output_tokens=5))
    assert not total.HasField("reasoning_tokens")
    assert not total.HasField("cost_usd")

    add_usage(total, Usage(input_tokens=20, output_tokens=7, reasoning_tokens=4, cost_usd=0.5))
    add_usage(total, Usage(input_tokens=1, output_tokens=1, reasoning_tokens=0, cost_usd=0.0))

    assert total == Usage(input_tokens=31, output_tokens=13, reasoning_tokens=4, cost_usd=0.5)


def test_a_zero_optional_still_counts_as_reported() -> None:
    total = Usage()

    add_usage(total, Usage(reasoning_tokens=0, cost_usd=0.0))

    assert total.HasField("reasoning_tokens")
    assert total.HasField("cost_usd")


def test_chats_are_recorded_in_the_order_they_start() -> None:
    record = TrajectoryRecorder()

    first = record.start_chat("main", [system("Be fast."), user("Rewrite it.")])
    second = record.start_chat("verify", [])

    trajectory = record.trajectory()
    assert (first, second) == (0, 1)
    assert [chat.label for chat in trajectory.chats] == ["main", "verify"]
    assert list(trajectory.chats[0].initial_messages) == [system("Be fast."), user("Rewrite it.")]
    assert not trajectory.chats[0].HasField("forked_from_step")


def test_a_step_takes_its_call_record_and_its_usage_joins_the_total() -> None:
    record = TrajectoryRecorder()
    record.start_chat("main", [])
    usage = Usage(input_tokens=1200, output_tokens=300, reasoning_tokens=250, cost_usd=0.002)
    message = assistant(ToolCall(id="c1", name="submit", arguments_json="{}"))

    step = record.add_step(0, call_record("llm-1", usage), message)
    step.tool_results.add(tool_call_id="c1", name="submit", content="Submitted.")
    record.add_step(0, call_record("llm-2", usage, latency_s=2.0), assistant())

    trajectory = record.trajectory()
    assert record.step_count == 2
    assert [step.index for step in trajectory.steps] == [0, 1]
    first = trajectory.steps[0]
    assert (first.chat, first.call_id, first.latency_s) == (0, "llm-1", 1.5)
    assert first.started_at.seconds == 1_760_500_000
    assert first.usage == usage
    assert first.assistant == message
    assert [result.content for result in first.tool_results] == ["Submitted."]
    assert trajectory.usage == Usage(
        input_tokens=2400, output_tokens=600, reasoning_tokens=500, cost_usd=0.004
    )
    assert record.usage == trajectory.usage


def test_a_failed_calls_usage_joins_the_total_without_a_step() -> None:
    record = TrajectoryRecorder()

    record.add_usage(Usage(cost_usd=0.0))

    assert record.step_count == 0
    assert record.usage == Usage(cost_usd=0.0)


def test_the_usage_and_the_trajectory_are_copies() -> None:
    record = TrajectoryRecorder()
    record.start_chat("main", [])

    record.usage.input_tokens = 99
    record.trajectory().chats.add(label="extra")

    assert record.usage.input_tokens == 0
    assert len(record.trajectory().chats) == 1


def test_stop_records_why_the_agent_stopped() -> None:
    record = TrajectoryRecorder()
    record.stop(SUBMITTED)
    assert record.trajectory().stop_reason == SUBMITTED
    assert record.trajectory().error_message == ""

    record.stop(TOKENS, error_message="ContextOverflow: the prompt does not fit")  # A later loop.

    trajectory = record.trajectory()
    assert trajectory.stop_reason == TOKENS
    assert trajectory.error_message == "ContextOverflow: the prompt does not fit"


def test_the_recorder_leaves_the_runners_fields_unset() -> None:
    record = TrajectoryRecorder()
    record.start_chat("main", [user("Rewrite it.")])
    record.stop(SUBMITTED)

    trajectory = record.trajectory()

    # The runner fills these in (T06): the identity, the task, the prediction, the grade and
    # the timestamps.
    assert (trajectory.run_id, trajectory.instance_id, trajectory.seed) == ("", "", 0)
    for field in ("task", "prediction", "grade", "started_at", "finished_at"):
        assert not trajectory.HasField(field)


def full_trajectory() -> Trajectory:
    """A trajectory with every field set, as the loop and then the runner fill them."""
    record = TrajectoryRecorder()
    record.start_chat("main", [system("Be fast."), user("Rewrite it.")])
    usage = Usage(input_tokens=1200, output_tokens=300, reasoning_tokens=0, cost_usd=0.0)
    call = ToolCall(
        id="call_abc",
        name="submit",
        arguments_json='{"sql": "SELECT 1"',
        repaired_arguments_json='{"sql": "SELECT 1"}',
        source=ToolCallSource.TOOL_CALL_SOURCE_REASONING_XML,
    )
    step = record.add_step(0, call_record("llm-1", usage), assistant(call))
    step.estimated_prompt_tokens = 1180
    step.folds.append(ContextFold(step=0, tool_call_id="call_abc", replacement="[folded]"))
    step.tool_results.append(
        ToolResult(
            tool_call_id="call_abc",
            name="submit",
            content="Submitted.",
            truncated=True,
            elapsed_s=0.25,
            submission=PREDICTION,
        )
    )
    step.injected.append(user("Two steps left."))
    record.stop(SUBMITTED)
    trajectory = record.trajectory()
    trajectory.run_id, trajectory.instance_id, trajectory.seed = "run-1", "q7", 2
    trajectory.prediction.CopyFrom(PREDICTION)
    trajectory.started_at.FromSeconds(1_760_500_000)
    trajectory.finished_at.FromSeconds(1_760_500_090)
    return trajectory


def test_a_trajectory_survives_a_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "trajectories.jsonl"
    trajectory = full_trajectory()

    append_jsonl(path, trajectory)

    [read] = read_jsonl(path, Trajectory)
    assert read == trajectory
    # The optional zero and the folded step's fields come back as they went.
    assert read.steps[0].usage.HasField("cost_usd")
    assert read.steps[0].folds[0].replacement == "[folded]"


@pytest.mark.parametrize("label", ["", " "])
def test_a_chat_needs_a_label(label: str) -> None:
    with pytest.raises(ValueError, match="label"):
        TrajectoryRecorder().start_chat(label, [])


def test_a_step_needs_a_chat() -> None:
    record = TrajectoryRecorder()
    record.start_chat("main", [])

    with pytest.raises(ValueError, match="no chat 1"):
        record.add_step(1, call_record("llm-1", Usage()), assistant())
