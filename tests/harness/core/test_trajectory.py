"""Tests for goaly.harness.core.trajectory: building a task's trajectory, and writing it."""

from pathlib import Path

import pytest

from goaly.contracts.gen.llm_call_pb2 import LlmCallRecord, Usage
from goaly.contracts.gen.task_pb2 import EffiSqlPrediction, EffiSqlTaskInput, Prediction, TaskInput
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
from goaly.contracts.jsonl import read_jsonl
from goaly.harness.core.trajectory import (
    JsonlTrajectorySink,
    TrajectoryRecorder,
    TrajectorySink,
    add_usage,
)

SUBMITTED = StopReason.STOP_REASON_SUBMITTED
CRASHED = StopReason.STOP_REASON_CRASHED
TASK = TaskInput(effisql=EffiSqlTaskInput(db="tpch", base_sql="SELECT 1"))
OTHER_TASK = TaskInput(effisql=EffiSqlTaskInput(db="tpch", base_sql="SELECT 2"))
PREDICTION = Prediction(effisql=EffiSqlPrediction(sql="SELECT 1"))


class ListSink:
    """A TrajectorySink that keeps what it is given."""

    def __init__(self) -> None:
        self.written: list[Trajectory] = []

    def write(self, trajectory: Trajectory) -> None:
        self.written.append(trajectory)


def recorder(sink: TrajectorySink) -> TrajectoryRecorder:
    return TrajectoryRecorder(sink, run_id="run-1", instance_id="q7", seed=2)


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


def test_the_first_chat_records_the_task_and_starts_the_clock() -> None:
    record = recorder(ListSink())

    first = record.start_chat(TASK, "main", [system("Be fast."), user("Rewrite it.")])
    second = record.start_chat(OTHER_TASK, "verify", [])

    trajectory = record.snapshot()
    assert (first, second) == (0, 1)
    assert trajectory.task == TASK  # The first chat's, which every chat of the task shares.
    assert trajectory.HasField("started_at")
    assert [chat.label for chat in trajectory.chats] == ["main", "verify"]
    assert list(trajectory.chats[0].initial_messages) == [system("Be fast."), user("Rewrite it.")]
    assert not trajectory.chats[0].HasField("forked_from_step")


def test_a_step_takes_its_call_record_and_its_usage_joins_the_total() -> None:
    record = recorder(ListSink())
    record.start_chat(TASK, "main", [])
    usage = Usage(input_tokens=1200, output_tokens=300, reasoning_tokens=250, cost_usd=0.002)
    message = assistant(ToolCall(id="c1", name="submit", arguments_json="{}"))

    step = record.add_step(0, call_record("llm-1", usage), message)
    step.tool_results.add(tool_call_id="c1", name="submit", content="Submitted.")
    record.add_step(0, call_record("llm-2", usage, latency_s=2.0), assistant())

    trajectory = record.snapshot()
    assert record.step_count == 2
    assert [step.index for step in trajectory.steps] == [0, 1]
    first = trajectory.steps[0]
    assert (first.chat, first.call_id, first.latency_s) == (0, "llm-1", 1.5)
    assert first.started_at.seconds == 1_760_500_000
    assert first.usage == usage
    assert first.assistant == message
    assert [result.content for result in first.tool_results] == ["Submitted."]
    assert record.usage == Usage(
        input_tokens=2400, output_tokens=600, reasoning_tokens=500, cost_usd=0.004
    )


def test_a_failed_calls_usage_joins_the_total_without_a_step() -> None:
    record = recorder(ListSink())

    record.add_usage(Usage(cost_usd=0.0))

    assert record.step_count == 0
    assert record.usage == Usage(cost_usd=0.0)


def test_the_usage_is_a_copy() -> None:
    record = recorder(ListSink())

    record.usage.input_tokens = 99

    assert record.usage.input_tokens == 0


def test_finish_writes_a_copy_with_the_prediction_and_why_it_stopped() -> None:
    sink = ListSink()
    record = recorder(sink)
    record.start_chat(TASK, "main", [user("Rewrite it.")])

    record.finish(SUBMITTED, prediction=PREDICTION)
    record.add_step(0, call_record("llm-1", Usage()), assistant())  # After the write.

    [written] = sink.written
    assert (written.run_id, written.instance_id, written.seed) == ("run-1", "q7", 2)
    assert written.prediction == PREDICTION
    assert written.stop_reason == SUBMITTED
    assert written.error_message == ""
    assert written.HasField("finished_at")
    assert not written.steps  # The sink's copy is its own.


def test_finishing_again_writes_the_whole_trajectory_again() -> None:
    sink = ListSink()
    record = recorder(sink)
    record.start_chat(TASK, "main", [])
    record.finish(SUBMITTED, prediction=PREDICTION)
    record.start_chat(TASK, "verify", [])

    record.finish(CRASHED, error_message="EnvironmentFailure: the database died")

    first, last = sink.written
    assert len(first.chats) == 1
    assert len(last.chats) == 2  # The last line for the task holds everything.
    assert not last.HasField("prediction")  # The policy crashed, so it returned nothing.
    assert last.stop_reason == CRASHED
    assert last.error_message == "EnvironmentFailure: the database died"


def full_trajectory() -> Trajectory:
    """A trajectory with every field set, as the loop and the runner fill them."""
    sink = ListSink()
    record = recorder(sink)
    record.start_chat(TASK, "main", [system("Be fast."), user("Rewrite it.")])
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
    record.finish(SUBMITTED, prediction=PREDICTION, error_message="")
    [trajectory] = sink.written
    return trajectory


def test_a_trajectory_survives_a_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "trajectories.jsonl"
    sink: TrajectorySink = JsonlTrajectorySink(path)  # mypy checks that it is a TrajectorySink.
    trajectory = full_trajectory()

    sink.write(trajectory)
    sink.write(Trajectory(run_id="run-1", instance_id="q8"))

    first, second = read_jsonl(path, Trajectory)
    assert first == trajectory
    assert second.instance_id == "q8"
    # The optional zero and the folded step's fields come back as they went.
    assert first.steps[0].usage.HasField("cost_usd")
    assert first.steps[0].folds[0].replacement == "[folded]"


def test_the_jsonl_sink_creates_its_file(tmp_path: Path) -> None:
    path = tmp_path / "trajectories.jsonl"

    JsonlTrajectorySink(path).write(full_trajectory())

    assert path.read_text(encoding="utf-8").count("\n") == 1


@pytest.mark.parametrize("label", ["", " "])
def test_a_chat_needs_a_label(label: str) -> None:
    with pytest.raises(ValueError, match="label"):
        recorder(ListSink()).start_chat(TASK, label, [])


def test_a_step_needs_a_chat() -> None:
    record = recorder(ListSink())
    record.start_chat(TASK, "main", [])

    with pytest.raises(ValueError, match="no chat 1"):
        record.add_step(1, call_record("llm-1", Usage()), assistant())
