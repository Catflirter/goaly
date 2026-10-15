"""Tests for the run and experiment records: trajectory, result, manifest and experiment."""

import json
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from google.protobuf.message import Message
from google.protobuf.timestamp_pb2 import Timestamp

from goaly.contracts.gen.experiment_pb2 import (
    Decision,
    DecisionThresholds,
    ExperimentRecord,
    Guard,
    GuardResult,
    ImplementerExit,
    ImplementerRun,
    Lane,
    LaneEvent,
    LaneEventKind,
    Lesson,
    PhaseResult,
    RejectionReason,
    SmokeTest,
)
from goaly.contracts.gen.manifest_pb2 import RunManifest
from goaly.contracts.gen.task_pb2 import Prediction, Split
from goaly.contracts.gen.trajectory_pb2 import (
    Chat,
    ChatMessage,
    ChatRole,
    ContextFold,
    Step,
    ToolCall,
    ToolCallSource,
    ToolResult,
    Trajectory,
)
from goaly.contracts.jsonl import append_jsonl, from_json_line, read_jsonl, to_json_line

# Rebuilding the prompt a step's model call was sent, by the rule in Step's comment
# (proto/goaly/v1/trajectory.proto). These helpers are that rule's reference implementation.

# For a tool message: its step's index and tool_call_id. For an assistant message: its step's
# index. None for any other message.
type Origin = tuple[int, str] | int | None
type Located = tuple[Origin, ChatMessage]

XML_SOURCES = {
    ToolCallSource.TOOL_CALL_SOURCE_REASONING_XML,
    ToolCallSource.TOOL_CALL_SOURCE_CONTENT_XML,
}
# Qwen3-Coder's call markup: a block ends at its closing tag, at the next block, or at the end.
TOOL_CALL_BLOCK = re.compile(r"<tool_call>.*?(?:</tool_call>|(?=<tool_call>)|\Z)", re.DOTALL)
FUNCTION_BLOCK = re.compile(r"<function=[^>\n]*>.*?(?:</function>|(?=<function=)|\Z)", re.DOTALL)


def tool_message(tool_call_id: str, content: str) -> ChatMessage:
    return ChatMessage(role=ChatRole.CHAT_ROLE_TOOL, tool_call_id=tool_call_id, content=content)


def added_by(step: Step) -> list[Located]:
    """What ``step`` adds to its chat: its assistant message, tool messages and injected ones."""
    added: list[Located] = [(step.index, step.assistant)]
    added += [
        ((step.index, result.tool_call_id), tool_message(result.tool_call_id, result.content))
        for result in step.tool_results
    ]
    added += [(None, message) for message in step.injected]
    return added


def unfolded(trajectory: Trajectory, step: Step) -> tuple[list[Located], list[ContextFold]]:
    """The messages of ``step``'s prompt before folding, and the folds in effect at it."""
    chat = trajectory.chats[step.chat]
    messages: list[Located] = []
    folds: list[ContextFold] = []
    if chat.HasField("forked_from_step"):
        forked_from = trajectory.steps[chat.forked_from_step]
        messages, folds = unfolded(trajectory, forked_from)
        messages += added_by(forked_from)
    messages += [(None, message) for message in chat.initial_messages]
    for earlier in trajectory.steps[: step.index]:
        if earlier.chat == step.chat:
            messages += added_by(earlier)
            folds += earlier.folds
    return messages, [*folds, *step.folds]


def without_markup(text: str, *, reasoning: bool) -> str:
    """``text`` without its <tool_call> blocks, or, in a content with none, its <function> ones."""
    markup = TOOL_CALL_BLOCK if reasoning or TOOL_CALL_BLOCK.search(text) else FUNCTION_BLOCK
    return markup.sub("", text)


def replayed(assistant: ChatMessage, *, keep_reasoning: bool) -> ChatMessage:
    """An earlier assistant message as a later prompt carries it."""
    message = ChatMessage()
    message.CopyFrom(assistant)
    if any(call.source in XML_SOURCES for call in assistant.tool_calls):
        message.content = without_markup(message.content, reasoning=False)
        message.reasoning_content = without_markup(message.reasoning_content, reasoning=True)
    if not keep_reasoning:
        message.reasoning_content = ""
    return message


def prompt(trajectory: Trajectory, index: int) -> list[ChatMessage]:
    """The messages the model was sent at step ``index``: folds, XML and reasoning rules applied."""
    step = trajectory.steps[index]
    messages, folds = unfolded(trajectory, step)
    replacements: dict[Origin, str] = {
        (fold.step, fold.tool_call_id): fold.replacement for fold in folds
    }
    chat = trajectory.chats[step.chat]
    assistants = [origin for origin, _ in messages if isinstance(origin, int)]
    if chat.HasField("keep_turns"):  # An executor chat: only the latest steps keep reasoning.
        assistants = assistants[max(len(assistants) - chat.keep_turns, 0) :]
    sent = []
    for origin, message in messages:
        if origin in replacements:
            sent.append(tool_message(message.tool_call_id, replacements[origin]))
        elif isinstance(origin, int):
            sent.append(replayed(message, keep_reasoning=origin in assistants))
        else:
            sent.append(message)
    return sent


SYSTEM = ChatMessage(role=ChatRole.CHAT_ROLE_SYSTEM, content="You make slow queries faster.")
TASK = ChatMessage(role=ChatRole.CHAT_ROLE_USER, content="Speed up: SELECT count(*) FROM lineitem")


def make_step(
    index: int,
    chat: int,
    *calls: str,
    injected: Sequence[ChatMessage] = (),
    folds: Sequence[ContextFold] = (),
) -> Step:
    """Step ``index``, in chat ``chat``, whose model calls a tool once per ID in ``calls``."""
    return Step(
        index=index,
        chat=chat,
        call_id=f"call-{index}",
        assistant=ChatMessage(
            role=ChatRole.CHAT_ROLE_ASSISTANT,
            content=f"answer {index}",
            reasoning_content=f"thinking {index}",
            tool_calls=[
                ToolCall(id=call, name="execute_sql", arguments_json="{}") for call in calls
            ],
        ),
        tool_results=[
            ToolResult(tool_call_id=call, name="execute_sql", content=f"rows for {call}")
            for call in calls
        ],
        injected=injected,
        folds=folds,
    )


def tool(call: str) -> ChatMessage:
    """The unfolded tool message that answers ``call`` in a step built by make_step."""
    return tool_message(call, f"rows for {call}")


def test_a_step_sees_only_its_own_chat() -> None:
    verify = ChatMessage(role=ChatRole.CHAT_ROLE_USER, content="Check this answer: answer 0")
    hint = ChatMessage(role=ChatRole.CHAT_ROLE_USER, content="Inspect the schema first.")
    trajectory = Trajectory(
        chats=[
            Chat(label="main", initial_messages=[SYSTEM, TASK]),
            Chat(label="verify", initial_messages=[verify]),
        ],
        steps=[
            make_step(0, 0, "t1", injected=[hint]),
            make_step(1, 1, "v1"),
            make_step(2, 0, "t2"),
            make_step(3, 1),
        ],
    )
    steps = trajectory.steps

    assert prompt(trajectory, 0) == [SYSTEM, TASK]
    assert prompt(trajectory, 1) == [verify]
    assert prompt(trajectory, 2) == [SYSTEM, TASK, steps[0].assistant, tool("t1"), hint]
    assert prompt(trajectory, 3) == [verify, steps[1].assistant, tool("v1")]


def test_chats_forked_from_a_step_continue_its_context() -> None:
    # Self-consistency: three samples drawn from step 1's context, in which t1 was folded.
    nudge = ChatMessage(role=ChatRole.CHAT_ROLE_USER, content="Try another approach.")
    fold_t1 = ContextFold(step=0, tool_call_id="t1", replacement="3 rows")
    fold_t2 = ContextFold(step=1, tool_call_id="t2", replacement="9 rows")
    trajectory = Trajectory(
        chats=[
            Chat(label="main", initial_messages=[SYSTEM, TASK]),
            Chat(label="sample-1", forked_from_step=1),
            Chat(label="sample-2", forked_from_step=1),
            Chat(label="sample-3", forked_from_step=1, initial_messages=[nudge]),
        ],
        steps=[
            make_step(0, 0, "t1"),
            make_step(1, 0, "t2", folds=[fold_t1]),
            make_step(2, 1, "s1"),
            make_step(3, 2),
            make_step(4, 3),
            make_step(5, 1, folds=[fold_t2]),
            make_step(6, 0),
        ],
    )
    steps = trajectory.steps
    folded_t1 = tool_message("t1", "3 rows")
    at_fork = [SYSTEM, TASK, steps[0].assistant, folded_t1, steps[1].assistant, tool("t2")]

    assert prompt(trajectory, 1) == [SYSTEM, TASK, steps[0].assistant, folded_t1]
    assert prompt(trajectory, 2) == at_fork
    assert prompt(trajectory, 3) == at_fork
    assert prompt(trajectory, 4) == [*at_fork, nudge]
    # sample-1's own fold applies to sample-1 alone.
    assert prompt(trajectory, 5) == [
        *at_fork[:5],
        tool_message("t2", "9 rows"),
        steps[2].assistant,
        tool("s1"),
    ]
    assert prompt(trajectory, 6) == at_fork  # The main chat sees none of the samples.


def without_reasoning(message: ChatMessage) -> ChatMessage:
    """``message`` as a prompt carries it once its step is out of the reasoning window."""
    sent = ChatMessage()
    sent.CopyFrom(message)
    sent.reasoning_content = ""
    return sent


def test_an_executor_chat_replays_reasoning_for_its_last_keep_turns_steps_only() -> None:
    trajectory = Trajectory(
        chats=[Chat(label="main", initial_messages=[SYSTEM, TASK], keep_turns=2)],
        steps=[
            make_step(0, 0, "t0"),
            make_step(1, 0, "t1"),
            make_step(2, 0, "t2"),
            make_step(3, 0),
        ],
    )
    steps = trajectory.steps

    assert prompt(trajectory, 2) == [
        *(SYSTEM, TASK, steps[0].assistant, tool("t0"), steps[1].assistant, tool("t1"))
    ]
    assert prompt(trajectory, 3) == [
        *(SYSTEM, TASK, without_reasoning(steps[0].assistant), tool("t0")),
        *(steps[1].assistant, tool("t1"), steps[2].assistant, tool("t2")),
    ]
    # The trajectory keeps every message in full.
    assert steps[0].assistant.reasoning_content == "thinking 0"


def test_a_chat_without_keep_turns_replays_reasoning_unchanged() -> None:
    # As GLM's reasoning is (spec §5.1).
    trajectory = Trajectory(
        chats=[Chat(label="main", initial_messages=[SYSTEM, TASK])],
        steps=[make_step(0, 0, "t0"), make_step(1, 0, "t1"), make_step(2, 0)],
    )
    steps = trajectory.steps

    assert prompt(trajectory, 2) == [
        *(SYSTEM, TASK, steps[0].assistant, tool("t0"), steps[1].assistant, tool("t1"))
    ]


def test_a_forked_chat_counts_its_reasoning_window_through_the_forked_context() -> None:
    trajectory = Trajectory(
        chats=[
            Chat(label="main", initial_messages=[SYSTEM, TASK], keep_turns=2),
            Chat(label="sample-1", forked_from_step=1, keep_turns=2),
        ],
        steps=[
            make_step(0, 0, "t0"),
            make_step(1, 0, "t1"),
            make_step(2, 1, "s2"),
            make_step(3, 1),
        ],
    )
    steps = trajectory.steps

    # At step 2 the window holds steps 0 and 1, both from the forked context.
    assert prompt(trajectory, 2) == [
        *(SYSTEM, TASK, steps[0].assistant, tool("t0"), steps[1].assistant, tool("t1"))
    ]
    # At step 3 it holds step 1, from the forked context, and step 2, the chat's own.
    assert prompt(trajectory, 3) == [
        *(SYSTEM, TASK, without_reasoning(steps[0].assistant), tool("t0")),
        *(steps[1].assistant, tool("t1"), steps[2].assistant, tool("s2")),
    ]


def xml_step(index: int, source: ToolCallSource.ValueType, content: str, reasoning: str) -> Step:
    """Step ``index`` of chat 0, whose one call ``t{index}`` was parsed from XML."""
    step = make_step(index, 0, f"t{index}")
    step.assistant.content = content
    step.assistant.reasoning_content = reasoning
    step.assistant.tool_calls[0].source = source
    return step


def test_a_call_parsed_from_xml_is_replayed_only_as_a_structured_call() -> None:
    call = (
        "<tool_call><function=execute_sql><parameter=sql>SELECT 1</parameter></function>"
        "</tool_call>"
    )
    bare = "<function=execute_sql><parameter=sql>SELECT 2</parameter></function>"
    content_xml = ToolCallSource.TOOL_CALL_SOURCE_CONTENT_XML
    reasoning_xml = ToolCallSource.TOOL_CALL_SOURCE_REASONING_XML
    standard = make_step(3, 0, "t3")
    standard.assistant.content = call  # Its call came from the standard field.
    trajectory = Trajectory(
        chats=[Chat(label="main", initial_messages=[SYSTEM, TASK], keep_turns=8)],
        steps=[
            xml_step(0, content_xml, f"Run it. {call}", f"Plan. {call}"),
            xml_step(1, content_xml, f"Again. {bare}", ""),
            # Cut off: the block runs to the end of the text. A bare block in reasoning stays.
            xml_step(2, reasoning_xml, "", f"{bare} <tool_call>x"),
            standard,
            make_step(4, 0),
        ],
    )

    sent = [
        message for message in prompt(trajectory, 4) if message.role == ChatRole.CHAT_ROLE_ASSISTANT
    ]

    assert [(message.content, message.reasoning_content) for message in sent] == [
        ("Run it. ", "Plan. "),
        ("Again. ", ""),
        ("", f"{bare} "),
        (call, "thinking 3"),
    ]
    assert [len(message.tool_calls) for message in sent] == [1, 1, 1, 1]


def test_every_lane_has_its_own_thresholds() -> None:
    lanes = {value.name.removeprefix("LANE_").lower() for value in Lane.DESCRIPTOR.values}
    variants = DecisionThresholds.DESCRIPTOR.oneofs_by_name["benchmark"].fields

    # A lane is decided by its own variant's thresholds, so each lane needs exactly one.
    assert {field.name for field in variants} == lanes - {"unspecified"} == {"effisql", "swe"}


def test_a_tool_result_says_whether_a_tool_submitted() -> None:
    # An empty prediction is still a submission: for SWE, an empty patch means "no change".
    submitted = ToolResult(tool_call_id="t1", name="submit", submission=Prediction())
    plain = ToolResult(tool_call_id="t2", name="execute_sql", content="3 rows")

    assert from_json_line(to_json_line(submitted), ToolResult).HasField("submission")
    assert "submission" not in json.loads(to_json_line(plain))
    assert not from_json_line(to_json_line(plain), ToolResult).HasField("submission")


def test_a_manifest_records_the_environment_settings_that_can_change_results() -> None:
    settings = {
        "postgres_image": "postgres@sha256:" + "ab" * 32,
        "shared_buffers": "2GB",
        "jit": "off",
        "max_parallel_workers_per_gather": "2",
        "cpuset.grade-1": "6,7,18,19",
    }
    manifest = RunManifest(run_id="run-1", environment_settings=settings)

    line = to_json_line(manifest)

    assert json.loads(line)["environment_settings"] == settings
    assert from_json_line(line, RunManifest) == manifest
    assert json.loads(to_json_line(RunManifest()))["environment_settings"] == {}


# The RSI ledger: experiments, lane events and lessons (spec §8.2, §8.5-§8.8).


def test_a_guard_result_names_the_pack_code_allowlist() -> None:
    result = GuardResult(
        guard=Guard.GUARD_PACK_CODE_ALLOWLIST, passed=False, detail="policy.py:3: import os"
    )

    line = to_json_line(result)

    assert json.loads(line)["guard"] == "GUARD_PACK_CODE_ALLOWLIST"
    assert from_json_line(line, GuardResult) == result


def test_a_phase_lists_the_tasks_a_hard_coding_check_flagged() -> None:
    # One flagged prediction fails the candidate outright (spec §8.2 step 4).
    screening = PhaseResult(
        split=Split.SPLIT_TRAIN, task_count=20, flagged_instance_ids=["17", "203"]
    )
    record = ExperimentRecord(
        experiment_id="exp-3",
        screening=screening,
        decision=Decision.DECISION_REJECTED,
        rejection_reasons=[RejectionReason.REJECTION_REASON_HARD_CODED],
    )

    line = to_json_line(record)

    fields = json.loads(line)
    assert fields["screening"]["flagged_instance_ids"] == ["17", "203"]
    assert fields["rejection_reasons"] == ["REJECTION_REASON_HARD_CODED"]
    assert from_json_line(line, ExperimentRecord) == record


def set_fields(*messages: Message) -> set[str]:
    """The names of the fields that hold something other than their default in ``messages``."""
    return {field.name for message in messages for field, _ in message.ListFields()}


def test_an_experiment_records_its_implementer_run_and_smoke_tests() -> None:
    run = ImplementerRun(
        exit=ImplementerExit.IMPLEMENTER_EXIT_SUBMITTED,
        step_count=23,
        wall_time_s=812.5,
        trajectory_path="rsi/effisql/exp-3/implementer.traj.json",
        smoke_tests=[
            SmokeTest(
                instance_id="17",
                run_id="smoke-exp-3-17",
                score=0.75,
                detail="credit 0.75; stop reason: submitted; 9 steps; $0.02; no tool error",
            ),
            SmokeTest(instance_id="250", detail="refused: 250 is not in train"),
        ],
        summary="Added a skill that turns correlated subqueries into joins; unit tests pass.",
    )
    record = ExperimentRecord(experiment_id="exp-3", lane=Lane.LANE_EFFISQL, implementer=run)

    read = from_json_line(to_json_line(record), ExperimentRecord)

    assert read.implementer == run
    assert set_fields(run) == set(ImplementerRun.DESCRIPTOR.fields_by_name)
    assert set_fields(run.smoke_tests[0]) == set(SmokeTest.DESCRIPTOR.fields_by_name)


def test_an_implementer_that_never_started_leaves_no_run() -> None:
    line = to_json_line(ExperimentRecord(experiment_id="exp-3"))

    assert "implementer" not in json.loads(line)
    assert not from_json_line(line, ExperimentRecord).HasField("implementer")


def test_an_implementer_run_that_was_never_measured_has_no_step_count_or_wall_time() -> None:
    # An INTERRUPTED run found after a crash: its steps and time are unknown, not 0.
    unmeasured = to_json_line(ImplementerRun(exit=ImplementerExit.IMPLEMENTER_EXIT_INTERRUPTED))
    measured = to_json_line(ImplementerRun(step_count=0, wall_time_s=0.0))

    assert {"step_count", "wall_time_s"}.isdisjoint(json.loads(unmeasured))
    assert (json.loads(measured)["step_count"], json.loads(measured)["wall_time_s"]) == (0, 0.0)
    read = from_json_line(unmeasured, ImplementerRun)
    assert not read.HasField("step_count")
    assert not read.HasField("wall_time_s")


def test_a_smoke_test_that_never_ran_the_candidate_has_no_score() -> None:
    # The host refused the call, or a guard failed: no run, so no score. A score of 0 was earned.
    refused = to_json_line(SmokeTest(instance_id="250", detail="refused: 250 is not in train"))
    scored_zero = to_json_line(SmokeTest(instance_id="17", run_id="smoke-exp-3-17", score=0.0))

    assert "score" not in json.loads(refused)
    assert json.loads(scored_zero)["score"] == 0.0
    assert not from_json_line(refused, SmokeTest).HasField("score")
    assert from_json_line(scored_zero, SmokeTest).HasField("score")


def at(hour: int, minute: int) -> Timestamp:
    """A point in time on 2026-10-15, in UTC."""
    timestamp = Timestamp()
    timestamp.FromDatetime(datetime(2026, 10, 15, hour, minute, tzinfo=UTC))
    return timestamp


def test_lane_events_are_appended_to_lane_events_jsonl_and_read_back(tmp_path: Path) -> None:
    lane, kind = Lane.LANE_EFFISQL, LaneEventKind
    pr_opened = "effisql: results PR #12 is open"
    events = [
        LaneEvent(
            lane=lane,
            occurred_at=at(16, 0),
            kind=kind.LANE_EVENT_KIND_STARTED,
            deadline_at=at(23, 59),  # Absolute, from config: a restart never moves it.
        ),
        LaneEvent(
            lane=lane,
            occurred_at=at(17, 31),
            kind=kind.LANE_EVENT_KIND_CRASHED,  # The supervisor writes the cause here.
            detail="killed after 10 minutes without a heartbeat",
        ),
        LaneEvent(
            lane=lane,
            occurred_at=at(17, 32),
            kind=kind.LANE_EVENT_KIND_RESTARTED,  # Only the backoff: the cause is the crash's.
            detail="restarted after a 30 s backoff",
        ),
        LaneEvent(
            lane=lane,
            occurred_at=at(19, 0),
            kind=kind.LANE_EVENT_KIND_RESULTS_PR,
            detail="3 accepted changes since the last results PR",
            pr_number=12,
            pr_url="https://github.com/Catflirter/goaly/pull/12",
            head_commit="4f2c0de1b6a8e3f9c7d5b1a2e4f6c8d0a9b7e5f3",
            experiment_ids=["exp-3", "exp-7", "exp-8"],
        ),
        LaneEvent(
            lane=lane,
            occurred_at=at(19, 1),
            kind=kind.LANE_EVENT_KIND_NOTIFY_FAILED,
            detail="HTTP 502 from ntfy.sh",
            notification=pr_opened,
            notification_id="notify-12",
        ),
        LaneEvent(
            lane=lane,
            occurred_at=at(19, 2),
            kind=kind.LANE_EVENT_KIND_NOTIFIED,
            notification=pr_opened,
            notification_id="notify-12",  # The same notification, delivered on its retry.
        ),
        LaneEvent(
            lane=lane,
            occurred_at=at(23, 40),
            kind=kind.LANE_EVENT_KIND_STOPPED,
            detail="4 rounds in a row with no accepted change",
        ),
    ]
    path = tmp_path / "lane_events.jsonl"

    for event in events:
        append_jsonl(path, event)

    assert path.read_bytes().count(b"\n") == len(events)
    assert list(read_jsonl(path, LaneEvent)) == events
    assert set_fields(*events) == set(LaneEvent.DESCRIPTOR.fields_by_name)


def test_a_deleted_lesson_keeps_its_earlier_lines_and_its_last_line_says_when(
    tmp_path: Path,
) -> None:
    standing = Lesson(
        lesson_id="lesson-4",
        lane=Lane.LANE_EFFISQL,
        text="Rewriting IN subqueries as joins helped on 3 tasks (exp-3).",
        experiment_ids=["exp-3"],
        created_at=at(18, 0),
        updated_at=at(18, 0),
    )
    deleted = Lesson()
    deleted.CopyFrom(standing)
    deleted.updated_at.CopyFrom(at(21, 15))
    deleted.deleted_at.CopyFrom(at(21, 15))
    path = tmp_path / "lessons.jsonl"

    append_jsonl(path, standing)
    append_jsonl(path, deleted)

    assert "deleted_at" not in json.loads(to_json_line(standing))
    assert list(read_jsonl(path, Lesson)) == [standing, deleted]
