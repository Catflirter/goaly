"""Tests for the run and experiment records: trajectory, result, manifest and experiment."""

import json
from collections.abc import Sequence

from goaly.contracts.gen.experiment_pb2 import (
    DecisionThresholds,
    Guard,
    GuardResult,
    Lane,
    PhaseResult,
)
from goaly.contracts.gen.manifest_pb2 import RunManifest
from goaly.contracts.gen.task_pb2 import Prediction
from goaly.contracts.gen.trajectory_pb2 import (
    Chat,
    ChatMessage,
    ChatRole,
    ContextFold,
    Step,
    ToolCall,
    ToolResult,
    Trajectory,
)
from goaly.contracts.jsonl import from_json_line, to_json_line

# Rebuilding the prompt a step's model call was sent, by the rule in Step's comment
# (proto/goaly/v1/trajectory.proto). These helpers are that rule's reference implementation.

type Origin = tuple[int, str] | None  # For a tool message: its step's index and tool_call_id.
type Located = tuple[Origin, ChatMessage]


def tool_message(tool_call_id: str, content: str) -> ChatMessage:
    return ChatMessage(role=ChatRole.CHAT_ROLE_TOOL, tool_call_id=tool_call_id, content=content)


def added_by(step: Step) -> list[Located]:
    """What ``step`` adds to its chat: its assistant message, tool messages and injected ones."""
    added: list[Located] = [(None, step.assistant)]
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


def prompt(trajectory: Trajectory, index: int) -> list[ChatMessage]:
    """The messages the model was sent at step ``index``, folds applied."""
    messages, folds = unfolded(trajectory, trajectory.steps[index])
    replacements: dict[Origin, str] = {
        (fold.step, fold.tool_call_id): fold.replacement for fold in folds
    }
    return [
        tool_message(message.tool_call_id, replacements[origin])
        if origin in replacements
        else message
        for origin, message in messages
    ]


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


def test_an_optional_statistic_keeps_its_presence_through_jsonl() -> None:
    # A sign-test p of exactly 0 was computed; a gain probability never set was not.
    line = to_json_line(PhaseResult(sign_test_p=0.0))

    fields = json.loads(line)
    read = from_json_line(line, PhaseResult)

    assert fields["sign_test_p"] == 0.0
    assert "gain_probability" not in fields
    assert read.HasField("sign_test_p")
    assert not read.HasField("gain_probability")


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
