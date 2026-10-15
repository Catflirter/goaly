"""Tests for goaly.harness.core.context_window: estimating a chat's prompt, and folding it."""

import json
import math
from typing import Any

import pytest
from pydantic import ValidationError

from goaly.contracts.gen.trajectory_pb2 import ChatMessage, ChatRole, ContextFold, ToolCall
from goaly.contracts.interfaces import JsonValue
from goaly.harness.core.context_window import ChatPrompt, ContextWindow

SYSTEM, USER = ChatRole.CHAT_ROLE_SYSTEM, ChatRole.CHAT_ROLE_USER
ASSISTANT, TOOL = ChatRole.CHAT_ROLE_ASSISTANT, ChatRole.CHAT_ROLE_TOOL
OPENING = [
    ChatMessage(role=SYSTEM, content="You rewrite slow queries."),
    ChatMessage(role=USER, content="Rewrite: SELECT 1"),
]
TOOLS: list[dict[str, JsonValue]] = [
    {"type": "function", "function": {"name": "execute_sql", "description": "Run SQL."}}
]


def window(**changes: Any) -> ContextWindow:
    return ContextWindow.model_validate({"fold_above_tokens": 1000, **changes})


def assistant(call_id: str, arguments: str = '{"sql": "SELECT 1"}') -> ChatMessage:
    call = ToolCall(id=call_id, name="execute_sql", arguments_json=arguments)
    return ChatMessage(role=ASSISTANT, content="Let me look.", tool_calls=[call])


def tool(call_id: str, content: str) -> ChatMessage:
    return ChatMessage(role=TOOL, content=content, tool_call_id=call_id)


def chat(outputs: list[str], **changes: Any) -> ChatPrompt:
    """A chat whose step i made one call, c<i>, whose tool output is outputs[i]."""
    prompt = ChatPrompt(OPENING, TOOLS, window(**changes))
    for step, output in enumerate(outputs):
        prompt.add_turn(step, assistant(f"c{step}"))
        prompt.add(tool(f"c{step}", output))
    return prompt


def contents(prompt: ChatPrompt) -> list[str]:
    return [message.content for message in prompt.messages() if message.role == TOOL]


def test_the_defaults_are_the_specs() -> None:
    defaults = ContextWindow()

    assert (defaults.fold_above_tokens, defaults.keep_turns, defaults.folded_chars) == (
        96_000,  # Of Qwen's 128K (spec §5.1).
        8,
        500,
    )


@pytest.mark.parametrize(
    "config",
    [
        {"fold_above_tokens": 0},
        {"keep_turns": -1},
        {"folded_chars": 0},
        {"fold_above": 96_000},  # Unknown fields are typos.
    ],
)
def test_invalid_windows_are_rejected(config: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ContextWindow.model_validate(config)


def test_the_prompt_is_the_opening_then_each_turn_in_order() -> None:
    prompt = ChatPrompt(OPENING, TOOLS, window())
    prompt.add_turn(0, assistant("c0"))
    prompt.add(tool("c0", "rows: 1"))
    prompt.add(ChatMessage(role=USER, content="Two steps left."))

    assert prompt.messages() == [
        *OPENING,
        assistant("c0"),
        tool("c0", "rows: 1"),
        ChatMessage(role=USER, content="Two steps left."),
    ]


def test_before_any_call_the_estimate_counts_the_opening_and_the_tools() -> None:
    prompt = ChatPrompt(OPENING, TOOLS, window())

    characters = sum(len(message.content) for message in OPENING) + len(json.dumps(TOOLS))
    assert prompt.estimated_tokens == math.ceil(characters / 4)


def test_after_a_call_the_estimate_is_its_input_tokens_plus_what_came_since() -> None:
    prompt = ChatPrompt(OPENING, TOOLS, window())
    prompt.sent(700)
    reply = assistant("c0", arguments="{}")
    prompt.add_turn(0, reply)
    prompt.add(tool("c0", "x" * 40))

    added = len("Let me look.") + len("execute_sql") + len("{}") + 40
    assert prompt.estimated_tokens == 700 + math.ceil(added / 4)


def test_the_reasoning_counts_toward_the_estimate() -> None:
    prompt = ChatPrompt(OPENING, TOOLS, window())
    prompt.sent(700)

    prompt.add_turn(0, ChatMessage(role=ASSISTANT, reasoning_content="r" * 400))

    assert prompt.estimated_tokens == 800


def test_a_call_with_no_usage_keeps_the_estimate() -> None:
    prompt = chat(["x" * 400])
    before = prompt.estimated_tokens

    prompt.sent(0)  # The gateway reported no usage.

    assert prompt.estimated_tokens == before


def test_a_prompt_exactly_at_the_threshold_does_not_fold() -> None:
    prompt = chat(["x" * 4000] * 3, keep_turns=1)  # The threshold is 1,000.

    prompt.sent(1000)

    assert prompt.fold() == []


def test_a_message_needs_a_turn_to_join() -> None:
    prompt = ChatPrompt(OPENING, TOOLS, window())

    with pytest.raises(ValueError, match="add_turn"):
        prompt.add(tool("c0", "rows: 1"))


def test_nothing_is_folded_up_to_the_threshold() -> None:
    prompt = chat(["x" * 2000] * 12, fold_above_tokens=100_000)

    assert prompt.fold() == []
    assert contents(prompt) == ["x" * 2000] * 12


def test_older_outputs_fold_oldest_first_until_the_prompt_is_under_the_threshold() -> None:
    prompt = chat(["x" * 4000] * 12, keep_turns=8)
    prompt.sent(1100)  # 100 over: one fold of 4,000 characters, about 870 tokens, is enough.

    folds = prompt.fold()

    assert [(fold.step, fold.tool_call_id) for fold in folds] == [(0, "c0")]
    marker = "\n[... folded: the first 500 of 4000 characters ...]"
    assert folds[0] == ContextFold(step=0, tool_call_id="c0", replacement="x" * 500 + marker)
    assert contents(prompt) == ["x" * 500 + marker] + ["x" * 4000] * 11
    assert prompt.estimated_tokens <= 1000


def test_the_last_turns_and_the_other_messages_never_fold() -> None:
    prompt = ChatPrompt(OPENING, TOOLS, window(keep_turns=2))
    for step in range(4):
        prompt.add_turn(step, assistant(f"c{step}", arguments='{"sql": "' + "s" * 3000 + '"}'))
        prompt.add(tool(f"c{step}", "x" * 3000))
        prompt.add(ChatMessage(role=USER, content="g" * 3000))
    prompt.sent(50_000)  # Far over, so it folds all it may.

    folds = prompt.fold()

    assert [fold.step for fold in folds] == [0, 1]  # Not steps 2 and 3, the last 2 turns.
    messages = prompt.messages()
    assert messages[:2] == OPENING
    assert [len(m.content) for m in messages if m.role == USER][1:] == [3000] * 4  # After the task.
    assert all(len(m.tool_calls[0].arguments_json) > 3000 for m in messages if m.role == ASSISTANT)


def test_an_output_no_longer_than_its_fold_stays_whole() -> None:
    # 550 characters fold to 500 and a 50-character marker: no shorter, so no fold.
    assert len("\n[... folded: the first 500 of 550 characters ...]") == 50
    prompt = chat(["x" * 550, "y" * 3000], keep_turns=0)
    prompt.sent(50_000)

    folds = prompt.fold()

    assert [fold.tool_call_id for fold in folds] == ["c1"]


def test_an_output_folds_once() -> None:
    prompt = chat(["x" * 4000] * 3, keep_turns=1)
    prompt.sent(50_000)
    first = prompt.fold()

    prompt.sent(50_000)

    assert [fold.step for fold in first] == [0, 1]
    assert prompt.fold() == []


def test_folding_everything_older_ignores_the_threshold() -> None:
    prompt = chat(["x" * 4000] * 4, keep_turns=1, fold_above_tokens=100_000)

    folds = prompt.fold(everything_older=True)

    assert [fold.step for fold in folds] == [0, 1, 2]


def test_with_no_turns_to_keep_every_output_may_fold() -> None:
    prompt = chat(["x" * 4000] * 2, keep_turns=0)

    folds = prompt.fold(everything_older=True)

    assert [fold.step for fold in folds] == [0, 1]


def test_folding_lowers_the_estimate_by_the_characters_it_takes_out() -> None:
    prompt = chat(["x" * 4000] * 2, keep_turns=1)
    prompt.sent(5000)

    [fold] = prompt.fold(everything_older=True)

    assert prompt.estimated_tokens == math.ceil(5000 - (4000 - len(fold.replacement)) / 4)


def test_a_fold_names_the_global_step_of_its_turn() -> None:
    prompt = ChatPrompt(OPENING, TOOLS, window(keep_turns=0))
    prompt.add_turn(7, assistant("c0"))  # Another chat's steps came first.
    prompt.add(tool("c0", "x" * 4000))

    [fold] = prompt.fold(everything_older=True)

    assert (fold.step, fold.tool_call_id) == (7, "c0")


def thinking(step: int, reasoning: str) -> ChatMessage:
    call = ToolCall(id=f"c{step}", name="execute_sql", arguments_json="{}")
    return ChatMessage(
        role=ASSISTANT, content="Done.", reasoning_content=reasoning, tool_calls=[call]
    )


def test_reasoning_goes_out_only_within_the_last_turns() -> None:
    prompt = ChatPrompt(OPENING, TOOLS, window(keep_turns=2))
    for step in range(4):
        prompt.add_turn(step, thinking(step, f"thought {step}"))
        prompt.add(tool(f"c{step}", "rows: 1"))

    replies = [message for message in prompt.messages() if message.role == ASSISTANT]

    assert [reply.reasoning_content for reply in replies] == ["", "", "thought 2", "thought 3"]
    assert [reply.content for reply in replies] == ["Done."] * 4  # Only the reasoning goes.
    assert [reply.tool_calls[0].id for reply in replies] == ["c0", "c1", "c2", "c3"]


def test_reasoning_the_gateway_counted_stays_in_the_estimate_when_it_leaves() -> None:
    # The count may never have held it, as the chat template may have dropped it: an estimate
    # that took it off could go below the prompt's size.
    prompt = ChatPrompt(OPENING, TOOLS, window(keep_turns=1))
    prompt.add_turn(0, thinking(0, "r" * 400))
    prompt.sent(1000)

    prompt.add_turn(1, thinking(1, ""))

    added = len("Done.") + len("execute_sql") + len("{}")
    assert prompt.estimated_tokens == math.ceil(1000 + added / 4)


def test_reasoning_added_since_the_last_count_comes_off_the_estimate_when_it_leaves() -> None:
    prompt = ChatPrompt(OPENING, TOOLS, window(keep_turns=1))
    before = prompt.estimated_tokens
    prompt.add_turn(0, thinking(0, "r" * 400))
    prompt.sent(0)  # The gateway reported no usage, so no count covers the turn.

    prompt.add_turn(1, thinking(1, ""))

    added = 2 * (len("Done.") + len("execute_sql") + len("{}"))
    assert abs(prompt.estimated_tokens - (before + added / 4)) <= 1  # The 400 came off.


def test_the_estimate_never_goes_below_0() -> None:
    prompt = chat(["x" * 4000] * 2, keep_turns=1)
    prompt.sent(100)  # A count far below the characters' estimate, as for dense text.

    prompt.fold(everything_older=True)

    assert prompt.estimated_tokens == 0


def test_a_window_of_0_replays_reasoning_unchanged() -> None:
    # Chat.keep_turns 0 marks a chat whose replies replay unchanged (trajectory.proto).
    prompt = ChatPrompt(OPENING, TOOLS, window(keep_turns=0))

    for step in range(3):
        prompt.add_turn(step, thinking(step, f"thought {step}"))

    replies = [m.reasoning_content for m in prompt.messages() if m.role == ASSISTANT]
    assert replies == ["thought 0", "thought 1", "thought 2"]


def test_the_reasoning_tail_is_the_chats_last_turns_before_the_next_call() -> None:
    # The steps are global: other chats' steps came between, and do not count.
    prompt = ChatPrompt(OPENING, TOOLS, window(keep_turns=2))
    kept = []
    for step in (0, 3, 4, 9):
        prompt.add_turn(step, thinking(step, f"thought {step}"))
        kept.append([m.reasoning_content for m in prompt.messages() if m.role == ASSISTANT])

    assert kept == [
        ["thought 0"],
        ["thought 0", "thought 3"],
        ["", "thought 3", "thought 4"],
        ["", "", "thought 4", "thought 9"],
    ]
