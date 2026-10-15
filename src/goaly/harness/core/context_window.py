"""Context management (spec §5.1): keeping a chat's prompt within the model's context.

Qwen has a 128K-token context. Once the prompt of a chat's next call would pass 96K tokens, the
loop folds older tool outputs, oldest first, until the prompt is back under: a folded output
keeps its first 500 characters and a marker that gives its full length. The system prompt, the
task and the last 8 turns, each an assistant message with its tool results, stay as they are.
The fold is deterministic, with no model call, and only the prompt is folded: the trajectory
keeps every output whole and records each fold (Step.folds), so every step's prompt can still be
rebuilt from it.

Only the turns of that verbatim tail go out with their reasoning: an assistant message that
leaves it is sent without its reasoning_content from then on, whatever the prompt's size. The
trajectory keeps the reasoning, and records the window as Chat.keep_turns, so the rule rebuilds
the prompt (Step's comment in trajectory.proto). A window of 0 replays the chat's reasoning
unchanged, as the contract reads it.

A prompt's size is estimated as the previous call's input tokens, as the gateway counted them,
plus a token for every 4 characters added since.
"""

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveInt

from goaly.contracts.gen.trajectory_pb2 import ChatMessage, ChatRole, ContextFold
from goaly.contracts.interfaces import JsonValue

__all__ = ["ChatPrompt", "ContextWindow"]

# About how many characters make a token, for text the gateway has not counted yet.
_CHARS_PER_TOKEN = 4


class ContextWindow(BaseModel):
    """When the loop folds a chat's prompt, and what it keeps (spec §5.1); from the pack's config.

    Once the next call's prompt would pass ``fold_above_tokens``, older tool outputs fold to
    their first ``folded_chars`` characters, and the last ``keep_turns`` turns never fold. The
    default threshold leaves 32K of Qwen's 128K for the reply and for the estimate's error.

    ``keep_turns`` is also the chat's replay window (Chat.keep_turns): only the last
    ``keep_turns`` replies keep their reasoning, and every reply loses its call markup. With 0,
    any tool output may fold, but the replies replay unchanged, as the contract reads a window
    of 0: the chat is then no executor chat.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    fold_above_tokens: PositiveInt = 96_000
    keep_turns: NonNegativeInt = 8
    folded_chars: PositiveInt = 500


@dataclass
class _Turn:
    """One step's part of a prompt: its assistant message, then its tool and injected messages."""

    step: int  # The step's index in Trajectory.steps.
    messages: list[ChatMessage]
    folded: set[int] = field(default_factory=set)  # The indexes of the messages folded.


class ChatPrompt:
    """One chat's prompt, as the loop sends it: the opening, then a turn per step, some folded.

    It estimates the size of the next call's prompt, and folds older tool outputs once that
    passes the window's threshold.
    """

    def __init__(
        self,
        opening: Sequence[ChatMessage],
        tools: Sequence[Mapping[str, JsonValue]],
        window: ContextWindow,
    ) -> None:
        self._window = window
        self._opening = list(opening)
        self._turns: list[_Turn] = []
        self._counted = 0  # How many turns the gateway's last count of the prompt took in.
        tool_chars = len(json.dumps(tools, ensure_ascii=False))
        self._estimate = (sum(map(_chars, opening)) + tool_chars) / _CHARS_PER_TOKEN

    def messages(self) -> list[ChatMessage]:
        """Return the next call's prompt: the opening, then every turn, as folded."""
        return [*self._opening, *(message for turn in self._turns for message in turn.messages)]

    def add_turn(self, step: int, assistant: ChatMessage) -> None:
        """Start the turn of the step at ``step`` in Trajectory.steps, with its assistant reply.

        The turn this pushes out of the last ``keep_turns`` loses its reasoning, unless the
        window is 0, which replays reasoning unchanged.
        """
        self._turns.append(_Turn(step, [assistant]))
        self._estimate += _chars(assistant) / _CHARS_PER_TOKEN
        leaving = len(self._turns) - 1 - self._window.keep_turns
        if self._window.keep_turns > 0 and leaving >= 0:
            self._drop_reasoning(leaving)

    def add(self, message: ChatMessage) -> None:
        """Add a tool message, or an injected one, to the last turn."""
        self._turns[-1].messages.append(message)
        self._estimate += _chars(message) / _CHARS_PER_TOKEN

    def sent(self, input_tokens: int) -> None:
        """Take the gateway's count of the prompt just sent; 0 means it reported none."""
        if input_tokens > 0:
            self._estimate = input_tokens
            self._counted = len(self._turns)

    @property
    def estimated_tokens(self) -> int:
        """The estimated size of the next call's prompt, which the threshold is compared with."""
        return max(math.ceil(self._estimate), 0)

    def fold(self, *, everything_older: bool = False) -> list[ContextFold]:
        """Fold older tool outputs, oldest first, and return the folds made.

        Past the window's threshold, outputs fold until the estimate is back at or under it.
        ``everything_older`` folds every output that may fold, whatever the estimate, as when
        the gateway said the prompt does not fit. An output in the last turns, one already
        folded, and one no longer than its fold stay as they are.
        """
        folds: list[ContextFold] = []
        older = self._turns[: max(len(self._turns) - self._window.keep_turns, 0)]
        for turn in older:
            for index, message in enumerate(turn.messages):
                if not everything_older and self._fits():
                    return folds
                if message.role != ChatRole.CHAT_ROLE_TOOL or index in turn.folded:
                    continue
                replacement = _folded(message.content, self._window.folded_chars)
                if len(replacement) >= len(message.content):
                    continue
                # A new message: a caller may still hold the old one, from an earlier prompt.
                turn.messages[index] = ChatMessage(
                    role=message.role, content=replacement, tool_call_id=message.tool_call_id
                )
                turn.folded.add(index)
                self._estimate -= (len(message.content) - len(replacement)) / _CHARS_PER_TOKEN
                folds.append(
                    ContextFold(
                        step=turn.step, tool_call_id=message.tool_call_id, replacement=replacement
                    )
                )
        return folds

    def _drop_reasoning(self, index: int) -> None:
        """Send the reply of turn ``index`` without its reasoning from now on.

        Its reasoning comes off the estimate only if the estimate added it since the gateway's
        last count: that count may never have held it, as the chat template may have dropped it.
        """
        turn = self._turns[index]
        reply = turn.messages[0]
        if reply.reasoning_content:
            plain = ChatMessage()  # A new message, as a fold makes.
            plain.CopyFrom(reply)
            plain.reasoning_content = ""
            turn.messages[0] = plain
            if index >= self._counted:
                self._estimate -= len(reply.reasoning_content) / _CHARS_PER_TOKEN

    def _fits(self) -> bool:
        return self.estimated_tokens <= self._window.fold_above_tokens


def _folded(content: str, keep: int) -> str:
    """``content`` folded: its first ``keep`` characters, and a marker with its length."""
    return f"{content[:keep]}\n[... folded: the first {keep} of {len(content)} characters ...]"


def _chars(message: ChatMessage) -> int:
    """The characters of ``message`` that the model reads: its text, reasoning and calls."""
    calls = sum(
        len(call.name) + len(call.repaired_arguments_json or call.arguments_json)
        for call in message.tool_calls
    )
    return len(message.content) + len(message.reasoning_content) + calls
