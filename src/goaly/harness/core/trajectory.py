"""Trajectories (spec §5.1): a task's chats and steps, as the agent loop records them in memory.

A :class:`TrajectoryRecorder` builds the part of a task's Trajectory that the loop owns: its
chats, one Step per model call holding only what the call added to its chat, the usage of every
model call, the stop reason and the error message. The runner, the single writer of
trajectories.jsonl (T06), reads it from the task's context once the policy returns or raises,
and fills in the rest: the run, task and seed, the prediction, the grade and the timestamps.

Versioning is the contract's: a Trajectory is a ``goaly.v1`` message, whose package names the
schema version, and the proto files' rules keep old lines readable (spec §3.3).
"""

from collections.abc import Sequence

from goaly.contracts.gen.llm_call_pb2 import LlmCallRecord, Usage
from goaly.contracts.gen.trajectory_pb2 import ChatMessage, Step, StopReason, Trajectory

__all__ = ["TrajectoryRecorder", "add_usage"]


def add_usage(total: Usage, usage: Usage) -> None:
    """Add ``usage`` to ``total``, as llm_call.proto sums a Usage.

    Tokens add up. An optional field is set in the sum once any summed usage reported it, and
    holds the total of the reported values.
    """
    total.input_tokens += usage.input_tokens
    total.output_tokens += usage.output_tokens
    if usage.HasField("reasoning_tokens"):
        total.reasoning_tokens += usage.reasoning_tokens
    if usage.HasField("cost_usd"):
        total.cost_usd += usage.cost_usd


class TrajectoryRecorder:
    """Records one task's chats and steps, in memory, as its loops run.

    Every loop a policy runs for the task starts a chat. Its steps go into the trajectory's one
    list of steps, in the order the calls were made, and each step's usage joins the
    trajectory's.
    """

    def __init__(self) -> None:
        self._trajectory = Trajectory()

    def start_chat(self, label: str, messages: Sequence[ChatMessage], *, keep_turns: int) -> int:
        """Start a chat named ``label`` that begins with ``messages``, and return its index.

        ``keep_turns`` is the chat's replay window (Chat.keep_turns): more than 0 for an
        executor chat, whose prompts keep the reasoning of their last ``keep_turns`` replies; 0
        for a chat whose replies replay unchanged. Raise ValueError for a blank label, one that
        another chat of the task has, or a negative ``keep_turns``.
        """
        if not label.strip():
            raise ValueError("a chat needs a label, such as 'main'")
        if any(chat.label == label for chat in self._trajectory.chats):
            raise ValueError(f"the task already has a chat named {label!r}")
        if keep_turns < 0:
            raise ValueError(f"a replay window cannot be negative, as keep_turns={keep_turns} is")
        self._trajectory.chats.add(label=label, initial_messages=messages, keep_turns=keep_turns)
        return len(self._trajectory.chats) - 1

    def add_step(self, chat: int, record: LlmCallRecord, assistant: ChatMessage) -> Step:
        """Add the step of the model call ``record``, whose reply is ``assistant``.

        The step takes the call's ID, start time, latency and usage from its record, and the
        usage joins the trajectory's. Return the step, for the caller to add the tool results
        and injected messages to.
        """
        if not 0 <= chat < len(self._trajectory.chats):
            raise ValueError(f"there is no chat {chat}")
        steps = self._trajectory.steps
        step = steps.add(
            index=len(steps),
            chat=chat,
            call_id=record.call_id,
            started_at=record.started_at,
            latency_s=record.latency_s,
            usage=record.usage,
            assistant=assistant,
        )
        add_usage(self._trajectory.usage, record.usage)
        return step

    def add_usage(self, usage: Usage) -> None:
        """Add the usage of a model call that has no step, as one that failed."""
        add_usage(self._trajectory.usage, usage)

    @property
    def step_count(self) -> int:
        """How many steps the trajectory holds."""
        return len(self._trajectory.steps)

    @property
    def usage(self) -> Usage:
        """A copy of the usage of every model call so far."""
        usage = Usage()
        usage.CopyFrom(self._trajectory.usage)
        return usage

    def stop(self, stop_reason: StopReason.ValueType, *, error_message: str = "") -> None:
        """Record why the agent stopped, and what went wrong if it stopped on an error.

        A later loop of the same task records its own stop over this one.
        """
        self._trajectory.stop_reason = stop_reason
        self._trajectory.error_message = error_message

    def trajectory(self) -> Trajectory:
        """Return a copy of the trajectory so far."""
        trajectory = Trajectory()
        trajectory.CopyFrom(self._trajectory)
        return trajectory
