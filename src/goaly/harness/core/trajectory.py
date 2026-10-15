"""Trajectories (spec §5.1): building one task's Trajectory as the agent runs, and writing it.

A :class:`TrajectoryRecorder` holds a task's Trajectory, in the contract's shape: its chats,
then every model call as a Step that holds only what the call added to its chat. Each loop ends
by handing the whole trajectory so far to a :class:`TrajectorySink`. The last line written for a
task is therefore the complete one, as Trajectory's comment says readers take it.

Versioning is the contract's: the lines are ``goaly.v1.Trajectory`` messages in protobuf's JSON
mapping, whose package names the schema version, and the proto files' rules keep old lines
readable (spec §3.3).
"""

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from goaly.contracts.gen.llm_call_pb2 import LlmCallRecord, Usage
from goaly.contracts.gen.task_pb2 import Prediction, TaskInput
from goaly.contracts.gen.trajectory_pb2 import ChatMessage, Step, StopReason, Trajectory
from goaly.contracts.jsonl import append_jsonl

__all__ = ["JsonlTrajectorySink", "TrajectoryRecorder", "TrajectorySink", "add_usage"]


class TrajectorySink(Protocol):
    """Where the agent loop writes a task's trajectory: the runner's single writer (T06)."""

    def write(self, trajectory: Trajectory) -> None:
        """Take ``trajectory``, which is the sink's own copy.

        The loop calls this on the event loop, also while its task is being cancelled, so it
        must not block for long: the runner's single writer only queues the trajectory.
        """


class JsonlTrajectorySink:
    """A TrajectorySink that appends each trajectory to a JSONL file at once, for scripts and tests.

    Each write blocks the event loop for a moment; a run's trajectories go through the runner's
    single writer instead (spec §6.2).
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()  # append_jsonl does not lock, and threads may share this.

    def write(self, trajectory: Trajectory) -> None:
        with self._lock:
            append_jsonl(self.path, trajectory)


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
    """Builds one task's trajectory, for one run and replicate seed, and writes it to a sink.

    Every loop a policy runs for the task starts a chat. Its steps go into the one list of
    steps, in the order the calls were made, and each step's usage joins the trajectory's.
    """

    def __init__(self, sink: TrajectorySink, *, run_id: str, instance_id: str, seed: int) -> None:
        self._sink = sink
        self._trajectory = Trajectory(run_id=run_id, instance_id=instance_id, seed=seed)

    def start_chat(self, task: TaskInput, label: str, messages: Sequence[ChatMessage]) -> int:
        """Start a chat named ``label`` with ``messages``, and return its index.

        The first chat records ``task``, the task every chat of the trajectory works on, and
        the trajectory's start time.
        """
        if not label.strip():
            raise ValueError("a chat needs a label, such as 'main'")
        trajectory = self._trajectory
        if not trajectory.chats:
            trajectory.task.CopyFrom(task)
            trajectory.started_at.GetCurrentTime()
        trajectory.chats.add(label=label, initial_messages=messages)
        return len(trajectory.chats) - 1

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

    def snapshot(self) -> Trajectory:
        """Return a copy of the trajectory so far."""
        trajectory = Trajectory()
        trajectory.CopyFrom(self._trajectory)
        return trajectory

    def finish(
        self,
        stop_reason: StopReason.ValueType,
        *,
        prediction: Prediction | None = None,
        error_message: str = "",
    ) -> None:
        """Record why the agent stopped and what it returned, and write a copy to the sink.

        ``prediction`` is None when the policy returns nothing, as when it crashed. A later
        loop of the same task finishes again, and so writes the whole trajectory again.
        """
        trajectory = self._trajectory
        trajectory.stop_reason = stop_reason
        trajectory.error_message = error_message
        if prediction is None:
            trajectory.ClearField("prediction")
        else:
            trajectory.prediction.CopyFrom(prediction)
        trajectory.finished_at.GetCurrentTime()
        self._sink.write(self.snapshot())
