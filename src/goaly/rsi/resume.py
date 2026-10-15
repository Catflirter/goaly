"""Picking a lane up after a crash or a stop (spec §8.4, §8.7).

A restarted lane calls these before its next round, and a stopping lane at its deadline:

- :func:`remove_worktrees` removes the lane's leftover worktrees; their branches keep any commit.
- :func:`reconcile_best` makes the ledger's best version and the ``best/<lane>`` tag agree. The
  round moves the tag before it records the acceptance (goaly.rsi.round.shielded_accept), so a
  crash in between leaves the tag on a pending experiment whose confirmation passed: that
  experiment is recorded as accepted.
- :func:`abort_experiments` records pending experiments as aborted and queues their hypotheses
  again: those interrupted before their commit, when a lane resumes, since their run is lost;
  every pending one, when the deadline stops the lane. An experiment interrupted after its commit
  resumes at its next step instead (goaly.rsi.round).
"""

import asyncio
import logging
from collections.abc import Callable, Mapping
from datetime import datetime

from google.protobuf.timestamp_pb2 import Timestamp

from goaly.contracts.gen.experiment_pb2 import (
    DECISION_ABORTED,
    DECISION_ACCEPTED,
    IMPLEMENTER_EXIT_INTERRUPTED,
    REJECTION_REASON_OUTRANKED,
    ExperimentRecord,
    Lane,
)
from goaly.contracts.gen.llm_call_pb2 import CallPurpose, Usage
from goaly.rsi.hypothesis_queue import HypothesisQueue
from goaly.rsi.ledger import Ledger
from goaly.rsi.workspace import Workspace, WorkspaceError

__all__ = [
    "MAX_REQUEUES",
    "LaneStateError",
    "UsageSource",
    "abort_experiments",
    "abort_record",
    "finish",
    "merge_usage",
    "reconcile_best",
    "remove_worktrees",
    "requeue",
    "timestamp",
]

MAX_REQUEUES = 2  # How often a hypothesis goes back in the queue before it is dropped.

_logger = logging.getLogger(__name__)

# The calls recorded under a run ID, summed by purpose.
type UsageSource = Callable[[str], Mapping[CallPurpose.ValueType, Usage]]


class LaneStateError(Exception):
    """The ledger and the best tag disagree in a way no crash explains; a person must look."""


def remove_worktrees(workspace: Workspace, lane: Lane.ValueType) -> int:
    """Remove the lane's leftover worktrees, keeping their branches; return how many."""
    leftovers = workspace.worktrees(lane)
    for worktree in leftovers:
        workspace.remove(worktree)
    return len(leftovers)


def reconcile_best(
    ledger: Ledger,
    workspace: Workspace,
    lane: Lane.ValueType,
    *,
    v0_commit: str,
    now: datetime,
) -> str:
    """Make ``best/<lane>`` and the ledger agree, and return the best commit.

    With nothing accepted, the best is ``v0_commit``, a full commit ID. A missing tag is set to
    the ledger's best. A tag on a pending experiment whose confirmation passed records that
    experiment as accepted. A tag behind the ledger's best moves forward to it. Anything else
    raises LaneStateError.
    """
    tag = workspace.best_commit(lane)
    recorded = ledger.best_version(lane) or v0_commit
    if tag == recorded:
        return recorded
    if tag is None:
        workspace.set_best(lane, recorded, expected=None)
        _logger.warning("set the missing best tag to %s", recorded)
        return recorded
    confirmed = [
        record
        for record in ledger.pending_experiments(lane)
        if record.candidate_commit == tag and record.HasField("confirmation")
    ]
    if confirmed:
        record = confirmed[-1]
        record.decision = DECISION_ACCEPTED
        record.finished_at.CopyFrom(timestamp(now))
        ledger.record_experiment(record)
        _logger.warning("recorded %s as accepted: the best tag was on it", record.experiment_id)
        return tag
    try:
        workspace.set_best(lane, recorded, expected=tag)
    except WorkspaceError as error:
        raise LaneStateError(
            f"the best tag is on {tag}, which is neither the ledger's best {recorded}, nor one "
            "of its ancestors, nor a confirmed candidate"
        ) from error
    _logger.warning("moved the best tag forward from %s to the ledger's %s", tag, recorded)
    return recorded


def abort_experiments(
    ledger: Ledger,
    queue: HypothesisQueue,
    lane: Lane.ValueType,
    reason: str,
    *,
    now: datetime,
    uncommitted_only: bool,
    usage: UsageSource = lambda _: {},
    max_requeues: int = MAX_REQUEUES,
) -> list[ExperimentRecord]:
    """Record the lane's pending experiments as aborted, or only those without a commit, and
    queue their hypotheses again; return them.
    """
    aborted = []
    for record in ledger.pending_experiments(lane):
        if uncommitted_only and record.candidate_commit:
            continue
        if not record.candidate_commit and not record.HasField("implementer"):
            # Its run was cut short and never measured (ImplementerRun.step_count).
            record.implementer.exit = IMPLEMENTER_EXIT_INTERRUPTED
        abort_record(
            ledger,
            queue,
            record,
            reason,
            now=now,
            usage=usage(record.experiment_id),
            max_requeues=max_requeues,
        )
        aborted.append(record)
    return aborted


def abort_record(
    ledger: Ledger,
    queue: HypothesisQueue,
    record: ExperimentRecord,
    reason: str,
    *,
    now: datetime,
    usage: Mapping[CallPurpose.ValueType, Usage],
    max_requeues: int = MAX_REQUEUES,
) -> None:
    """Queue ``record``'s hypothesis again, within its cap, then record it as aborted (spec
    §8.7). The push comes first: a crash between the two leaves the hypothesis queued and the
    experiment pending, which the restarted lane aborts again, and the queue keeps one copy.
    """
    _logger.warning("experiment %s aborted: %s", record.experiment_id, reason)
    requeue(ledger, queue, record, max_requeues=max_requeues)
    record.decision = DECISION_ABORTED
    del record.rejection_reasons[:]
    record.decision_reason = "; ".join(filter(None, [record.decision_reason, f"aborted: {reason}"]))
    record.finished_at.CopyFrom(timestamp(now))
    merge_usage(record, usage)
    ledger.record_experiment(record)


def requeue(
    ledger: Ledger, queue: HypothesisQueue, record: ExperimentRecord, *, max_requeues: int
) -> bool:
    """Queue ``record``'s hypothesis again, unless it went back ``max_requeues`` times already,
    as earlier experiments that were aborted or outranked show; then it is dropped and logged.
    Return whether it was queued.
    """
    name = record.hypothesis.hypothesis_id
    earlier = [
        other
        for other in ledger.latest_experiments(record.lane).values()
        if other.hypothesis.hypothesis_id == name
        and other.experiment_id != record.experiment_id
        and (
            other.decision == DECISION_ABORTED
            or REJECTION_REASON_OUTRANKED in other.rejection_reasons
        )
    ]
    if len(earlier) >= max_requeues:
        _logger.warning(
            "dropped hypothesis %s: it went back in the queue %d times already",
            name,
            len(earlier),
            extra={"event": "hypothesis_dropped", "hypothesis_id": name},
        )
        return False
    queue.push([record.hypothesis])
    return True


async def finish[T](work: Callable[[], T]) -> T:
    """Run the blocking ``work`` in a thread and return its result. If the caller is cancelled
    meanwhile, wait for the work to finish anyway, then let the cancellation through: a ledger
    write or a tag move is never left running behind a caller that has moved on.
    """
    future = asyncio.ensure_future(asyncio.to_thread(work))
    cancelled = False
    while not future.done():
        try:
            await asyncio.shield(future)
        except asyncio.CancelledError:
            cancelled = True
    result = future.result()  # Raises the work's own error, if it failed.
    if cancelled:
        raise asyncio.CancelledError
    return result


def merge_usage(record: ExperimentRecord, usage: Mapping[CallPurpose.ValueType, Usage]) -> None:
    """Set the record's usage for each purpose in ``usage``, which sums its calls so far."""
    for purpose, total in usage.items():
        entry = next((e for e in record.usage_by_purpose if e.purpose == purpose), None)
        if entry is None:
            entry = record.usage_by_purpose.add(purpose=purpose)
        entry.usage.CopyFrom(total)


def timestamp(moment: datetime) -> Timestamp:
    """``moment`` as a protobuf Timestamp."""
    stamp = Timestamp()
    stamp.FromDatetime(moment)
    return stamp
