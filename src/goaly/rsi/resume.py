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

import logging
from collections.abc import Callable, Mapping
from datetime import datetime

from google.protobuf.timestamp_pb2 import Timestamp

from goaly.contracts.gen.experiment_pb2 import (
    DECISION_ABORTED,
    DECISION_ACCEPTED,
    IMPLEMENTER_EXIT_INTERRUPTED,
    ExperimentRecord,
    Lane,
)
from goaly.contracts.gen.llm_call_pb2 import CallPurpose, Usage
from goaly.rsi.hypothesis_queue import HypothesisQueue
from goaly.rsi.ledger import Ledger
from goaly.rsi.workspace import Workspace, WorkspaceError

__all__ = [
    "LaneStateError",
    "UsageSource",
    "abort_experiments",
    "abort_record",
    "merge_usage",
    "reconcile_best",
    "remove_worktrees",
]

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
        record.finished_at.CopyFrom(_timestamp(now))
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
        abort_record(ledger, queue, record, reason, now=now, usage=usage(record.experiment_id))
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
) -> None:
    """Record ``record`` as aborted, and queue its hypothesis again (spec §8.7)."""
    _logger.warning("experiment %s aborted: %s", record.experiment_id, reason)
    record.decision = DECISION_ABORTED
    del record.rejection_reasons[:]
    record.decision_reason = "; ".join(filter(None, [record.decision_reason, f"aborted: {reason}"]))
    record.finished_at.CopyFrom(_timestamp(now))
    merge_usage(record, usage)
    ledger.record_experiment(record)
    queue.push([record.hypothesis])


def merge_usage(record: ExperimentRecord, usage: Mapping[CallPurpose.ValueType, Usage]) -> None:
    """Set the record's usage for each purpose in ``usage``, which sums its calls so far."""
    for purpose, total in usage.items():
        entry = next((e for e in record.usage_by_purpose if e.purpose == purpose), None)
        if entry is None:
            entry = record.usage_by_purpose.add(purpose=purpose)
        entry.usage.CopyFrom(total)


def _timestamp(moment: datetime) -> Timestamp:
    stamp = Timestamp()
    stamp.FromDatetime(moment)
    return stamp
