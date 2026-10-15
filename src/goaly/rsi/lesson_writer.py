"""The lesson writer (spec §8.2 step 8, §8.6): GLM turns one decided experiment into a lesson.

:func:`write_lesson` makes one structured call through goaly.rsi.structured, with purpose
CALL_PURPOSE_RSI_LESSON and the experiment ID as run ID. The lesson says what was tried, what
happened and why, and cites the experiment ID; the store adds the citation.

The writer sees the hypothesis, how the implementer and the guards ended, and the decision with
each phase's aggregates. It never sees per-task val data: no val task ID, no val trajectory and
no per-task val result, because lessons reach the analyst, who must stay val-clean (spec §7).
The decision_reason it is shown is written by goaly.rsi.decision.describe, from aggregates only.
"""

from typing import Annotated, Final

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from goaly.contracts.gen.experiment_pb2 import (
    ChangeKind,
    Decision,
    ExperimentRecord,
    Guard,
    ImplementerExit,
    PhaseResult,
    RejectionReason,
)
from goaly.contracts.gen.llm_call_pb2 import CallPurpose
from goaly.contracts.gen.trajectory_pb2 import ChatMessage, ChatRole
from goaly.harness.core.llm import CallTags, LlmClient
from goaly.harness.core.llm_models import IMPROVEMENT_MODEL, ModelParams
from goaly.rsi.structured import StructuredReply, complete_structured

__all__ = [
    "LESSON_ATTEMPT_TIMEOUT_S",
    "LESSON_MAX_TOKENS",
    "LessonAnswer",
    "lesson_prompt",
    "write_lesson",
]

# GLM can reason past 12K tokens before it answers (T19), so a lesson call may use 24K.
LESSON_MAX_TOKENS: Final = 24_000
# Each attempt's limit, as the analyst's: 24K tokens outlast the client's default 300 s under load.
LESSON_ATTEMPT_TIMEOUT_S: Final = 900.0
_SUMMARY_CHARS: Final = 600

_SYSTEM: Final = """\
You keep the lessons of an RSI loop that improves an agent harness for Effi-SQL, a benchmark of
making slow PostgreSQL queries faster. Each lesson is one short paragraph that a later analyst
reads before proposing changes: what was tried, what happened and why, and what to do or avoid
next time. Be concrete about the change and honest about the evidence: a rejection on noise is
weak evidence, a flagged hard-coded prediction is strong evidence of a bad idea. Cite the
experiment ID given. Never invent task IDs or numbers that are not in the record."""


class LessonAnswer(BaseModel):
    """The lesson writer's answer."""

    model_config = ConfigDict(extra="forbid")

    lesson: Annotated[str, StringConstraints(strip_whitespace=True, min_length=20)] = Field(
        description="The lesson: one paragraph, at most about 120 words, citing the experiment ID.",
        max_length=2000,
    )


def lesson_prompt(record: ExperimentRecord) -> list[ChatMessage]:
    """The lesson call's messages for ``record``, which must be decided, not aborted."""
    if record.decision not in (Decision.DECISION_ACCEPTED, Decision.DECISION_REJECTED):
        raise ValueError(f"experiment {record.experiment_id} is not accepted or rejected")
    hypothesis = record.hypothesis
    lines = [
        f"Experiment {record.experiment_id}, round {record.round}.",
        f"Hypothesis {hypothesis.hypothesis_id} ({ChangeKind.Name(hypothesis.change_kind)}):",
        f"- targets: {hypothesis.failure_pattern}",
        f"- change: {hypothesis.change}",
        f"- expected effect: {hypothesis.expected_effect}",
        f"- risk: {hypothesis.risk}",
    ]
    if record.HasField("implementer"):
        run = record.implementer
        lines.append(f"Implementer: {ImplementerExit.Name(run.exit)}.")
        if run.summary:
            lines.append(f"Its summary: {run.summary[:_SUMMARY_CHARS]}")
    for guard in record.guards:
        lines.append(f"Guard {Guard.Name(guard.guard)}: {'passed' if guard.passed else 'FAILED'}.")
        if not guard.passed and guard.detail:
            lines.append(f"  {guard.detail[:_SUMMARY_CHARS]}")
    phases = (
        ("Screen (train)", record.screening, record.HasField("screening")),
        ("Validation (val)", record.validation, record.HasField("validation")),
        ("Confirmation (val, both seeds)", record.confirmation, record.HasField("confirmation")),
    )
    lines.extend(f"{name}: {_aggregates(phase)}" for name, phase, ran in phases if ran)
    reasons = ", ".join(RejectionReason.Name(reason) for reason in record.rejection_reasons)
    lines.append(f"Decision: {Decision.Name(record.decision)}{f' ({reasons})' if reasons else ''}.")
    if record.decision_reason:
        lines.append(f"Reasoning: {record.decision_reason}")
    lines.append("Write the lesson.")
    return [
        ChatMessage(role=ChatRole.CHAT_ROLE_SYSTEM, content=_SYSTEM),
        ChatMessage(role=ChatRole.CHAT_ROLE_USER, content="\n".join(lines)),
    ]


async def write_lesson(
    client: LlmClient, record: ExperimentRecord
) -> StructuredReply[LessonAnswer]:
    """Ask GLM for ``record``'s lesson. Raises StructuredOutputError after 3 invalid replies, and
    the LlmError of a gateway failure that outlasted the client's retries.
    """
    tags = CallTags(CallPurpose.CALL_PURPOSE_RSI_LESSON, run_id=record.experiment_id)
    llm = client.bind(model=IMPROVEMENT_MODEL, tags=tags)
    return await complete_structured(
        llm,
        lesson_prompt(record),
        LessonAnswer,
        params=ModelParams(max_tokens=LESSON_MAX_TOKENS),
        attempt_timeout_s=LESSON_ATTEMPT_TIMEOUT_S,
    )


def _aggregates(phase: PhaseResult) -> str:
    """A phase's aggregates. A flagged task is counted, never named: val IDs must not leak."""
    text = (
        f"{phase.task_count} tasks x {phase.seed_count} seeds, R-VES"
        f" {100 * phase.candidate_score:.1f} vs the best's {100 * phase.best_score:.1f}"
        f" (delta {100 * phase.delta:+.1f} points); wins {phase.wins}, losses {phase.losses},"
        f" {phase.regressions} regressions"
    )
    if phase.flagged_instance_ids:
        text += (
            f"; {len(phase.flagged_instance_ids)} predictions flagged as hard-coded results"
            " (hard_coded or literal_result)"
        )
    return text
