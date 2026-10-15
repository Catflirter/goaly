"""The RSI failure analyst (spec §8.2 step 1): GLM turns failed train tasks into 3 hypotheses.

:func:`analyze` makes the round's one structured call, through goaly.rsi.structured, with
purpose CALL_PURPOSE_RSI_ANALYST, the round ID as run ID, no tools and no agent loop. GLM's answer
must be an :class:`AnalystAnswer`: the failures grouped by pattern, and exactly 3 hypotheses,
which analyze maps to Hypothesis contracts. The answer is checked against the input it answers:
its patterns may name only failed train tasks the input listed, and its hypothesis IDs must be new
to the ledger and the queue. After 3 invalid replies the round's analysis is skipped, and
AnalysisSkipped says so with the calls' IDs and cost, for T21 to record.

goaly.rsi.analyst_input builds the :class:`AnalystInput`; goaly.rsi.hypothesis_queue keeps the
hypotheses a round does not build.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Annotated, Final, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationInfo,
    field_validator,
    model_validator,
)

from goaly.contracts.gen.experiment_pb2 import (
    CHANGE_KIND_CONFIG,
    CHANGE_KIND_POLICY,
    CHANGE_KIND_PROMPT,
    CHANGE_KIND_REMOVAL,
    CHANGE_KIND_SKILL,
    CHANGE_KIND_TOOL,
    HYPOTHESIS_SOURCE_ANALYST,
    HYPOTHESIS_SOURCE_IDEA_BANK,
    ChangeKind,
    Hypothesis,
    HypothesisSource,
)
from goaly.contracts.gen.llm_call_pb2 import CallPurpose, Usage
from goaly.contracts.gen.trajectory_pb2 import ChatMessage, ChatRole
from goaly.harness.core.llm import CallTags, LlmClient
from goaly.harness.core.llm_models import IMPROVEMENT_MODEL, ModelParams
from goaly.rsi.ids import check_id
from goaly.rsi.structured import StructuredOutputError, complete_structured

__all__ = [
    "ANALYST_ATTEMPT_TIMEOUT_S",
    "ANALYST_MAX_TOKENS",
    "HYPOTHESIS_COUNT",
    "Analysis",
    "AnalysisSkipped",
    "AnalystAnswer",
    "AnalystInput",
    "FailurePattern",
    "ProposedHypothesis",
    "analyze",
]

HYPOTHESIS_COUNT: Final = 3
# GLM may think past GLM_PROFILE's 12K tokens on this prompt: a 2026-10-15 smoke run spent all 12K
# on reasoning. A reply cut off anyway is retried, its reasoning replayed (goaly.rsi.structured).
ANALYST_MAX_TOKENS: Final = 24_000
# 24K tokens at GLM's ~45 tokens/s under load (spec §6.2) take about 530 s, past the client's
# default 300 s per attempt.
ANALYST_ATTEMPT_TIMEOUT_S: Final = 900.0

_logger = logging.getLogger(__name__)

_CHANGE_KINDS: Final[Mapping[str, ChangeKind.ValueType]] = {
    "prompt": CHANGE_KIND_PROMPT,
    "skill": CHANGE_KIND_SKILL,
    "tool": CHANGE_KIND_TOOL,
    "policy": CHANGE_KIND_POLICY,
    "config": CHANGE_KIND_CONFIG,
    "removal": CHANGE_KIND_REMOVAL,
}
_SOURCES: Final[Mapping[str, HypothesisSource.ValueType]] = {
    "idea_bank": HYPOTHESIS_SOURCE_IDEA_BANK,
    "analysis": HYPOTHESIS_SOURCE_ANALYST,
}


# The answer GLM must give.


def _valid_id(value: str) -> str:
    check_id(value, "id")
    return value


type _Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
type _Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
type _Change = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)
]


class FailurePattern(BaseModel):
    """A group of failed train tasks that fail the same way."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: _Name = Field(description="A short name, which hypotheses cite as target_pattern.")
    task_ids: tuple[str, ...] = Field(
        min_length=1, description="The failed train tasks in the pattern, by their IDs."
    )
    description: _Text = Field(description="What goes wrong in these tasks, and why.")

    @field_validator("task_ids")
    @classmethod
    def _check_tasks(cls, task_ids: tuple[str, ...], info: ValidationInfo) -> tuple[str, ...]:
        failed: frozenset[str] = (info.context or {}).get("failed_task_ids", frozenset())
        unknown = [task for task in task_ids if task not in failed]
        if unknown:
            raise ValueError(f"not failed train tasks of this input: {', '.join(unknown)}")
        return tuple(dict.fromkeys(task_ids))


class ProposedHypothesis(BaseModel):
    """One pack change to test, as GLM proposes it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: Annotated[str, AfterValidator(_valid_id)] = Field(
        description="A new unique ID: 1-64 letters, digits, '_' or '-', starting with a letter "
        "or a digit."
    )
    target_pattern: _Name = Field(description="The exact name of the failure pattern it targets.")
    change_kind: Literal["prompt", "skill", "tool", "policy", "config", "removal"]
    change: _Change = Field(description="Exactly what to change, in which file.")
    expected_effect: _Text
    risk: _Text
    source: Literal["idea_bank", "analysis"] = Field(
        description="idea_bank if the idea comes from the idea bank, analysis if from your own "
        "analysis of the failures."
    )


class AnalystAnswer(BaseModel):
    """The analyst's answer: the failures grouped by pattern, and exactly 3 hypotheses."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    failure_patterns: tuple[FailurePattern, ...] = Field(min_length=1, max_length=12)
    hypotheses: tuple[ProposedHypothesis, ...] = Field(
        min_length=HYPOTHESIS_COUNT, max_length=HYPOTHESIS_COUNT
    )

    @model_validator(mode="after")
    def _check_references(self, info: ValidationInfo) -> Self:
        names = [pattern.name for pattern in self.failure_patterns]
        if len(set(names)) < len(names):
            raise ValueError("two failure patterns have the same name")
        ids = [hypothesis.id for hypothesis in self.hypotheses]
        if len(set(ids)) < len(ids):
            raise ValueError("two hypotheses have the same id")
        taken: frozenset[str] = (info.context or {}).get("taken_ids", frozenset())
        reused = [hypothesis_id for hypothesis_id in ids if hypothesis_id in taken]
        if reused:
            raise ValueError(f"hypothesis ids already used in the ledger or queue: {reused}")
        for hypothesis in self.hypotheses:
            if hypothesis.target_pattern not in names:
                raise ValueError(
                    f"hypothesis {hypothesis.id} targets {hypothesis.target_pattern!r}, which is "
                    "no failure pattern's name"
                )
        return self


# What the analyst is shown: goaly.rsi.analyst_input builds it.


@dataclass(frozen=True, slots=True)
class AnalystInput:
    """What the analyst is shown, rendered, and the IDs its answer is checked against."""

    system: str
    user: str
    failed_task_ids: frozenset[str]  # Every failed train task, shown or dropped.
    shown_task_ids: tuple[str, ...]  # The failures summarized, in priority order.
    dropped_task_ids: tuple[str, ...]  # The failures dropped to fit the budget, in that order.
    taken_ids: frozenset[str]  # Hypothesis IDs the ledger or the queue already use.

    def messages(self) -> list[ChatMessage]:
        return [
            ChatMessage(role=ChatRole.CHAT_ROLE_SYSTEM, content=self.system),
            ChatMessage(role=ChatRole.CHAT_ROLE_USER, content=self.user),
        ]


# The call.


@dataclass(frozen=True, slots=True)
class Analysis:
    """A round's analysis: the failure patterns and the 3 hypotheses, and the calls it took."""

    patterns: tuple[FailurePattern, ...]
    hypotheses: tuple[Hypothesis, ...]
    call_ids: tuple[str, ...]
    usage: Usage = field(default_factory=Usage)


@dataclass(frozen=True, slots=True)
class AnalysisSkipped:
    """The round's analysis was skipped; T21 records it. ``reason`` says why, in words."""

    reason: str
    call_ids: tuple[str, ...] = ()
    usage: Usage = field(default_factory=Usage)


async def analyze(
    client: LlmClient, analyst_input: AnalystInput, *, round_id: str
) -> Analysis | AnalysisSkipped:
    """Ask GLM for the round's hypotheses: one structured call, retried while invalid.

    The calls are recorded with purpose CALL_PURPOSE_RSI_ANALYST and ``round_id`` as run ID.
    Each call may use ANALYST_MAX_TOKENS, and each attempt ANALYST_ATTEMPT_TIMEOUT_S. With no
    failed train task there is nothing to analyze, and no call is made. A gateway failure that
    outlasted the client's retries raises its LlmError.
    """
    check_id(round_id, "round_id")
    if not analyst_input.failed_task_ids:
        return AnalysisSkipped("the best fails no train task, so there is nothing to analyze")
    tags = CallTags(CallPurpose.CALL_PURPOSE_RSI_ANALYST, run_id=round_id)
    llm = client.bind(model=IMPROVEMENT_MODEL, tags=tags)
    context = {
        "failed_task_ids": analyst_input.failed_task_ids,
        "taken_ids": analyst_input.taken_ids,
    }
    try:
        reply = await complete_structured(
            llm,
            analyst_input.messages(),
            AnalystAnswer,
            context=context,
            params=ModelParams(max_tokens=ANALYST_MAX_TOKENS),
            attempt_timeout_s=ANALYST_ATTEMPT_TIMEOUT_S,
        )
    except StructuredOutputError as error:
        cost = error.usage.cost_usd if error.usage.HasField("cost_usd") else None
        _logger.warning(
            "skipped the round's analysis: every reply was invalid",
            extra={
                "event": "analyst_skipped",
                "round_id": round_id,
                "call_ids": list(error.call_ids),
                "cost_usd": cost,
                "problems": error.errors,
            },
        )
        return AnalysisSkipped(
            f"{len(error.call_ids)} invalid replies: {error.errors}", error.call_ids, error.usage
        )
    answer = reply.value
    patterns = {pattern.name: pattern for pattern in answer.failure_patterns}
    hypotheses = tuple(
        _contract(proposed, patterns[proposed.target_pattern]) for proposed in answer.hypotheses
    )
    return Analysis(answer.failure_patterns, hypotheses, reply.call_ids, reply.usage)


def _contract(proposed: ProposedHypothesis, pattern: FailurePattern) -> Hypothesis:
    return Hypothesis(
        hypothesis_id=proposed.id,
        failure_pattern=f"{pattern.name}: {pattern.description}",
        target_instance_ids=pattern.task_ids,
        change_kind=_CHANGE_KINDS[proposed.change_kind],
        change=proposed.change,
        expected_effect=proposed.expected_effect,
        risk=proposed.risk,
        source=_SOURCES[proposed.source],
    )
