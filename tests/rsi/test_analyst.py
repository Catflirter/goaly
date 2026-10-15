"""Tests for goaly.rsi.analyst: the analyst's one structured call and the answer it must give.

GLM is FakeGlm behind core's real client. The input here is written by hand; the tests of
goaly.rsi.analyst_input build real ones.
"""

import json
import logging
from typing import Any

import pytest

from goaly.contracts.gen.experiment_pb2 import (
    CHANGE_KIND_POLICY,
    CHANGE_KIND_PROMPT,
    CHANGE_KIND_SKILL,
    HYPOTHESIS_SOURCE_ANALYST,
    HYPOTHESIS_SOURCE_IDEA_BANK,
    Hypothesis,
)
from goaly.contracts.gen.llm_call_pb2 import CallPurpose
from goaly.contracts.gen.trajectory_pb2 import ChatMessage, ChatRole
from goaly.rsi.analyst import Analysis, AnalysisSkipped, AnalystInput, analyze
from tests.rsi.fake_llm import COST_PER_CALL, FakeGlm

pytestmark = pytest.mark.anyio

ROUND = "round-2"
INPUT = AnalystInput(
    system="You analyze failures.",
    user="Tasks 101 to 104 failed.",
    failed_task_ids=frozenset({"101", "102", "103", "104"}),
    shown_task_ids=("103", "102", "104", "101"),
    dropped_task_ids=(),
    taken_ids=frozenset({"h-old", "h-queued"}),
)

# The call.


def hypothesis(hypothesis_id: str, pattern: str, kind: str, source: str) -> dict[str, str]:
    return {
        "id": hypothesis_id,
        "target_pattern": pattern,
        "change_kind": kind,
        "change": f"In prompt.md, change {hypothesis_id}.",
        "expected_effect": "Two more tasks reach tau >= 2.",
        "risk": "Slower easy tasks.",
        "source": source,
    }


def answer(**changes: Any) -> str:
    data: dict[str, Any] = {
        "failure_patterns": [
            {"name": "wrong rewrites", "task_ids": ["102", "103"], "description": "Rows differ."},
            {"name": "timid rewrites", "task_ids": ["101", "104"], "description": "No speedup."},
        ],
        "hypotheses": [
            hypothesis("h-r2-1", "wrong rewrites", "skill", "idea_bank"),
            hypothesis("h-r2-2", "timid rewrites", "prompt", "analysis"),
            hypothesis("h-r2-3", "timid rewrites", "policy", "analysis"),
        ],
    }
    data.update(changes)
    return json.dumps(data)


async def run(gateway: FakeGlm, analyst_input: AnalystInput = INPUT) -> Analysis | AnalysisSkipped:
    async with gateway.client() as client:
        return await analyze(client, analyst_input, round_id=ROUND)


async def test_a_valid_answer_becomes_hypotheses() -> None:
    gateway = FakeGlm(answer())
    analysis = await run(gateway)
    assert isinstance(analysis, Analysis)
    assert [pattern.name for pattern in analysis.patterns] == ["wrong rewrites", "timid rewrites"]
    first, second, third = analysis.hypotheses
    assert first == Hypothesis(
        hypothesis_id="h-r2-1",
        failure_pattern="wrong rewrites: Rows differ.",
        target_instance_ids=["102", "103"],
        change_kind=CHANGE_KIND_SKILL,
        change="In prompt.md, change h-r2-1.",
        expected_effect="Two more tasks reach tau >= 2.",
        risk="Slower easy tasks.",
        source=HYPOTHESIS_SOURCE_IDEA_BANK,
    )
    assert (second.change_kind, second.source) == (CHANGE_KIND_PROMPT, HYPOTHESIS_SOURCE_ANALYST)
    assert third.change_kind == CHANGE_KIND_POLICY
    assert list(third.target_instance_ids) == ["101", "104"]
    [record] = gateway.recorder.records
    assert (record.purpose, record.run_id) == (CallPurpose.CALL_PURPOSE_RSI_ANALYST, ROUND)
    assert analysis.call_ids == (record.call_id,)
    assert analysis.usage.cost_usd == COST_PER_CALL
    [body] = gateway.bodies
    assert body["model"] == "glm-5.3"
    assert "tools" not in body
    assert [message["role"] for message in body["messages"]] == ["system", "user"]


@pytest.mark.parametrize(
    ("invalid", "says"),
    [
        (answer(hypotheses=json.loads(answer())["hypotheses"][:2]), "at least 3"),
        (
            answer(
                failure_patterns=[
                    {"name": "wrong rewrites", "task_ids": ["77701"], "description": "x"},
                    {"name": "timid rewrites", "task_ids": ["105"], "description": "y"},
                ]
            ),
            "not failed train tasks of this input: 77701",
        ),
        (
            answer(hypotheses=[hypothesis("h-old", "wrong rewrites", "skill", "analysis")] * 3),
            "same id",
        ),
        (
            answer(
                hypotheses=[
                    hypothesis("h-old", "wrong rewrites", "skill", "analysis"),
                    hypothesis("h-2", "wrong rewrites", "skill", "analysis"),
                    hypothesis("h-3", "wrong rewrites", "skill", "analysis"),
                ]
            ),
            "already used in the ledger or queue",
        ),
        (
            answer(
                hypotheses=[
                    hypothesis("h-1", "wrong rewrites", "skill", "analysis"),
                    hypothesis("h-2", "nonexistent", "skill", "analysis"),
                    hypothesis("h-3", "wrong rewrites", "skill", "analysis"),
                ]
            ),
            "no failure pattern's name",
        ),
        (
            answer(hypotheses=[hypothesis("bad id!", "wrong rewrites", "tool", "analysis")] * 3),
            "letters, digits",
        ),
        (
            answer(hypotheses=[hypothesis("h-1", "wrong rewrites", "magic", "analysis")] * 3),
            "change_kind",
        ),
        (
            answer(
                failure_patterns=[
                    {"name": "wrong rewrites", "task_ids": ["102"], "description": "x"},
                    {"name": "wrong rewrites", "task_ids": ["103"], "description": "y"},
                ]
            ),
            "same name",
        ),
    ],
)
async def test_an_invalid_answer_is_retried_then_accepted(invalid: str, says: str) -> None:
    gateway = FakeGlm(invalid, answer())
    analysis = await run(gateway)
    assert isinstance(analysis, Analysis)
    assert len(analysis.call_ids) == 2
    assert analysis.usage.cost_usd == 2 * COST_PER_CALL
    assert says in gateway.bodies[1]["messages"][-1]["content"]


async def test_three_invalid_answers_skip_the_round(
    caplog: pytest.LogCaptureFixture,
) -> None:
    gateway = FakeGlm("no", "still no", '{"hypotheses": []}')
    with caplog.at_level(logging.WARNING, logger="goaly.rsi.analyst"):
        skipped = await run(gateway)
    assert isinstance(skipped, AnalysisSkipped)
    call_ids = tuple(record.call_id for record in gateway.recorder.records)
    assert skipped.call_ids == call_ids
    assert len(call_ids) == 3
    assert skipped.usage.cost_usd == 3 * COST_PER_CALL
    assert skipped.reason.startswith("3 invalid replies")
    [log] = [record for record in caplog.records if record.msg.startswith("skipped")]
    assert vars(log)["call_ids"] == list(call_ids)
    assert vars(log)["cost_usd"] == 3 * COST_PER_CALL
    assert vars(log)["round_id"] == ROUND


async def test_no_failure_means_no_call() -> None:
    nothing = AnalystInput("s", "u", frozenset(), (), (), frozenset())
    gateway = FakeGlm()
    skipped = await run(gateway, nothing)
    assert isinstance(skipped, AnalysisSkipped)
    assert skipped.call_ids == ()
    assert gateway.bodies == []


async def test_the_round_id_must_be_an_id() -> None:
    async with FakeGlm().client() as client:
        with pytest.raises(ValueError, match="round_id"):
            await analyze(client, INPUT, round_id="round 2")


def test_messages_are_a_system_and_a_user_message() -> None:
    analyst_input = INPUT
    assert analyst_input.messages() == [
        ChatMessage(role=ChatRole.CHAT_ROLE_SYSTEM, content=analyst_input.system),
        ChatMessage(role=ChatRole.CHAT_ROLE_USER, content=analyst_input.user),
    ]
