"""Tests for goaly.rsi.structured: JSON answers validated by pydantic, retried in the same chat."""

import json
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, SecretStr, ValidationInfo, field_validator

from goaly.contracts.gen.llm_call_pb2 import CallPurpose, Usage
from goaly.contracts.gen.trajectory_pb2 import ChatMessage, ChatRole
from goaly.harness.core.llm import CallTags, LlmClient
from goaly.harness.core.llm_errors import ClientError
from goaly.harness.core.llm_models import IMPROVEMENT_MODEL, ModelParams
from goaly.harness.core.settings import Settings
from goaly.rsi.structured import (
    StructuredOutputError,
    add_usage,
    complete_structured,
    format_instructions,
)
from tests.rsi.fake_llm import COST_PER_CALL, FakeGlm, GlmReply, Recorder

pytestmark = pytest.mark.anyio

LESSON = CallTags(CallPurpose.CALL_PURPOSE_RSI_LESSON, run_id="round-3")
SYSTEM = ChatMessage(role=ChatRole.CHAT_ROLE_SYSTEM, content="You write lessons.")
USER = ChatMessage(role=ChatRole.CHAT_ROLE_USER, content="Write one lesson.")


class Answer(BaseModel):
    text: str
    count: int

    @field_validator("count")
    @classmethod
    def _below_limit(cls, count: int, info: ValidationInfo) -> int:
        limit = (info.context or {}).get("limit")
        if limit is not None and count > limit:
            raise ValueError(f"count must be at most {limit}")
        return count


VALID = json.dumps({"text": "Push predicates down.", "count": 2})


async def ask(gateway: FakeGlm, **options: Any) -> Any:
    async with gateway.client() as client:
        llm = client.bind(model=IMPROVEMENT_MODEL, tags=LESSON)
        return await complete_structured(llm, [SYSTEM, USER], Answer, **options)


async def test_a_valid_reply_is_returned_after_one_call() -> None:
    gateway = FakeGlm(VALID)
    reply = await ask(gateway)
    assert reply.value == Answer(text="Push predicates down.", count=2)
    assert reply.call_ids == (gateway.recorder.records[0].call_id,)
    assert reply.usage.cost_usd == COST_PER_CALL
    assert reply.usage.input_tokens == 1000
    [body] = gateway.bodies
    assert "response_format" not in body
    assert body["messages"][0] == {"role": "system", "content": "You write lessons."}
    last = body["messages"][1]["content"]
    assert last == f"Write one lesson.\n\n{format_instructions(Answer)}"
    assert '"count"' in last  # The JSON Schema is in the prompt.
    assert USER.content == "Write one lesson."  # The caller's message is left as it was.


async def test_calls_carry_the_bound_purpose_and_run() -> None:
    gateway = FakeGlm(VALID)
    await ask(gateway)
    [record] = gateway.recorder.records
    assert record.purpose == CallPurpose.CALL_PURPOSE_RSI_LESSON
    assert record.run_id == "round-3"
    assert record.model == IMPROVEMENT_MODEL


@pytest.mark.parametrize(
    "content",
    [
        f"```json\n{VALID}\n```",
        f"Here is the answer:\n```\n{VALID}\n```\nDone.",
        f"Sure. {VALID} Hope that helps.",
        f"The rewrite:\n```sql\nSELECT 1\n```\nAnswer:\n```json\n{VALID}\n```",
        f"```sql\nSELECT 1\n```\nThen:\n```\n{VALID}\n```",
        f"```JSON\n{VALID}\n```",
    ],
)
async def test_json_is_read_from_a_fence_or_from_the_text(content: str) -> None:
    reply = await ask(FakeGlm(content))
    assert reply.value.count == 2


async def test_an_invalid_reply_is_retried_in_the_same_chat_with_its_reasoning() -> None:
    first = GlmReply('{"text": "x", "count": "many"}', reasoning="  Exact <reasoning>\n kept. ")
    gateway = FakeGlm(first, VALID)
    reply = await ask(gateway)
    assert reply.value.count == 2
    assert len(reply.call_ids) == 2
    assert reply.call_ids == tuple(record.call_id for record in gateway.recorder.records)
    assert reply.usage.cost_usd == 2 * COST_PER_CALL
    assert reply.usage.reasoning_tokens == 300
    retry = gateway.bodies[1]["messages"]
    assert retry[:2] == gateway.bodies[0]["messages"]
    assert retry[2] == {
        "role": "assistant",
        "content": first.content,
        "reasoning_content": "  Exact <reasoning>\n kept. ",
    }
    assert retry[3]["role"] == "user"
    assert "count" in retry[3]["content"]
    assert "valid integer" in retry[3]["content"]
    assert len(retry) == 4


async def test_three_invalid_replies_raise_with_every_call() -> None:
    gateway = FakeGlm("not json", '{"text": "x"}', "{}")
    with pytest.raises(StructuredOutputError) as caught:
        await ask(gateway)
    error = caught.value
    assert error.call_ids == tuple(record.call_id for record in gateway.recorder.records)
    assert len(error.call_ids) == 3
    assert error.usage.cost_usd == 3 * COST_PER_CALL
    assert "text" in error.errors
    assert "count" in error.errors
    assert len(gateway.bodies[2]["messages"]) == 6  # Each retry continues the same chat.


async def test_max_attempts_bounds_the_calls() -> None:
    gateway = FakeGlm("nope")
    with pytest.raises(StructuredOutputError):
        await ask(gateway, max_attempts=1)
    assert len(gateway.bodies) == 1


@pytest.mark.parametrize(
    ("reply", "says"),
    [
        (GlmReply(VALID[:10], finish_reason="length"), "cut off"),
        (GlmReply(VALID, finish_reason="length"), "cut off"),  # Never trust a cut-off answer.
        (GlmReply(""), "empty"),
    ],
)
async def test_a_cut_off_or_empty_reply_is_retried(reply: GlmReply, says: str) -> None:
    gateway = FakeGlm(reply, VALID)
    await ask(gateway)
    assert says in gateway.bodies[1]["messages"][-1]["content"]


async def test_a_cut_off_reply_is_retried_with_its_reasoning_replayed() -> None:
    cut = GlmReply("", reasoning="Thinking until the tokens ran out", finish_reason="length")
    gateway = FakeGlm(cut, VALID)
    reply = await ask(gateway)
    assert reply.value.count == 2
    assert len(reply.call_ids) == 2
    retry = gateway.bodies[1]["messages"]
    assert retry[2] == {
        "role": "assistant",
        "content": "",
        "reasoning_content": "Thinking until the tokens ran out",
    }
    assert "cut off at the token limit" in retry[3]["content"]


async def test_an_invalid_reply_quotes_the_likeliest_candidates_problems() -> None:
    content = '```sql\nSELECT 1\n```\n```json\n{"text": "x", "count": "many"}\n```'
    gateway = FakeGlm(content, VALID)
    await ask(gateway)
    feedback = gateway.bodies[1]["messages"][-1]["content"]
    assert "count: Input should be a valid integer" in feedback


async def test_the_context_reaches_the_validators() -> None:
    gateway = FakeGlm(VALID, json.dumps({"text": "y", "count": 1}))
    reply = await ask(gateway, context={"limit": 1})
    assert reply.value.count == 1
    assert "at most 1" in gateway.bodies[1]["messages"][-1]["content"]


async def test_params_are_sent_with_every_attempt() -> None:
    gateway = FakeGlm("nope", VALID)
    await ask(gateway, params=ModelParams(max_tokens=777))
    assert [body["max_tokens"] for body in gateway.bodies] == [777, 777]


async def test_many_problems_are_quoted_in_part() -> None:
    class Wide(BaseModel):
        values: list[int]

    gateway = FakeGlm(json.dumps({"values": ["x" * 300] * 25}), json.dumps({"values": [1]}))
    async with gateway.client() as client:
        llm = client.bind(model=IMPROVEMENT_MODEL, tags=LESSON)
        await complete_structured(llm, [USER], Wide)
    feedback = gateway.bodies[1]["messages"][-1]["content"]
    assert "and 5 more" in feedback
    assert "x" * 201 not in feedback


async def test_a_gateway_failure_raises_its_llm_error() -> None:
    async def refuse(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "bad parameter"}})

    settings = Settings(api_key=SecretStr("sk-test-never-sent"), base_url="https://gw.test/v1")
    async with LlmClient(settings, Recorder(), transport=httpx.MockTransport(refuse)) as client:
        llm = client.bind(model=IMPROVEMENT_MODEL, tags=LESSON)
        with pytest.raises(ClientError):
            await complete_structured(llm, [USER], Answer)


@pytest.mark.parametrize("messages", [[], [SYSTEM]])
async def test_the_chat_must_end_with_a_user_message(messages: list[ChatMessage]) -> None:
    gateway = FakeGlm()
    async with gateway.client() as client:
        llm = client.bind(model=IMPROVEMENT_MODEL, tags=LESSON)
        with pytest.raises(ValueError, match="end with a user message"):
            await complete_structured(llm, messages, Answer)
        with pytest.raises(ValueError, match="at least 1"):
            await complete_structured(llm, [USER], Answer, max_attempts=0)
    assert gateway.bodies == []


def test_add_usage_sets_optional_fields_only_when_reported() -> None:
    total = Usage(input_tokens=1, output_tokens=2)
    add_usage(total, Usage(input_tokens=10, output_tokens=20))
    assert total == Usage(input_tokens=11, output_tokens=22)
    assert not total.HasField("cost_usd")
    assert not total.HasField("reasoning_tokens")
    add_usage(total, Usage(reasoning_tokens=5, cost_usd=0.5))
    add_usage(total, Usage(reasoning_tokens=0, cost_usd=0.0))
    assert total.reasoning_tokens == 5
    assert total.cost_usd == 0.5
