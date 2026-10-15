"""A fake GLM behind core's real LlmClient, for goaly.rsi's structured calls.

FakeGlm is an httpx transport handler: it answers each chat completion with the next scripted
reply, shaped as the gateway sends GLM's, and keeps every request body, so a test can check what
the client sent, such as the reasoning it replayed. No test reaches the network.
"""

import json
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import SecretStr

from goaly.contracts.gen.llm_call_pb2 import LlmCallRecord
from goaly.harness.core.llm import LlmClient
from goaly.harness.core.llm_models import IMPROVEMENT_MODEL
from goaly.harness.core.settings import Settings

COST_PER_CALL = 0.25


@dataclass(frozen=True)
class GlmReply:
    """One scripted reply: its content, its reasoning and why the model stopped."""

    content: str
    reasoning: str = "Let me think about the failures."
    finish_reason: str = "stop"


class Recorder:
    """A CallRecorder that keeps the records it gets."""

    def __init__(self) -> None:
        self.records: list[LlmCallRecord] = []

    def record(self, record: LlmCallRecord) -> None:
        self.records.append(record)


class FakeGlm:
    """A gateway that answers with its scripted GLM replies, in order."""

    def __init__(self, *replies: GlmReply | str) -> None:
        self.replies = [GlmReply(reply) if isinstance(reply, str) else reply for reply in replies]
        self.bodies: list[dict[str, Any]] = []
        self.recorder = Recorder()

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        reply = self.replies.pop(0)
        body = {
            "id": f"chatcmpl-{len(self.bodies)}",
            "object": "chat.completion",
            "model": IMPROVEMENT_MODEL,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": reply.finish_reason,
                    "message": {
                        "role": "assistant",
                        "content": reply.content,
                        "reasoning_content": reply.reasoning,
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 200,
                "completion_tokens_details": {"reasoning_tokens": 150},
            },
        }
        headers = {"x-litellm-response-cost": str(COST_PER_CALL)}
        return httpx.Response(200, json=body, headers=headers)

    def client(self) -> LlmClient:
        """A real LlmClient whose requests this gateway answers."""
        settings = Settings(api_key=SecretStr("sk-test-never-sent"), base_url="https://gw.test/v1")
        return LlmClient(settings, self.recorder, transport=httpx.MockTransport(self.handle))
