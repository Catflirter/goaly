"""Call Qwen and GLM through the real gateway, and print how each call went.

    uv run --locked python scripts/llm_smoke.py

A manual check, not run in CI, that the LLM client and the gateway still agree. The key comes
through load_settings(), as everywhere in goaly. It makes four calls:

1. Qwen, as the executor calls it, with the default request;
2. Qwen again with a tool, which it should call;
3. Qwen once more, sent back its reasoning and its tool call, then the tool's result;
4. GLM, through complete_sync from a worker thread, as RSI's implementer (T20) calls it.

For each call the script prints only the model, the status, the token counts and the cost:
never the key, and not the replies.

Exit status: 0 if every call succeeded, 1 if one failed or Qwen called no tool, and 2 if the
settings are missing or invalid.
"""

import asyncio
import contextlib
import sys

from goaly.contracts.gen.llm_call_pb2 import CallPurpose, LlmCallRecord, LlmCallStatus
from goaly.contracts.gen.trajectory_pb2 import ChatMessage, ChatRole
from goaly.contracts.interfaces import JsonValue
from goaly.harness.core.llm import CallTags, LlmClient
from goaly.harness.core.llm_errors import LlmError
from goaly.harness.core.llm_models import EXECUTOR_MODEL, IMPROVEMENT_MODEL
from goaly.harness.core.settings import SettingsError, load_settings

RUN_ID = "llm-smoke"
CALLS = 4
QUESTION = [
    ChatMessage(
        role=ChatRole.CHAT_ROLE_USER, content="What is 17 * 23? Answer with the number only."
    )
]
TOOL_TASK = [
    ChatMessage(
        role=ChatRole.CHAT_ROLE_USER,
        content="Use the execute_sql tool to count the rows of the orders table.",
    )
]
TOOLS: list[dict[str, JsonValue]] = [
    {
        "type": "function",
        "function": {
            "name": "execute_sql",
            "description": "Run a SQL query and return its rows.",
            "parameters": {
                "type": "object",
                "properties": {"sql": {"type": "string"}},
                "required": ["sql"],
            },
        },
    }
]
TOOL_RESULT = "42"


class Records:
    """A CallRecorder that keeps the records it gets."""

    def __init__(self) -> None:
        self.records: list[LlmCallRecord] = []

    def record(self, record: LlmCallRecord) -> None:
        self.records.append(record)


async def smoke(llm: LlmClient) -> bool:
    """Make the calls, and return whether Qwen called the tool, so that its second turn ran.

    A failed call does not stop the others: its record tells of it.
    """
    exec_tags = CallTags(CallPurpose.CALL_PURPOSE_EXEC, run_id=RUN_ID)
    with contextlib.suppress(LlmError):
        await llm.complete(QUESTION, model=EXECUTOR_MODEL, tags=exec_tags)
    tool_turn_ran = await _tool_turns(llm, exec_tags)
    impl_tags = CallTags(CallPurpose.CALL_PURPOSE_RSI_IMPL, run_id=RUN_ID)
    with contextlib.suppress(LlmError):
        await asyncio.to_thread(
            llm.complete_sync, QUESTION, model=IMPROVEMENT_MODEL, tags=impl_tags
        )
    return tool_turn_ran


async def _tool_turns(llm: LlmClient, tags: CallTags) -> bool:
    """Have Qwen call the tool, then send it its own turn back with the tool's result."""
    try:
        first = await llm.complete(TOOL_TASK, model=EXECUTOR_MODEL, tags=tags, tools=TOOLS)
    except LlmError:
        return False
    if not first.message.tool_calls:
        return False
    results = [
        ChatMessage(role=ChatRole.CHAT_ROLE_TOOL, tool_call_id=call.id, content=TOOL_RESULT)
        for call in first.message.tool_calls
    ]
    history = [*TOOL_TASK, first.message, *results]  # Its reasoning goes back as it came.
    with contextlib.suppress(LlmError):
        await llm.complete(history, model=EXECUTOR_MODEL, tags=tags, tools=TOOLS)
    return True


def describe(record: LlmCallRecord) -> str:
    """Return one line about a call: its model, status, token counts and cost, nothing else."""
    usage = record.usage
    status = LlmCallStatus.Name(record.status).removeprefix("LLM_CALL_STATUS_")
    reasoning = "reasoning not reported"
    if usage.HasField("reasoning_tokens"):
        reasoning = f"{usage.reasoning_tokens} reasoning"
    cost = f"${usage.cost_usd:.6f}" if usage.HasField("cost_usd") else "not reported"
    tokens = f"{usage.input_tokens} in, {usage.output_tokens} out ({reasoning})"
    return f"{record.model}: {status}; tokens {tokens}; cost {cost}"


async def main() -> int:
    try:
        settings = load_settings()
    except SettingsError as error:  # Its message says what to set, and never quotes a value.
        sys.stderr.write(f"llm_smoke: {error}\n")
        return 2
    records = Records()
    async with LlmClient(settings, records) as llm:
        tool_turn_ran = await smoke(llm)
    for record in records.records:
        sys.stdout.write(describe(record) + "\n")
    failed = any(record.status != LlmCallStatus.LLM_CALL_STATUS_OK for record in records.records)
    if not failed and not tool_turn_ran:
        sys.stderr.write("llm_smoke: Qwen called no tool, so its second tool turn did not run\n")
    return 0 if len(records.records) == CALLS and not failed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
