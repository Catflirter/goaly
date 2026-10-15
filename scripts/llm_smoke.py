"""Call Qwen and GLM through the real gateway, and print how each call went.

    uv run --locked python scripts/llm_smoke.py

A manual check, not run in CI, that the LLM client and the gateway still agree. The key comes
through load_settings(), as everywhere in goaly. It makes four calls:

1. Qwen, as the executor calls it, with the default request;
2. Qwen again with a tool, which it should call;
3. Qwen once more, sent back its reasoning and its tool call, then the tool's result;
4. GLM, through complete_sync from a worker thread, as RSI's implementer (T20) calls it.

For each call the script prints only the model, the status, the token counts and the cost:
never the key, and not the replies. Warnings, such as a model being throttled, go to stderr as
JSON lines.

Exit status: 0 if every call succeeded, 1 if one failed, one was refused before it began, or
the third call did not send Qwen back both its reasoning and its tool call, and 2 if the
settings are missing or invalid.
"""

import asyncio
import sys
from collections.abc import Awaitable

from goaly.contracts.gen.llm_call_pb2 import CallPurpose, LlmCallRecord, LlmCallStatus
from goaly.contracts.gen.trajectory_pb2 import ChatMessage, ChatRole
from goaly.contracts.interfaces import JsonValue
from goaly.harness.core.llm import CallTags, LlmClient, LlmReply, MemoryCallRecorder
from goaly.harness.core.llm_errors import LlmError
from goaly.harness.core.llm_models import EXECUTOR_MODEL, IMPROVEMENT_MODEL
from goaly.harness.core.logs import configure_logging
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


async def smoke(llm: LlmClient) -> list[str]:
    """Make the calls, and return what kept them from showing that the client and gateway agree.

    A failed call does not stop the others, and is not in the list: its record tells of it.
    """
    problems: list[str] = []
    exec_tags = CallTags(CallPurpose.CALL_PURPOSE_EXEC, run_id=RUN_ID)
    await _call(llm.complete(QUESTION, model=EXECUTOR_MODEL, tags=exec_tags), problems)
    await _tool_turns(llm, exec_tags, problems)
    impl_tags = CallTags(CallPurpose.CALL_PURPOSE_RSI_IMPL, run_id=RUN_ID)
    worker = asyncio.to_thread(llm.complete_sync, QUESTION, model=IMPROVEMENT_MODEL, tags=impl_tags)
    await _call(worker, problems)
    return problems


async def _tool_turns(llm: LlmClient, tags: CallTags, problems: list[str]) -> None:
    """Have Qwen call the tool, then send it its own turn back with the tool's result."""
    asking = llm.complete(TOOL_TASK, model=EXECUTOR_MODEL, tags=tags, tools=TOOLS)
    first = await _call(asking, problems)
    if first is None:
        return
    if not first.message.tool_calls:
        problems.append("Qwen called no tool, so its second tool turn did not run")
        return
    if not first.message.reasoning_content:  # The client sends none back for an empty one.
        problems.append("Qwen returned no reasoning_content, so none was replayed")
    results = [
        ChatMessage(role=ChatRole.CHAT_ROLE_TOOL, tool_call_id=call.id, content=TOOL_RESULT)
        for call in first.message.tool_calls
    ]
    history = [*TOOL_TASK, first.message, *results]  # Its reasoning goes back as it came.
    await _call(llm.complete(history, model=EXECUTOR_MODEL, tags=tags, tools=TOOLS), problems)


async def _call(call: Awaitable[LlmReply], problems: list[str]) -> LlmReply | None:
    """Return the call's reply, or None if it failed.

    A call refused before it began writes no record, so it is added to ``problems``.
    """
    try:
        return await call
    except LlmError as error:
        if error.record is None:  # ModelNotAllowed: the client sent nothing.
            problems.append(f"a call was refused before it began: {error}")
    except ValueError as error:  # A message the client cannot put on the wire.
        problems.append(f"a call was refused before it began: {error}")
    return None


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
    """Run the calls against the gateway in the settings, and return the exit status above."""
    configure_logging("WARNING", stream=sys.stderr)
    try:
        settings = load_settings()
    except SettingsError as error:  # Its message says what to set, and never quotes a value.
        sys.stderr.write(f"llm_smoke: {error}\n")
        return 2
    records = MemoryCallRecorder()
    async with LlmClient(settings, records) as llm:
        problems = await smoke(llm)
    for record in records.records:
        sys.stdout.write(describe(record) + "\n")
    for problem in problems:
        sys.stderr.write(f"llm_smoke: {problem}\n")
    failed = any(record.status != LlmCallStatus.LLM_CALL_STATUS_OK for record in records.records)
    return 0 if len(records.records) == CALLS and not failed and not problems else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
