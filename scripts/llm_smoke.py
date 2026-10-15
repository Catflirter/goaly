"""Call Qwen once and GLM once through the real gateway, and print how each call went.

    uv run --locked python scripts/llm_smoke.py

A manual check, not run in CI, that the LLM client and the gateway still agree. The key comes
through load_settings(), as everywhere in goaly. For each call the script prints only the
model, the status, the token counts and the cost: never the key, and not the reply. Qwen is
called as the executor calls it. GLM is called through complete_sync from a worker thread, as
RSI's implementer (T20) calls it.

Exit status: 0 if both calls succeeded, 1 if either failed, and 2 if the settings are missing
or invalid.
"""

import asyncio
import contextlib
import sys

from goaly.contracts.gen.llm_call_pb2 import CallPurpose, LlmCallRecord, LlmCallStatus
from goaly.contracts.gen.trajectory_pb2 import ChatMessage, ChatRole
from goaly.harness.core.llm import CallTags, LlmClient
from goaly.harness.core.llm_errors import LlmError
from goaly.harness.core.llm_models import EXECUTOR_MODEL, IMPROVEMENT_MODEL
from goaly.harness.core.settings import SettingsError, load_settings

RUN_ID = "llm-smoke"
QUESTION = [
    ChatMessage(
        role=ChatRole.CHAT_ROLE_USER, content="What is 17 * 23? Answer with the number only."
    )
]


class Records:
    """A CallRecorder that keeps the records it gets."""

    def __init__(self) -> None:
        self.records: list[LlmCallRecord] = []

    def record(self, record: LlmCallRecord) -> None:
        self.records.append(record)


async def smoke(llm: LlmClient) -> None:
    """Make the two calls. A failure does not stop the other call: its record tells of it."""
    exec_tags = CallTags(CallPurpose.CALL_PURPOSE_EXEC, run_id=RUN_ID)
    with contextlib.suppress(LlmError):
        await llm.complete(QUESTION, model=EXECUTOR_MODEL, tags=exec_tags)
    impl_tags = CallTags(CallPurpose.CALL_PURPOSE_RSI_IMPL, run_id=RUN_ID)
    with contextlib.suppress(LlmError):
        await asyncio.to_thread(
            llm.complete_sync, QUESTION, model=IMPROVEMENT_MODEL, tags=impl_tags
        )


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
        await smoke(llm)
    for record in records.records:
        sys.stdout.write(describe(record) + "\n")
    succeeded = [record.status == LlmCallStatus.LLM_CALL_STATUS_OK for record in records.records]
    return 0 if len(succeeded) == 2 and all(succeeded) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
