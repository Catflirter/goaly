"""Tests for goaly.rsi.usage: the lane's call log, summed by run ID and purpose."""

from pathlib import Path

from goaly.contracts.gen.llm_call_pb2 import CallPurpose, LlmCallRecord, Usage
from goaly.rsi.usage import CallTally

IMPL = CallPurpose.CALL_PURPOSE_RSI_IMPL
LESSON = CallPurpose.CALL_PURPOSE_RSI_LESSON


def call(run_id: str, purpose: CallPurpose.ValueType, tokens: int, cost: float) -> LlmCallRecord:
    return LlmCallRecord(
        run_id=run_id,
        purpose=purpose,
        usage=Usage(input_tokens=tokens, output_tokens=1, cost_usd=cost),
    )


def test_calls_are_logged_and_summed_by_run_and_purpose(tmp_path: Path) -> None:
    path = tmp_path / "llm_calls.jsonl"
    tally = CallTally(path)
    tally.record(call("e1", IMPL, 10, 0.1))
    tally.record(call("e1", IMPL, 5, 0.2))
    tally.record(call("e1", LESSON, 7, 0.3))
    tally.record(call("e2", IMPL, 1, 0.0))
    tally.record(call("", IMPL, 99, 9.0))  # No run: logged, but not summed.
    usage = tally.usage("e1")
    assert usage[IMPL].input_tokens == 15
    assert usage[IMPL].cost_usd == 0.1 + 0.2
    assert usage[LESSON].output_tokens == 1
    assert tally.usage("missing") == {}
    usage[IMPL].input_tokens = 0  # A copy: the tally keeps its sum.
    assert tally.usage("e1")[IMPL].input_tokens == 15
    # A restarted lane rebuilds the sums from the log.
    again = CallTally(path)
    assert again.usage("e1")[IMPL].input_tokens == 15
    assert len(path.read_text().splitlines()) == 5
