"""Tests for scripts/llm_smoke.py, the manual check of the LLM client against the real gateway.

No test reaches the real gateway. The scripted gateway stands in for it, and so does a local
HTTP server, which also takes the client through httpx's real transport.
"""

import json
import logging
import threading
from collections.abc import Iterator, Set
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
import pytest

from goaly.contracts.gen.llm_call_pb2 import CallPurpose, LlmCallRecord, LlmCallStatus, Usage
from goaly.harness.core.llm import MemoryCallRecorder
from scripts import llm_smoke
from scripts.llm_smoke import TOOLS, describe, main, smoke
from tests.harness.core.llm_fakes import GLM, KEY, QWEN, Gateway, completion, connect, failure

pytestmark = pytest.mark.anyio

OK = LlmCallStatus.LLM_CALL_STATUS_OK
CLIENT_ERROR = LlmCallStatus.LLM_CALL_STATUS_CLIENT_ERROR
EXEC = CallPurpose.CALL_PURPOSE_EXEC
RSI_IMPL = CallPurpose.CALL_PURPOSE_RSI_IMPL
REPLY = "a reply the script must not print"
REASONING = "The tool counts rows: call it.\n"
ARGUMENTS = '{"sql": "SELECT count(*) FROM orders"}'
LINE = "tokens 26 in, 309 out (304 reasoning); cost $0.000313"

type Seen = list[tuple[str, str, str]]  # Each request's path, Authorization header and model.


def tool_call(reasoning: str = REASONING) -> httpx.Response:
    """Qwen's first tool turn, as the gateway answers it: its reasoning and one tool call."""
    call = {
        "index": 0,
        "function": {"arguments": ARGUMENTS, "name": "execute_sql"},
        "id": "call_1",
        "type": "function",
    }
    return completion("\n\n", reasoning=reasoning, tool_calls=[call], finish_reason="tool_calls")


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    """main() configures logging for the process; undo it, so later tests log as before."""
    root = logging.getLogger()
    level, handlers = root.level, list(root.handlers)
    yield
    for handler in list(root.handlers):
        if handler not in handlers:
            root.removeHandler(handler)
    root.setLevel(level)


@contextmanager
def stand_in_gateway(
    failing: Set[str] = frozenset(),
    *,
    calls_tools: bool = True,
    reasons: bool = True,
    cost: str | None = None,
) -> Iterator[tuple[str, Seen]]:
    """Serve chat completions on a local port, answering 400 for the ``failing`` models.

    A request with tools whose last message is the user's gets a tool call, if ``calls_tools``,
    with Qwen's reasoning, if ``reasons``. Every reply says it cost ``cost``, if given.
    Yields the base URL and the list of requests it saw.
    """
    seen: Seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, self.headers["Authorization"], body["model"]))
            if body["model"] in failing:
                reply = failure(400, "refused")
            elif calls_tools and "tools" in body and body["messages"][-1]["role"] == "user":
                reply = tool_call(REASONING if reasons else "")
            else:
                reply = completion(REPLY)
            self.send_response(reply.status_code)
            self.send_header("content-type", reply.headers["content-type"])
            own_cost = reply.headers["x-litellm-response-cost"]
            self.send_header("x-litellm-response-cost", own_cost if cost is None else cost)
            self.send_header("content-length", str(len(reply.content)))
            self.end_headers()
            self.wfile.write(reply.content)

        def log_message(self, format: str, *args: object) -> None:
            """Keep the test output clean."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


async def test_calls_qwen_plainly_then_through_a_tool_turn_then_glm_from_a_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = Gateway(completion(), tool_call(), completion(), completion())
    records = MemoryCallRecorder()
    async with connect(gateway, recorder=records) as llm:
        callers: list[int] = []
        complete_sync = llm.complete_sync

        def spy(*args: Any, **kwargs: Any) -> Any:
            callers.append(threading.get_ident())
            return complete_sync(*args, **kwargs)

        monkeypatch.setattr(llm, "complete_sync", spy)
        problems = await smoke(llm)

    assert problems == []
    assert callers  # GLM's call went through complete_sync,
    assert threading.get_ident() not in callers  # from a worker thread.
    plain, first, second, glm = gateway.bodies
    assert [body["model"] for body in (plain, first, second, glm)] == [QWEN, QWEN, QWEN, GLM]
    assert "tools" not in plain
    # The whole default Qwen body: the profile's own sampling and thinking switch, no overrides.
    assert {key: value for key, value in first.items() if key != "messages"} == {
        "model": QWEN,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": 16384,
        "chat_template_kwargs": {"enable_thinking": True},
        "tools": TOOLS,
    }
    assert second["tools"] == TOOLS
    # The second turn: Qwen's reasoning and tool call replayed, then the tool's result.
    assert second["messages"][1:] == [
        {
            "role": "assistant",
            "content": "\n\n",
            "reasoning_content": REASONING,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "execute_sql", "arguments": ARGUMENTS},
                }
            ],
        },
        {"role": "tool", "content": "42", "tool_call_id": "call_1"},
    ]
    assert [(record.model, record.purpose, record.run_id) for record in records.records] == [
        (QWEN, EXEC, "llm-smoke"),
        (QWEN, EXEC, "llm-smoke"),
        (QWEN, EXEC, "llm-smoke"),
        (GLM, RSI_IMPL, "llm-smoke"),
    ]


async def test_reports_a_tool_turn_with_no_tool_call() -> None:
    gateway = Gateway(completion(), completion(), completion())
    async with connect(gateway) as llm:
        problems = await smoke(llm)

    assert problems == ["Qwen called no tool, so its second tool turn did not run"]
    assert [body["model"] for body in gateway.bodies] == [QWEN, QWEN, GLM]


async def test_reports_a_tool_turn_with_no_reasoning_to_replay() -> None:
    gateway = Gateway(completion(), tool_call(reasoning=""), completion(), completion())
    async with connect(gateway) as llm:
        problems = await smoke(llm)

    assert problems == ["Qwen returned no reasoning_content, so none was replayed"]
    assert len(gateway.requests) == 4  # The tool turn still ran: only the replay is missing.


async def test_reports_a_call_refused_before_it_began(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_smoke, "IMPROVEMENT_MODEL", QWEN)  # RSI may not call Qwen.
    gateway, records = Gateway(completion(), tool_call(), completion()), MemoryCallRecorder()
    async with connect(gateway, recorder=records) as llm:
        problems = await smoke(llm)

    assert problems == [
        "a call was refused before it began: "
        f"CALL_PURPOSE_RSI_IMPL may not call {QWEN}: it may call only {GLM}"
    ]
    assert len(records.records) == 3


async def test_a_failed_call_does_not_stop_the_others() -> None:
    gateway = Gateway(failure(400), tool_call(), failure(400), completion())
    records = MemoryCallRecorder()
    async with connect(gateway, recorder=records) as llm:
        problems = await smoke(llm)

    assert problems == []  # The failures are in the records.
    assert [record.status for record in records.records] == [CLIENT_ERROR, OK, CLIENT_ERROR, OK]


async def test_a_failed_first_tool_turn_has_no_second() -> None:
    gateway, records = Gateway(completion(), failure(400), completion()), MemoryCallRecorder()
    async with connect(gateway, recorder=records) as llm:
        problems = await smoke(llm)

    assert problems == []
    assert [record.status for record in records.records] == [OK, CLIENT_ERROR, OK]


def test_describes_a_call_by_model_status_tokens_and_cost() -> None:
    usage = Usage(input_tokens=26, output_tokens=309, reasoning_tokens=304, cost_usd=0.0003129)
    record = LlmCallRecord(model=QWEN, status=OK, usage=usage)

    assert describe(record) == f"qwen3.6-35b-a3b: OK; {LINE}"


def test_says_what_the_gateway_did_not_report() -> None:
    record = LlmCallRecord(model=GLM, status=CLIENT_ERROR)

    assert describe(record) == (
        "glm-5.3: CLIENT_ERROR; tokens 0 in, 0 out (reasoning not reported); cost not reported"
    )


async def test_prints_one_line_per_call_and_exits_0_when_all_succeed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with stand_in_gateway() as (base_url, seen):
        monkeypatch.setenv("GOALY_API_KEY", KEY)
        monkeypatch.setenv("GOALY_BASE_URL", base_url)
        status = await main()

    printed = capsys.readouterr()
    assert status == 0
    assert seen == [
        ("/v1/chat/completions", f"Bearer {KEY}", model) for model in [QWEN] * 3 + [GLM]
    ]
    assert printed.out.splitlines() == [f"{model}: OK; {LINE}" for model in [QWEN] * 3 + [GLM]]
    assert printed.err == ""
    assert KEY not in printed.out
    assert REPLY not in printed.out
    assert REASONING.strip() not in printed.out


async def test_exits_1_when_a_call_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with stand_in_gateway(failing={GLM}) as (base_url, _):
        monkeypatch.setenv("GOALY_API_KEY", KEY)
        monkeypatch.setenv("GOALY_BASE_URL", base_url)
        status = await main()

    printed = capsys.readouterr()
    assert status == 1
    statuses = [line.split(";")[0] for line in printed.out.splitlines()]
    assert statuses == [f"{QWEN}: OK"] * 3 + [f"{GLM}: CLIENT_ERROR"]
    assert printed.err == ""


@pytest.mark.parametrize(
    ("calls_tools", "reasons", "note", "lines"),
    [
        pytest.param(
            False,
            True,
            "Qwen called no tool, so its second tool turn did not run",
            3,
            id="no-tool-call",
        ),
        pytest.param(
            True,
            False,
            "Qwen returned no reasoning_content, so none was replayed",
            4,
            id="no-reasoning",
        ),
    ],
)
async def test_exits_1_and_says_why_when_the_replay_was_not_proved(
    calls_tools: bool,
    reasons: bool,
    note: str,
    lines: int,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with stand_in_gateway(calls_tools=calls_tools, reasons=reasons) as (base_url, _):
        monkeypatch.setenv("GOALY_API_KEY", KEY)
        monkeypatch.setenv("GOALY_BASE_URL", base_url)
        status = await main()

    printed = capsys.readouterr()
    assert status == 1
    assert len(printed.out.splitlines()) == lines
    assert printed.err == f"llm_smoke: {note}\n"


async def test_logs_warnings_to_stderr_as_json_lines(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with stand_in_gateway(cost="n/a") as (base_url, _):
        monkeypatch.setenv("GOALY_API_KEY", KEY)
        monkeypatch.setenv("GOALY_BASE_URL", base_url)
        status = await main()

    printed = capsys.readouterr()
    assert status == 0  # A cost the gateway did not report fails no call.
    warnings = [json.loads(line) for line in printed.err.splitlines()]
    assert [(entry["level"], entry["event"]) for entry in warnings] == [
        ("WARNING", "llm_bad_cost")
    ] * 4


async def test_without_a_key_says_what_to_set_and_exits_2(
    capsys: pytest.CaptureFixture[str],
) -> None:
    status = await main()  # The conftest leaves no GOALY_* variable and an empty HOME.

    printed = capsys.readouterr()
    assert status == 2
    assert printed.out == ""
    assert printed.err.startswith("llm_smoke: GOALY_API_KEY is not set.")
