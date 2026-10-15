"""Tests for scripts/llm_smoke.py, the manual check of the LLM client against the real gateway.

No test reaches the real gateway. The scripted gateway stands in for it, and so does a local
HTTP server, which also takes the client through httpx's real transport.
"""

import json
import threading
from collections.abc import Iterator, Set
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
import pytest

from goaly.contracts.gen.llm_call_pb2 import CallPurpose, LlmCallRecord, LlmCallStatus, Usage
from scripts.llm_smoke import TOOLS, Records, describe, main, smoke
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


def tool_call() -> httpx.Response:
    """Qwen's first tool turn, as the gateway answers it: its reasoning and one tool call."""
    call = {
        "index": 0,
        "function": {"arguments": ARGUMENTS, "name": "execute_sql"},
        "id": "call_1",
        "type": "function",
    }
    return completion("\n\n", reasoning=REASONING, tool_calls=[call], finish_reason="tool_calls")


@contextmanager
def stand_in_gateway(
    failing: Set[str] = frozenset(), *, calls_tools: bool = True
) -> Iterator[tuple[str, Seen]]:
    """Serve chat completions on a local port, answering 400 for the ``failing`` models.

    A request with tools whose last message is the user's gets a tool call, if ``calls_tools``.
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
                reply = tool_call()
            else:
                reply = completion(REPLY)
            self.send_response(reply.status_code)
            for name in ("content-type", "x-litellm-response-cost"):
                self.send_header(name, reply.headers[name])
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


async def test_calls_qwen_plainly_then_through_a_tool_turn_then_glm_from_a_thread() -> None:
    gateway, records = Gateway(completion(), tool_call(), completion(), completion()), Records()
    async with connect(gateway, recorder=records) as llm:
        tool_turn_ran = await smoke(llm)

    assert tool_turn_ran
    plain, first, second, glm = gateway.bodies
    assert [body["model"] for body in (plain, first, second, glm)] == [QWEN, QWEN, QWEN, GLM]
    assert "tools" not in plain
    assert first["tools"] == second["tools"] == TOOLS
    # The default Qwen body: the profile's own sampling and thinking switch, no overrides.
    defaults: dict[str, Any] = {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": 16384,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    assert {key: first[key] for key in defaults} == defaults
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


async def test_skips_the_second_tool_turn_if_qwen_calls_no_tool() -> None:
    gateway, records = Gateway(completion(), completion(), completion()), Records()
    async with connect(gateway, recorder=records) as llm:
        tool_turn_ran = await smoke(llm)

    assert not tool_turn_ran
    assert [body["model"] for body in gateway.bodies] == [QWEN, QWEN, GLM]


async def test_a_failed_call_does_not_stop_the_others() -> None:
    gateway = Gateway(failure(400), tool_call(), failure(400), completion())
    records = Records()
    async with connect(gateway, recorder=records) as llm:
        tool_turn_ran = await smoke(llm)

    assert tool_turn_ran
    assert [record.status for record in records.records] == [CLIENT_ERROR, OK, CLIENT_ERROR, OK]


async def test_a_failed_first_tool_turn_has_no_second() -> None:
    gateway, records = Gateway(completion(), failure(400), completion()), Records()
    async with connect(gateway, recorder=records) as llm:
        tool_turn_ran = await smoke(llm)

    assert not tool_turn_ran
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


async def test_exits_1_when_qwen_calls_no_tool(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with stand_in_gateway(calls_tools=False) as (base_url, _):
        monkeypatch.setenv("GOALY_API_KEY", KEY)
        monkeypatch.setenv("GOALY_BASE_URL", base_url)
        status = await main()

    printed = capsys.readouterr()
    assert status == 1
    assert len(printed.out.splitlines()) == 3
    assert printed.err == "llm_smoke: Qwen called no tool, so its second tool turn did not run\n"


async def test_without_a_key_says_what_to_set_and_exits_2(
    capsys: pytest.CaptureFixture[str],
) -> None:
    status = await main()  # The conftest leaves no GOALY_* variable and an empty HOME.

    printed = capsys.readouterr()
    assert status == 2
    assert printed.out == ""
    assert printed.err.startswith("llm_smoke: GOALY_API_KEY is not set.")
