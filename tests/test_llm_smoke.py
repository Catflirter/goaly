"""Tests for scripts/llm_smoke.py, the manual check of the LLM client against the real gateway.

No test reaches the real gateway. The scripted gateway stands in for it, and so does a local
HTTP server, which also takes the client through httpx's real transport.
"""

import json
import threading
from collections.abc import Iterator, Set
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from goaly.contracts.gen.llm_call_pb2 import CallPurpose, LlmCallRecord, LlmCallStatus, Usage
from scripts.llm_smoke import Records, describe, main, smoke
from tests.harness.core.llm_fakes import GLM, KEY, QWEN, Gateway, completion, connect, failure

pytestmark = pytest.mark.anyio

OK = LlmCallStatus.LLM_CALL_STATUS_OK
CLIENT_ERROR = LlmCallStatus.LLM_CALL_STATUS_CLIENT_ERROR
REPLY = "a reply the script must not print"

type Seen = list[tuple[str, str, str]]  # Each request's path, Authorization header and model.


@contextmanager
def stand_in_gateway(failing: Set[str] = frozenset()) -> Iterator[tuple[str, Seen]]:
    """Serve chat completions on a local port, answering 400 for the ``failing`` models.

    Yields the base URL and the list of requests it saw.
    """
    seen: Seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            model = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["model"]
            seen.append((self.path, self.headers["Authorization"], model))
            reply = failure(400, "refused") if model in failing else completion(REPLY)
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


async def test_calls_qwen_as_the_executor_then_glm_as_rsi() -> None:
    gateway, records = Gateway(completion(), completion()), Records()
    async with connect(gateway, recorder=records) as llm:
        await smoke(llm)

    assert [body["model"] for body in gateway.bodies] == [QWEN, GLM]
    assert [(record.model, record.purpose, record.run_id) for record in records.records] == [
        (QWEN, CallPurpose.CALL_PURPOSE_EXEC, "llm-smoke"),
        (GLM, CallPurpose.CALL_PURPOSE_RSI_IMPL, "llm-smoke"),
    ]


async def test_a_failed_call_does_not_stop_the_other() -> None:
    gateway, records = Gateway(failure(400), completion()), Records()
    async with connect(gateway, recorder=records) as llm:
        await smoke(llm)

    assert [record.status for record in records.records] == [CLIENT_ERROR, OK]


def test_describes_a_call_by_model_status_tokens_and_cost() -> None:
    usage = Usage(input_tokens=26, output_tokens=309, reasoning_tokens=304, cost_usd=0.0003129)
    record = LlmCallRecord(model=QWEN, status=OK, usage=usage)

    assert describe(record) == (
        "qwen3.6-35b-a3b: OK; tokens 26 in, 309 out (304 reasoning); cost $0.000313"
    )


def test_says_what_the_gateway_did_not_report() -> None:
    record = LlmCallRecord(model=GLM, status=CLIENT_ERROR)

    assert describe(record) == (
        "glm-5.3: CLIENT_ERROR; tokens 0 in, 0 out (reasoning not reported); cost not reported"
    )


async def test_prints_one_line_per_call_and_exits_0_when_both_succeed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with stand_in_gateway() as (base_url, seen):
        monkeypatch.setenv("GOALY_API_KEY", KEY)
        monkeypatch.setenv("GOALY_BASE_URL", base_url)
        status = await main()

    printed = capsys.readouterr()
    assert status == 0
    assert seen == [
        ("/v1/chat/completions", f"Bearer {KEY}", QWEN),
        ("/v1/chat/completions", f"Bearer {KEY}", GLM),
    ]
    assert printed.out.splitlines() == [
        "qwen3.6-35b-a3b: OK; tokens 26 in, 309 out (304 reasoning); cost $0.000313",
        "glm-5.3: OK; tokens 26 in, 309 out (304 reasoning); cost $0.000313",
    ]
    assert printed.err == ""
    assert KEY not in printed.out
    assert REPLY not in printed.out


async def test_exits_1_when_a_call_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with stand_in_gateway(failing={GLM}) as (base_url, _):
        monkeypatch.setenv("GOALY_API_KEY", KEY)
        monkeypatch.setenv("GOALY_BASE_URL", base_url)
        status = await main()

    assert status == 1
    lines = capsys.readouterr().out.splitlines()
    assert [line.split(";")[0] for line in lines] == [
        "qwen3.6-35b-a3b: OK",
        "glm-5.3: CLIENT_ERROR",
    ]


async def test_without_a_key_says_what_to_set_and_exits_2(
    capsys: pytest.CaptureFixture[str],
) -> None:
    status = await main()  # The conftest leaves no GOALY_* variable and an empty HOME.

    printed = capsys.readouterr()
    assert status == 2
    assert printed.out == ""
    assert printed.err.startswith("llm_smoke: GOALY_API_KEY is not set.")
