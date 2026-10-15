"""Tests for goaly.harness.core.llm_proxy: the key-injecting proxy for sandboxed solving.

The proxy forwards to a scripted gateway on httpx.MockTransport, so no test reaches the network.
On the sandbox side, a real LlmClient or a raw httpx client talks to it over loopback HTTP.
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from goaly.contracts.gen.llm_call_pb2 import LlmCallStatus
from goaly.harness.core import llm_proxy
from goaly.harness.core.llm import LlmClient, RetryPolicy
from goaly.harness.core.llm_errors import ConnectionFailed, RateLimited
from goaly.harness.core.llm_proxy import LlmProxy
from goaly.harness.core.logs import JsonFormatter
from goaly.harness.core.settings import Settings
from tests.harness.core.llm_fakes import (
    EXEC,
    GLM,
    KEY,
    QWEN,
    Gateway,
    Recorder,
    completion,
    failure,
    hang,
    settings,
    user,
)

pytestmark = pytest.mark.anyio

BODY = {"model": QWEN, "messages": [{"role": "user", "content": "a prompt nobody may log"}]}


@asynccontextmanager
async def running(gateway: Gateway, **options: Any) -> AsyncIterator[LlmProxy]:
    async with LlmProxy(
        settings(), transport=httpx.MockTransport(gateway.handle), **options
    ) as proxy:
        yield proxy


def sandbox_client(proxy: LlmProxy, recorder: Recorder | None = None) -> LlmClient:
    """The LLM client as the sandbox builds it: from the environment the proxy hands it."""
    env = proxy.container_env()
    sandbox = Settings(api_key=SecretStr(env["GOALY_API_KEY"]), base_url=env["GOALY_BASE_URL"])
    return LlmClient(
        sandbox, Recorder() if recorder is None else recorder, retry=RetryPolicy(max_retries=0)
    )


def raw(proxy: LlmProxy, token: str | None = None) -> httpx.AsyncClient:
    """A plain HTTP client of the proxy, with the run's token unless told another."""
    token = proxy.container_env()["GOALY_API_KEY"] if token is None else token
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return httpx.AsyncClient(base_url=proxy.base_url, headers=headers)


def requests_logged(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    keys = ("request", "model", "status")
    return [
        {key: getattr(entry, key) for key in keys}
        for entry in caplog.records
        if getattr(entry, "event", None) == "llm_proxy_request"
    ]


# Forwarding.


async def test_forwards_a_call_with_the_real_key_and_passes_the_reply_through() -> None:
    gateway = Gateway(completion())
    async with running(gateway) as proxy, sandbox_client(proxy) as llm:
        reply = await llm.complete([user("hi")], model=QWEN, tags=EXEC)

    (request,) = gateway.requests
    assert request.url == "https://gateway.test/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert gateway.bodies[0]["model"] == QWEN
    assert reply.message.content == "391"
    assert reply.usage.cost_usd == 0.0003129  # x-litellm-response-cost came through.
    assert reply.usage.reasoning_tokens == 304


async def test_passes_a_failure_and_its_retry_after_through_unchanged() -> None:
    gateway = Gateway(failure(429, "slow down", retry_after="7"))
    async with running(gateway) as proxy, sandbox_client(proxy) as llm:
        with pytest.raises(RateLimited, match="HTTP 429: slow down") as raised:
            await llm.complete([user("hi")], model=QWEN, tags=EXEC)

    assert raised.value.record is not None
    (attempt,) = raised.value.record.attempts
    assert (attempt.status, attempt.http_status) == (
        LlmCallStatus.LLM_CALL_STATUS_RATE_LIMITED,
        429,
    )
    assert attempt.retry_after_s == 7.0


async def test_the_sandbox_gets_no_key_and_each_run_its_own_token() -> None:
    async with running(Gateway()) as proxy, running(Gateway()) as other:
        env, other_env = proxy.container_env(), other.container_env()

    assert set(env) == {"GOALY_API_KEY", "GOALY_BASE_URL"}
    assert env["GOALY_BASE_URL"] == proxy.base_url
    assert env["GOALY_BASE_URL"].startswith("http://127.0.0.1:")
    assert KEY not in json.dumps(env)
    assert env["GOALY_API_KEY"] != other_env["GOALY_API_KEY"]


# Refusals.


async def test_refuses_any_model_but_the_executor(caplog: pytest.LogCaptureFixture) -> None:
    gateway = Gateway()
    async with running(gateway) as proxy, raw(proxy) as http:
        response = await http.post("/chat/completions", json={**BODY, "model": GLM})

    assert response.status_code == 403
    assert "only qwen3.6-35b-a3b" in response.json()["error"]["message"]
    assert gateway.requests == []
    (refused,) = [
        entry for entry in caplog.records if getattr(entry, "event", None) == "llm_proxy_refused"
    ]
    assert refused.levelno == logging.WARNING
    assert getattr(refused, "model", None) == GLM


@pytest.mark.parametrize(
    "token",
    [
        pytest.param("not-the-token", id="wrong"),
        pytest.param("", id="none"),
    ],
)
async def test_refuses_a_request_without_the_runs_token(token: str) -> None:
    gateway = Gateway()
    async with running(gateway) as proxy, raw(proxy, token) as http:
        response = await http.post("/chat/completions", json=BODY)

    assert response.status_code == 401
    assert gateway.requests == []


async def test_refuses_the_real_key_too() -> None:
    gateway = Gateway()
    async with running(gateway) as proxy, raw(proxy, KEY) as http:
        response = await http.post("/chat/completions", json=BODY)

    assert response.status_code == 401


@pytest.mark.parametrize(
    ("method", "path", "status"),
    [
        pytest.param("POST", "/v1/chat/completions", 404, id="other-path"),
        pytest.param("POST", "/models", 404, id="models"),
        pytest.param("GET", "/chat/completions", 405, id="get"),
    ],
)
async def test_serves_only_post_chat_completions(method: str, path: str, status: int) -> None:
    gateway = Gateway()
    async with running(gateway) as proxy, raw(proxy) as http:
        response = await http.request(method, path, json=BODY)

    assert response.status_code == status
    assert gateway.requests == []


@pytest.mark.parametrize(
    ("content", "status"),
    [
        pytest.param(b"{not json", 400, id="not-json"),
        pytest.param(b"[1, 2]", 400, id="not-an-object"),
        pytest.param(b"{}", 403, id="no-model"),
        pytest.param(b'{"model": "' + b"x" * 200 + b'"}', 413, id="too-large"),
    ],
)
async def test_refuses_a_body_it_cannot_judge(content: bytes, status: int) -> None:
    gateway = Gateway()
    async with running(gateway, max_body_bytes=100) as proxy, raw(proxy) as http:
        response = await http.post("/chat/completions", content=content)

    assert response.status_code == status
    assert gateway.requests == []


# When the gateway fails.


@pytest.mark.parametrize(
    ("reply", "status"),
    [
        pytest.param(httpx.ConnectError("refused"), 502, id="unreachable"),
        pytest.param(httpx.ReadTimeout("slow"), 504, id="timed-out"),
    ],
)
async def test_answers_a_gateway_failure_as_a_gateway_would(reply: Exception, status: int) -> None:
    async with running(Gateway(reply)) as proxy, raw(proxy) as http:
        response = await http.post("/chat/completions", json=BODY)

    assert response.status_code == status
    assert response.json()["error"]["message"]


async def test_times_out_a_forward_that_takes_too_long() -> None:
    async with running(Gateway(hang), timeout_s=0.05) as proxy, raw(proxy) as http:
        response = await http.post("/chat/completions", json=BODY)

    assert response.status_code == 504


# Logs and secrets.


async def test_logs_each_request_without_the_key_the_token_or_the_body(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    gateway = Gateway(completion())
    async with running(gateway) as proxy:
        token = proxy.container_env()["GOALY_API_KEY"]
        async with raw(proxy) as http:
            await http.post("/chat/completions", json=BODY)
            await http.post("/chat/completions", json={**BODY, "model": GLM})
        async with raw(proxy, "not-the-token") as http:
            await http.post("/chat/completions", json=BODY)

    assert requests_logged(caplog) == [
        {"request": 1, "model": QWEN, "status": 200},
        {"request": 2, "model": GLM, "status": 403},
        {"request": 3, "model": "", "status": 401},
    ]
    latencies = [
        getattr(entry, "latency_s", None)
        for entry in caplog.records
        if getattr(entry, "event", None) == "llm_proxy_request"
    ]
    assert all(isinstance(value, float) and value >= 0 for value in latencies)
    text = "\n".join(JsonFormatter().format(entry) for entry in caplog.records)
    assert KEY not in text
    assert token not in text
    assert "a prompt nobody may log" not in text


# Concurrency and shutdown.


async def test_serves_calls_at_once() -> None:
    in_flight, peak = 0, 0

    async def answer(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.05)
        in_flight -= 1
        return completion()

    gateway = Gateway(*(answer for _ in range(12)))
    async with running(gateway) as proxy, sandbox_client(proxy) as llm:
        replies = await asyncio.gather(
            *(llm.complete([user(f"call {n}")], model=QWEN, tags=EXEC) for n in range(12))
        )

    assert len(replies) == 12
    assert peak > 1  # The proxy forwarded them together, not one after another.


async def test_stops_promptly_with_calls_in_flight() -> None:
    arrived = asyncio.Event()
    loop = asyncio.get_running_loop()

    async def hold(request: httpx.Request) -> httpx.Response:
        loop.call_soon_threadsafe(arrived.set)  # The proxy forwards on its own thread's loop.
        return await hang(request)

    async with asyncio.timeout(5):  # Stopping does not wait for the call to end.
        async with running(Gateway(hold)) as proxy:
            llm = sandbox_client(proxy)
            call = asyncio.create_task(llm.complete([user("hi")], model=QWEN, tags=EXEC))
            await arrived.wait()
    try:
        with pytest.raises(ConnectionFailed):  # The call in flight ends, with an error.
            await call
    finally:
        await llm.aclose()

    with pytest.raises(httpx.ConnectError):  # Nothing listens any more.
        async with raw(proxy) as http:
            await http.post("/chat/completions", json=BODY)


async def test_a_proxy_runs_once() -> None:
    proxy = LlmProxy(settings())
    async with proxy:
        pass

    with pytest.raises(RuntimeError, match="once"):
        async with proxy:
            pass


# Requests that are not well-formed, over a bare connection.


async def exchange(proxy: LlmProxy, data: bytes, *, end: bool = True) -> bytes:
    """Send ``data`` to the proxy on a bare connection, and return all it answers."""
    url = httpx.URL(proxy.base_url)
    reader, writer = await asyncio.open_connection(url.host, url.port)
    try:
        writer.write(data)
        await writer.drain()
        if end:
            writer.write_eof()
        return await reader.read()
    finally:
        writer.close()
        await writer.wait_closed()


def status_of(answer: bytes) -> int:
    return int(answer.split(b" ", 2)[1])


@pytest.mark.parametrize(
    ("data", "status"),
    [
        pytest.param(b"not a request\r\n\r\n", 400, id="no-request-line"),
        pytest.param(b"POST /chat/completions HTTP/1.1\r\nno colon\r\n\r\n", 400, id="no-colon"),
        pytest.param(b"POST /chat/completions HTTP/1.1\r\nAUTH\r\n\r\n", 411, id="no-length"),
        pytest.param(
            b"POST /chat/completions HTTP/1.1\r\nAUTH\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n0\r\n\r\n",
            411,
            id="chunked",
        ),
        pytest.param(
            b"POST /chat/completions HTTP/1.1\r\nAUTH\r\nContent-Length: ten\r\n\r\n",
            400,
            id="bad-length",
        ),
    ],
)
async def test_refuses_a_malformed_request(data: bytes, status: int) -> None:
    gateway = Gateway()
    async with running(gateway) as proxy:
        token = proxy.container_env()["GOALY_API_KEY"].encode()
        answer = await exchange(proxy, data.replace(b"AUTH", b"Authorization: Bearer " + token))

    assert status_of(answer) == status
    assert gateway.requests == []


async def test_refuses_headers_too_long_to_read() -> None:
    # Under twice the stream's limit, so asyncio's buffer holds all of it and the close is clean.
    head = b"POST /chat/completions HTTP/1.1\r\nX-Padding: " + b"x" * 70_000
    async with running(Gateway()) as proxy:
        answer = await exchange(proxy, head, end=False)

    assert status_of(answer) == 431


async def test_refuses_a_request_that_arrives_too_slowly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_proxy, "_ARRIVAL_S", 0.05)
    async with running(Gateway()) as proxy:
        token = proxy.container_env()["GOALY_API_KEY"].encode()
        head = b"POST /chat/completions HTTP/1.1\r\nAuthorization: Bearer " + token
        answer = await exchange(proxy, head + b"\r\nContent-Length: 100\r\n\r\n{", end=False)

    assert status_of(answer) == 408


async def test_logs_a_caller_that_went_away(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="goaly.harness.core.llm_proxy")
    async with running(Gateway()) as proxy:
        answer = await exchange(proxy, b"POST /chat/comp")

    assert answer == b""
    assert requests_logged(caplog) == [{"request": 1, "model": "", "status": 0}]


async def test_passes_on_a_status_with_no_standard_name() -> None:
    gateway = Gateway(httpx.Response(599, json={"error": {"message": "odd"}}))
    async with running(gateway) as proxy, raw(proxy) as http:
        response = await http.post("/chat/completions", json=BODY)

    assert response.status_code == 599
    assert response.json() == {"error": {"message": "odd"}}


# Starting and stopping.


async def test_raises_when_it_cannot_start(monkeypatch: pytest.MonkeyPatch) -> None:
    async def refuse(*args: object, **kwargs: object) -> object:
        raise OSError("no port")

    monkeypatch.setattr(asyncio, "start_server", refuse)
    with pytest.raises(OSError, match="no port"):
        async with LlmProxy(settings()):
            pass


@pytest.mark.parametrize(
    "options",
    [
        pytest.param({"max_body_bytes": 0}, id="no-body"),
        pytest.param({"timeout_s": 0.0}, id="no-time"),
    ],
)
def test_refuses_options_out_of_range(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="positive"):
        LlmProxy(settings(), **options)


def test_has_no_url_before_it_starts() -> None:
    with pytest.raises(RuntimeError, match="not started"):
        LlmProxy(settings()).container_env()
