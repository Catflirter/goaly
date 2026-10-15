"""Tests for goaly.harness.core.llm_proxy: the key-injecting proxy for sandboxed solving.

The proxy forwards to a scripted gateway on httpx.MockTransport, so no test reaches the network.
On the sandbox side, a real LlmClient or a raw httpx client talks to it over loopback HTTP or
over its Unix socket, and bare connections send it what no HTTP client would.
"""

import asyncio
import contextlib
import gzip
import json
import logging
import os
import socket
import stat
import struct
import tempfile
import threading
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from goaly.contracts.gen.llm_call_pb2 import CallPurpose, LlmCallRecord, LlmCallStatus
from goaly.contracts.jsonl import to_json_line
from goaly.harness.core import llm_proxy
from goaly.harness.core.llm import REQUEST_KEYS, THINKING_SWITCHES, LlmClient, RetryPolicy
from goaly.harness.core.llm_errors import ConnectionFailed, RateLimited
from goaly.harness.core.llm_models import ModelParams
from goaly.harness.core.llm_proxy import LlmProxy
from goaly.harness.core.logs import JsonFormatter
from goaly.harness.core.settings import Settings
from tests.harness.core.llm_fakes import (
    BASE_URL,
    EXEC,
    GLM,
    KEY,
    QWEN,
    Gateway,
    Recorder,
    attempts,
    completion,
    failure,
    hang,
    settings,
    user,
)

pytestmark = pytest.mark.anyio

OK = LlmCallStatus.LLM_CALL_STATUS_OK
TIMEOUT = LlmCallStatus.LLM_CALL_STATUS_TIMEOUT
SERVER_ERROR = LlmCallStatus.LLM_CALL_STATUS_SERVER_ERROR
CONNECTION_ERROR = LlmCallStatus.LLM_CALL_STATUS_CONNECTION_ERROR
INVALID_RESPONSE = LlmCallStatus.LLM_CALL_STATUS_INVALID_RESPONSE
CLIENT_ERROR = LlmCallStatus.LLM_CALL_STATUS_CLIENT_ERROR
CANCELLED = LlmCallStatus.LLM_CALL_STATUS_CANCELLED
PATH = "/chat/completions"
BODY = {"model": QWEN, "messages": [{"role": "user", "content": "a prompt nobody may log"}]}
SQL_TOOL: dict[str, Any] = {"type": "function", "function": {"name": "execute_sql"}}
# Every ModelParams field, set: the client then writes every key it ever writes.
EVERY_PARAM = ModelParams(
    temperature=0.7,
    top_p=0.9,
    top_k=30,
    max_tokens=512,
    presence_penalty=0.5,
    enable_thinking=False,
    preserve_thinking=True,
)


@asynccontextmanager
async def running(
    gateway: Gateway, recorder: Recorder | None = None, **options: Any
) -> AsyncIterator[LlmProxy]:
    async with LlmProxy(
        settings(),
        Recorder() if recorder is None else recorder,
        transport=httpx.MockTransport(gateway.handle),
        **options,
    ) as proxy:
        yield proxy


def sandbox_client(proxy: LlmProxy, *, base_url: str | None = None) -> LlmClient:
    """The LLM client as the sandbox builds it: from the environment the proxy hands it."""
    env = proxy.container_env()
    uds = env.get("GOALY_LLM_UDS")
    sandbox = Settings(
        api_key=SecretStr(env["GOALY_API_KEY"]),
        base_url=env["GOALY_BASE_URL"] if base_url is None else base_url,
        llm_uds=None if uds is None else Path(uds),
    )
    return LlmClient(sandbox, Recorder(), retry=RetryPolicy(max_retries=0))


def raw(proxy: LlmProxy, token: str | None = None, *, uds: Path | None = None) -> httpx.AsyncClient:
    """A plain HTTP client of the proxy, with the run's token unless told another."""
    token = proxy.container_env()["GOALY_API_KEY"] if token is None else token
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    transport = None if uds is None else httpx.AsyncHTTPTransport(uds=str(uds))
    return httpx.AsyncClient(
        base_url=proxy.base_url, headers=headers, transport=transport, trust_env=False
    )


def logged(caplog: pytest.LogCaptureFixture, event: str) -> list[logging.LogRecord]:
    return [entry for entry in caplog.records if getattr(entry, "event", None) == event]


def requests_logged(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    keys = ("request", "model", "status")
    return [
        {key: getattr(entry, key) for key in keys} for entry in logged(caplog, "llm_proxy_request")
    ]


def with_headers(reply: httpx.Response, **headers: str) -> httpx.Response:
    """``reply`` with more headers, as the gateway might send them."""
    return httpx.Response(
        reply.status_code, content=reply.content, headers={**reply.headers, **headers}
    )


class Chunks(httpx.AsyncByteStream):
    """A reply body that arrives as it would off the network, chunk by chunk."""

    def __init__(self, *chunks: bytes) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


@pytest.fixture(autouse=True)
def _no_unhandled_errors(caplog: pytest.LogCaptureFixture) -> Iterator[None]:
    """Fail a test whose proxy let an error escape a handler: asyncio only logs those."""
    yield
    escaped = [entry for entry in caplog.get_records("call") if entry.name == "asyncio"]
    assert [entry.getMessage() for entry in escaped if entry.levelno >= logging.ERROR] == []


@pytest.fixture
def short_dir() -> Iterator[Path]:
    """A directory with a short path: a Unix socket's whole path must fit in 107 bytes."""
    with tempfile.TemporaryDirectory(prefix="goaly-", dir="/tmp") as name:
        yield Path(name)


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


async def test_forwards_every_key_the_client_writes_and_nothing_else() -> None:
    # A field added to ModelParams must join llm.REQUEST_KEYS, and EVERY_PARAM, before it works.
    assert set(EVERY_PARAM.model_dump(exclude_none=True)) == set(ModelParams.model_fields)
    gateway = Gateway(completion())
    async with running(gateway) as proxy, sandbox_client(proxy) as llm:
        await llm.complete(
            [user("hi")], model=QWEN, tags=EXEC, tools=[SQL_TOOL], params=EVERY_PARAM
        )

    (body,) = gateway.bodies
    assert set(body) == REQUEST_KEYS | {"stream", "n"}
    assert set(body["chat_template_kwargs"]) == set(THINKING_SWITCHES)
    assert (body["max_tokens"], body["stream"], body["n"]) == (512, False, 1)
    assert body["tools"] == [SQL_TOOL]


async def test_the_sandbox_gets_no_key_and_each_run_its_own_token() -> None:
    async with running(Gateway()) as proxy, running(Gateway()) as other:
        env, other_env = proxy.container_env(), other.container_env()

    assert set(env) == {"GOALY_API_KEY", "GOALY_BASE_URL"}
    assert env["GOALY_BASE_URL"] == proxy.base_url
    assert httpx.URL(env["GOALY_BASE_URL"]).host == "127.0.0.1"  # The socket's own address.
    assert KEY not in json.dumps(env)
    assert env["GOALY_API_KEY"] != other_env["GOALY_API_KEY"]


async def test_the_gateway_sees_none_of_the_containers_headers() -> None:
    gateway = Gateway(completion())
    async with running(gateway) as proxy, raw(proxy) as http:
        token = proxy.container_env()["GOALY_API_KEY"]
        headers = {
            "User-Agent": "pack-code",
            "X-Sandbox": "inside",
            "Cookie": "session=1",
            "X-Goaly-Call-Tags": '{"instance_id":"task-7"}',
        }
        response = await http.post(PATH, json=BODY, headers=headers)

    assert response.status_code == 200
    (request,) = gateway.requests
    names = {name.lower() for name in request.headers}
    assert names.isdisjoint({"x-sandbox", "cookie", "x-goaly-call-tags"})
    assert request.headers["user-agent"] != "pack-code"
    assert token not in str(request.headers.raw)


# The body (spec §6.1): rebuilt from the keys the client writes.


@pytest.mark.parametrize(
    ("extra", "key"),
    [
        pytest.param({"fallbacks": [GLM]}, "fallbacks", id="fallbacks"),
        pytest.param({"mock_testing_fallbacks": True}, "mock_testing_fallbacks", id="mock"),
        pytest.param({"user_config": {"model_list": []}}, "user_config", id="user-config"),
        pytest.param({"api_base": "http://elsewhere.test"}, "api_base", id="api-base"),
        pytest.param({"stream": True}, "stream", id="stream"),
        pytest.param({"n": 2}, "n", id="n"),
        pytest.param(
            {"chat_template_kwargs": {"thinking_budget": 1}},
            "chat_template_kwargs.thinking_budget",
            id="template-switch",
        ),
    ],
)
async def test_refuses_a_key_the_client_never_writes(
    extra: dict[str, Any], key: str, caplog: pytest.LogCaptureFixture
) -> None:
    gateway = Gateway()
    async with running(gateway) as proxy, raw(proxy) as http:
        response = await http.post(PATH, json={**BODY, **extra})

    assert response.status_code == 400
    assert response.json()["error"]["message"] == f"the proxy forwards no {key}"
    assert gateway.requests == []
    (refused,) = logged(caplog, "llm_proxy_refused")
    assert refused.levelno == logging.WARNING
    assert getattr(refused, "keys", None) == [key]
    assert "elsewhere" not in JsonFormatter().format(refused)  # Names only, never values.


@pytest.mark.parametrize(
    ("content", "message"),
    [
        pytest.param(
            b'{"model": "glm-5.3", "model": "qwen3.6-35b-a3b", "messages": []}',
            "the body gives a key twice",
            id="key-twice",
        ),
        pytest.param(
            b'{"model": "qwen3.6-35b-a3b", "messages": [{"role": "user", "role": "system"}]}',
            "the body gives a key twice",
            id="inner-key-twice",
        ),
        pytest.param(
            b'{"model": "qwen3.6-35b-a3b", "messages": [], "temperature": NaN}',
            "the body holds NaN, which is not JSON",
            id="nan",
        ),
        pytest.param(
            b'{"model": "qwen3.6-35b-a3b", "messages": [], "temperature": 1e999}',
            "the body holds 1e999, which is not a finite number",
            id="overflowing-number",
        ),
        pytest.param(
            b'{"model": "qwen3.6-35b-a3b", "messages": [{"role": "user", "x": -1e400}]}',
            "the body holds -1e400, which is not a finite number",
            id="inner-overflowing-number",
        ),
        pytest.param(
            b'{"model": "qwen3.6-35b-a3b", "messages": [], "chat_template_kwargs": []}',
            "chat_template_kwargs must be an object",
            id="switches-not-an-object",
        ),
        pytest.param(
            b'{"model": "qwen3.6-35b-a3b", "messages": [], "max_tokens": "100000"}',
            "max_tokens must be a positive integer",
            id="max-tokens-text",
        ),
        pytest.param(
            b'{"model": "qwen3.6-35b-a3b", "messages": [], "max_tokens": true}',
            "max_tokens must be a positive integer",
            id="max-tokens-bool",
        ),
        pytest.param(
            b'{"model": "qwen3.6-35b-a3b", "messages": [], "max_tokens": 0}',
            "max_tokens must be a positive integer",
            id="max-tokens-zero",
        ),
    ],
)
async def test_refuses_a_body_it_cannot_vouch_for(content: bytes, message: str) -> None:
    gateway = Gateway()
    async with running(gateway) as proxy, raw(proxy) as http:
        response = await http.post(PATH, content=content)

    assert (response.status_code, response.json()["error"]["message"]) == (400, message)
    assert gateway.requests == []


@pytest.mark.parametrize(
    ("asked", "sent"),
    [
        pytest.param({}, 16384, id="none"),
        pytest.param({"max_tokens": 100_000}, 16384, id="too-many"),
        pytest.param({"max_tokens": 512}, 512, id="fewer"),
    ],
)
async def test_caps_max_tokens_and_asks_for_one_whole_reply(
    asked: dict[str, int], sent: int
) -> None:
    gateway = Gateway(completion())
    async with running(gateway) as proxy, raw(proxy) as http:
        await http.post(PATH, json={**BODY, **asked})

    assert gateway.bodies == [{**BODY, "max_tokens": sent, "stream": False, "n": 1}]


async def test_refuses_any_model_but_the_executor(caplog: pytest.LogCaptureFixture) -> None:
    gateway = Gateway()
    async with running(gateway) as proxy, raw(proxy) as http:
        response = await http.post(PATH, json={**BODY, "model": GLM})

    assert response.status_code == 403
    assert "only qwen3.6-35b-a3b" in response.json()["error"]["message"]
    assert gateway.requests == []
    (refused,) = logged(caplog, "llm_proxy_refused")
    assert refused.levelno == logging.WARNING
    assert getattr(refused, "model", None) == GLM


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
        response = await http.post(PATH, content=content)

    assert response.status_code == status
    assert gateway.requests == []


# Who may call, and how.


@pytest.mark.parametrize(
    "token",
    [
        pytest.param("not-the-token", id="wrong"),
        pytest.param("", id="none"),
        pytest.param(KEY, id="the-real-key"),
    ],
)
async def test_refuses_a_request_without_the_runs_token(token: str) -> None:
    gateway = Gateway()
    async with running(gateway) as proxy, raw(proxy, token) as http:
        response = await http.post(PATH, json=BODY)

    assert response.status_code == 401
    assert gateway.requests == []


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


# The reply.


async def test_passes_the_reply_on_decoded_with_its_own_headers_and_framing() -> None:
    plain = completion().content
    reply = httpx.Response(
        200,
        content=gzip.compress(plain),
        headers={
            "Content-Type": "application/json",
            "Content-Encoding": "gzip",
            "X-Custom": "kept",
            "Connection": "keep-alive",
            "Keep-Alive": "timeout=5",
            "x-litellm-response-cost": "0.0003129",
        },
    )
    async with running(Gateway(reply)) as proxy, raw(proxy) as http:
        response = await http.post(PATH, json=BODY)

    assert response.content == plain  # Decoded, as the spec allows: byte-identical once decoded.
    assert response.headers["x-custom"] == "kept"
    assert response.headers["x-litellm-response-cost"] == "0.0003129"
    assert "content-encoding" not in response.headers
    assert "keep-alive" not in response.headers
    assert response.headers["connection"] == "close"
    assert response.headers["content-length"] == str(len(plain))


async def test_takes_the_key_out_of_what_it_passes_on() -> None:
    key = 'sk-with-"quotes"-0123456789'  # JSON escapes it, so it must go in both forms.
    echoed = json.dumps({"error": {"message": f"key {key} is not valid"}})
    gateway = Gateway(
        httpx.Response(401, content=f"{echoed} {key}".encode(), headers={"X-Echo": key})
    )
    host = Settings(api_key=SecretStr(key), base_url=BASE_URL)
    async with (
        LlmProxy(host, Recorder(), transport=httpx.MockTransport(gateway.handle)) as proxy,
        raw(proxy) as http,
    ):
        response = await http.post(PATH, json=BODY)

    assert response.status_code == 401
    assert key not in response.text
    assert json.dumps(key)[1:-1] not in response.text
    assert response.text.count("[redacted]") == 2
    assert response.headers["x-echo"] == "[redacted]"


async def test_passes_on_a_status_with_no_standard_name() -> None:
    gateway = Gateway(httpx.Response(599, json={"error": {"message": "odd"}}))
    async with running(gateway) as proxy, raw(proxy) as http:
        response = await http.post(PATH, json=BODY)

    assert response.status_code == 599
    assert response.json() == {"error": {"message": "odd"}}


async def test_answers_502_for_a_reply_over_its_size_limit_but_records_its_cost() -> None:
    reply = httpx.Response(
        200,
        headers={"Content-Type": "application/json", "x-litellm-response-cost": "0.0003129"},
        stream=Chunks(b"x" * 60, b"x" * 60),
    )
    gateway, recorder = Gateway(reply), Recorder()
    async with running(gateway, recorder, max_reply_bytes=100) as proxy, raw(proxy) as http:
        response = await http.post(PATH, json=BODY)

    assert response.status_code == 502
    assert "over 100 bytes" in response.json()["error"]["message"]
    (record,) = recorder.records
    assert (record.status, record.http_status) == (INVALID_RESPONSE, 200)
    assert record.usage.cost_usd == 0.0003129  # The gateway billed it all the same.


async def test_answers_502_for_a_reply_from_a_fallback_model() -> None:
    reply = with_headers(completion(), **{"x-litellm-attempted-fallbacks": "1"})
    gateway, recorder = Gateway(reply), Recorder()
    async with running(gateway, recorder) as proxy, raw(proxy) as http:
        response = await http.post(PATH, json=BODY)

    assert response.status_code == 502
    assert "fallback" in response.json()["error"]["message"]
    (record,) = recorder.records
    assert (record.status, record.http_status) == (INVALID_RESPONSE, 200)
    assert (record.usage.output_tokens, record.usage.cost_usd) == (309, 0.0003129)


async def test_passes_on_a_reply_that_tried_no_fallback() -> None:
    reply = with_headers(completion(), **{"x-litellm-attempted-fallbacks": "0"})
    async with running(Gateway(reply)) as proxy, raw(proxy) as http:
        response = await http.post(PATH, json=BODY)

    assert response.status_code == 200


# When the gateway fails.


@pytest.mark.parametrize(
    ("reply", "status", "outcome"),
    [
        pytest.param(httpx.ConnectError("refused"), 502, CONNECTION_ERROR, id="unreachable"),
        pytest.param(httpx.ReadTimeout("slow"), 504, TIMEOUT, id="timed-out"),
        pytest.param(
            httpx.Response(200, headers={"Content-Encoding": "gzip"}, stream=Chunks(b"no")),
            502,
            INVALID_RESPONSE,
            id="unreadable",
        ),
    ],
)
async def test_answers_a_gateway_failure_as_a_gateway_would(
    reply: Exception | httpx.Response, status: int, outcome: int
) -> None:
    recorder = Recorder()
    async with running(Gateway(reply), recorder) as proxy, raw(proxy) as http:
        response = await http.post(PATH, json=BODY)

    assert response.status_code == status
    assert response.json()["error"]["message"]
    (record,) = recorder.records
    assert attempts(record) == [(outcome, 0)]  # No reply that could be read came.
    assert (record.status, record.http_status) == attempts(record)[0]


async def test_times_out_a_forward_that_takes_too_long() -> None:
    async with running(Gateway(hang), timeout_s=0.05) as proxy, raw(proxy) as http:
        response = await http.post(PATH, json=BODY)

    assert response.status_code == 504


# Records (spec §6.1): one per request, refusals included, on the caller's loop.


async def test_records_a_call_with_its_tokens_cost_and_tags() -> None:
    gateway, recorder = Gateway(completion()), Recorder()
    async with (
        running(gateway, recorder, run_id="run-1") as proxy,
        sandbox_client(proxy) as llm,
    ):
        await llm.complete([user("hi")], model=QWEN, tags=EXEC)

    (record,) = recorder.records
    assert (record.run_id, record.instance_id, record.seed) == ("run-1", "task-7", 2)
    assert (record.purpose, record.model) == (CallPurpose.CALL_PURPOSE_EXEC, QWEN)
    assert (record.status, record.http_status) == (OK, 200)
    usage = record.usage
    assert (usage.input_tokens, usage.output_tokens, usage.reasoning_tokens) == (26, 309, 304)
    assert usage.cost_usd == 0.0003129
    assert attempts(record) == [(OK, 200)]
    assert record.latency_s >= record.attempts[0].latency_s > 0
    assert record.wait_s == record.attempts[0].wait_s >= 0
    assert record.started_at.seconds > 0
    assert len(record.call_id) == 32


async def test_records_every_refusal_too() -> None:
    recorder = Recorder()
    async with running(Gateway(), recorder) as proxy:
        async with raw(proxy) as http:
            await http.post(PATH, json={**BODY, "model": GLM})
            await http.post(PATH, json={**BODY, "fallbacks": [GLM]})
        async with raw(proxy, "not-the-token") as http:
            await http.post(PATH, json=BODY)

    calls = [(r.status, r.http_status, r.model, len(r.attempts)) for r in recorder.records]
    assert calls == [
        (CLIENT_ERROR, 403, GLM, 0),
        (CLIENT_ERROR, 400, QWEN, 0),
        (CLIENT_ERROR, 401, "", 0),
    ]
    assert not any(record.usage.HasField("cost_usd") for record in recorder.records)


async def test_records_a_call_before_its_answer_goes_out(monkeypatch: pytest.MonkeyPatch) -> None:
    async def gone(self: asyncio.StreamWriter) -> None:
        raise ConnectionResetError("the caller went away")

    monkeypatch.setattr(asyncio.StreamWriter, "drain", gone)  # Only the proxy's writers use it.
    gateway, recorder = Gateway(completion()), Recorder()
    async with running(gateway, recorder) as proxy, raw(proxy) as http:
        with contextlib.suppress(httpx.HTTPError):
            await http.post(PATH, json=BODY)

    (record,) = recorder.records  # The gateway billed the call, so it counts.
    assert (record.status, record.http_status, record.usage.cost_usd) == (OK, 200, 0.0003129)


async def test_hands_every_record_over_on_the_callers_loop_before_it_stops() -> None:
    class Noting(Recorder):
        def __init__(self) -> None:
            super().__init__()
            self.threads: set[int] = set()

        def record(self, record: LlmCallRecord) -> None:
            self.threads.add(threading.get_ident())
            super().record(record)

    recorder = Noting()
    gateway = Gateway(*(completion() for _ in range(12)))
    async with running(gateway, recorder) as proxy, sandbox_client(proxy) as llm:
        await asyncio.gather(
            *(llm.complete([user(f"call {n}")], model=QWEN, tags=EXEC) for n in range(12))
        )

    assert len(recorder.records) == 12
    assert len({record.call_id for record in recorder.records}) == 12
    assert recorder.threads == {threading.get_ident()}


async def test_a_broken_recorder_loses_only_its_records(caplog: pytest.LogCaptureFixture) -> None:
    class Broken:
        def record(self, record: LlmCallRecord) -> None:
            raise RuntimeError("disk full")

    async with (
        LlmProxy(
            settings(), Broken(), transport=httpx.MockTransport(Gateway(completion()).handle)
        ) as proxy,
        raw(proxy) as http,
    ):
        response = await http.post(PATH, json=BODY)

    assert response.status_code == 200
    (lost,) = logged(caplog, "llm_record_lost")
    assert lost.levelno == logging.ERROR


def test_loses_a_record_only_once_the_callers_loop_has_closed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    recorder = Recorder()
    gateway = Gateway(completion())
    proxy = LlmProxy(settings(), recorder, transport=httpx.MockTransport(gateway.handle))

    async def enter() -> None:
        await proxy.__aenter__()  # Then leave, closing the loop, with the proxy still serving.

    asyncio.run(enter())
    try:
        token = proxy.container_env()["GOALY_API_KEY"]
        response = httpx.post(
            proxy.base_url + PATH,
            json=BODY,
            headers={"Authorization": f"Bearer {token}"},
            trust_env=False,
        )
    finally:
        asyncio.run(proxy.__aexit__(None, None, None))

    assert response.status_code == 200
    assert recorder.records == []
    (lost,) = logged(caplog, "llm_record_lost")
    assert lost.levelno == logging.ERROR


# The tags (spec §6.1): the container's claim, good for attribution only.


@pytest.mark.parametrize(
    ("tags", "problem", "task"),
    [
        pytest.param(None, "missing", "", id="missing"),
        pytest.param("not json", "malformed", "", id="not-json"),
        pytest.param("[1]", "malformed", "", id="not-an-object"),
        pytest.param('{"instance_id": "t", "user": "x"}', "malformed", "", id="unknown-key"),
        pytest.param('{"instance_id": 7}', "malformed", "", id="task-not-text"),
        pytest.param('{"seed": true}', "malformed", "", id="seed-bool"),
        pytest.param('{"seed": 1.5}', "malformed", "", id="seed-not-int"),
        pytest.param('{"seed": 2147483648}', "malformed", "", id="seed-too-large"),
        pytest.param('{"instance_id": "\\ud800"}', "malformed", "", id="lone-surrogate"),
        pytest.param('{"instance_id": "' + "t" * 1100 + '"}', "malformed", "", id="too-long"),
        pytest.param(
            '{"run_id": "run-2", "instance_id": "task-7"}', "another run", "task-7", id="other-run"
        ),
    ],
)
async def test_records_a_call_with_untrustworthy_tags_under_its_own_run(
    tags: str | None, problem: str, task: str, caplog: pytest.LogCaptureFixture
) -> None:
    gateway, recorder = Gateway(completion()), Recorder()
    headers = {} if tags is None else {"X-Goaly-Call-Tags": tags}
    async with running(gateway, recorder, run_id="run-1") as proxy, raw(proxy) as http:
        response = await http.post(PATH, json=BODY, headers=headers)

    assert response.status_code == 200  # Tags never stop a call.
    (record,) = recorder.records
    assert (record.run_id, record.instance_id, record.seed) == ("run-1", task, 0)
    (warning,) = logged(caplog, "llm_proxy_bad_tags")
    assert getattr(warning, "problem", None) == problem


async def test_without_a_run_of_its_own_records_the_run_the_tags_claim(
    caplog: pytest.LogCaptureFixture,
) -> None:
    gateway, recorder = Gateway(completion()), Recorder()
    async with running(gateway, recorder) as proxy, sandbox_client(proxy) as llm:
        await llm.complete([user("hi")], model=QWEN, tags=EXEC)

    (record,) = recorder.records
    assert (record.run_id, record.instance_id, record.seed) == ("run-1", "task-7", 2)
    assert logged(caplog, "llm_proxy_bad_tags") == []


# Limits.


async def test_forwards_at_most_its_share_and_a_few_more_at_once() -> None:
    in_flight, peak = 0, 0

    async def answer(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.2)
        in_flight -= 1
        return completion()

    gateway, recorder = Gateway(*(answer for _ in range(8))), Recorder()
    async with running(gateway, recorder, share=1) as proxy, raw(proxy) as http:
        responses = await asyncio.gather(*(http.post(PATH, json=BODY) for _ in range(8)))

    assert [response.status_code for response in responses] == [200] * 8
    assert peak == 5  # Its share of 1, plus 4.
    assert max(record.wait_s for record in recorder.records) > 0.1  # The rest waited.


async def test_answers_503_past_its_connection_limit() -> None:
    gateway, recorder = Gateway(), Recorder()
    async with running(gateway, recorder, share=1) as proxy:  # 10 connections at most.
        url = httpx.URL(proxy.base_url)
        idle = [await asyncio.open_connection(url.host, url.port) for _ in range(10)]
        try:
            async with raw(proxy) as http:
                response = await http.post(PATH, json=BODY)
        finally:
            for _, writer in idle:
                writer.close()
                await writer.wait_closed()

    assert response.status_code == 503
    assert gateway.requests == []
    assert sorted(record.http_status for record in recorder.records) == [0] * 10 + [503]


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


def request(token: str, body: bytes, *headers: bytes) -> bytes:
    """A whole POST to the proxy, with ``token`` and any more ``headers``."""
    head = [
        b"POST /chat/completions HTTP/1.1",
        b"Authorization: Bearer " + token.encode(),
        b"Content-Length: %d" % len(body),
        *headers,
    ]
    return b"\r\n".join([*head, b"", body])


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
        pytest.param(
            b"POST /chat/completions HTTP/1.1\r\nAUTH\r\nContent-Length: "
            + b"9" * 5000
            + b"\r\n\r\n",
            413,
            id="length-of-many-digits",
        ),
        pytest.param(
            b"POST /chat/completions HTTP/1.1\r\nAUTH\r\nContent-Length: "
            + b"0" * 30
            + b"2\r\n\r\n{}",
            403,  # Read as 2 bytes, then refused for naming no model.
            id="length-with-leading-zeros",
        ),
        pytest.param(
            b"POST /chat/completions HTTP/1.1\r\nAUTH\r\nContent-Length: 2\r\n"
            b"Content-Length: 50\r\n\r\n{}",
            400,
            id="length-twice",
        ),
        pytest.param(
            b"POST /chat/completions HTTP/1.1\r\nAUTH\r\nAUTH\r\nContent-Length: 2\r\n\r\n{}",
            400,
            id="token-twice",
        ),
        pytest.param(
            b"POST /chat/completions HTTP/1.1\r\nAUTH\r\nContent-Length: 2\r\n"
            b"Transfer-Encoding: chunked\r\nTransfer-Encoding: identity\r\n\r\n{}",
            400,
            id="transfer-encoding-twice",
        ),
        pytest.param(
            b"POST /chat/completions HTTP/1.1\r\nAUTH\r\nContent-Length: 2\r\n"
            b"X-Goaly-Call-Tags: {}\r\nX-Goaly-Call-Tags: {}\r\n\r\n{}",
            400,
            id="tags-twice",
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


async def test_answers_500_rather_than_nothing_when_it_fails(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def broken(response: httpx.Response, max_bytes: int) -> bytes:
        raise RuntimeError(f"a bug that quotes {KEY}")

    monkeypatch.setattr(llm_proxy, "_read_reply", broken)
    recorder = Recorder()
    async with running(Gateway(completion()), recorder) as proxy, raw(proxy) as http:
        response = await http.post(PATH, json=BODY)

    assert response.status_code == 500
    (record,) = recorder.records
    assert attempts(record) == [(SERVER_ERROR, 0)]
    (failed,) = logged(caplog, "llm_proxy_failed")
    assert getattr(failed, "error", None) == "RuntimeError"
    assert KEY not in JsonFormatter().format(failed)  # Its type only, never its text.


# Callers that go away.


async def test_logs_a_caller_that_went_away(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="goaly.harness.core.llm_proxy")
    recorder = Recorder()
    async with running(Gateway(), recorder) as proxy:
        answer = await exchange(proxy, b"POST /chat/comp")

    assert answer == b""
    assert requests_logged(caplog) == [{"request": 1, "model": "", "status": 0}]
    (record,) = recorder.records
    assert (record.status, record.http_status) == (CANCELLED, 0)


@pytest.mark.parametrize(
    "reset", [pytest.param(False, id="closed"), pytest.param(True, id="reset")]
)
async def test_a_caller_that_hangs_up_cancels_its_forward(reset: bool) -> None:
    arrived, cancelled = threading.Event(), threading.Event()

    async def hold(request: httpx.Request) -> httpx.Response:
        arrived.set()
        try:
            return await hang(request)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    recorder = Recorder()
    async with running(Gateway(hold), recorder) as proxy:
        url = httpx.URL(proxy.base_url)
        _, writer = await asyncio.open_connection(url.host, url.port)
        token = proxy.container_env()["GOALY_API_KEY"]
        writer.write(request(token, json.dumps(BODY).encode()) + b"and some bytes past it")
        await writer.drain()
        assert await asyncio.to_thread(arrived.wait, 5)
        if reset:  # Close with an RST, as a crashed process's socket would.
            linger = struct.pack("ii", 1, 0)
            writer.get_extra_info("socket").setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, linger)
        writer.close()
        await writer.wait_closed()
        assert await asyncio.to_thread(cancelled.wait, 5)  # The forward ended with the caller.

    (record,) = recorder.records
    assert (record.status, record.http_status) == (CANCELLED, 0)
    assert attempts(record) == [(CANCELLED, 0)]


# Concurrency, starting and stopping.


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

    recorder = Recorder()
    async with asyncio.timeout(5):  # Stopping does not wait for the call to end.
        async with running(Gateway(hold), recorder) as proxy:
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
            await http.post(PATH, json=BODY)
    (record,) = recorder.records
    assert attempts(record) == [(CANCELLED, 0)]


async def test_a_proxy_runs_once() -> None:
    proxy = LlmProxy(settings(), Recorder())
    async with proxy:
        pass

    with pytest.raises(RuntimeError, match="once"):
        async with proxy:
            pass


async def test_raises_when_it_cannot_listen(monkeypatch: pytest.MonkeyPatch) -> None:
    async def refuse(*args: object, **kwargs: object) -> object:
        raise OSError("no port")

    monkeypatch.setattr(asyncio, "start_server", refuse)
    with pytest.raises(OSError, match="no port"):
        async with LlmProxy(settings(), Recorder()):
            pass


async def enter_and_leave(proxy: LlmProxy) -> None:
    async with proxy:
        pass


async def test_raises_when_it_cannot_make_its_gateway_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(*args: object, **kwargs: object) -> httpx.AsyncClient:
        raise ValueError("no such SSL_CERT_FILE")

    monkeypatch.setattr(httpx, "AsyncClient", refuse)
    with pytest.raises(ValueError, match="SSL_CERT_FILE"):  # It once waited here forever.
        await asyncio.wait_for(enter_and_leave(LlmProxy(settings(), Recorder())), 5)


async def test_a_start_given_up_on_stops_the_proxy_once_it_serves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = threading.Event()
    start_server = asyncio.start_server

    async def slow(*args: Any, **kwargs: Any) -> asyncio.Server:
        await asyncio.to_thread(release.wait, 5)
        return await start_server(*args, **kwargs)

    monkeypatch.setattr(asyncio, "start_server", slow)
    before = set(threading.enumerate())
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(enter_and_leave(LlmProxy(settings(), Recorder())), 0.05)
    started = [thread for thread in threading.enumerate() if thread not in before]
    (thread,) = [thread for thread in started if thread.name == "llm-proxy"]
    release.set()
    await asyncio.to_thread(thread.join, 5)

    assert not thread.is_alive()


@pytest.mark.parametrize(
    "options",
    [
        pytest.param({"share": 0}, id="no-share"),
        pytest.param({"max_body_bytes": 0}, id="no-body"),
        pytest.param({"max_reply_bytes": 0}, id="no-reply"),
        pytest.param({"timeout_s": 0.0}, id="no-time"),
    ],
)
def test_refuses_options_out_of_range(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="positive"):
        LlmProxy(settings(), Recorder(), **options)


def test_has_no_url_before_it_starts() -> None:
    with pytest.raises(RuntimeError, match="not started"):
        LlmProxy(settings(), Recorder()).container_env()


@contextmanager
def local_gateway() -> Iterator[str]:
    """A gateway on a local port that answers every call with a completion; yields its URL."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers["Content-Length"]))
            reply = completion()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(reply.content)))
            self.end_headers()
            self.wfile.write(reply.content)

        def log_message(self, format: str, *args: object) -> None:
            """Keep the test output clean."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


async def test_reaches_the_gateway_directly_whatever_proxy_the_environment_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")  # Nothing listens there.
    with local_gateway() as base_url:
        host = Settings(api_key=SecretStr(KEY), base_url=base_url)
        async with LlmProxy(host, Recorder()) as proxy, raw(proxy) as http:
            response = await http.post(PATH, json=BODY)

    assert response.status_code == 200


# The Unix socket (V28): the solve container can open no IP socket.


async def test_serves_the_sandbox_client_over_its_unix_socket(short_dir: Path) -> None:
    path = short_dir / "llm" / "proxy.sock"
    gateway, recorder = Gateway(completion(), completion()), Recorder()
    for _ in range(2):  # The second run finds the socket's directory there, and takes it.
        async with running(gateway, recorder, uds=path) as proxy:
            assert proxy.container_env()["GOALY_LLM_UDS"] == str(path)
            assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
            assert stat.S_ISSOCK(path.stat().st_mode)
            # No TCP: a host that does not resolve, so only the socket can carry the call.
            async with sandbox_client(proxy, base_url="http://llm-proxy.invalid") as llm:
                reply = await llm.complete([user("hi")], model=QWEN, tags=EXEC)
            assert reply.message.content == "391"
        assert not path.exists()  # The proxy removed its socket when it stopped.

    assert [record.status for record in recorder.records] == [OK, OK]


async def test_refuses_a_wrong_token_over_the_unix_socket_too(short_dir: Path) -> None:
    path = short_dir / "llm" / "proxy.sock"
    gateway = Gateway()
    async with (
        running(gateway, uds=path) as proxy,
        raw(proxy, "not-the-token", uds=path) as http,
    ):
        response = await http.post(PATH, json=BODY)

    assert response.status_code == 401
    assert gateway.requests == []


async def test_refuses_a_socket_directory_that_is_not_its_users(
    short_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = short_dir / "llm"
    directory.mkdir(mode=0o700)
    uid = os.getuid()
    monkeypatch.setattr(os, "getuid", lambda: uid + 1)  # As if another user had made it.

    with pytest.raises(PermissionError, match="mode 0700"):
        await enter_and_leave(LlmProxy(settings(), Recorder(), uds=directory / "proxy.sock"))


async def test_makes_its_socket_directory_0700_whatever_the_umask(short_dir: Path) -> None:
    path = short_dir / "llm" / "proxy.sock"
    umask = os.umask(0o277)  # mkdir alone would make the directory 0500.
    try:
        await enter_and_leave(LlmProxy(settings(), Recorder(), uds=path))
    finally:
        os.umask(umask)

    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


async def test_refuses_a_socket_directory_that_others_may_enter(short_dir: Path) -> None:
    directory = short_dir / "llm"
    directory.mkdir()
    directory.chmod(0o755)

    with pytest.raises(PermissionError, match="mode 0700"):
        async with LlmProxy(settings(), Recorder(), uds=directory / "proxy.sock"):
            pass
    assert list(directory.iterdir()) == []


# Logs and secrets.


async def test_logs_each_request_without_the_key_the_token_or_the_body(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    gateway, recorder = Gateway(completion()), Recorder()
    async with running(gateway, recorder) as proxy:
        token = proxy.container_env()["GOALY_API_KEY"]
        async with raw(proxy) as http:
            await http.post(PATH, json=BODY)
            await http.post(PATH, json={**BODY, "model": GLM})
        async with raw(proxy, "not-the-token") as http:
            await http.post(PATH, json=BODY)

    assert requests_logged(caplog) == [
        {"request": 1, "model": QWEN, "status": 200},
        {"request": 2, "model": GLM, "status": 403},
        {"request": 3, "model": "", "status": 401},
    ]
    latencies = [getattr(entry, "latency_s", None) for entry in logged(caplog, "llm_proxy_request")]
    assert all(isinstance(value, float) and value >= 0 for value in latencies)
    text = "\n".join(JsonFormatter().format(entry) for entry in caplog.records)
    text += "\n".join(to_json_line(record) for record in recorder.records)
    assert KEY not in text
    assert token not in text
    assert "a prompt nobody may log" not in text
