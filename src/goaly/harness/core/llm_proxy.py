"""A key-injecting proxy to the model gateway, for code that must not hold the key (spec V24).

Candidate code solves its tasks in a sandbox container that holds no secret, and its LLM client
talks to this proxy instead of the gateway. The proxy runs in the trusted host process, listens
on 127.0.0.1 on a port of its own, and forwards each call to the gateway with the real key:

- it needs the run's bearer token, which it makes itself: any other gets 401, the real key too;
- it serves only ``POST /chat/completions``, the client's one request: any other path gets 404
  and any other method 405;
- it forwards only the executor model: any other gets 403, logged as ``llm_proxy_refused``;
- it passes the gateway's reply on as it came: the status, the headers (``x-litellm-response-cost``
  and ``Retry-After`` among them) and the body. The reply is read whole and sent with its exact
  length, so one cut off on the way in becomes a 502, never a truncated body;
- it bounds a request's size, and a forward's time, past the client's own 300 s.

It logs one line per request: its number, model, status and latency, but never the key, the
token or a body::

    async with LlmProxy(load_settings()) as proxy:
        run_the_sandbox(env=proxy.container_env())  # Its GOALY_API_KEY is the run's token.

The proxy serves on a thread of its own, with its own event loop. Then a moment of blocking work
in the caller's loop, such as the runner's, never holds up the sandbox's calls, and the proxy's
traffic never slows the caller.
"""

import asyncio
import concurrent.futures
import functools
import hmac
import http
import itertools
import json
import logging
import secrets
import threading
from collections.abc import Callable
from types import TracebackType
from typing import Final, Self, cast

import httpx
from pydantic import SecretStr

from goaly.harness.core.llm_models import EXECUTOR_MODEL
from goaly.harness.core.settings import Settings

__all__ = ["LlmProxy"]

_logger = logging.getLogger(__name__)

_PATH: Final = "/chat/completions"
_HEAD_BYTES: Final = 64 * 1024  # The longest request line and headers it reads.
_ARRIVAL_S: Final = 30.0  # How long a request may take to arrive in full.
# Headers about one hop of the connection, or about the body's encoding and length, which the
# proxy sets itself: it sends the body decoded, with its own length, and closes the connection.
_NOT_PASSED_ON: Final = frozenset(
    {b"connection", b"keep-alive", b"proxy-authenticate", b"proxy-authorization", b"te"}
    | {b"trailer", b"transfer-encoding", b"upgrade", b"content-encoding", b"content-length"}
)


class _Refusal(Exception):  # noqa: N818 - A refusal is an answer, not an error.
    """The proxy answers the request itself, with ``status``."""

    def __init__(self, status: int, message: str, model: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.model = model


class LlmProxy:
    """A key-injecting proxy to the gateway that ``settings`` name; see the module docstring.

    Each instance serves one run, with its own token, and runs once. ``transport`` is for tests.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        max_body_bytes: int = 16 * 2**20,
        timeout_s: float = 330.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if max_body_bytes < 1 or not timeout_s > 0:
            raise ValueError("the proxy needs a positive body size limit and timeout")
        self._settings = settings
        self._max_body_bytes = max_body_bytes
        self._timeout_s = timeout_s
        self._transport = transport
        self._token = SecretStr(secrets.token_urlsafe(32))
        self._requests = itertools.count(1)
        self._started = False
        self._port: int | None = None
        # The port it listens on, and how to stop it from another thread.
        self._ready: concurrent.futures.Future[tuple[int, Callable[[], object]]] = (
            concurrent.futures.Future()
        )
        self._thread = threading.Thread(target=self._run, name="llm-proxy", daemon=True)
        self._stop_serving: Callable[[], object] = lambda: None  # Set once it serves.
        self._handlers: set[asyncio.Task[object]] = set()

    @property
    def base_url(self) -> str:
        """Where the sandbox reaches the proxy."""
        if self._port is None:
            raise RuntimeError("the proxy has not started")
        return f"http://127.0.0.1:{self._port}"

    def container_env(self) -> dict[str, str]:
        """Return the sandbox client's settings: the run's token for a key, this proxy for a URL."""
        return {"GOALY_API_KEY": self._token.get_secret_value(), "GOALY_BASE_URL": self.base_url}

    async def __aenter__(self) -> Self:
        if self._started:
            raise RuntimeError("a proxy runs once; make another for the next run")
        self._started = True
        self._thread.start()
        port, stop_serving = await asyncio.wrap_future(self._ready)
        self._port, self._stop_serving = port, stop_serving
        _logger.info("LLM proxy started", extra={"event": "llm_proxy_started", "port": self._port})
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop serving. Calls still in flight are cut off, and their connections closed."""
        self._stop_serving()
        await asyncio.to_thread(self._thread.join)
        _logger.info("LLM proxy stopped", extra={"event": "llm_proxy_stopped"})

    def _run(self) -> None:
        asyncio.run(self._serve())

    async def _serve(self) -> None:
        try:
            loop, stop = asyncio.get_running_loop(), asyncio.Event()
            server = await asyncio.start_server(self._handle, "127.0.0.1", 0, limit=_HEAD_BYTES)
        except BaseException as error:  # noqa: BLE001 - The caller must hear of any failure here.
            self._ready.set_exception(error)
            return
        key = self._settings.api_key.get_secret_value()
        async with httpx.AsyncClient(
            base_url=self._settings.base_url,
            headers={"Authorization": f"Bearer {key}"},
            timeout=self._timeout_s,
            transport=self._transport,
        ) as self._upstream:
            port = server.sockets[0].getsockname()[1]
            self._ready.set_result((port, functools.partial(loop.call_soon_threadsafe, stop.set)))
            await stop.wait()
            server.close()
            handlers = list(self._handlers)
            for handler in handlers:
                handler.cancel()
            await asyncio.gather(*handlers, return_exceptions=True)
            await server.wait_closed()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = cast(asyncio.Task[object], asyncio.current_task())  # start_server made it one.
        self._handlers.add(task)
        number, began = next(self._requests), asyncio.get_running_loop().time()
        status, model = 0, ""
        try:
            try:
                status, model = await self._answer(reader, writer, number)
            except _Refusal as refusal:
                status, model = refusal.status, refusal.model
                writer.write(_error(refusal.status, str(refusal)))
                await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass  # The caller went away; status 0 says that no answer reached it.
        finally:
            self._handlers.discard(task)
            writer.close()
            _logger.info(
                "LLM proxy request",
                extra={
                    "event": "llm_proxy_request",
                    "request": number,
                    "model": model,
                    "status": status,
                    "latency_s": asyncio.get_running_loop().time() - began,
                },
            )

    async def _answer(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, number: int
    ) -> tuple[int, str]:
        """Check the request, then forward it; refuse it with a _Refusal otherwise."""
        try:
            async with asyncio.timeout(_ARRIVAL_S):
                method, path, headers = await _read_head(reader)
                expected = f"Bearer {self._token.get_secret_value()}".encode()
                if not hmac.compare_digest(headers.get("authorization", b""), expected):
                    raise _Refusal(401, "the proxy needs this run's token")
                if path != _PATH:
                    raise _Refusal(404, f"the proxy serves only {_PATH}")
                if method != "POST":
                    raise _Refusal(405, f"the proxy serves only POST {_PATH}")
                body = await _read_body(reader, headers, self._max_body_bytes)
        except TimeoutError:
            raise _Refusal(408, "the request took too long to arrive") from None
        model = _model_of(body)
        if model != EXECUTOR_MODEL:
            _logger.warning(
                "LLM proxy refused a model other than the executor",
                extra={"event": "llm_proxy_refused", "request": number, "model": model},
            )
            raise _Refusal(403, f"the proxy forwards only {EXECUTOR_MODEL}", model)
        return await self._forward(body, writer), model

    async def _forward(self, body: bytes, writer: asyncio.StreamWriter) -> int:
        """Send ``body`` to the gateway, pass its reply on, and return the reply's status."""
        try:
            async with asyncio.timeout(self._timeout_s):
                response = await self._upstream.post(
                    _PATH, content=body, headers={"Content-Type": "application/json"}
                )
        except (TimeoutError, httpx.TimeoutException):
            raise _Refusal(504, "the gateway did not answer in time") from None
        except httpx.RequestError as error:
            raise _Refusal(502, f"the gateway failed: {type(error).__name__}") from None
        writer.write(_response(response.status_code, response.headers.raw, response.content))
        await writer.drain()
        return response.status_code


async def _read_head(reader: asyncio.StreamReader) -> tuple[str, str, dict[str, bytes]]:
    """Read a request line and headers: its method, path and headers, by lowercase name."""
    try:
        head = await reader.readuntil(b"\r\n\r\n")
    except asyncio.LimitOverrunError:
        raise _Refusal(431, "the request's headers are too long") from None
    request_line, *lines = head[:-4].split(b"\r\n")
    parts = request_line.decode("latin-1").split(" ")
    if len(parts) != 3 or not parts[2].startswith("HTTP/1."):
        raise _Refusal(400, "not an HTTP/1.1 request")
    headers: dict[str, bytes] = {}
    for line in lines:
        name, colon, value = line.partition(b":")
        if not colon:
            raise _Refusal(400, "a malformed header")
        headers[name.strip().lower().decode("latin-1")] = value.strip()
    return parts[0], parts[1], headers


async def _read_body(
    reader: asyncio.StreamReader, headers: dict[str, bytes], max_bytes: int
) -> bytes:
    if "transfer-encoding" in headers or "content-length" not in headers:
        raise _Refusal(411, "the proxy needs a Content-Length")
    length = headers["content-length"]
    if not length.isdigit():
        raise _Refusal(400, "a malformed Content-Length")
    if int(length) > max_bytes:
        raise _Refusal(413, f"the body is over {max_bytes} bytes")
    return await reader.readexactly(int(length))


def _model_of(body: bytes) -> str:
    """Return the model a request body names, "" if it names none, cut short for the log."""
    try:
        payload = json.loads(body)
    except (ValueError, RecursionError):
        raise _Refusal(400, "the body is not JSON") from None
    if not isinstance(payload, dict):
        raise _Refusal(400, "the body is not a JSON object")
    model = payload.get("model")
    return model[:100] if isinstance(model, str) else ""


def _response(status: int, headers: list[tuple[bytes, bytes]], body: bytes) -> bytes:
    """A whole response: ``status``, the ``headers`` to pass on, its own framing, and ``body``."""
    lines = [f"HTTP/1.1 {status} {_reason(status)}".encode()]
    lines += [name + b": " + value for name, value in headers if name.lower() not in _NOT_PASSED_ON]
    lines += [b"Content-Length: %d" % len(body), b"Connection: close"]
    return b"\r\n".join([*lines, b"", body])


def _error(status: int, message: str) -> bytes:
    """A whole response for a refusal, its body shaped as the gateway's own errors."""
    error = {"message": message, "type": None, "param": None, "code": str(status)}
    body = json.dumps({"error": error}).encode()
    return _response(status, [(b"content-type", b"application/json")], body)


def _reason(status: int) -> str:
    try:
        return http.HTTPStatus(status).phrase
    except ValueError:  # A status the standard library does not know.
        return ""
