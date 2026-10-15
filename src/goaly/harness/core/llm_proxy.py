"""A key-injecting proxy to the model gateway, for code that must not hold the key (spec §6.1).

Candidate code solves its tasks in a sandbox container that holds no secret, and its LLM client
talks to this proxy instead of the gateway. The proxy runs in the trusted host process. It
listens on 127.0.0.1, on a port of its own, and also on a Unix socket when given one: the solve
container can open no IP socket at all (V28). It forwards each call to the gateway with the
real key:

- it needs the run's bearer token, which it makes itself: any other gets 401, the real key too;
- it serves only ``POST /chat/completions``, the client's one request: any other path gets 404,
  and any other method 405;
- it rebuilds the body from the keys core's client writes (``llm.REQUEST_KEYS``). Any other key
  gets 400, logged as ``llm_proxy_refused``; so do a key given twice and a number that is not
  finite (NaN, or one too large for a float, which would go on as Infinity). It sets ``stream`` to
  false and ``n`` to 1, and caps ``max_tokens`` at Qwen's profile, so pack code can ask for no
  fallback model, no second reply and no longer one;
- it forwards only the executor model: any other gets 403, logged the same way;
- it passes the gateway's reply on: its status, its headers (``x-litellm-response-cost`` and
  ``Retry-After`` among them) and its body, with the key taken out of all three. The body goes on
  decoded, without its Content-Encoding, and with its exact length, so a reply cut off on the
  way in becomes a 502, never a truncated body. So does a reply over the size limit, or one the
  gateway got from a fallback model;
- it bounds a request's size, a reply's size, the calls it forwards at once (the run's share of
  the executor), the connections it holds, and each forward's time, past the client's own 300 s.
  A caller that hangs up cancels its forward.

It hands one ``LlmCallRecord`` per request to a ``CallRecorder``, refusals included. The decision
rules count cost and tokens from these records, as the container's own call log is not trusted
(spec §6.1). Each holds the model, the status, the latency, the tokens and the cost. Its run is
the proxy's own. Its task and seed come from the ``X-Goaly-Call-Tags`` header that core's client
sends: they are the container's claim, good for attribution only. The proxy also logs one line
per request, with its number, model, status and latency, but never the key, the token or a
body::

    async with LlmProxy(load_settings(), JsonlCallRecorder(path), run_id=run_id) as proxy:
        run_the_sandbox(env=proxy.container_env())  # Its GOALY_API_KEY is the run's token.

The proxy serves on a thread of its own, with its own event loop. Then a moment of blocking work
in the caller's loop, such as the runner's, never holds up the sandbox's calls, and the proxy's
traffic never slows the caller. It hands each record over on the caller's loop, as the client
does, so the recorder needs no lock.

This module reads the real key, so pack code must never import it: the pack-code allowlist
leaves it out, with the other LLM modules (spec §8.5).
"""

import asyncio
import concurrent.futures
import contextlib
import functools
import hmac
import http
import itertools
import json
import logging
import math
import os
import secrets
import stat
import threading
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Final, Self, cast

import httpx
from pydantic import SecretStr

from goaly.contracts.gen.llm_call_pb2 import (
    CallPurpose,
    LlmCallAttempt,
    LlmCallRecord,
    LlmCallStatus,
)
from goaly.harness.core.llm import (
    CALL_TAGS_HEADER,
    REQUEST_KEYS,
    THINKING_SWITCHES,
    CallRecorder,
    judge_response,
)
from goaly.harness.core.llm_models import EXECUTOR_MODEL, QWEN_PROFILE
from goaly.harness.core.llm_throttle import DEFAULT_SHARES
from goaly.harness.core.settings import Settings

__all__ = ["LlmProxy"]

_logger = logging.getLogger(__name__)

_S = LlmCallStatus
_HOST: Final = "127.0.0.1"
_PATH: Final = "/chat/completions"
_HEAD_BYTES: Final = 64 * 1024  # The longest request line and headers it reads.
_ARRIVAL_S: Final = 30.0  # How long a request may take to arrive in full.
_LINGER_S: Final = 1.0  # How long it reads on after answering; see _linger.
_TAGS_HEADER: Final = CALL_TAGS_HEADER.lower()
_TAGS_BYTES: Final = 1024  # The longest X-Goaly-Call-Tags it reads.
_TAG_KEYS: Final = frozenset({"run_id", "instance_id", "seed", "purpose"})
_SLACK: Final = 4  # Forwards past the share, for calls whose clients have just given up.
_MAX_TOKENS: Final = cast(int, QWEN_PROFILE.params.max_tokens)  # 16384 (spec §5.4).
# A request may give each of these once: with two, the proxy and the gateway could each read
# a different one.
_ONCE: Final = frozenset({"authorization", "content-length", "transfer-encoding", _TAGS_HEADER})
_JSON_HEADERS: Final = {"Content-Type": "application/json"}
# Headers about one hop of the connection, or about the body's encoding and length, which the
# proxy sets itself: it sends the body decoded, with its own length, and closes the connection.
_NOT_PASSED_ON: Final = frozenset(
    {b"connection", b"keep-alive", b"proxy-authenticate", b"proxy-authorization", b"te"}
    | {b"trailer", b"transfer-encoding", b"upgrade", b"content-encoding", b"content-length"}
)

type _Ready = concurrent.futures.Future[tuple[str, Callable[[], object]]]


class _Refusal(Exception):  # noqa: N818 - A refusal is an answer, not an error.
    """The proxy answers the request itself, with ``status``; ``call_status`` goes on record."""

    def __init__(
        self,
        status: int,
        message: str,
        call_status: LlmCallStatus.ValueType = _S.LLM_CALL_STATUS_CLIENT_ERROR,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.call_status = call_status


@dataclass(frozen=True, slots=True)
class _Serving:
    """What the proxy's handlers share: the gateway's client, the forward slots, the caller."""

    upstream: httpx.AsyncClient
    slots: asyncio.Semaphore
    caller: asyncio.AbstractEventLoop  # Where the recorder takes its records.


class LlmProxy:
    """A key-injecting proxy to the gateway that ``settings`` name; see the module docstring.

    Each instance serves one run, with its own token, and runs once.

    - ``recorder`` gets each request's record, on the event loop that entered the proxy.
    - ``run_id`` is the run that every record names. Without it, a record names the run that its
      request's tags claim.
    - ``uds`` is a Unix socket to listen on as well. The proxy makes its directory, mode 0700, or
      checks that the one there is this user's own with that mode, and removes the socket when
      it stops.
    - ``share`` is how many calls it forwards at once: the run's share of the executor's budget
      (spec §6.2), plus a few. More calls wait, and past twice that many connections it answers
      503.
    - ``transport`` is for tests.
    """

    def __init__(
        self,
        settings: Settings,
        recorder: CallRecorder,
        *,
        run_id: str | None = None,
        uds: Path | None = None,
        share: int = DEFAULT_SHARES[EXECUTOR_MODEL],
        max_body_bytes: int = 4 * 2**20,
        max_reply_bytes: int = 4 * 2**20,
        timeout_s: float = 330.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if min(share, max_body_bytes, max_reply_bytes) < 1 or not timeout_s > 0:
            raise ValueError("the proxy needs a positive share, size limits and timeout")
        self._settings = settings
        self._recorder = recorder
        self._run_id = run_id
        self._uds = uds
        self._forwards = share + _SLACK
        self._max_connections = 2 * self._forwards
        self._max_body_bytes = max_body_bytes
        self._max_reply_bytes = max_reply_bytes
        self._timeout_s = timeout_s
        self._transport = transport
        self._token = SecretStr(secrets.token_urlsafe(32))
        self._requests = itertools.count(1)
        self._started = False
        self._base_url: str | None = None
        self._ready: _Ready = concurrent.futures.Future()  # Its URL, and how to stop it.
        self._stop_serving: Callable[[], object] = lambda: None  # Set once it serves.
        self._join: Callable[[], None] = lambda: None  # Set once its thread starts.
        self._handlers: set[asyncio.Task[object]] = set()

    @property
    def base_url(self) -> str:
        """Where the sandbox reaches the proxy over TCP; over the Unix socket, its host only."""
        if self._base_url is None:
            raise RuntimeError("the proxy has not started")
        return self._base_url

    def container_env(self) -> dict[str, str]:
        """Return the sandbox client's settings: the run's token for a key, and this proxy.

        With a Unix socket, GOALY_LLM_UDS names it by its path on the host. A container that
        mounts its directory somewhere else must change that value.
        """
        env = {"GOALY_API_KEY": self._token.get_secret_value(), "GOALY_BASE_URL": self.base_url}
        if self._uds is not None:
            env["GOALY_LLM_UDS"] = str(self._uds)
        return env

    async def __aenter__(self) -> Self:
        if self._started:
            raise RuntimeError("a proxy runs once; make another for the next run")
        self._started = True
        thread = threading.Thread(
            target=self._run, args=(asyncio.get_running_loop(),), name="llm-proxy", daemon=True
        )
        thread.start()
        self._join = thread.join
        try:
            # Shielded, so that a start abandoned here leaves _ready to the thread to resolve.
            ready = await asyncio.shield(asyncio.wrap_future(self._ready))
        except BaseException:
            self._ready.add_done_callback(_stop_once_serving)  # The thread must not serve on.
            raise
        self._base_url, self._stop_serving = ready
        _logger.info(
            "LLM proxy started",
            extra={"event": "llm_proxy_started", "url": self._base_url, "uds": str(self._uds)},
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop serving. Calls in flight are cut off, and recorded as cancelled.

        Every record is with the recorder by the time this returns.
        """
        self._stop_serving()
        await asyncio.to_thread(self._join)
        _logger.info("LLM proxy stopped", extra={"event": "llm_proxy_stopped"})

    def _run(self, caller: asyncio.AbstractEventLoop) -> None:
        asyncio.run(self._serve(caller))

    async def _serve(self, caller: asyncio.AbstractEventLoop) -> None:
        """Serve until told to stop. A failure to start goes to __aenter__, through _ready."""
        async with contextlib.AsyncExitStack() as stack:  # Closes, at the end, what it opened.
            try:
                upstream = await stack.enter_async_context(
                    httpx.AsyncClient(
                        base_url=self._settings.base_url,
                        headers={
                            "Authorization": f"Bearer {self._settings.api_key.get_secret_value()}"
                        },
                        timeout=self._timeout_s,
                        limits=httpx.Limits(
                            max_connections=self._forwards,
                            max_keepalive_connections=self._forwards,
                        ),
                        transport=self._transport,
                        trust_env=False,  # The settings say where the gateway is: no .netrc.
                    )
                )
                serving = _Serving(upstream, asyncio.Semaphore(self._forwards), caller)
                handle = functools.partial(self._handle, serving)
                servers: list[asyncio.Server] = []
                stack.push_async_callback(self._shut, servers)
                servers.append(await asyncio.start_server(handle, _HOST, 0, limit=_HEAD_BYTES))
                if self._uds is not None:
                    _private_directory(self._uds.parent)
                    servers.append(
                        await asyncio.start_unix_server(handle, self._uds, limit=_HEAD_BYTES)
                    )
                host, port = servers[0].sockets[0].getsockname()[:2]
            except BaseException as error:  # noqa: BLE001 - __aenter__ raises it.
                self._ready.set_exception(error)
                return
            loop, stop = asyncio.get_running_loop(), asyncio.Event()
            stop_serving = functools.partial(loop.call_soon_threadsafe, stop.set)
            self._ready.set_result((f"http://{host}:{port}", stop_serving))
            await stop.wait()

    async def _shut(self, servers: list[asyncio.Server]) -> None:
        """Stop listening, cut off the calls in flight, and remove the Unix socket it made."""
        for server in servers:
            server.close()
        handlers = list(self._handlers)
        for handler in handlers:
            handler.cancel()
        await asyncio.gather(*handlers, return_exceptions=True)
        for server in servers:
            await server.wait_closed()
        if self._uds is not None and len(servers) == 2:
            self._uds.unlink(missing_ok=True)

    async def _handle(
        self, serving: _Serving, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Serve one connection: one request, one record, one answer and one log line."""
        task = cast(asyncio.Task[object], asyncio.current_task())  # start_server made it one.
        self._handlers.add(task)
        loop = asyncio.get_running_loop()
        number, began, answered = next(self._requests), loop.time(), 0
        record = LlmCallRecord(
            call_id=uuid.uuid4().hex,
            run_id=self._run_id or "",
            purpose=CallPurpose.CALL_PURPOSE_EXEC,
        )
        record.started_at.GetCurrentTime()
        try:
            try:
                answered, answer = await self._reply(serving, reader, record, number)
            finally:  # On record before any answer goes out: a caller that leaves loses none.
                self._close_record(serving, record, loop.time() - began)
            writer.write(answer)
            await writer.drain()
            await _linger(reader, writer)
        except (ConnectionError, asyncio.IncompleteReadError):
            pass  # The caller went away; its record says how far its call got.
        finally:
            self._handlers.discard(task)
            writer.close()
            _logger.info(
                "LLM proxy request",
                extra={
                    "event": "llm_proxy_request",
                    "request": number,
                    "call_id": record.call_id,
                    "model": record.model,
                    "status": answered,
                    "outcome": LlmCallStatus.Name(record.status),
                    "latency_s": record.latency_s,
                },
            )

    async def _reply(
        self,
        serving: _Serving,
        reader: asyncio.StreamReader,
        record: LlmCallRecord,
        number: int,
    ) -> tuple[int, bytes]:
        """Return the answer to the request on ``reader``: its status, and the whole of it.

        ``record`` is filled in on the way. Only a caller that goes away, and cancellation,
        raise.
        """
        try:
            if len(self._handlers) > self._max_connections:
                busy = _S.LLM_CALL_STATUS_SERVER_ERROR
                raise _Refusal(503, "the proxy holds too many connections", busy)
            return await self._answer(serving, reader, record, number)
        except _Refusal as refusal:
            return _refused(record, refusal)
        except (ConnectionError, asyncio.IncompleteReadError):
            raise
        except Exception as error:  # noqa: BLE001 - A bug: answer, not leave the caller waiting.
            _logger.error(
                "the LLM proxy failed on a request",
                extra={
                    "event": "llm_proxy_failed",
                    "request": number,
                    "error": type(error).__name__,
                },
            )
            failure = _Refusal(500, "the proxy failed", _S.LLM_CALL_STATUS_SERVER_ERROR)
            return _refused(record, failure)

    async def _answer(
        self,
        serving: _Serving,
        reader: asyncio.StreamReader,
        record: LlmCallRecord,
        number: int,
    ) -> tuple[int, bytes]:
        """Check the request, then forward it; refuse it with a _Refusal otherwise."""
        try:
            async with asyncio.timeout(_ARRIVAL_S):
                method, path, headers = await _read_head(reader)
                expected = f"Bearer {self._token.get_secret_value()}".encode()
                if not hmac.compare_digest(headers.get("authorization", b""), expected):
                    raise _Refusal(401, "the proxy needs this run's token")
                self._tag(record, headers.get(_TAGS_HEADER), number)
                if path != _PATH:
                    raise _Refusal(404, f"the proxy serves only {_PATH}")
                if method != "POST":
                    raise _Refusal(405, f"the proxy serves only POST {_PATH}")
                body = await _read_body(reader, headers, self._max_body_bytes)
        except TimeoutError:
            raise _Refusal(408, "the request took too long to arrive") from None
        return await self._forward(serving, reader, self._rebuild(body, record, number), record)

    def _tag(self, record: LlmCallRecord, raw: bytes | None, number: int) -> None:
        """Put the task and seed that the request's tags claim on ``record``.

        The run they claim goes on it too, if the proxy has no run of its own.
        """
        tags = _read_tags(raw)
        if tags is None or (self._run_id is not None and tags[0] != self._run_id):
            problem = "missing" if raw is None else "malformed" if tags is None else "another run"
            _logger.warning(
                "LLM proxy request whose tags cannot be trusted for attribution",
                extra={"event": "llm_proxy_bad_tags", "request": number, "problem": problem},
            )
        if tags is not None:
            run_id, record.instance_id, record.seed = tags
            if self._run_id is None:
                record.run_id = run_id

    def _rebuild(self, body: bytes, record: LlmCallRecord, number: int) -> bytes:
        """Return the body to forward: only the keys the client writes, ``stream`` and ``n``."""
        payload = _parse(body)
        model = payload.get("model")
        record.model = model[:100] if isinstance(model, str) else ""
        switches = payload.get("chat_template_kwargs", {})
        if not isinstance(switches, dict):
            raise _Refusal(400, "chat_template_kwargs must be an object")
        unknown = sorted(set(payload) - REQUEST_KEYS) + sorted(
            f"chat_template_kwargs.{name}" for name in set(switches) - set(THINKING_SWITCHES)
        )
        if unknown:
            keys = [key[:40] for key in unknown[:10]]
            _logger.warning(
                "LLM proxy refused keys the client never sends",
                extra={"event": "llm_proxy_refused", "request": number, "keys": keys},
            )
            raise _Refusal(400, f"the proxy forwards no {', '.join(keys)}")
        if model != EXECUTOR_MODEL:
            _logger.warning(
                "LLM proxy refused a model other than the executor",
                extra={"event": "llm_proxy_refused", "request": number, "model": record.model},
            )
            raise _Refusal(403, f"the proxy forwards only {EXECUTOR_MODEL}")
        asked = payload.get("max_tokens", _MAX_TOKENS)
        if isinstance(asked, bool) or not isinstance(asked, int) or asked < 1:
            raise _Refusal(400, "max_tokens must be a positive integer")
        payload.update(max_tokens=min(asked, _MAX_TOKENS), stream=False, n=1)
        return json.dumps(payload, separators=(",", ":")).encode()

    async def _forward(
        self, serving: _Serving, reader: asyncio.StreamReader, body: bytes, record: LlmCallRecord
    ) -> tuple[int, bytes]:
        """Forward ``body``, and return the answer to pass on, unless the caller hangs up first."""
        exchange = asyncio.create_task(self._exchange(serving, body, record))
        hang_up = asyncio.create_task(_hang_up(reader))
        try:
            await asyncio.wait({exchange, hang_up}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            hang_up.cancel()
            exchange.cancel()  # Nothing to cancel if it has answered.
            # Once both are over, the reader is free again, and the attempt is on record.
            await asyncio.wait({exchange, hang_up})
        if exchange.cancelled():
            raise ConnectionResetError("the caller hung up")
        return exchange.result()

    async def _exchange(
        self, serving: _Serving, body: bytes, record: LlmCallRecord
    ) -> tuple[int, bytes]:
        """Wait for a forward slot, send ``body`` to the gateway, and return the answer."""
        attempt = record.attempts.add()
        attempt.started_at.GetCurrentTime()
        loop = asyncio.get_running_loop()
        began, sent = loop.time(), None
        try:
            async with serving.slots:
                sent = loop.time()
                return await self._send(serving.upstream, body, record, attempt)
        except asyncio.CancelledError:
            attempt.status = _S.LLM_CALL_STATUS_CANCELLED
            raise
        finally:
            now = loop.time()
            attempt.wait_s = (now if sent is None else sent) - began
            attempt.latency_s = 0.0 if sent is None else now - sent

    async def _send(
        self,
        upstream: httpx.AsyncClient,
        body: bytes,
        record: LlmCallRecord,
        attempt: LlmCallAttempt,
    ) -> tuple[int, bytes]:
        """Send ``body`` once, judge the gateway's reply as the client would, and return it."""
        try:
            async with (
                asyncio.timeout(self._timeout_s),
                upstream.stream("POST", _PATH, content=body, headers=_JSON_HEADERS) as response,
            ):
                content = await _read_reply(response, self._max_reply_bytes)
        except (TimeoutError, httpx.TimeoutException):
            attempt.status = _S.LLM_CALL_STATUS_TIMEOUT
            raise _Refusal(504, "the gateway did not answer in time") from None
        except httpx.TransportError as error:
            attempt.status = _S.LLM_CALL_STATUS_CONNECTION_ERROR
            raise _Refusal(502, f"the gateway failed: {type(error).__name__}") from None
        except httpx.RequestError as error:  # For example, a body that cannot be decoded.
            attempt.status = _S.LLM_CALL_STATUS_INVALID_RESPONSE
            unreadable = f"the gateway's reply is unreadable: {type(error).__name__}"
            raise _Refusal(502, unreadable) from None
        code, secret = response.status_code, self._settings.api_key.get_secret_value()
        attempt.http_status = code
        headers = [
            (name, value)
            for name, value in response.headers.raw
            if name.lower() not in _NOT_PASSED_ON
        ]
        if content is None:  # Too long to pass on, but billed all the same.
            empty = httpx.Response(code, headers=headers)
            attempt.status = judge_response(empty, record, secret=secret)
            raise _Refusal(502, f"the gateway's reply is over {self._max_reply_bytes} bytes")
        reply = httpx.Response(code, headers=headers, content=content)
        attempt.status = judge_response(reply, record, secret=secret)
        if response.headers.get("x-litellm-attempted-fallbacks", "0").strip() != "0":
            attempt.status = _S.LLM_CALL_STATUS_INVALID_RESPONSE
            raise _Refusal(502, "the gateway answered with a fallback model")
        passed_on = [(name, _without(secret, value)) for name, value in headers]
        return code, _response(code, passed_on, _without(secret, content))

    def _close_record(self, serving: _Serving, record: LlmCallRecord, latency_s: float) -> None:
        """Complete ``record``, and hand a copy of it to the recorder, on the caller's loop."""
        if record.attempts:  # It was forwarded: the call ended as its attempt did.
            record.status = record.attempts[-1].status
            record.http_status = record.attempts[-1].http_status
        if not record.status:  # No answer at all: the caller went away, or the proxy stopped.
            record.status = _S.LLM_CALL_STATUS_CANCELLED
        record.latency_s = latency_s
        record.wait_s = sum(attempt.wait_s for attempt in record.attempts)
        copy = LlmCallRecord()
        copy.CopyFrom(record)
        try:
            serving.caller.call_soon_threadsafe(self._deliver, copy)
        except RuntimeError:  # The caller's loop has closed, so nothing can take the record.
            _logger.error(
                "the caller's loop has closed, so this call's record is lost",
                extra={
                    "event": "llm_record_lost",
                    "model": record.model,
                    "call_id": record.call_id,
                },
            )

    def _deliver(self, record: LlmCallRecord) -> None:
        try:
            self._recorder.record(record)
        except Exception:  # A broken recorder must not break the caller's loop.
            _logger.exception(
                "the call recorder failed, so this call's record is lost",
                extra={
                    "event": "llm_record_lost",
                    "model": record.model,
                    "call_id": record.call_id,
                },
            )


def _stop_once_serving(ready: _Ready) -> None:
    """Stop a proxy whose start was abandoned, as soon as it serves, if it ever does."""
    if ready.exception() is None:
        ready.result()[1]()


def _refused(record: LlmCallRecord, refusal: _Refusal) -> tuple[int, bytes]:
    """Put ``refusal`` on record, unless the gateway's reply is already there, and answer it."""
    if not record.attempts:  # Refused before it was forwarded.
        record.status, record.http_status = refusal.call_status, refusal.status
    elif not record.attempts[-1].status:
        record.attempts[-1].status = refusal.call_status
    return refusal.status, _error(refusal.status, str(refusal))


def _private_directory(directory: Path) -> None:
    """Make ``directory`` with mode 0700, or check that the one there is ours, with that mode."""
    try:
        directory.mkdir(mode=0o700)
    except FileExistsError:
        pass
    else:
        directory.chmod(0o700)  # The umask may have taken bits off mkdir's mode.
    info = directory.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise PermissionError(
            f"the proxy's socket directory must be this user's alone, mode 0700: {directory}"
        )


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
        key = name.strip().lower().decode("latin-1")
        if key in _ONCE and key in headers:
            raise _Refusal(400, f"the request gives {key} twice")
        headers[key] = value.strip()
    return parts[0], parts[1], headers


async def _read_body(
    reader: asyncio.StreamReader, headers: dict[str, bytes], max_bytes: int
) -> bytes:
    if "transfer-encoding" in headers or "content-length" not in headers:
        raise _Refusal(411, "the proxy needs a Content-Length")
    length = headers["content-length"]
    if not length.isdigit():
        raise _Refusal(400, "a malformed Content-Length")
    digits = length.lstrip(b"0") or b"0"
    if len(digits) > len(str(max_bytes)) or int(digits) > max_bytes:  # int() of a few digits.
        raise _Refusal(413, f"the body is over {max_bytes} bytes")
    return await reader.readexactly(int(digits))


def _read_tags(raw: bytes | None) -> tuple[str, str, int] | None:
    """Return the run, task and seed in an X-Goaly-Call-Tags value, or None for an unusable one."""
    if raw is None or len(raw) > _TAGS_BYTES:
        return None
    try:
        tags = json.loads(raw)
    except (ValueError, RecursionError):
        return None
    if not isinstance(tags, dict) or not set(tags) <= _TAG_KEYS:
        return None
    run_id, task, seed = tags.get("run_id", ""), tags.get("instance_id", ""), tags.get("seed", 0)
    if not (isinstance(run_id, str) and isinstance(task, str)) or isinstance(seed, bool):
        return None
    if not isinstance(seed, int) or not -(2**31) <= seed < 2**31:  # LlmCallRecord.seed: int32.
        return None
    try:
        (run_id + task).encode()
    except UnicodeEncodeError:  # A lone surrogate, which no record can hold.
        return None
    return run_id, task, seed


def _parse(body: bytes) -> dict[str, object]:
    """Return the JSON object in ``body``; refuse anything else, a key given twice included."""
    try:
        payload = json.loads(
            body, object_pairs_hook=_object, parse_constant=_constant, parse_float=_finite
        )
    except (ValueError, RecursionError):
        raise _Refusal(400, "the body is not JSON") from None
    if not isinstance(payload, dict):
        raise _Refusal(400, "the body is not a JSON object")
    return payload


def _object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build a JSON object, refusing a key given twice: the gateway might read the other one."""
    payload = dict(pairs)
    if len(payload) < len(pairs):
        raise _Refusal(400, "the body gives a key twice")
    return payload


def _constant(name: str) -> object:
    """Refuse NaN and Infinity, which Python reads but JSON does not allow."""
    raise _Refusal(400, f"the body holds {name}, which is not JSON")


def _finite(text: str) -> float:
    """Read a number, refusing one too large for a float: it would go on as Infinity."""
    number = float(text)
    if not math.isfinite(number):
        raise _Refusal(400, f"the body holds {text[:20]}, which is not a finite number")
    return number


async def _hang_up(reader: asyncio.StreamReader) -> None:
    """Return once the caller has closed the connection, or it broke."""
    with contextlib.suppress(OSError):
        while await reader.read(_HEAD_BYTES):
            pass  # Anything after the request is ignored: the proxy answers one request.


async def _linger(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Let the caller read its answer: stop writing, then read on, briefly, until it closes.

    A refusal may come before the request has all arrived. Closing with the rest of it unread
    would make the kernel reset the connection, and the caller could lose the answer.
    """
    with contextlib.suppress(OSError, TimeoutError):
        writer.write_eof()
        async with asyncio.timeout(_LINGER_S):
            while await reader.read(_HEAD_BYTES):
                pass


async def _read_reply(response: httpx.Response, max_bytes: int) -> bytes | None:
    """Return a reply's body, decoded, or None if it is longer than ``max_bytes``."""
    chunks: list[bytes] = []
    size = 0
    async for chunk in response.aiter_bytes():
        size += len(chunk)
        if size > max_bytes:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


def _without(secret: str, data: bytes) -> bytes:
    """Return ``data`` with ``secret`` taken out, both as it is and as JSON escapes it."""
    for form in {secret, json.dumps(secret)[1:-1]}:
        data = data.replace(form.encode(), b"[redacted]")
    return data


def _response(status: int, headers: Sequence[tuple[bytes, bytes]], body: bytes) -> bytes:
    """A whole response: ``status``, the ``headers`` to pass on, its own framing, and ``body``."""
    lines = [f"HTTP/1.1 {status} {_reason(status)}".encode()]
    lines += [name + b": " + value for name, value in headers]
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
