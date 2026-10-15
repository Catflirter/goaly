"""Notifications to the project lead, over HTTPS to an ntfy.sh topic (spec §8.7).

A notification goes out when a lane's automatic recovery fails, when the gateway's circuit
breaker trips repeatedly, when a results PR is opened, and when a lane stops. The lead
subscribes to the topic in the ntfy app; its name is a secret, ``GOALY_NTFY_TOPIC``, read through
goaly.harness.core.settings.

:class:`QueuedNotifier` never blocks the loop that calls it: :meth:`~QueuedNotifier.notify` only
queues, and a worker thread delivers. Each attempt is written to the lane's event log
(``lane_events.jsonl``) as a LaneEvent: NOTIFIED once delivered, NOTIFY_FAILED for each failed
attempt, both carrying the notification's ID, so a failure is tied to its later delivery. A
failed delivery is retried after 10 s, 30 s, 1, 2 and 5 minutes, then every 10 minutes, for as
long as the process runs. :meth:`~QueuedNotifier.close` attempts every queued one at once, and keeps
retrying failures after 1, 2, 4 and 8 s while its budget lasts.

A notification never carries a secret or task data. Its text is built by goaly's code from
aggregates, and :func:`sanitize` replaces any known secret in it and keeps only plain words,
before it is queued, logged or sent. :meth:`~QueuedNotifier.resume` queues again what an earlier
process recorded as not delivered.

The topic goes in the JSON body of a request to the server's root, never in a URL, so neither
httpx's request log nor an error message can show it.
"""

import enum
import heapq
import itertools
import logging
import re
import secrets
import threading
import time
from collections.abc import Callable, Collection, Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import Protocol
from urllib.parse import urlsplit

import httpx
from google.protobuf.timestamp_pb2 import Timestamp

from goaly.contracts.gen.experiment_pb2 import (
    LANE_EVENT_KIND_NOTIFIED,
    LANE_EVENT_KIND_NOTIFY_FAILED,
    Lane,
    LaneEvent,
)
from goaly.harness.core.settings import Settings
from goaly.rsi.lanes import lane_info
from goaly.rsi.ledger import Ledger

__all__ = [
    "CLOSE_RETRY_DELAYS_S",
    "RETRY_DELAYS_S",
    "Channel",
    "DeliveryError",
    "FakeChannel",
    "Notification",
    "NotificationKind",
    "Notifier",
    "NtfyChannel",
    "QueuedNotifier",
    "notifier_from_settings",
    "redact",
    "sanitize",
]

# After the k-th failed attempt, the next comes RETRY_DELAYS_S[k - 1] later; the last repeats.
RETRY_DELAYS_S: tuple[float, ...] = (10.0, 30.0, 60.0, 120.0, 300.0, 600.0)
# While closing, the retries come after these, as long as close()'s budget allows.
CLOSE_RETRY_DELAYS_S: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
_MAX_TEXT = 1000  # Characters; ntfy.sh accepts 4096 bytes of message.
_MAX_DETAIL = 300
_REDACTED = "[redacted]"
_UNSAFE = re.compile(r"[^A-Za-z0-9 .,:;!()#/_+\-=%?&@\[\]]+")
_SPACES = re.compile(r" {2,}")
_OFF = "GOALY_NTFY_TOPIC is not set"
_RESUME_WINDOW = timedelta(hours=12)
_BREAKER_QUIET_S = 1800.0  # One breaker notification per lane in 30 minutes is enough.

_logger = logging.getLogger(__name__)


class NotificationKind(enum.StrEnum):
    """Why the lead is notified (spec §8.7)."""

    RECOVERY_FAILED = "recovery-failed"  # 5 crashes within an hour: the lane was given up.
    BREAKER_TRIPPED = "breaker-tripped"  # The gateway's circuit breaker tripped repeatedly.
    RESULTS_PR = "results-pr"  # A results PR was opened.
    STOPPED = "stopped"  # The lane stopped by a stopping rule (spec §8.4).
    TEST = "test"  # A test notification, sent once by hand.


_TITLES = {
    NotificationKind.RECOVERY_FAILED: "recovery failed",
    NotificationKind.BREAKER_TRIPPED: "gateway breaker tripped",
    NotificationKind.RESULTS_PR: "results PR opened",
    NotificationKind.STOPPED: "lane stopped",
    NotificationKind.TEST: "test",
}
_URGENT = frozenset({NotificationKind.RECOVERY_FAILED, NotificationKind.BREAKER_TRIPPED})
_NOTIFICATION_EVENTS = frozenset({LANE_EVENT_KIND_NOTIFIED, LANE_EVENT_KIND_NOTIFY_FAILED})


@dataclass(frozen=True)
class Notification:
    """One notification, already redacted."""

    notification_id: str
    lane: Lane.ValueType
    kind: NotificationKind
    title: str
    text: str


class DeliveryError(Exception):
    """A notification could not be delivered; the message holds no secret and no URL."""


class Channel(Protocol):
    """Where notifications are delivered."""

    def send(self, notification: Notification) -> None:
        """Deliver ``notification``, or raise DeliveryError."""
        ...


class Notifier(Protocol):
    """What a lane and the supervisor call to notify the project lead."""

    def notify(self, lane: Lane.ValueType, kind: NotificationKind, text: str) -> None:
        """Queue a notification and return at once; never raises for a delivery problem.

        ``text`` is built by goaly's code from counts, rules, PR numbers and links. It never
        comes from an exception, a task, a query or a model's output: it reaches the lead's
        phone and the next results PR. The notifier also passes it through :func:`sanitize`.
        """
        ...

    def close(self, timeout_s: float = 30.0) -> None:
        """Give every queued notification a last attempt, waiting at most ``timeout_s``."""
        ...


class NtfyChannel:
    """Delivers to an ntfy.sh topic: a JSON POST to the server's root, topic in the body."""

    def __init__(
        self,
        topic: str,
        *,
        server: str = "https://ntfy.sh",
        timeout_s: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        parts = urlsplit(server)
        if parts.scheme != "https" or not parts.hostname or parts.path not in {"", "/"}:
            raise ValueError("the ntfy server must be an https URL with no path")
        self._topic = topic
        self._url = f"https://{parts.netloc}/"
        self._client = httpx.Client(
            timeout=timeout_s, follow_redirects=False, transport=transport, trust_env=False
        )

    def send(self, notification: Notification) -> None:
        info = lane_info(notification.lane)
        body = {
            "topic": self._topic,
            "title": f"goaly {info.name}: {notification.title}",
            "message": notification.text,
            "tags": ["robot"],
            "priority": 4 if notification.kind in _URGENT else 3,
        }
        try:
            response = self._client.post(self._url, json=body)
        except httpx.HTTPError as error:
            # The type alone: an httpx message may quote the request, and so the body.
            raise DeliveryError(f"the request failed: {type(error).__name__}") from None
        if response.status_code != httpx.codes.OK:
            raise DeliveryError(f"ntfy answered HTTP {response.status_code}")

    def close(self) -> None:
        self._client.close()


class FakeChannel:
    """A channel for tests: records what it delivers, and fails the first ``failures`` sends."""

    def __init__(self, failures: int = 0) -> None:
        self.failures = failures
        self.attempts: list[Notification] = []
        self.delivered: list[Notification] = []
        self._lock = threading.Lock()

    def send(self, notification: Notification) -> None:
        with self._lock:
            self.attempts.append(notification)
            if self.failures > 0:
                self.failures -= 1
                raise DeliveryError("the fake channel is down")
            self.delivered.append(notification)


def sanitize(text: str, secret_values: Iterable[str]) -> str:
    """Return ``text`` fit for a notification: redacted, then cut down to plain words.

    Only letters, digits, spaces and ``.,:;!()#/_+-=%?&@[]`` are kept: anything else becomes a
    space, and a run of spaces one, so quoted SQL, a query's result or a traceback cannot pass
    through whole. Callers still build every text from code: counts, rules, PR numbers and links,
    never from an exception, a task, a query or a model's output.
    """
    text = redact(text, secret_values)
    text = _SPACES.sub(" ", _UNSAFE.sub(" ", text)).strip()
    return text if len(text) <= _MAX_TEXT else text[: _MAX_TEXT - 1] + "…"


def redact(text: str, secret_values: Iterable[str]) -> str:
    """Return ``text`` with every one of ``secret_values`` and every control character replaced,
    cut to the notification limit."""
    for value in sorted({value for value in secret_values if value}, key=len, reverse=True):
        text = text.replace(value, _REDACTED)
    text = "".join(char if char.isprintable() or char == "\n" else " " for char in text)
    return text if len(text) <= _MAX_TEXT else text[: _MAX_TEXT - 1] + "…"


@dataclass(order=True)
class _Attempt:
    due: float
    sequence: int
    notification: Notification = field(compare=False)
    failures: int = field(default=0, compare=False)


class QueuedNotifier:
    """Queues notifications and delivers them on a worker thread, recording each attempt.

    ``channel`` None means notifications are off, as when GOALY_NTFY_TOPIC is unset: each one is
    then recorded once as NOTIFY_FAILED, saying so, and never retried. ``secret_values``, or the
    values a callable returns at each notification, are replaced in every text. ``now`` stamps
    the events and ``monotonic`` times the retries. :meth:`close` retries failures after
    ``close_retry_delays_s`` for as long as its budget allows.
    """

    def __init__(
        self,
        channel: Channel | None,
        ledger: Ledger,
        *,
        secret_values: Sequence[str] | Callable[[], Sequence[str]] = (),
        retry_delays_s: Sequence[float] = RETRY_DELAYS_S,
        close_retry_delays_s: Sequence[float] = CLOSE_RETRY_DELAYS_S,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        for delays in (retry_delays_s, close_retry_delays_s):
            if not delays or any(delay < 0 for delay in delays):
                raise ValueError("retry delays need at least one delay, none negative")
        self._channel = channel
        self._ledger = ledger
        self._secret_values = secret_values
        self._delays = tuple(retry_delays_s)
        self._close_delays = tuple(close_retry_delays_s)
        self._now = now
        self._monotonic = monotonic
        self._queue: list[_Attempt] = []
        self._sequence = itertools.count()
        self._condition = threading.Condition()
        self._closing = False
        self._deadline = 0.0  # Once closing: when the last attempt must have started.
        self._busy = False
        self._worker: threading.Thread | None = None
        self._last_breaker: dict[Lane.ValueType, float] = {}

    def notify(self, lane: Lane.ValueType, kind: NotificationKind, text: str) -> None:
        try:
            self._queue_notification(lane, kind, text)
        except Exception:  # Never let a notification stop the loop that sent it.
            _logger.exception("could not queue a notification", extra={"kind": str(kind)})

    def resume(self, lane: Lane.ValueType, kinds: Collection[NotificationKind]) -> int:
        """Queue again ``lane``'s notifications of ``kinds`` that an earlier process recorded
        as not delivered, within the last 12 hours, keeping their IDs; return how many.

        Run it once per process, for the kinds that process sends, so that no other process
        sends the same ones again. Notifications that were off are not resent.
        """
        latest: dict[str, LaneEvent] = {}
        for event in self._ledger.lane_events(lane):
            if event.kind in _NOTIFICATION_EVENTS and event.notification_id:
                latest[event.notification_id] = event
        cutoff = self._now() - _RESUME_WINDOW
        count = 0
        for notification_id, event in latest.items():
            kind = _kind_of(notification_id)
            if (
                kind is None
                or event.kind != LANE_EVENT_KIND_NOTIFY_FAILED
                or event.detail == _OFF
                or kind not in kinds
                or event.occurred_at.ToDatetime(tzinfo=UTC) < cutoff
            ):
                continue
            title = _TITLES[kind]
            text = event.notification.removeprefix(f"{title}: ")
            self._enqueue(Notification(notification_id, lane, kind, title, text))
            count += 1
        return count

    def close(self, timeout_s: float = 30.0) -> None:
        with self._condition:
            if not self._closing:
                self._closing = True
                self._deadline = self._monotonic() + timeout_s
                now = self._monotonic()
                for attempt in self._queue:  # A retry waiting for later gets its chance now.
                    attempt.due = min(attempt.due, now)
                heapq.heapify(self._queue)
            self._condition.notify_all()
            worker = self._worker
        if worker is not None:
            worker.join(timeout_s + 5.0)  # The last attempt may start just before the deadline.
            if worker.is_alive():
                _logger.warning("notifications were still being delivered at exit")
        closer = getattr(self._channel, "close", None)
        if callable(closer):
            closer()

    def pending(self) -> int:
        """How many notifications are queued or being delivered."""
        with self._condition:
            return len(self._queue) + int(self._busy)

    def _secrets(self) -> Sequence[str]:
        values = self._secret_values
        return values() if callable(values) else values

    def _queue_notification(self, lane: Lane.ValueType, kind: NotificationKind, text: str) -> None:
        info = lane_info(lane)
        kind = NotificationKind(kind)
        with self._condition:
            if self._closing:
                raise RuntimeError("the notifier is closed")
            if kind is NotificationKind.BREAKER_TRIPPED:
                last = self._last_breaker.get(lane)
                if last is not None and self._monotonic() - last < _BREAKER_QUIET_S:
                    _logger.info("breaker notification skipped: one went out recently")
                    return
                self._last_breaker[lane] = self._monotonic()
        self._enqueue(
            Notification(
                notification_id=f"{kind.value}-{secrets.token_hex(6)}",
                lane=lane,
                kind=kind,
                title=_TITLES[kind],
                text=text.strip() or f"{info.title}: {_TITLES[kind]}",
            )
        )

    def _enqueue(self, notification: Notification) -> None:
        clean = replace(notification, text=sanitize(notification.text, self._secrets()))
        with self._condition:
            if self._closing:  # Checked again: close() may have run since the first check.
                raise RuntimeError("the notifier is closed")
            heapq.heappush(self._queue, _Attempt(self._monotonic(), next(self._sequence), clean))
            if self._worker is None:
                self._worker = threading.Thread(
                    target=self._run, name="goaly-notifier", daemon=True
                )
                self._worker.start()
            self._condition.notify_all()

    def _run(self) -> None:
        while True:
            with self._condition:
                final = False
                while True:
                    now = self._monotonic()
                    if self._closing and not self._queue:
                        return
                    if not self._queue:
                        self._condition.wait()
                        continue
                    if self._closing and now >= self._deadline:
                        final = True  # Past the budget: one last attempt each, no retry.
                        break
                    wait = self._queue[0].due - now
                    if wait <= 0:
                        break
                    if self._closing:
                        wait = min(wait, self._deadline - now)
                    self._condition.wait(wait)
                attempt = heapq.heappop(self._queue)
                self._busy = True
            try:
                self._attempt(attempt, final=final)
            except Exception:  # The worker must outlive any bug in one attempt.
                _logger.exception("a notification attempt failed unexpectedly")
            finally:
                with self._condition:
                    self._busy = False

    def _attempt(self, attempt: _Attempt, *, final: bool = False) -> None:
        notification = attempt.notification
        if self._channel is None:
            self._record(notification, delivered=False, detail=_OFF)
            _logger.warning("notifications are off: GOALY_NTFY_TOPIC is not set")
            return
        try:
            self._channel.send(notification)
        except Exception as error:  # noqa: BLE001  (a channel bug must not kill the worker)
            reason = str(error) if isinstance(error, DeliveryError) else type(error).__name__
        else:
            self._record(notification, delivered=True, detail="")
            return
        attempt.failures += 1
        with self._condition:  # Decide and queue at once, so close() sees the retry.
            closing = self._closing
            delays = self._close_delays if closing else self._delays
            delay = delays[min(attempt.failures, len(delays)) - 1]
            retry = not final and (not closing or self._monotonic() + delay < self._deadline)
            if retry:
                attempt.due = self._monotonic() + delay
                heapq.heappush(self._queue, attempt)
                self._condition.notify_all()
        later = f"retry in {delay:g} s" if retry else "no retry: the process is exiting"
        detail = f"attempt {attempt.failures}: {reason}; {later}"
        self._record(notification, delivered=False, detail=detail)

    def _record(self, notification: Notification, *, delivered: bool, detail: str) -> None:
        detail = redact(detail, self._secrets())[:_MAX_DETAIL]
        event = LaneEvent(
            lane=notification.lane,
            kind=LANE_EVENT_KIND_NOTIFIED if delivered else LANE_EVENT_KIND_NOTIFY_FAILED,
            detail=detail,
            notification=f"{notification.title}: {notification.text}",
            notification_id=notification.notification_id,
        )
        stamp = Timestamp()
        stamp.FromDatetime(self._now())
        event.occurred_at.CopyFrom(stamp)
        extra = {"notification_id": notification.notification_id, "kind": str(notification.kind)}
        if delivered:
            _logger.info("notification delivered", extra=extra)
        else:
            _logger.warning("notification not delivered", extra={**extra, "reason": detail})
        try:
            self._ledger.record_lane_event(event)
        except Exception:  # The ledger's own failure is logged; the loop goes on.
            _logger.exception("could not record a notification event", extra=extra)


def _kind_of(notification_id: str) -> NotificationKind | None:
    try:
        return NotificationKind(notification_id.rpartition("-")[0])
    except ValueError:
        return None


def notifier_from_settings(
    settings: Settings, ledger: Ledger, *, server: str = "https://ntfy.sh"
) -> QueuedNotifier:
    """Build the notifier for ``settings``: ntfy.sh on GOALY_NTFY_TOPIC, or off if it is unset.

    Every secret the settings hold is redacted from the texts.
    """
    secret_values = [settings.api_key.get_secret_value()]
    for value in (settings.ntfy_topic, settings.github_token):
        if value is not None:
            secret_values.append(value.get_secret_value())
    channel: Channel | None = None
    if settings.ntfy_topic is not None:
        channel = NtfyChannel(settings.ntfy_topic.get_secret_value(), server=server)
    return QueuedNotifier(channel, ledger, secret_values=secret_values)
