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
long as the process runs; :meth:`~QueuedNotifier.close` gives each one left a last attempt.

A notification never carries a secret or task data. Its text is built by goaly's code from
aggregates, and any known secret in it is replaced before it is queued, logged or sent.

The topic goes in the JSON body of a request to the server's root, never in a URL, so neither
httpx's request log nor an error message can show it.
"""

import enum
import heapq
import itertools
import logging
import secrets
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
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
]

# After the k-th failed attempt, the next comes RETRY_DELAYS_S[k - 1] later; the last repeats.
RETRY_DELAYS_S: tuple[float, ...] = (10.0, 30.0, 60.0, 120.0, 300.0, 600.0)
_MAX_TEXT = 1000  # Characters; ntfy.sh accepts 4096 bytes of message.
_MAX_DETAIL = 300
_REDACTED = "[redacted]"
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
        """Queue a notification and return at once; never raises for a delivery problem."""
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
    then recorded once as NOTIFY_FAILED, saying so, and never retried. ``secret_values`` are
    replaced in every text; ``now`` stamps the events and ``monotonic`` times the retries.
    """

    def __init__(
        self,
        channel: Channel | None,
        ledger: Ledger,
        *,
        secret_values: Sequence[str] = (),
        retry_delays_s: Sequence[float] = RETRY_DELAYS_S,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not retry_delays_s or any(delay < 0 for delay in retry_delays_s):
            raise ValueError("retry_delays_s needs at least one delay, none negative")
        self._channel = channel
        self._ledger = ledger
        self._secret_values = tuple(secret_values)
        self._delays = tuple(retry_delays_s)
        self._now = now
        self._monotonic = monotonic
        self._queue: list[_Attempt] = []
        self._sequence = itertools.count()
        self._condition = threading.Condition()
        self._closing = False
        self._busy = False
        self._worker: threading.Thread | None = None
        self._last_breaker: dict[Lane.ValueType, float] = {}

    def notify(self, lane: Lane.ValueType, kind: NotificationKind, text: str) -> None:
        try:
            self._queue_notification(lane, kind, text)
        except Exception:  # Never let a notification stop the loop that sent it.
            _logger.exception("could not queue a notification", extra={"kind": str(kind)})

    def close(self, timeout_s: float = 30.0) -> None:
        with self._condition:
            self._closing = True
            self._condition.notify_all()
            worker = self._worker
        if worker is not None:
            worker.join(timeout_s)
            if worker.is_alive():
                _logger.warning("notifications were still being delivered at exit")

    def pending(self) -> int:
        """How many notifications are queued or being delivered."""
        with self._condition:
            return len(self._queue) + int(self._busy)

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
        notification = Notification(
            notification_id=f"{kind.value}-{secrets.token_hex(6)}",
            lane=lane,
            kind=kind,
            title=_TITLES[kind],
            text=redact(text.strip() or f"{info.title}: {_TITLES[kind]}", self._secret_values),
        )
        if self._channel is None:
            self._record(notification, delivered=False, detail="GOALY_NTFY_TOPIC is not set")
            _logger.warning("notifications are off: GOALY_NTFY_TOPIC is not set")
            return
        with self._condition:
            heapq.heappush(
                self._queue, _Attempt(self._monotonic(), next(self._sequence), notification)
            )
            if self._worker is None:
                self._worker = threading.Thread(
                    target=self._run, name="goaly-notifier", daemon=True
                )
                self._worker.start()
            self._condition.notify_all()

    def _run(self) -> None:
        while True:
            with self._condition:
                while True:
                    if not self._queue:
                        if self._closing:
                            return
                        self._condition.wait()
                        continue
                    wait = self._queue[0].due - self._monotonic()
                    if wait <= 0 or self._closing:
                        break
                    self._condition.wait(wait)
                attempt = heapq.heappop(self._queue)
                self._busy = True
                closing = self._closing
            try:
                self._attempt(attempt, last=closing)
            finally:
                with self._condition:
                    self._busy = False

    def _attempt(self, attempt: _Attempt, *, last: bool) -> None:
        notification = attempt.notification
        try:
            if self._channel is None:  # Never queued then; kept for the type checker.
                raise DeliveryError("GOALY_NTFY_TOPIC is not set")
            self._channel.send(notification)
        except Exception as error:  # noqa: BLE001  (a channel bug must not kill the worker)
            reason = str(error) if isinstance(error, DeliveryError) else type(error).__name__
            attempt.failures += 1
            delay = self._delays[min(attempt.failures, len(self._delays)) - 1]
            later = "no retry: the process is exiting" if last else f"retry in {delay:g} s"
            self._record(
                notification,
                delivered=False,
                detail=f"attempt {attempt.failures}: {reason}; {later}",
            )
            if not last:
                attempt.due = self._monotonic() + delay
                with self._condition:
                    heapq.heappush(self._queue, attempt)
            return
        self._record(notification, delivered=True, detail="")

    def _record(self, notification: Notification, *, delivered: bool, detail: str) -> None:
        detail = redact(detail, self._secret_values)[:_MAX_DETAIL]
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
