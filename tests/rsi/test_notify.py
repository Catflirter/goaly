"""Tests for goaly.rsi.notify: the queued notifier, its ledger events and the ntfy channel."""

import json
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from goaly.contracts.gen.experiment_pb2 import (
    LANE_EFFISQL,
    LANE_EVENT_KIND_NOTIFIED,
    LANE_EVENT_KIND_NOTIFY_FAILED,
    LANE_UNSPECIFIED,
)
from goaly.harness.core.settings import Settings
from goaly.rsi.ledger import Ledger
from goaly.rsi.notify import (
    DeliveryError,
    FakeChannel,
    Notification,
    NotificationKind,
    NtfyChannel,
    QueuedNotifier,
    notifier_from_settings,
    redact,
)

TOPIC = "goaly-test-topic-0123456789"


@pytest.fixture
def ledger(tmp_path: Path) -> Ledger:
    return Ledger(tmp_path / "ledger")


@pytest.fixture
def notifiers() -> Iterator[list[QueuedNotifier]]:
    made: list[QueuedNotifier] = []
    yield made
    for notifier in made:
        notifier.close(timeout_s=5)


def make(
    channel: FakeChannel | None,
    ledger: Ledger,
    made: list[QueuedNotifier],
    **kwargs: object,
) -> QueuedNotifier:
    notifier = QueuedNotifier(channel, ledger, retry_delays_s=(0.0,), **kwargs)  # type: ignore[arg-type]  # Test-only keyword pass-through.
    made.append(notifier)
    return notifier


def wait_for(condition: object, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if condition():  # type: ignore[operator]  # A zero-argument callable.
            return
        time.sleep(0.01)
    raise AssertionError("the condition did not hold in time")


def test_a_delivered_notification_is_recorded_as_notified(
    ledger: Ledger, notifiers: list[QueuedNotifier]
) -> None:
    channel = FakeChannel()
    notifier = make(channel, ledger, notifiers)
    notifier.notify(LANE_EFFISQL, NotificationKind.STOPPED, "Effi-SQL stopped at its deadline.")
    notifier.close()
    (event,) = ledger.lane_events(LANE_EFFISQL)
    assert event.kind == LANE_EVENT_KIND_NOTIFIED
    assert event.notification == "lane stopped: Effi-SQL stopped at its deadline."
    assert event.notification_id.startswith("stopped-")
    assert event.notification_id == channel.delivered[0].notification_id
    assert event.HasField("occurred_at")


@pytest.mark.parametrize("kind", list(NotificationKind))
def test_every_kind_reaches_the_channel(
    kind: NotificationKind, ledger: Ledger, notifiers: list[QueuedNotifier]
) -> None:
    channel = FakeChannel()
    notifier = make(channel, ledger, notifiers)
    notifier.notify(LANE_EFFISQL, kind, "")
    notifier.close()
    assert [sent.kind for sent in channel.delivered] == [kind]
    assert channel.delivered[0].text  # A blank text gets a default.


def test_a_failed_delivery_is_retried_and_tied_to_its_delivery(
    ledger: Ledger, notifiers: list[QueuedNotifier]
) -> None:
    channel = FakeChannel(failures=2)
    notifier = make(channel, ledger, notifiers)
    notifier.notify(LANE_EFFISQL, NotificationKind.RESULTS_PR, "Results PR #7 is open.")
    wait_for(lambda: len(channel.delivered) == 1)
    notifier.close()
    events = ledger.lane_events(LANE_EFFISQL)
    kinds = [event.kind for event in events]
    assert kinds == [
        LANE_EVENT_KIND_NOTIFY_FAILED,
        LANE_EVENT_KIND_NOTIFY_FAILED,
        LANE_EVENT_KIND_NOTIFIED,
    ]
    assert len({event.notification_id for event in events}) == 1
    assert events[0].detail == "attempt 1: the fake channel is down; retry in 0 s"
    assert events[2].detail == ""


def test_notify_never_blocks_on_a_slow_channel(
    ledger: Ledger, notifiers: list[QueuedNotifier]
) -> None:
    release = threading.Event()

    class Slow(FakeChannel):
        def send(self, notification: Notification) -> None:
            release.wait(5)
            super().send(notification)

    channel = Slow()
    notifier = make(channel, ledger, notifiers)
    started = time.monotonic()
    for _ in range(3):
        notifier.notify(LANE_EFFISQL, NotificationKind.STOPPED, "stopped")
    assert time.monotonic() - started < 0.5
    assert notifier.pending() == 3
    release.set()
    notifier.close()
    assert len(channel.delivered) == 3
    assert notifier.pending() == 0


def test_close_gives_a_waiting_retry_one_last_attempt(
    ledger: Ledger, notifiers: list[QueuedNotifier]
) -> None:
    channel = FakeChannel(failures=1)
    notifier = QueuedNotifier(channel, ledger, retry_delays_s=(3600.0,))
    notifiers.append(notifier)
    notifier.notify(LANE_EFFISQL, NotificationKind.STOPPED, "stopped")
    wait_for(lambda: len(channel.attempts) == 1)
    started = time.monotonic()
    notifier.close()
    assert time.monotonic() - started < 2
    assert len(channel.delivered) == 1
    assert [event.kind for event in ledger.lane_events()][-1] == LANE_EVENT_KIND_NOTIFIED


def test_close_records_a_last_failure_without_a_retry(
    ledger: Ledger, notifiers: list[QueuedNotifier]
) -> None:
    channel = FakeChannel(failures=5)
    notifier = QueuedNotifier(channel, ledger, retry_delays_s=(3600.0,))
    notifiers.append(notifier)
    notifier.notify(LANE_EFFISQL, NotificationKind.STOPPED, "stopped")
    wait_for(lambda: len(channel.attempts) == 1)
    notifier.close()
    details = [event.detail for event in ledger.lane_events()]
    assert details == [
        "attempt 1: the fake channel is down; retry in 3600 s",
        "attempt 2: the fake channel is down; no retry: the process is exiting",
    ]


def test_a_channel_bug_is_recorded_and_the_worker_goes_on(
    ledger: Ledger, notifiers: list[QueuedNotifier]
) -> None:
    class Broken(FakeChannel):
        calls = 0

        def send(self, notification: Notification) -> None:
            Broken.calls += 1
            if Broken.calls == 1:
                raise RuntimeError(f"bug with {TOPIC}")
            super().send(notification)

    channel = Broken()
    notifier = make(channel, ledger, notifiers)
    notifier.notify(LANE_EFFISQL, NotificationKind.STOPPED, "stopped")
    wait_for(lambda: len(channel.delivered) == 1)
    notifier.close()
    failed = ledger.lane_events()[0]
    assert failed.detail == "attempt 1: RuntimeError; retry in 0 s"  # Only the type.


def test_without_a_channel_each_notification_is_recorded_once_as_failed(
    ledger: Ledger, notifiers: list[QueuedNotifier]
) -> None:
    notifier = make(None, ledger, notifiers)
    notifier.notify(LANE_EFFISQL, NotificationKind.RECOVERY_FAILED, "Effi-SQL was given up.")
    notifier.close()
    (event,) = ledger.lane_events()
    assert event.kind == LANE_EVENT_KIND_NOTIFY_FAILED
    assert event.detail == "GOALY_NTFY_TOPIC is not set"
    assert event.notification == "recovery failed: Effi-SQL was given up."


def test_secrets_are_redacted_from_text_events_and_channel(
    ledger: Ledger, notifiers: list[QueuedNotifier]
) -> None:
    channel = FakeChannel()
    notifier = make(channel, ledger, notifiers, secret_values=("sk-secret", TOPIC))
    notifier.notify(LANE_EFFISQL, NotificationKind.STOPPED, f"key sk-secret on {TOPIC}\x07!")
    notifier.close()
    assert channel.delivered[0].text == "key [redacted] on [redacted] !"
    raw = (ledger.directory / "lane_events.jsonl").read_text(encoding="utf-8")
    assert "sk-secret" not in raw
    assert TOPIC not in raw


def test_redact_cuts_long_texts() -> None:
    assert len(redact("x" * 5000, ())) == 1000
    assert redact("a\nb", ["", "zz"]) == "a\nb"


def test_breaker_notifications_are_rate_limited(
    ledger: Ledger, notifiers: list[QueuedNotifier]
) -> None:
    clock = [1000.0]
    channel = FakeChannel()
    notifier = make(channel, ledger, notifiers, monotonic=lambda: clock[0])
    notifier.notify(LANE_EFFISQL, NotificationKind.BREAKER_TRIPPED, "tripped")
    notifier.notify(LANE_EFFISQL, NotificationKind.BREAKER_TRIPPED, "tripped again")
    clock[0] += 1801
    notifier.notify(LANE_EFFISQL, NotificationKind.BREAKER_TRIPPED, "tripped later")
    notifier.close()
    assert [sent.text for sent in channel.delivered] == ["tripped", "tripped later"]


def test_notify_after_close_and_a_bad_lane_never_raise(
    ledger: Ledger, notifiers: list[QueuedNotifier], caplog: pytest.LogCaptureFixture
) -> None:
    notifier = make(FakeChannel(), ledger, notifiers)
    notifier.notify(LANE_UNSPECIFIED, NotificationKind.STOPPED, "no lane")
    notifier.close()
    notifier.notify(LANE_EFFISQL, NotificationKind.STOPPED, "too late")
    assert ledger.lane_events() == []
    assert caplog.text.count("could not queue a notification") == 2


def test_a_ledger_failure_is_logged_and_the_worker_goes_on(
    tmp_path: Path, notifiers: list[QueuedNotifier], caplog: pytest.LogCaptureFixture
) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("", encoding="utf-8")
    channel = FakeChannel()
    notifier = make(channel, Ledger(blocker / "ledger"), notifiers)
    notifier.notify(LANE_EFFISQL, NotificationKind.STOPPED, "one")
    notifier.notify(LANE_EFFISQL, NotificationKind.STOPPED, "two")
    notifier.close()
    assert len(channel.delivered) == 2
    assert caplog.text.count("could not record a notification event") == 2


def test_retry_delays_must_be_sane(ledger: Ledger) -> None:
    with pytest.raises(ValueError, match="at least one delay"):
        QueuedNotifier(FakeChannel(), ledger, retry_delays_s=())
    with pytest.raises(ValueError, match="none negative"):
        QueuedNotifier(FakeChannel(), ledger, retry_delays_s=(-1.0,))


# The ntfy channel, against httpx.MockTransport.


def notification(kind: NotificationKind = NotificationKind.STOPPED) -> Notification:
    return Notification("stopped-abc", LANE_EFFISQL, kind, "lane stopped", "Effi-SQL stopped.")


def test_ntfy_posts_json_to_the_root_with_the_topic_in_the_body() -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"id": "x"})

    channel = NtfyChannel(TOPIC, transport=httpx.MockTransport(handle))
    channel.send(notification(NotificationKind.RECOVERY_FAILED))
    channel.close()
    (request,) = requests
    assert str(request.url) == "https://ntfy.sh/"
    assert TOPIC not in str(request.url)
    assert request.method == "POST"
    body = json.loads(request.content)
    assert body == {
        "topic": TOPIC,
        "title": "goaly effisql: lane stopped",
        "message": "Effi-SQL stopped.",
        "tags": ["robot"],
        "priority": 4,
    }


def test_ntfy_reports_a_status_without_the_topic() -> None:
    channel = NtfyChannel(TOPIC, transport=httpx.MockTransport(lambda _: httpx.Response(429)))
    with pytest.raises(DeliveryError, match="HTTP 429") as caught:
        channel.send(notification())
    assert TOPIC not in str(caught.value)


def test_ntfy_reports_a_transport_error_by_type_only() -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"cannot reach {request.content!r}")

    channel = NtfyChannel(TOPIC, transport=httpx.MockTransport(fail))
    with pytest.raises(DeliveryError) as caught:
        channel.send(notification())
    assert str(caught.value) == "the request failed: ConnectError"
    assert caught.value.__cause__ is None


@pytest.mark.parametrize("server", ["http://ntfy.sh", "https://ntfy.sh/topic", "ntfy.sh"])
def test_ntfy_needs_an_https_server_without_a_path(server: str) -> None:
    with pytest.raises(ValueError, match="https URL"):
        NtfyChannel(TOPIC, server=server)


def test_notifier_from_settings_redacts_every_secret(ledger: Ledger) -> None:
    settings = Settings(
        api_key=SecretStr("sk-key"),
        ntfy_topic=SecretStr(TOPIC),
        github_token=SecretStr("ghu_token"),
    )
    notifier = notifier_from_settings(settings, ledger)
    assert isinstance(notifier._channel, NtfyChannel)
    assert set(notifier._secret_values) == {"sk-key", TOPIC, "ghu_token"}
    off = notifier_from_settings(Settings(api_key=SecretStr("sk-key")), ledger)
    assert off._channel is None
