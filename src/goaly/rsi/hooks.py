"""T22's hooks for the lane (goaly.rsi.lane.LaneHooks): results PRs, notifications, heartbeat.

Config names :func:`lane_hooks` as ``hooks = "goaly.rsi.hooks:lane_hooks"``. The lane then:

- beats its heartbeat in :func:`state_dir` for its whole life, which the supervisor watches;
- after every round, opens a periodic results PR when one is due;
- when it stops, opens the final results PR, then notifies the lead that it stopped, and gives
  the notification a last chance to go out before it exits;
- notifies the lead when the gateway's circuit breaker opens repeatedly: 3 times within 30
  minutes, as core's LLM client logs it in the lane process (:class:`BreakerWatch`).

Git and HTTP run in a worker thread, so the lane's event loop, and its heartbeat, go on.
"""

import asyncio
import logging
import time
from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from goaly.contracts.gen.experiment_pb2 import LANE_EFFISQL, Lane
from goaly.harness.core.settings import load_settings
from goaly.rsi.cli import LaneSettings
from goaly.rsi.heartbeat import heartbeat, heartbeat_file
from goaly.rsi.lane import LaneHooks, StopInfo
from goaly.rsi.lanes import lane_info
from goaly.rsi.ledger import Ledger
from goaly.rsi.notify import NotificationKind, Notifier, QueuedNotifier, notifier_from_settings
from goaly.rsi.results_pr import (
    GitHubApi,
    GitHubRepo,
    HttpsRemote,
    ResultsPublisher,
    ResultsPublisherImpl,
    runtime_github_token,
)
from goaly.rsi.round import RoundOutcome

__all__ = ["GITHUB_REPO", "LANES", "BreakerWatch", "build_hooks", "lane_hooks", "state_dir"]

LANES = {"effisql": LANE_EFFISQL}  # A config's lane name, as goaly.rsi.cli knows them.
GITHUB_REPO = GitHubRepo("Catflirter", "goaly")
# The kinds a lane process sends, and resends after a crash; the supervisor resends its own.
_LANE_KINDS = frozenset(
    {NotificationKind.STOPPED, NotificationKind.RESULTS_PR, NotificationKind.BREAKER_TRIPPED}
)
_CLOSE_S = 60.0  # How long a stopping lane waits for its last notifications.
_THROTTLE_LOGGER = "goaly.harness.core.llm_throttle"
_BREAKER_OPEN = "llm_breaker_open"  # The event core logs when a breaker opens.

_logger = logging.getLogger(__name__)


class BreakerWatch(logging.Handler):
    """Watches core's breaker log and notifies when it opens ``times`` times in ``window_s``.

    The notifier sends at most one breaker notification per lane in 30 minutes.
    """

    def __init__(
        self,
        notifier: Notifier,
        lane: Lane.ValueType,
        *,
        times: int = 3,
        window_s: float = 1800.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(logging.WARNING)
        self._notifier = notifier
        self._lane = lane
        self._times = times
        self._window_s = window_s
        self._clock = clock
        self._opened: deque[float] = deque()

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(record, "event", None) != _BREAKER_OPEN:
            return
        now = self._clock()
        self._opened.append(now)
        while self._opened and now - self._opened[0] > self._window_s:
            self._opened.popleft()
        if len(self._opened) >= self._times:
            minutes = self._window_s / 60
            self._notifier.notify(
                self._lane,
                NotificationKind.BREAKER_TRIPPED,
                f"{lane_info(self._lane).title}: the gateway circuit breaker opened "
                f"{len(self._opened)} times within {minutes:g} minutes.",
            )


def state_dir(settings: LaneSettings) -> Path:
    """The supervisor's directory for the lane's config: heartbeats, pid files and logs."""
    return (settings.repo / settings.runs_dir / "rsi" / "supervisor").resolve()


def lane_hooks(settings: LaneSettings) -> LaneHooks:
    """Build the real lane's hooks: GitHub results PRs and ntfy.sh notifications."""
    gateway = load_settings()
    ledger = Ledger(settings.ledger())
    notifier = notifier_from_settings(gateway, ledger)
    publisher = ResultsPublisherImpl(
        ledger,
        settings.repo.resolve(),
        github=GitHubApi(GITHUB_REPO, runtime_github_token),
        remote=HttpsRemote(GITHUB_REPO, runtime_github_token),
        notifier=notifier,
    )
    return build_hooks(settings, publisher, notifier)


def build_hooks(
    settings: LaneSettings, publisher: ResultsPublisher, notifier: Notifier
) -> LaneHooks:
    """Wire ``publisher`` and ``notifier`` into the lane's hooks."""
    lane_value = LANES[settings.lane_name]
    beat = heartbeat_file(state_dir(settings), lane_value)

    async def after_round(_: RoundOutcome) -> None:
        now = datetime.now(UTC)
        if await asyncio.to_thread(publisher.due, lane_value, now):
            await asyncio.to_thread(publisher.publish, lane_value, final=False, now=now)

    async def results_pr(stop: StopInfo) -> None:
        await asyncio.to_thread(publisher.publish, stop.lane, final=True, now=datetime.now(UTC))

    async def notify(stop: StopInfo) -> None:
        title = lane_info(stop.lane).title
        aborted = f", {len(stop.aborted)} experiment(s) aborted" if stop.aborted else ""
        notifier.notify(
            stop.lane, NotificationKind.STOPPED, f"{title} stopped: {stop.rule}{aborted}."
        )
        await asyncio.to_thread(notifier.close, _CLOSE_S)

    @asynccontextmanager
    async def running() -> AsyncIterator[None]:
        watch = BreakerWatch(notifier, lane_value)
        throttle = logging.getLogger(_THROTTLE_LOGGER)
        throttle.addHandler(watch)
        try:
            async with heartbeat(beat):
                if isinstance(notifier, QueuedNotifier):  # Resend what a crashed run could not.
                    try:
                        await asyncio.to_thread(notifier.resume, lane_value, _LANE_KINDS)
                    except Exception:
                        _logger.exception("could not resend undelivered notifications")
                yield
        finally:
            throttle.removeHandler(watch)

    return LaneHooks(results_pr=results_pr, notify=notify, after_round=after_round, running=running)
