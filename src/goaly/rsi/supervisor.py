"""The supervisor: ``goaly rsi run [--detach]`` runs every lane until it stops (spec §8.7).

This machine has no systemd and no cron, so goaly supervises itself:

- ``run --detach`` starts a minimal shell loop in its own session, with standard input closed
  and output to ``supervisor.log``, and returns. The loop runs the supervisor, and starts it
  again 30 s after it crashes; it ends when the supervisor exits 0 (every lane stopped), 3 (a
  configuration error) or 4 (another supervisor holds the lock).
- ``run`` is the supervisor itself. A lock file keeps it to one instance. It runs each lane as
  ``python -m goaly.rsi lane`` in its own session, with no terminal input, and watches it:
  - a lane that exits 0 stopped by rule, and is never restarted;
  - a lane that exits 3 has an unusable configuration: it is not restarted, and the lead is
    notified;
  - any other exit, a heartbeat 10 minutes old (the lane is then killed), or a start that fails
    the preflight checks (goaly.rsi.preflight) is a crash, recorded as a CRASHED LaneEvent;
  - a crashed lane starts again after 30 s, 1, 2 and 5 minutes, then every 10 minutes, and each
    restart is recorded as RESTARTED. A lane that ran for an hour starts the series again;
  - at 5 crashes within an hour, read from the ledger so a restarted supervisor counts them
    too, automatic recovery has failed: the lane is given up, RECOVERY_FAILED is recorded, and
    the lead is notified.

The supervisor exits 0 once every lane has stopped or been given up. On SIGTERM it kills its
lanes and exits 0, which ends the loop too. A lane left running by a supervisor that died is
killed when the next one starts, since only its own children can be watched; the lane then
resumes from the ledger.
"""

import argparse
import contextlib
import json
import logging
import os
import signal
import subprocess
import sys
import time
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import FrameType
from typing import IO, Final, cast

from google.protobuf.timestamp_pb2 import Timestamp
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from goaly.contracts.gen.experiment_pb2 import (
    LANE_EVENT_KIND_CRASHED,
    LANE_EVENT_KIND_RECOVERY_FAILED,
    LANE_EVENT_KIND_RESTARTED,
    LANE_EVENT_KIND_STARTED,
    LANE_EVENT_KIND_STOPPED,
    Lane,
    LaneEvent,
    LaneEventKind,
)
from goaly.harness.core.logs import configure_logging
from goaly.harness.core.settings import SettingsError
from goaly.harness.core.settings import load_settings as load_goaly_settings
from goaly.rsi.cli import EXIT_CONFIG, ConfigError, LaneSettings, import_object
from goaly.rsi.cli import load_settings as load_lane_settings
from goaly.rsi.heartbeat import STALE_AFTER_S, heartbeat_age, heartbeat_file
from goaly.rsi.hooks import LANES, state_dir
from goaly.rsi.lanes import lane_info
from goaly.rsi.ledger import Ledger
from goaly.rsi.locks import LockTimeoutError, file_lock
from goaly.rsi.notify import NotificationKind, Notifier, QueuedNotifier, notifier_from_settings
from goaly.rsi.preflight import PreflightConfig, check_lane

__all__ = ["EXIT_LOCKED", "Supervisor", "SupervisorConfig", "crash_streak", "main"]

EXIT_DONE: Final = 0
EXIT_LOCKED: Final = 4
# "$@" is the supervisor's command. 0, 3 and 4 end the loop; anything else restarts it.
LOOP: Final = (
    'while :; do "$@"; code=$?; case $code in 0|3|4) exit "$code";; esac; '
    'echo "goaly rsi: the supervisor exited with $code; restarting" >&2; '
    'sleep "$GOALY_SUPERVISOR_RESTART_S"; done'
)

_logger = logging.getLogger(__name__)


class SupervisorConfig(BaseModel):
    """The supervisor's timings (spec §8.7); a TOML file passed with --supervisor-config."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    poll_s: float = Field(default=5.0, gt=0)
    heartbeat_stale_s: float = Field(default=STALE_AFTER_S, gt=0)
    backoffs_s: tuple[float, ...] = Field(default=(30.0, 60.0, 120.0, 300.0, 600.0), min_length=1)
    max_crashes: int = Field(default=5, ge=1)
    crash_window_s: float = Field(default=3600.0, gt=0)
    healthy_run_s: float = Field(default=3600.0, gt=0)  # A run this long resets the backoff.
    kill_wait_s: float = Field(default=10.0, gt=0)
    restart_s: float = Field(default=30.0, ge=0)  # The shell loop's pause before a restart.
    preflight: PreflightConfig = PreflightConfig()
    # "module:function" to run instead of goaly.rsi.preflight.check_lane, called with the lane's
    # settings and returning its problems: for dry runs with no Docker or PostgreSQL.
    preflight_function: str = ""


type Preflight = Callable[[LaneSettings], list[str]]
type LaneCommand = Callable[[Path, LaneSettings], list[str]]
type NotifierFactory = Callable[[LaneSettings], Notifier]


def lane_command(config: Path, settings: LaneSettings) -> list[str]:
    """``python -m goaly.rsi lane --lane <name> --config <config>``, as goaly.rsi.cli takes it."""
    name = settings.lane_name
    return [sys.executable, "-m", "goaly.rsi", "lane", "--lane", name, "--config", str(config)]


def crash_streak(events: Sequence[LaneEvent], healthy_s: float) -> int:
    """How many crashes in a row the lane has had, from its events, oldest first.

    A run that lasted ``healthy_s`` before it crashed starts a new series, and so does a stop.
    """
    streak = 0
    started: datetime | None = None
    for event in events:
        moment = event.occurred_at.ToDatetime(tzinfo=UTC)
        if event.kind == LANE_EVENT_KIND_STARTED:
            started = moment
        elif event.kind == LANE_EVENT_KIND_CRASHED:
            healthy = started is not None and moment - started >= timedelta(seconds=healthy_s)
            streak = 1 if healthy else streak + 1
            started = None
        elif event.kind in (LANE_EVENT_KIND_STOPPED, LANE_EVENT_KIND_RECOVERY_FAILED):
            streak = 0
    return streak


@dataclass
class _Slot:
    config: Path
    settings: LaneSettings
    lane: Lane.ValueType
    ledger: Ledger
    notifier: Notifier
    process: subprocess.Popen[bytes] | None = None
    started: float = 0.0
    start_at: float = 0.0
    backoff_s: float | None = None  # Set while a restart waits.
    done: bool = False

    @property
    def state(self) -> Path:
        return state_dir(self.settings)

    @property
    def beat(self) -> Path:
        return heartbeat_file(self.state, self.lane)

    @property
    def pid_file(self) -> Path:
        return self.state / f"{lane_info(self.lane).name}.pid"


class Supervisor:
    """Runs one process per lane until every lane has stopped or been given up."""

    def __init__(
        self,
        lanes: Sequence[tuple[Path, LaneSettings]],
        *,
        config: SupervisorConfig = SupervisorConfig(),  # noqa: B008  (frozen)
        preflight: Preflight | None = None,
        command: LaneCommand = lane_command,
        notifiers: NotifierFactory | None = None,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._config = config
        if preflight is None and config.preflight_function:
            preflight = cast(Preflight, import_object(config.preflight_function, "preflight"))
        self._preflight = preflight or (lambda s: _default_preflight(s, config.preflight))
        self._command = command
        self._clock = clock
        self._now = now
        self._stopping = False
        make_notifier = notifiers or _default_notifier
        self._slots = [
            _Slot(
                path,
                settings,
                LANES[settings.lane_name],
                Ledger(settings.ledger()),
                make_notifier(settings),
            )
            for path, settings in lanes
        ]

    def stop(self) -> None:
        """Ask :meth:`run` to kill the lanes and return; safe from a signal handler."""
        self._stopping = True

    def run(self) -> int:
        """Supervise until every lane is done; return the exit code, 0."""
        for slot in self._slots:
            _kill_orphan(slot.pid_file)
        try:
            while not self._stopping and not all(slot.done for slot in self._slots):
                for slot in self._slots:
                    if not slot.done and not self._stopping:
                        self._tend(slot)
                time.sleep(self._config.poll_s)
        finally:
            for slot in self._slots:
                if slot.process is not None:
                    self._kill(slot)
                slot.notifier.close()
        return EXIT_DONE

    def _tend(self, slot: _Slot) -> None:
        if slot.process is None:
            if self._clock() >= slot.start_at:
                self._start(slot)
            return
        code = slot.process.poll()
        if code is None:
            age = heartbeat_age(slot.beat)
            age = self._clock() - slot.started if age is None else age
            if age > self._config.heartbeat_stale_s:
                self._kill(slot)
                self._crashed(slot, f"its heartbeat was {age:.0f} s old, so it was killed")
            return
        slot.process = None
        slot.pid_file.unlink(missing_ok=True)
        if code == EXIT_DONE:
            _logger.info("the lane stopped", extra={"lane": slot.settings.lane_name})
            slot.done = True
        elif code == EXIT_CONFIG:
            self._record(slot, LANE_EVENT_KIND_CRASHED, "a configuration error (exit 3)")
            self._give_up(slot, "its configuration is unusable (exit 3), so it is not restarted")
        else:
            cause = f"killed by signal {-code}" if code < 0 else f"exited with {code}"
            self._crashed(slot, f"the lane process {cause}")

    def _start(self, slot: _Slot) -> None:
        problems = self._preflight(slot.settings)
        if problems:
            self._crashed(slot, f"the preflight checks failed: {'; '.join(problems)}"[:1000])
            return
        if slot.backoff_s is not None:
            self._record(
                slot, LANE_EVENT_KIND_RESTARTED, f"after a backoff of {slot.backoff_s:g} s"
            )
            slot.backoff_s = None
        slot.state.mkdir(parents=True, exist_ok=True)
        slot.beat.unlink(missing_ok=True)
        log: IO[bytes]
        with (slot.state / f"{slot.settings.lane_name}.log").open("ab") as log:
            slot.process = subprocess.Popen(  # noqa: S603  (the lane's own command)
                self._command(slot.config, slot.settings),
                cwd=slot.settings.repo.resolve(),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        slot.started = self._clock()
        slot.pid_file.write_text(f"{slot.process.pid}\n", encoding="utf-8")
        _logger.info(
            "lane started", extra={"lane": slot.settings.lane_name, "pid": slot.process.pid}
        )

    def _crashed(self, slot: _Slot, detail: str) -> None:
        _logger.warning(
            "the lane crashed", extra={"lane": slot.settings.lane_name, "detail": detail}
        )
        self._record(slot, LANE_EVENT_KIND_CRASHED, detail)
        events = self._events(slot)
        window = self._now() - timedelta(seconds=self._config.crash_window_s)
        recent = [
            event
            for event in events
            if event.kind == LANE_EVENT_KIND_CRASHED
            and event.occurred_at.ToDatetime(tzinfo=UTC) >= window
        ]
        if len(recent) >= self._config.max_crashes:
            hours = self._config.crash_window_s / 3600
            self._give_up(
                slot, f"{len(recent)} crashes within {hours:g} h; the last: {detail}"[:1000]
            )
            return
        streak = max(1, crash_streak(events, self._config.healthy_run_s))
        backoffs = self._config.backoffs_s
        slot.backoff_s = backoffs[min(streak, len(backoffs)) - 1]
        slot.start_at = self._clock() + slot.backoff_s

    def _give_up(self, slot: _Slot, detail: str) -> None:
        slot.done = True
        self._record(slot, LANE_EVENT_KIND_RECOVERY_FAILED, detail)
        title = lane_info(slot.lane).title
        slot.notifier.notify(
            slot.lane,
            NotificationKind.RECOVERY_FAILED,
            f"{title}: automatic recovery failed and the lane is stopped: {detail}",
        )

    def _kill(self, slot: _Slot) -> None:
        process = slot.process
        if process is None:
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        try:
            process.wait(self._config.kill_wait_s)
        except subprocess.TimeoutExpired:
            _logger.error("a killed lane did not exit", extra={"pid": process.pid})
        slot.process = None
        slot.pid_file.unlink(missing_ok=True)

    def _events(self, slot: _Slot) -> list[LaneEvent]:
        try:
            return slot.ledger.lane_events(slot.lane)
        except Exception:
            _logger.exception("could not read the lane's events")
            return []

    def _record(self, slot: _Slot, kind: LaneEventKind.ValueType, detail: str) -> None:
        stamp = Timestamp()
        stamp.FromDatetime(self._now())
        event = LaneEvent(lane=slot.lane, kind=kind, detail=detail, occurred_at=stamp)
        try:
            slot.ledger.record_lane_event(event)
        except Exception:  # The supervisor goes on; the log keeps what happened.
            _logger.exception("could not record a lane event", extra={"detail": detail})


def _default_preflight(settings: LaneSettings, config: PreflightConfig) -> list[str]:
    return check_lane(
        LANES[settings.lane_name], settings.repo.resolve(), settings.worktree_root, config
    )


def _default_notifier(settings: LaneSettings) -> Notifier:
    """ntfy.sh from the goaly settings, resending the recovery failures an earlier supervisor
    could not deliver. Unusable settings turn notifications off rather than stop supervising:
    each one is then recorded as not delivered."""
    ledger = Ledger(settings.ledger())
    try:
        notifier = notifier_from_settings(load_goaly_settings(), ledger)
    except SettingsError as error:
        _logger.error(
            "notifications are off: the settings are unusable", extra={"reason": str(error)}
        )
        notifier = QueuedNotifier(None, ledger)
    try:
        notifier.resume(LANES[settings.lane_name], {NotificationKind.RECOVERY_FAILED})
    except Exception:
        _logger.exception("could not resend undelivered notifications")
    return notifier


def _kill_orphan(pid_file: Path) -> None:
    """Kill a lane an earlier supervisor left running, if the pid is still that lane."""
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
        command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except (OSError, ValueError):
        pid_file.unlink(missing_ok=True)
        return
    if b"goaly.rsi" in command and b"lane" in command:
        _logger.warning("killing a lane an earlier supervisor left running", extra={"pid": pid})
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pid, signal.SIGKILL)
    pid_file.unlink(missing_ok=True)


def main(argv: Sequence[str]) -> int:
    """``goaly rsi run``: see the module's description."""
    parser = argparse.ArgumentParser(prog="goaly rsi run", description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, action="append", help="a lane's config, repeatable")
    parser.add_argument("--supervisor-config", type=Path, help="a SupervisorConfig TOML file")
    parser.add_argument("--detach", action="store_true", help="start in the background, return")
    args = parser.parse_args(argv)
    configs = [path.resolve() for path in (args.config or [Path("rsi/lane-effisql.toml")])]
    try:
        lanes = [(path, load_lane_settings(path)) for path in configs]
        config = SupervisorConfig()
        if args.supervisor_config is not None:
            text = args.supervisor_config.read_text(encoding="utf-8")
            config = SupervisorConfig.model_validate(tomllib.loads(text))
        if config.preflight_function:
            import_object(config.preflight_function, "preflight")  # Fail now, not in the loop.
    except (ConfigError, OSError, tomllib.TOMLDecodeError, ValidationError) as error:
        print(f"goaly rsi run: configuration error: {error}", file=sys.stderr)  # noqa: T201
        return EXIT_CONFIG
    state = state_dir(lanes[0][1])
    state.mkdir(parents=True, exist_ok=True)
    lock = state / "supervisor.lock"
    if args.detach:
        return _detach(argv, state, lock, config)
    configure_logging(lanes[0][1].log_level)
    try:
        with file_lock(lock, timeout_s=0):
            (state / "supervisor.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
            supervisor = Supervisor(lanes, config=config)

            def on_signal(signum: int, frame: FrameType | None) -> None:
                supervisor.stop()

            signal.signal(signal.SIGTERM, on_signal)
            signal.signal(signal.SIGINT, on_signal)
            return supervisor.run()
    except LockTimeoutError:
        _logger.warning("another supervisor is running", extra={"lock": str(lock)})
        return EXIT_LOCKED


def _detach(argv: Sequence[str], state: Path, lock: Path, config: SupervisorConfig) -> int:
    try:
        with file_lock(lock, timeout_s=0):
            pass
    except LockTimeoutError:
        print(json.dumps({"running": True, "state": str(state)}))  # noqa: T201  (its output)
        return EXIT_DONE
    command = [sys.executable, "-m", "goaly.rsi", "run", *(a for a in argv if a != "--detach")]
    environment = {**os.environ, "GOALY_SUPERVISOR_RESTART_S": f"{config.restart_s:g}"}
    with (state / "supervisor.log").open("ab") as log:
        loop = subprocess.Popen(  # noqa: S603  (sh with goaly's own loop)
            ["/bin/sh", "-c", LOOP, "goaly-supervisor", *command],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=environment,
        )
    (state / "loop.pid").write_text(f"{loop.pid}\n", encoding="utf-8")
    print(json.dumps({"running": True, "pid": loop.pid, "log": str(state / "supervisor.log")}))  # noqa: T201
    return EXIT_DONE
