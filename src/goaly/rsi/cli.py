"""The lane's command line: ``goaly rsi lane --lane effisql`` and ``goaly rsi status``.

``python -m goaly.rsi`` runs it too, until goaly's own CLI (T06) mounts it as ``goaly rsi``.

- ``lane`` runs the lane until a stopping rule holds (goaly.rsi.lane). It never reads standard
  input, so it runs unattended (spec §8.0).
- ``status`` prints the lane's state as JSON and changes nothing.

Both read one TOML file, ``--config``, whose top-level keys are :class:`LaneSettings`. Its
``[lane]`` table is the LaneConfig, the deadline among it; ``wiring`` names the function that
builds the lane's stores, agents and evaluator. Exit codes: 0 when the lane stopped by rule, 3 for
a configuration error, 1 if the lane failed in any other way, which the supervisor counts as a
crash (spec §8.7).
"""

import argparse
import asyncio
import importlib
import json
import logging
import sys
import tomllib
from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from goaly.contracts.gen.experiment_pb2 import LANE_EFFISQL, Decision, LaneEventKind
from goaly.harness.core.logs import configure_logging
from goaly.rsi.lane import LaneConfig, LaneHooks, LaneRunner, RoundLog
from goaly.rsi.ledger import Ledger, default_ledger_dir
from goaly.rsi.round import RoundDeps
from goaly.rsi.workspace import Workspace

__all__ = ["EXIT_CONFIG", "EXIT_CRASHED", "EXIT_STOPPED", "ConfigError", "LaneSettings", "main"]

EXIT_STOPPED: Final = 0
EXIT_CRASHED: Final = 1
EXIT_CONFIG: Final = 3
DEFAULT_CONFIG: Final = Path("rsi/lane-effisql.toml")
_LANES: Final = {"effisql": LANE_EFFISQL}

_logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """The lane's configuration is unusable; the lane exits with 3."""


class LaneSettings(BaseModel):
    """The lane's config file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lane_name: Literal["effisql"] = Field(default="effisql", alias="name")
    repo: Path = Path()
    ledger_dir: Path | None = None  # Default: <repo>/rsi/ledger.
    worktree_root: Path | None = None  # Default: ~/goaly-rsi-worktrees.
    runs_dir: Path = Path("runs")  # Relative to the repository.
    v0: str = "baseline-v0"  # v0's tag or commit.
    # "module:function", called with these settings; it returns an async context manager that
    # yields the lane's RoundDeps and LaneHooks (goaly.rsi.wiring.build is the real lane's).
    wiring: str = "goaly.rsi.wiring:build"
    # The real wiring's "module:function" for T11's evaluator and T22's hooks.
    evaluator: str = ""
    hooks: str = ""
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    lane: LaneConfig

    def ledger(self) -> Path:
        return self.ledger_dir or default_ledger_dir(self.repo)


type Wiring = Callable[[LaneSettings], AbstractAsyncContextManager[tuple[RoundDeps, LaneHooks]]]


def load_settings(path: Path) -> LaneSettings:
    """Read the config file; raise ConfigError if it is missing or invalid."""
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        return LaneSettings.model_validate(data)
    except (OSError, tomllib.TOMLDecodeError, ValidationError) as error:
        raise ConfigError(f"{path}: {error}") from error


def import_object(spec: str, what: str) -> object:
    """Import ``module:name``; raise ConfigError if it cannot be found."""
    module_name, _, name = spec.partition(":")
    if not module_name or not name:
        raise ConfigError(f"{what} must be 'module:function', not {spec!r}")
    try:
        return getattr(importlib.import_module(module_name), name)
    except (ImportError, AttributeError) as error:
        raise ConfigError(f"cannot load {what} {spec!r}: {error}") from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="goaly rsi", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name, text in (("lane", "run a lane until it stops"), ("status", "show a lane's state")):
        command = commands.add_parser(name, help=text)
        command.add_argument("--lane", choices=sorted(_LANES), default="effisql")
        command.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.config)
        if settings.lane_name != args.lane:
            raise ConfigError(
                f"{args.config} configures lane {settings.lane_name}, not {args.lane}"
            )
        configure_logging(settings.log_level)
        if args.command == "status":
            print(json.dumps(status(settings), indent=2))  # noqa: T201  (the command's output)
            return EXIT_STOPPED
        return asyncio.run(_run_lane(settings))
    except ConfigError as error:
        print(f"goaly rsi: configuration error: {error}", file=sys.stderr)  # noqa: T201
        return EXIT_CONFIG
    except Exception:
        _logger.exception("the lane failed")
        return EXIT_CRASHED


async def _run_lane(settings: LaneSettings) -> int:
    # Past the deadline the lane only stops, so it needs none of what a round needs: a restart
    # after the freeze writes its stop record and calls the final results PR whatever is down.
    if datetime.now(UTC) >= settings.lane.deadline:
        from goaly.rsi.wiring import build_stopped  # Here: goaly.rsi.wiring imports this module.

        wiring: Wiring = build_stopped
    else:
        wiring = cast("Wiring", import_object(settings.wiring, "wiring"))
    async with wiring(settings) as (deps, hooks):
        return await LaneRunner(deps, settings.lane, hooks=hooks).run()


def status(settings: LaneSettings) -> dict[str, object]:
    """The lane's state, read from the ledger, the round log and the best tag."""
    lane = _LANES[settings.lane_name]
    ledger = Ledger(settings.ledger())
    latest = ledger.latest_experiments(lane)
    log = RoundLog(settings.ledger(), lane)
    state = log.state([record.round for record in latest.values()])
    events = ledger.lane_events(lane)
    best_tag = Workspace(settings.repo, root=settings.worktree_root).best_commit(lane)
    decisions: dict[str, int] = {}
    for record in latest.values():
        name = Decision.Name(record.decision)
        decisions[name] = decisions.get(name, 0) + 1
    return {
        "lane": settings.lane_name,
        "deadline": settings.lane.deadline.isoformat(),
        "best_tag": best_tag,
        "ledger_best": ledger.best_version(lane),
        "next_round": state.next_round,
        "unfinished_round": state.unfinished,
        "rounds_without_accept": state.without_accept,
        "experiments": decisions,
        "pending": [
            r.experiment_id for r in latest.values() if r.decision == Decision.DECISION_PENDING
        ],
        "last_event": None
        if not events
        else {
            "kind": LaneEventKind.Name(events[-1].kind),
            "detail": events[-1].detail,
            "at": events[-1].occurred_at.ToJsonString(),
        },
    }
