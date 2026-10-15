"""Tests for goaly.rsi.wiring: the real lane's configuration checks, before anything starts."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from goaly.rsi.cli import ConfigError, load_settings
from goaly.rsi.lane import LaneHooks
from goaly.rsi.wiring import build
from tests.rsi.dry_run import write_config

pytestmark = pytest.mark.anyio


def settings(tmp_path: Path, **lines: str) -> object:
    _, config = write_config(tmp_path, datetime.now(UTC) + timedelta(hours=1))
    text = "".join(
        line
        for line in config.read_text().splitlines(keepends=True)
        if not line.startswith(("wiring =", "hooks =", "evaluator ="))
    )
    extra = "".join(f'{key} = "{value}"\n' for key, value in lines.items())
    config.write_text(extra + text, encoding="utf-8")
    return load_settings(config)


async def test_the_real_lane_needs_an_evaluator(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="evaluator must name"):
        async with build(settings(tmp_path)):  # type: ignore[arg-type]  # LaneSettings.
            pass


async def test_the_real_lane_needs_the_gateway_settings(tmp_path: Path) -> None:
    configured = settings(
        tmp_path, evaluator="tests.rsi.test_wiring:evaluator", hooks="tests.rsi.test_wiring:hooks"
    )
    with pytest.raises(ConfigError):  # The test HOME has no ~/.config/goaly/.env.
        async with build(configured):  # type: ignore[arg-type]  # LaneSettings.
            pass


def evaluator(**values: object) -> object:
    raise AssertionError("never reached")


def hooks(settings: object) -> LaneHooks:
    return LaneHooks()
