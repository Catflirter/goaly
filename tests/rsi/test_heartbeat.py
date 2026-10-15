"""Tests for goaly.rsi.heartbeat."""

import asyncio
import time
from pathlib import Path

import pytest

from goaly.contracts.gen.experiment_pb2 import LANE_EFFISQL
from goaly.rsi.heartbeat import beat, heartbeat, heartbeat_age, heartbeat_file

pytestmark = pytest.mark.anyio


def test_heartbeat_file_is_named_for_the_lane(tmp_path: Path) -> None:
    assert heartbeat_file(tmp_path, LANE_EFFISQL) == tmp_path / "effisql.heartbeat"


def test_age_of_a_missing_or_garbled_file_is_none(tmp_path: Path) -> None:
    path = tmp_path / "beat"
    assert heartbeat_age(path) is None
    path.write_text("not a time", encoding="utf-8")
    assert heartbeat_age(path) is None


def test_beat_records_the_time(tmp_path: Path) -> None:
    path = tmp_path / "beat"
    beat(path)
    age = heartbeat_age(path, now=time.time() + 100)
    assert age is not None
    assert 99 < age < 102
    assert heartbeat_age(path, now=0) == 0.0


async def test_heartbeat_beats_while_the_block_runs(tmp_path: Path) -> None:
    path = tmp_path / "state" / "effisql.heartbeat"
    async with heartbeat(path, interval_s=0.02):
        first = path.read_text(encoding="utf-8")
        await asyncio.sleep(0.1)
        assert path.read_text(encoding="utf-8") != first
    last = path.read_text(encoding="utf-8")
    await asyncio.sleep(0.06)
    assert path.read_text(encoding="utf-8") == last  # Stopped with the block.


async def test_a_failed_beat_is_logged_and_tried_again(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "effisql.heartbeat"
    async with heartbeat(path, interval_s=0.02):
        path.unlink()
        path.mkdir()  # os.replace onto a directory fails.
        await asyncio.sleep(0.07)
        path.rmdir()
        await asyncio.sleep(0.05)
        assert path.is_file()
    assert "could not write the heartbeat" in caplog.text
