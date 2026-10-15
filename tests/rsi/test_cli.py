"""Tests for goaly.rsi.cli, in-process: config loading, the lane check and status."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from goaly.rsi.cli import ConfigError, import_object, load_settings, main, status
from tests.rsi.dry_run import write_config


def test_status_reads_a_fresh_lane(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _, config = write_config(tmp_path, datetime.now(UTC) + timedelta(hours=1))
    state = status(load_settings(config))
    assert state["next_round"] == 1
    assert state["last_event"] is None
    assert state["ledger_best"] is None
    assert main(["status", "--config", str(config)]) == 0
    assert '"rounds_without_accept": 0' in capsys.readouterr().out
    assert not (tmp_path / "ledger").exists()  # Reading changed nothing.


def test_config_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["lane", "--config", str(tmp_path / "missing.toml")]) == 3
    bad = tmp_path / "bad.toml"
    bad.write_text("this is = = not toml", encoding="utf-8")
    assert main(["status", "--config", str(bad)]) == 3
    assert "configuration error" in capsys.readouterr().err
    with pytest.raises(ConfigError, match="module:function"):
        import_object("no_colon", "wiring")


def test_a_crashing_lane_exits_with_1(tmp_path: Path) -> None:
    _, config = write_config(tmp_path, datetime.now(UTC) + timedelta(hours=1))
    text = config.read_text().replace("tests.rsi.dry_run:build", "tests.rsi.test_cli:explode")
    config.write_text(text, encoding="utf-8")
    assert main(["lane", "--config", str(config)]) == 1


def explode(settings: object) -> object:
    raise RuntimeError("Docker is down")


def test_the_shipped_config_is_valid() -> None:
    settings = load_settings(Path(__file__).resolve().parents[2] / "rsi" / "lane-effisql.toml")
    assert settings.lane.deadline == datetime(2026, 10, 15, 20, 0, tzinfo=UTC)
    assert settings.lane.round.decision.min_delta == 0.05
    assert settings.lane.round.k == 3
    # The integration names T11's evaluator, T22's hooks and T12's v0 tag.
    assert settings.evaluator == "goaly.bench.effisql.evaluate:make_evaluator"
    assert settings.hooks == "goaly.rsi.hooks:lane_hooks"
    assert settings.v0 == "v0"
