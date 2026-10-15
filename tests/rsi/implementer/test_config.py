"""Tests for goaly.rsi.implementer.config: implementer.yaml's limits and templates."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from goaly.rsi.implementer.config import DEFAULT_CONFIG, Limits, load_config


def test_the_shipped_config_holds_the_spec_limits() -> None:
    config = load_config()

    # Spec §8.8: 60 steps, $10 of GLM, 30 minutes, 3 format errors in a row, 120 s per command,
    # and at most 2 goaly-smoke runs.
    assert config.limits == Limits(
        step_limit=60,
        cost_limit_usd=10.0,
        wall_time_limit_s=1800,
        max_consecutive_format_errors=3,
        command_timeout_s=120,
        smoke_limit=2,
    )
    assert "output.returncode" in config.model.observation_template
    assert "finish_reason" in config.model.format_error_template


def test_an_unknown_key_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "implementer.yaml"
    path.write_text(DEFAULT_CONFIG.read_text(encoding="utf-8") + "extra: 1\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="extra"):
        load_config(path)


def test_a_limit_must_be_positive(tmp_path: Path) -> None:
    path = tmp_path / "implementer.yaml"
    text = DEFAULT_CONFIG.read_text(encoding="utf-8")
    path.write_text(text.replace("step_limit: 60", "step_limit: 0"), encoding="utf-8")

    with pytest.raises(ValidationError, match="step_limit"):
        load_config(path)


def test_a_missing_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")
