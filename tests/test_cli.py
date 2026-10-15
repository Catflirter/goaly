"""Tests for goaly.cli, the composition root: the `goaly` command and what it ships with."""

import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

from goaly import cli
from goaly.harness.core.llm_budgets import load_budgets
from goaly.harness.core.manifest import CORE_DIR

GOALY = Path(sysconfig.get_path("scripts")) / "goaly"
QWEN, GLM = "qwen3.6-35b-a3b", "glm-5.3"


def test_the_console_script_runs_the_goaly_command() -> None:
    # The arguments are fixed strings, not untrusted input.
    completed = subprocess.run(  # noqa: S603
        [GOALY, "run", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env={**os.environ, "PYTHONWARNINGS": "error"},
    )

    assert completed.returncode == 0, completed.stderr
    assert "--bench" in completed.stdout
    assert "--phase" in completed.stdout


def test_main_runs_goaly_run_with_the_registered_harnesses(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    argv = ["goaly", "run", "--bench", "effisql", "--split", "val", "--phase", "rsi"]
    monkeypatch.setattr(sys, "argv", [*argv, "--out", str(tmp_path / "run")])

    status = cli.main()

    # T11 registers Effi-SQL; until then, no benchmark is.
    assert status == 2
    assert "no benchmark 'effisql'; none is registered" in capsys.readouterr().err
    assert cli.registry().benchmarks() == []


def test_the_budgets_shipped_are_the_specs() -> None:
    budgets = load_budgets(cli.BUDGETS_FILE)

    assert budgets.budgets == {QWEN: 48, GLM: 16}
    shares = {
        phase: {process: budgets.share(phase, process) for process in processes}
        for phase, processes in budgets.phases.items()
    }
    assert shares == {
        "baseline": {"effisql": {QWEN: 40}, "background": {QWEN: 8}},
        "rsi": {"effisql": {QWEN: 32, GLM: 8}, "background": {QWEN: 8}},
        "final": {"effisql": {QWEN: 40}},
    }


def test_the_budgets_are_outside_the_harness_hash() -> None:
    # Changing a share must not change the harness hash, or the evaluation cache would miss.
    assert not cli.BUDGETS_FILE.is_relative_to(CORE_DIR)
