"""Tests for goaly.cli, the composition root: the `goaly` command and what it ships with."""

import logging
import os
import subprocess
import sys
import sysconfig
from collections.abc import Iterator
from pathlib import Path

import pytest

from goaly import cli
from goaly.bench.effisql.harness import open_grading, open_harness
from goaly.harness.core.llm_budgets import load_budgets
from goaly.harness.core.manifest import CORE_DIR

GOALY = Path(sysconfig.get_path("scripts")) / "goaly"
QWEN, GLM = "qwen3.6-35b-a3b", "glm-5.3"


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    """cli.main configures logging for the process: put it back after each test."""
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    for handler in list(root.handlers):
        if handler not in handlers:
            root.removeHandler(handler)
    root.setLevel(level)


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
    assert "--budget-phase" in completed.stdout


def test_main_runs_goaly_run_with_the_registered_harnesses(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    argv = ["goaly", "run", "--bench", "nosuch", "--split", "val", "--budget-phase", "rsi"]
    monkeypatch.setattr(sys, "argv", [*argv, "--out", str(tmp_path / "run")])
    confined: list[bool] = []
    monkeypatch.setattr(cli, "confine_to_python_cpus", lambda: confined.append(True))

    status = cli.main()

    assert status == 2
    assert "no benchmark 'nosuch'; the benchmarks are: effisql" in capsys.readouterr().err
    assert confined == [True]


def test_effisql_is_registered_with_its_v0_pack_and_its_grading() -> None:
    registry = cli.registry()

    assert registry.benchmarks() == ["effisql"]
    assert registry.harnesses("effisql") == ["v0"]
    assert registry.factory("effisql", None) is open_harness
    assert registry.grading("effisql") is open_grading


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
