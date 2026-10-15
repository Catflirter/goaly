"""Tests for the shared test configuration: the integration opt-in, the secret isolation and the
async test backend."""

from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

REPO_ROOT = Path(__file__).resolve().parents[1]

PROBE_TESTS = """
import os
from pathlib import Path

import pytest


def goaly_variables():
    return sorted(name for name in os.environ if name.upper().startswith("GOALY_"))


def test_unit(tmp_path):
    assert Path.home().is_relative_to(tmp_path)
    assert list(Path.home().iterdir()) == []
    assert goaly_variables() == []


@pytest.mark.integration
def test_integration(tmp_path):
    assert not Path.home().is_relative_to(tmp_path)
    assert goaly_variables() == ["GOALY_API_KEY", "goaly_api_key"]
"""

ASYNC_PROBE_TESTS = """
import asyncio

import pytest

pytestmark = pytest.mark.anyio


async def test_async(anyio_backend):
    assert anyio_backend == "asyncio"
    asyncio.get_running_loop()
"""


@pytest.fixture
def project(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> pytest.Pytester:
    """A pytest session with this repository's pytest configuration and conftest.py.

    The environment carries GOALY_* variables, as a developer's shell might, in both cases:
    pydantic-settings matches names case-insensitively.
    """
    pytester.makepyprojecttoml((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pytester.makeconftest((REPO_ROOT / "tests" / "conftest.py").read_text(encoding="utf-8"))
    pytester.makepyfile(test_probe=PROBE_TESTS)
    monkeypatch.setenv("GOALY_API_KEY", "not-a-real-key")
    monkeypatch.setenv("goaly_api_key", "not-a-real-key")
    return pytester


def test_integration_tests_are_skipped_by_default(project: pytest.Pytester) -> None:
    result = project.runpytest("-rs", "test_probe.py")

    result.assert_outcomes(passed=1, skipped=1)
    result.stdout.fnmatch_lines(["SKIPPED * integration test: pass --integration to run"])


def test_integration_flag_runs_integration_tests(project: pytest.Pytester) -> None:
    result = project.runpytest("--integration", "test_probe.py")

    result.assert_outcomes(passed=2)


def test_async_tests_run_once_on_asyncio(project: pytest.Pytester) -> None:
    project.makepyfile(test_async_probe=ASYNC_PROBE_TESTS)

    result = project.runpytest("test_async_probe.py")

    result.assert_outcomes(passed=1)
