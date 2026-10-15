"""Shared pytest configuration.

- Tests marked ``integration`` need Docker, a database or the real model API. They are skipped
  unless pytest runs with ``--integration``.
- Every other test runs with a fresh, empty ``HOME`` and no ``GOALY_*`` environment variables
  (in any case, as pydantic-settings matches names case-insensitively), so it can never read
  ``~/.config/goaly/.env`` or a real API key.
"""

import os
from pathlib import Path

import pytest

INTEGRATION_SKIP_REASON = "integration test: pass --integration to run"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="run the tests marked integration (Docker, a database or the real model API)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--integration"):
        return
    skip_integration = pytest.mark.skip(reason=INTEGRATION_SKIP_REASON)
    for item in items:
        if item.get_closest_marker("integration") is not None:
            item.add_marker(skip_integration)


@pytest.fixture(autouse=True)
def _isolate_from_real_secrets(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    if request.node.get_closest_marker("integration") is not None:
        return
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    for name in list(os.environ):
        if name.upper().startswith("GOALY_"):
            monkeypatch.delenv(name)
