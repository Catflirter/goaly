"""Shared fixtures for the goaly.rsi tests."""

import pytest

from goaly.rsi import workspace


@pytest.fixture(autouse=True)
def _git_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give git an author and committer: tests run with an empty HOME, and CI sets none."""
    for role in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{role}_NAME", "goaly tests")
        monkeypatch.setenv(f"GIT_{role}_EMAIL", "tests@goaly.invalid")


@pytest.fixture(autouse=True)
def _no_sandbox_left(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unit tests start no sandbox, so the workspace's check that an experiment's sandbox has
    stopped finds none without asking docker. Tests of that check pass a probe of their own.
    """
    monkeypatch.setattr(workspace, "containers", lambda **labels: [])
