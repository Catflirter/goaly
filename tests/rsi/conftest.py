"""Shared fixtures for the goaly.rsi tests."""

import pytest


@pytest.fixture(autouse=True)
def _git_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give git an author and committer: tests run with an empty HOME, and CI sets none."""
    for role in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{role}_NAME", "goaly tests")
        monkeypatch.setenv(f"GIT_{role}_EMAIL", "tests@goaly.invalid")
