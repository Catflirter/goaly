"""Shared pytest fixtures for the gpuctl tests."""
import pytest

from fakes import FakeShell


@pytest.fixture(autouse=True)
def fake_shells_use_every_scripted_result():
    """Fail a test that leaves FakeShell results unused: the commands it scripted never ran, so it proves nothing."""
    FakeShell.created.clear()
    yield
    shells = list(FakeShell.created)
    FakeShell.created.clear()
    for sh in shells:
        assert not sh.results, (f"FakeShell left {len(sh.results)} scripted result(s) unused after "
                                f"{len(sh.calls)} call(s) {[c.args for c in sh.calls]}: {sh.results}")
