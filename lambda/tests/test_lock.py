"""One gpuctl write at a time (stretch S4 review r2-1; spec section 10.4: gpu-down always wins over the autoscaler).

apply, down and sync-peers hold flock(2) on .run/gpuctl.lock for the whole command; a second one waits for it and
says so. The read-only commands never wait. The stretch autoscaler holds the lock itself around its re-check, override
write and `gpuctl apply`, and hands it to that apply through GPUCTL_LOCK_FD (an inherited descriptor of the same
open file), so its apply does not wait for its own lock.
"""
import fcntl
import io
import os
import threading
import time

import pytest

import gpuctl.__main__ as cli
from fakes import FakeInfra, FakeLambda
from gpuctl import lock
from gpuctl.__main__ import main
from test_cli import APPLY_ENV, ENV, NODE, TYPES, overrides  # noqa: F401 (the override-file fixture)


@pytest.fixture
def lock_path(tmp_path_factory, monkeypatch):
    path = tmp_path_factory.mktemp("run") / "gpuctl.lock"
    monkeypatch.setattr(cli, "LOCK_PATH", path)
    return path


def hold(path):
    """Another process's lock (a separate open file description): returns its fd."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


def is_free(path):
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False
    finally:
        os.close(fd)


class Out(io.StringIO):
    """A stdout a waiting thread writes to while the test polls it."""

    def wait_for(self, text, timeout=10.0):
        deadline = time.monotonic() + timeout
        while text not in self.getvalue():
            assert time.monotonic() < deadline, f"{text!r} never printed: {self.getvalue()!r}"
            time.sleep(0.01)


def run_in_thread(argv, env, fake, out):
    result = {}
    thread = threading.Thread(target=lambda: result.update(rc=main(argv, env=env, client=fake, infra=FakeInfra(),
                                                                   out=out)))
    thread.start()
    return thread, result


@pytest.mark.parametrize("argv, env, waits_for", [
    (["down", "--auto-approve"], ENV, "gpu-down runs as soon as it ends"),
    (["apply", "--auto-approve"], APPLY_ENV, "runs as soon as it ends"),
    (["sync-peers"], ENV, "runs as soon as it ends"),
])
def test_a_write_command_waits_for_the_running_one(lock_path, argv, env, waits_for):
    held = hold(lock_path)
    fake, out = FakeLambda(types=TYPES, instances=[NODE]), Out()
    thread, result = run_in_thread(argv, env, fake, out)
    try:
        out.wait_for(waits_for)
        time.sleep(0.2)
        assert fake.calls == [] and thread.is_alive()  # nothing read, nothing changed while it waits
        assert str(lock_path) in out.getvalue()
    finally:
        os.close(held)
    thread.join(10)
    assert not thread.is_alive() and result["rc"] == 0
    assert fake.calls  # it ran once the lock was free
    assert is_free(lock_path)  # and released it


def test_gpu_down_wins_after_the_running_apply(lock_path, overrides):
    """The autoscaler's apply holds the lock; gpu-down waits, then removes the override and every GPU node."""
    held = hold(lock_path)
    overrides.write_text('{"pools": {"a6000": 2}}')
    fake, out = FakeLambda(types=TYPES, instances=[NODE]), Out()
    thread, result = run_in_thread(["down", "--auto-approve"], ENV, fake, out)
    out.wait_for("Waiting")
    assert overrides.exists() and fake.terminated == []
    os.close(held)
    thread.join(10)
    assert result["rc"] == 0 and fake.terminated == ["id-k3s-gpu-a6000-1"] and not overrides.exists()


def test_the_lock_is_held_while_the_command_runs(lock_path, capsys):
    seen = []
    fake = FakeLambda(types=TYPES, instances=[NODE])
    real = fake.list_instances

    def list_instances():
        seen.append(is_free(lock_path))
        return real()

    fake.list_instances = list_instances
    assert main(["down", "--auto-approve"], env=ENV, client=fake, infra=FakeInfra()) == 0
    assert seen and not any(seen)
    assert is_free(lock_path)


@pytest.mark.parametrize("argv", [["plan"], ["list-nodes"], ["rules"], ["status"]])
def test_read_only_commands_never_wait(lock_path, argv, capsys):
    held = hold(lock_path)
    try:
        fake = FakeLambda(types=TYPES)
        assert main(argv, env=ENV, client=fake, infra=FakeInfra()) in (0, 3)
        assert "Waiting" not in capsys.readouterr().out
    finally:
        os.close(held)


def test_an_inherited_lock_is_used_without_waiting(lock_path, capsys):
    """The autoscaler holds the lock and passes its descriptor: its own gpuctl apply runs at once."""
    held = hold(lock_path)
    try:
        fake = FakeLambda(types=TYPES)
        env = {**APPLY_ENV, lock.LOCK_FD_ENV: str(held)}
        assert main(["apply", "--auto-approve"], env=env, client=fake, infra=FakeInfra()) == 0
        assert len(fake.launched) == 1 and "Waiting" not in capsys.readouterr().out
        assert not is_free(lock_path)  # still the autoscaler's until it closes it
    finally:
        os.close(held)


@pytest.mark.parametrize("value", ["not-a-number", "999", "-1", "OTHER"])
def test_a_bad_inherited_descriptor_falls_back_to_the_lock_file(lock_path, tmp_path, value):
    """GPUCTL_LOCK_FD naming no descriptor, or one of another file, is ignored: the command takes the lock itself."""
    other = None
    if value == "OTHER":
        other = os.open(tmp_path / "other", os.O_RDWR | os.O_CREAT, 0o600)
        value = str(other)
    held = hold(lock_path)
    fake, out = FakeLambda(types=TYPES), Out()
    thread, result = run_in_thread(["down", "--auto-approve"], {**ENV, lock.LOCK_FD_ENV: value}, fake, out)
    try:
        out.wait_for("Waiting")
    finally:
        os.close(held)
        thread.join(10)
        if other is not None:
            os.close(other)
    assert result["rc"] == 0


def test_try_lock_and_inherited_fd_round_trip(tmp_path):
    path = tmp_path / "run" / "gpuctl.lock"
    fd = lock.try_lock(path)
    assert fd is not None and not is_free(path)
    assert lock.try_lock(path) is None  # a second holder is refused, never waits
    assert lock.inherited(path, {lock.LOCK_FD_ENV: str(fd)}) is True
    os.close(fd)
    assert is_free(path)


def test_lock_path_is_under_run(tmp_path):
    """.run/ is gitignored runtime state; the empty lock file records nothing (spec section 5.4)."""
    assert lock.lock_path(tmp_path) == tmp_path / ".run" / "gpuctl.lock"


def test_a_lock_file_that_cannot_be_opened_refuses(tmp_path, monkeypatch, capsys):
    (tmp_path / "not-a-dir").write_text("")
    monkeypatch.setattr(cli, "LOCK_PATH", tmp_path / "not-a-dir" / "gpuctl.lock")
    fake = FakeLambda(types=TYPES, instances=[NODE])
    assert main(["down", "--auto-approve"], env=ENV, client=fake, infra=FakeInfra()) == 2
    assert "gpuctl.lock" in capsys.readouterr().err and fake.terminated == []
