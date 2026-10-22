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


def test_an_unset_or_empty_lock_fd_takes_the_lock_itself(lock_path):
    """No GPUCTL_LOCK_FD (an operator's make gpu-down): the command takes the lock through its own descriptor."""
    for env in (ENV, {**ENV, lock.LOCK_FD_ENV: ""}):
        fake = FakeLambda(types=TYPES)
        assert main(["down", "--auto-approve"], env=env, client=fake, infra=FakeInfra(), out=io.StringIO()) == 0
        assert is_free(lock_path)


@pytest.mark.parametrize("value, why", [
    ("not-a-number", "not a descriptor number"),
    ("-1", "not a descriptor number"),
    ("CLOSED", "not an open descriptor"),  # the parent set the variable but did not pass the descriptor through
    ("OTHER", "not a descriptor of"),  # e.g. an AUTOSCALER_ROOT whose lambda/ resolves to another checkout
    ("SAME_FILE", "does not hold the lock"),  # the lock file, but another open file description: not the caller's
])
def test_an_invalid_inherited_lock_fd_refuses_at_once(lock_path, tmp_path, capsys, value, why):
    """Review S4-r3-1: GPUCTL_LOCK_FD set but not the caller's lock. Waiting for the lock file would wait for the
    parent that holds it (the autoscaler waits for this command's output): a deadlock that would block every later
    make gpu-down. The command refuses at once instead, naming the variable."""
    opened = []
    held = hold(lock_path)  # the parent (the autoscaler) holds the lock
    if value == "CLOSED":
        value = "999"
        with pytest.raises(OSError):
            os.fstat(999)
    elif value == "OTHER":
        opened.append(os.open(tmp_path / "other", os.O_RDWR | os.O_CREAT, 0o600))
        value = str(opened[-1])
    elif value == "SAME_FILE":
        opened.append(os.open(lock_path, os.O_RDWR))
        value = str(opened[-1])
    fake, out = FakeLambda(types=TYPES, instances=[NODE]), Out()
    thread, result = run_in_thread(["down", "--auto-approve"], {**ENV, lock.LOCK_FD_ENV: value}, fake, out)
    try:
        thread.join(5)
        assert not thread.is_alive(), "it waits for the parent's lock"
    finally:
        os.close(held)
        thread.join(10)
        for fd in opened:
            os.close(fd)
    assert result["rc"] == 2 and fake.calls == [] and "Waiting" not in out.getvalue()
    err = capsys.readouterr().err
    assert f"GPUCTL_LOCK_FD={value}" in err and why in err and str(lock_path) in err


def test_inherited_says_why_a_descriptor_is_not_the_lock(lock_path, tmp_path):
    held = hold(lock_path)
    other = os.open(tmp_path / "other", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        assert lock.inherited(lock_path, {}) is False
        assert lock.inherited(lock_path, {lock.LOCK_FD_ENV: str(held)}) is True
        assert lock.inherited_problem(lock_path, {lock.LOCK_FD_ENV: str(held)}) is None
        assert lock.inherited_problem(lock_path, {}) is None  # unset: nothing inherited, nothing wrong
        problem = lock.inherited_problem(lock_path, {lock.LOCK_FD_ENV: str(other)})
        assert "not a descriptor of" in problem and str(lock_path) in problem
    finally:
        os.close(held)
        os.close(other)


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


def test_the_waiting_message_names_every_holder_and_how_to_find_it(lock_path):
    """Review S4-r3-3: an interactive make gpu-apply/gpu-down holds the lock at its y/N prompt; the message names
    every possible holder and how to find it."""
    held = hold(lock_path)
    fake, out = FakeLambda(types=TYPES), Out()
    thread, result = run_in_thread(["down", "--auto-approve"], ENV, fake, out)
    try:
        out.wait_for("Waiting")
        text = out.getvalue()
    finally:
        os.close(held)
        thread.join(10)
    for holder in ("make gpu-apply", "make gpu-down", "confirmation prompt", "autoscaler", "cpu-pool.sh",
                   f"fuser -v {lock_path}", "Ctrl-C stops waiting"):
        assert holder in text, holder
    assert result["rc"] == 0


def test_the_waiting_message_is_repeated_while_it_waits(lock_path, monkeypatch):
    monkeypatch.setattr(lock, "WAIT_REPEAT_S", 0.2)
    held = hold(lock_path)
    fake, out = FakeLambda(types=TYPES), Out()
    thread, result = run_in_thread(["down", "--auto-approve"], ENV, fake, out)
    try:
        out.wait_for("Still waiting")
        out.wait_for("Still waiting", timeout=10.0)
    finally:
        os.close(held)
        thread.join(10)
    text = out.getvalue()
    assert text.count("Waiting for") == 1 and text.count("Still waiting") >= 1
    assert all(str(lock_path) in line for line in text.splitlines() if "aiting" in line)
    assert result["rc"] == 0


DRIVER = """
import sys
from pathlib import Path
sys.path[:0] = [sys.argv[1], sys.argv[2]]
import gpuctl.__main__ as cli
from fakes import FakeInfra, FakeLambda
cli.LOCK_PATH = Path(sys.argv[3])
sys.exit(cli.main(["down", "--auto-approve"], env={"LAMBDA_API_KEY": "k"}, client=FakeLambda(), infra=FakeInfra()))
"""


def test_ctrl_c_while_waiting_exits_130_without_a_traceback(lock_path):
    """Review S4-r3-5: the message invites Ctrl-C; a real SIGINT while waiting ends with one line and exit 130."""
    import signal
    import subprocess
    import sys
    from pathlib import Path

    tests = Path(__file__).resolve().parent
    held = hold(lock_path)
    try:
        proc = subprocess.Popen([sys.executable, "-u", "-c", DRIVER, str(tests.parent), str(tests), str(lock_path)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, stdin=subprocess.DEVNULL)
        try:
            first = proc.stdout.readline()
            assert "Waiting for" in first
            proc.send_signal(signal.SIGINT)
            out, err = proc.communicate(timeout=10)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    finally:
        os.close(held)
    assert proc.returncode == 130
    assert "Traceback" not in err and "KeyboardInterrupt" not in err
    assert err.strip().splitlines() == [f"gpuctl: stopped waiting for {lock_path}: nothing was changed"]


def test_ctrl_c_during_a_command_exits_130_with_one_line(capsys):
    fake = FakeLambda(types=TYPES, instances=[NODE])

    def interrupted():
        raise KeyboardInterrupt

    fake.list_instances = interrupted
    assert main(["down", "--auto-approve"], env=ENV, client=fake, infra=FakeInfra()) == 130
    err = capsys.readouterr().err
    assert len(err.strip().splitlines()) == 1 and "interrupted (Ctrl-C)" in err and "make gpu-status" in err
