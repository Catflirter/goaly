"""Progress output is best effort (Task 21a review C0-1, C0-2, X0-1): `make gpu-apply | tee` (L21-4) means gpuctl's
stdout may be a pipe whose reader is gone (tee killed by the same Ctrl-C, `| head`, a demo filter that died). Python
ignores SIGPIPE, so every write then raises BrokenPipeError. A progress line must never stop the step it reports:
gpu-down still terminates every instance, a failed or interrupted apply still terminates what it launched, and both
still take the terminated IPs out of the firewall allow-lists.
"""
import subprocess
import sys

import pytest

import gpuctl.__main__ as cli
from fakes import FakeInfra, FakeLambda
from gpuctl.__main__ import main
from gpuctl.output import write
from test_cli import ENV, NODE, in_sync
from test_reconcile import SPEC, TYPES, inst, make_rec

IPS = {"id-k3s-gpu-a6000-1": "129.80.30.30", "id-k3s-gpu-a6000-2": "129.80.30.31"}


class DeadPipe:
    """A stdout whose reader goes away after `alive` writes (or when `die()` is called)."""

    def __init__(self, alive=0):
        self.alive = alive
        self.writes = []

    def die(self):
        self.alive = 0

    def write(self, text):
        if self.alive <= 0:
            raise BrokenPipeError(32, "Broken pipe")
        self.alive -= 1
        self.writes.append(text)
        return len(text)

    def flush(self):
        if self.alive <= 0:
            raise BrokenPipeError(32, "Broken pipe")


@pytest.fixture
def dead_stderr(monkeypatch):
    """stderr is gone too (`2>&1 | tee`)."""
    monkeypatch.setattr(sys, "stderr", DeadPipe())


def two_nodes():
    return [inst("k3s-gpu-a6000-1", ip=IPS["id-k3s-gpu-a6000-1"]),
            inst("k3s-gpu-a6000-2", ip=IPS["id-k3s-gpu-a6000-2"])]


@pytest.mark.parametrize("alive", [0, 3])
def test_down_terminates_every_instance_into_a_dead_stdout(alive, dead_stderr):  # C0-1
    fake, infra = FakeLambda(types=TYPES, instances=two_nodes()), FakeInfra()
    assert make_rec(fake, infra, out=DeadPipe(alive)).down(SPEC, auto_approve=True) == 0
    assert sorted(fake.terminated) == sorted(IPS)
    assert sorted(infra.deleted_nodes) == ["k3s-gpu-a6000-1", "k3s-gpu-a6000-2"]
    assert infra.applied_gpu_ips[-1] == [] and infra.events[-1] == "nft"


def test_ctrl_c_after_the_first_terminate_with_a_dead_stdout_still_reconciles_the_firewall(dead_stderr):  # X0-1 (b)
    fake, infra = FakeLambda(types=TYPES, instances=two_nodes()), FakeInfra()
    out = DeadPipe(alive=100)
    forget = infra.forget_host_key

    def interrupted_forget(name):  # Ctrl-C right after the first terminate: tee dies with it
        forget(name)
        out.die()
        raise KeyboardInterrupt

    infra.forget_host_key = interrupted_forget
    assert make_rec(fake, infra, out=out).down(SPEC, auto_approve=True) == 1
    [first] = fake.terminated  # the KeyboardInterrupt still stops the remaining deletes
    assert infra.applied_gpu_ips[-1] == sorted(ip for i, ip in IPS.items() if i != first)
    assert infra.events[-1] == "nft"  # sync-peers ran: the terminated IP left k3s_peers and the rulesets


def test_ctrl_c_during_join_with_a_dead_stdout_terminates_the_new_instance(dead_stderr):  # C0-2
    fake, infra = FakeLambda(types=TYPES), FakeInfra()
    out = DeadPipe(alive=100)
    join = infra.join_node

    def interrupted_join(name, ip, gpu_type, progress=None):
        out.die()  # stdout dies when the join step starts
        join(name, ip, gpu_type, progress=progress)
        raise KeyboardInterrupt

    infra.join_node = interrupted_join
    assert make_rec(fake, infra, out=out).apply(SPEC, auto_approve=True, keep_on_failure=False) == 1
    assert fake.terminated == fake.launched_ids == ["i-1"]
    assert infra.applied_gpu_ips[-1] == [] and infra.events[-1] == "nft"


def test_ctrl_c_while_waiting_for_the_ip_with_a_dead_stdout_terminates_the_new_instance(dead_stderr):  # X0-1 (a)
    fake, infra = FakeLambda(types=TYPES), FakeInfra()
    out = DeadPipe(alive=100)

    def interrupted_get(instance_id):
        out.die()
        raise KeyboardInterrupt

    fake.get_instance = interrupted_get
    assert make_rec(fake, infra, out=out).apply(SPEC, auto_approve=True, keep_on_failure=False) == 1
    assert fake.terminated == ["i-1"] and infra.joined == []


def test_a_join_streaming_into_a_dead_stdout_still_joins():
    fake, infra = FakeLambda(types=TYPES), FakeInfra()
    out = DeadPipe(alive=100)
    join = infra.join_node

    def join_then_die(name, ip, gpu_type, progress=None):
        out.die()
        join(name, ip, gpu_type, progress=progress)  # its "joined and Ready" line goes nowhere

    infra.join_node = join_then_die
    assert make_rec(fake, infra, out=out).apply(SPEC, auto_approve=True, keep_on_failure=False) == 0
    assert fake.terminated == [] and infra.joined[0][0] == "k3s-gpu-a6000-1"


def test_the_cli_into_a_dead_stdout_still_terminates(monkeypatch, tmp_path, dead_stderr):
    monkeypatch.setattr(cli, "OVERRIDES_PATH", tmp_path / ".autoscale-state.json")
    fake, infra = FakeLambda(types=TYPES, instances=two_nodes()), FakeInfra()
    assert main(["down", "--auto-approve"], env=ENV, client=fake, infra=infra, out=DeadPipe()) == 0
    assert sorted(fake.terminated) == sorted(IPS)


def test_status_into_a_dead_stdout_keeps_its_exit_code(dead_stderr):
    assert main(["status"], env=ENV, client=FakeLambda(types=TYPES, instances=[NODE]), infra=in_sync(NODE),
                out=DeadPipe()) == 0
    infra = in_sync(NODE)
    infra.nft_elements.append("198.51.100.99")  # a stale k3s_peers entry: still drift
    assert main(["status"], env=ENV, client=FakeLambda(types=TYPES, instances=[NODE]), infra=infra,
                out=DeadPipe()) == 3


def test_a_prompt_that_cannot_be_shown_declines(monkeypatch):
    monkeypatch.setattr(sys, "stdin", __import__("io").StringIO("yes\n"))
    fake = FakeLambda(types=TYPES, instances=two_nodes())
    assert main(["down"], env=ENV, client=fake, infra=FakeInfra(), out=DeadPipe(alive=3)) == 2
    assert fake.terminated == []


def test_write_reports_a_dead_stream():
    assert write(DeadPipe(alive=5), "x\n") is True
    assert write(DeadPipe(), "x\n") is False


def test_a_dead_process_stdout_is_replaced_by_dev_null():
    """With the real stdout gone, later writes and the interpreter's final flush neither fail nor change the exit
    code (Python would otherwise print 'Exception ignored ... BrokenPipeError' and exit 120)."""
    code = (
        "import os, sys\n"
        "from gpuctl.output import write\n"
        "r, w = os.pipe(); os.dup2(w, 1); os.close(w); os.close(r)\n"  # stdout is a pipe nobody reads
        "assert write(sys.stdout, 'one\\n') is False\n"
        "assert write(sys.stdout, 'two\\n') is True\n"  # now /dev/null
        "sys.stdout.write('buffered\\n')\n"
        "sys.exit(7)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            env={"PYTHONPATH": str(cli.REPO_ROOT / "lambda")})
    assert (result.returncode, result.stderr) == (7, "")
