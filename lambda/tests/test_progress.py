"""gpuctl apply's progress (live finding L21-4, progress.md 2026-10-21): `make gpu-apply | tee` showed nothing for
minutes because Python block-buffers a piped stdout, and the runbook needs each step's duration. gpuctl now flushes
every line (and scripts/gpuctl runs Python unbuffered, tests/test_gpuctl_wrapper.py), stamps the steps of apply and
down with the time since the run started and their durations, and shows join-node.sh's output (Task 14 deferred
minor) line by line instead of capturing it silently. join-node.sh never prints the join token: it is piped from
k3s-cp straight into the node.
"""
import io
import subprocess
import sys

import pytest

from fakes import FakeInfra, FakeLambda, FakeShell, fail, ok
from gpuctl.infra import CommandError, Infra, SubprocessShell
from test_infra import ROOT
from test_reconcile import SPEC, TYPES, Clock, active, booting, inst, make_rec


class Recorder(io.StringIO):
    """An output stream that remembers how much had been written at each flush."""

    def __init__(self):
        super().__init__()
        self.flushed = []

    def flush(self):
        self.flushed.append(len(self.getvalue()))
        super().flush()


def lines(out):
    return out.getvalue().splitlines()


# --- every line is flushed ------------------------------------------------------------------------

def test_every_apply_line_is_flushed_as_it_is_written():
    out = Recorder()
    assert make_rec(FakeLambda(types=TYPES), FakeInfra(), out=out).apply(SPEC, auto_approve=True,
                                                                          keep_on_failure=False) == 0
    ends = [i for i, ch in enumerate(out.getvalue()) if ch == "\n"]
    assert set(end + 1 for end in ends) <= set(out.flushed)  # each line was flushed when it was complete


def test_cli_lines_are_flushed(capsys, monkeypatch):
    import gpuctl.__main__ as cli
    out = Recorder()
    assert cli.main(["list-nodes"], env={"LAMBDA_API_KEY": "k"}, client=FakeLambda(instances=[inst()]), out=out) == 0
    assert out.flushed and out.flushed[-1] == len(out.getvalue())


# --- step stamps and durations --------------------------------------------------------------------

def test_apply_stamps_each_step_with_its_duration():
    clock = Clock()
    # the instance boots for two polls (20 s on the fake clock) before it is active with an IP
    fake = FakeLambda(types=TYPES, instance_progress={"i-1": [booting(), booting(), active("129.80.30.30")]})
    out = io.StringIO()
    assert make_rec(fake, FakeInfra(), out=out, clock=clock).apply(SPEC, auto_approve=True, keep_on_failure=False) == 0
    text = lines(out)
    for step in ("launch k3s-gpu-a6000-1", "wait for k3s-gpu-a6000-1 to be active with an IP",
                 "terraform apply gpu-access (1 GPU IP)", "sync-peers", "join k3s-gpu-a6000-1"):
        assert any(line.startswith("[+") and line.endswith(f"{step} ...") for line in text), (step, text)
        assert any(f"{step}: done in " in line for line in text), (step, text)
    assert "[+00:20] wait for k3s-gpu-a6000-1 to be active with an IP: done in 20s" in text
    assert "[+00:20] GPU nodes match the spec (apply took 20s)" in text


def test_down_stamps_each_delete():
    out = io.StringIO()
    assert make_rec(FakeLambda(types=TYPES, instances=[inst()]), FakeInfra(), out=out).down(SPEC, auto_approve=True) == 0
    assert "[+00:00] delete k3s-gpu-a6000-1 ..." in lines(out)
    assert "[+00:00] delete k3s-gpu-a6000-1: done in 0s" in lines(out)


def test_a_failed_step_says_how_long_it_ran():
    out = io.StringIO()
    infra = FakeInfra(join_fail={"k3s-gpu-a6000-1"})
    assert make_rec(FakeLambda(types=TYPES), infra, out=out).apply(SPEC, auto_approve=True, keep_on_failure=True) == 1
    assert "[+00:00] join k3s-gpu-a6000-1: failed after 0s" in lines(out)


@pytest.mark.parametrize("seconds, text", [(0, "0s"), (59.6, "59s"), (60, "1m00s"), (321, "5m21s"), (3725, "62m05s")])
def test_duration_format(seconds, text):
    from gpuctl.reconcile import duration
    assert duration(seconds) == text


# --- join-node.sh's output is shown as it runs ----------------------------------------------------

def test_join_output_is_shown_line_by_line_under_the_join_step():
    out = io.StringIO()
    assert make_rec(FakeLambda(types=TYPES), FakeInfra(), out=out).apply(SPEC, auto_approve=True,
                                                                          keep_on_failure=False) == 0
    text = lines(out)
    start = text.index("[+00:00] join k3s-gpu-a6000-1 ...")
    assert text[start + 1] == "    | k3s-gpu-a6000-1 joined and Ready"  # FakeInfra's join-node.sh output


def test_infra_join_node_streams_the_script_output():
    sh = FakeShell([ok(stdout="==> joining k3s-gpu-a6000-1\nk3s-gpu-a6000-1 joined and Ready\n")])
    seen = []
    Infra(sh, ROOT, {}).join_node("k3s-gpu-a6000-1", "129.80.30.30", "a6000", progress=seen.append)
    assert seen == ["==> joining k3s-gpu-a6000-1", "k3s-gpu-a6000-1 joined and Ready"]
    assert sh.calls[0].streamed


def test_infra_join_node_failure_keeps_the_last_lines_in_the_error():
    sh = FakeShell([fail(1, stdout="==> joining\n", stderr="error: node k3s-gpu-a6000-1 not Ready after 300s\n")])
    with pytest.raises(CommandError, match="not Ready after 300s"):
        Infra(sh, ROOT, {}).join_node("k3s-gpu-a6000-1", "129.80.30.30", "a6000", progress=lambda line: None)


def test_subprocess_shell_streams_stdout_and_stderr_in_order_as_they_come():
    code = ("import sys, time\nprint('one', flush=True)\nprint('two', file=sys.stderr, flush=True)\n"
            "time.sleep(0.2)\nprint('three', flush=True)\n")
    seen = []
    result = SubprocessShell().run([sys.executable, "-c", code], on_line=seen.append)
    assert seen == ["one", "two", "three"] and result.returncode == 0


def test_subprocess_shell_stream_failure_raises_with_the_tail():
    code = "import sys\nfor i in range(100): print(f'line {i}')\nsys.exit(3)\n"
    seen = []
    with pytest.raises(CommandError) as e:
        SubprocessShell().run([sys.executable, "-c", code], on_line=seen.append)
    assert e.value.returncode == 3 and len(seen) == 100
    assert "line 99" in e.value.stderr and "line 0\n" not in e.value.stderr  # the last lines only


def test_subprocess_shell_stream_unchecked_returns_the_code():
    assert SubprocessShell().run([sys.executable, "-c", "raise SystemExit(4)"], on_line=lambda line: None,
                                 check=False).returncode == 4


def test_subprocess_shell_stream_refuses_input_and_timeout():
    with pytest.raises(ValueError):
        SubprocessShell().run(["true"], on_line=print, input="x")
    with pytest.raises(ValueError):
        SubprocessShell().run(["true"], on_line=print, timeout=5)


def test_subprocess_shell_stream_stops_the_child_when_the_reader_fails(tmp_path):
    """If showing a line fails (a closed pipe: `make gpu-apply | head`), the child is stopped, not left running."""
    pidfile = tmp_path / "pid"
    code = (f"import os, time\nopen({str(pidfile)!r}, 'w').write(str(os.getpid()))\n"
            "print('first', flush=True)\ntime.sleep(60)\n")

    def broken(line):
        raise BrokenPipeError

    with pytest.raises(BrokenPipeError):
        SubprocessShell().run([sys.executable, "-c", code], on_line=broken)
    pid = int(pidfile.read_text())
    assert subprocess.run(["kill", "-0", str(pid)], capture_output=True).returncode != 0
