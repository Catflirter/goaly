"""Task 25 (Steps 1-4, Ruling R63): scripts/demo.sh (run, replay, reset, list), scripts/preflight.sh and their
Makefile targets. Offline only.

The brief's four tests run demo.sh from the repo itself (list and replay read no .env and reach nothing). Every
other test runs a throwaway repo copy (test_ops_scripts.Env) whose terraform, gcloud, kubectl, helm, ssh, scp and
scripts/gpuctl are tests/scripts/fake_tools.py, whose scripts/cpu-pool.sh and scripts/down.sh are stubs that only
log their arguments, and whose .env sets OPERATOR_IP (so no checkip lookup). The chapters' check functions are
replaced by bash stubs after `source scripts/demo.sh --source-only`: the checks themselves are tested in
test_checks_*.py; here it is the demo's own logic (which checks each chapter runs, in a subshell, through redact
and tee, the pauses, the recordings and timings) that is tested.

Fake secrets are assembled from pieces so this file itself never matches the gitleaks rules.
"""
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from test_ops_scripts import Env, make_kubeconfig

ROOT = Path(__file__).resolve().parents[2]
K3S_TOKEN = "K10" + "e" * 64 + "::server:" + "fedcba9876543210"
OPERATOR_IP = "203.0.113.7"
ADC_TOKEN = "ya29." + "AdcTokenMustNeverBePrinted" + "0123456789"
CHECKS = {1: ["check_a1_nodes", "check_a2_describe_nodes", "check_a3_system_pods"],
          2: ["check_b6_firewall_inventory", "check_b5_negative", "check_b2_wire"],
          3: ["check_b1_matrix"],
          4: ["check_c2_cross_node_curl", "check_c3_port_forward", "check_c4_qos", "check_c5_cgroup",
              "check_c7_limits_demo", "check_c8_netpol"],
          5: ["check_d2_gpu_capacity", "check_d4_gpu_smoke", "check_d5_vectoradd", "check_d6_overflow"]}
ALL_CHECKS = [f for fs in CHECKS.values() for f in fs]
RT = "--request-timeout=60s"


# --- the brief's tests (Step 1) ------------------------------------------------------------------

def demo(*args, env=None):
    return subprocess.run(["bash", str(ROOT / "scripts/demo.sh"), *args], capture_output=True, text=True,
                          cwd=ROOT, env={**os.environ, **(env or {})})


def parse(arg):
    return subprocess.run(["bash", "-c", f'source {ROOT}/scripts/demo.sh --source-only; parse_chapters "{arg}"'],
                          capture_output=True, text=True)


def test_parse_chapters():
    assert parse("1,3,5").stdout.split() == ["1", "3", "5"]
    default = ["1", "2", "3", "4", "5", "6", "7", "8"] if (ROOT / "k8s/scale-demo").exists() else ["1", "2", "3", "4", "5", "6", "8"]
    assert parse("").stdout.split() == default     # 第 7 章只有装了 stretch 才有；这样做完 stretch 后这个测试也不会坏
    r = parse("1,9"); assert r.returncode == 2 and "unknown chapter 9" in r.stderr


def test_list_marks_stretch_chapter():
    assert "7" in demo("list").stdout and "stretch" in demo("list").stdout


def test_replay_prints_recording(tmp_path):
    (tmp_path / "demo").mkdir(); (tmp_path / "demo/ch3.txt").write_text("matrix ok\n")
    assert demo("replay", "3", env={"VERIFICATION_FINAL": str(tmp_path)}).stdout == "matrix ok\n"


def test_replay_missing_recording(tmp_path):
    r = demo("replay", "4", env={"VERIFICATION_FINAL": str(tmp_path)})
    assert r.returncode == 1 and "no recording for chapter 4" in r.stderr


# --- parse_chapters, list, replay ----------------------------------------------------------------

def test_parse_chapters_keeps_the_order_given_and_drops_repeats():
    r = parse("5, 1 3,5")
    assert r.returncode == 0 and r.stdout == "5 1 3\n", r.stderr


def test_parse_chapters_takes_chapter_7_when_asked_for_it():
    assert parse("7").stdout.split() == ["7"]


@pytest.mark.parametrize("arg, bad", [("0", "0"), ("10", "10"), ("x", "x"), ("1,,2;", "2;"), ("01", "01")])
def test_parse_chapters_refuses_anything_but_1_to_8(arg, bad):
    r = parse(arg)
    assert r.returncode == 2 and f"unknown chapter {bad}" in r.stderr and r.stdout == ""


def test_list_shows_every_chapter_with_its_estimate():
    r = demo("list")
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    for n in range(1, 9):
        assert any(re.match(rf"\s*{n}\s", line) for line in lines), r.stdout
    assert "min" in r.stdout
    seven = next(line for line in lines if re.match(r"\s*7\s", line))
    assert "stretch" in seven


def test_replay_redacts_what_it_prints(tmp_path):
    (tmp_path / "demo").mkdir()
    (tmp_path / "demo/ch2.txt").write_text(f"join with {K3S_TOKEN}\n")
    r = demo("replay", "2", env={"VERIFICATION_FINAL": str(tmp_path)})
    assert r.returncode == 0 and r.stdout == "join with <redacted>\n", r.stderr


@pytest.mark.parametrize("args", [["replay"], ["replay", "9"], ["replay", "x"], ["replay", "1", "2"]])
def test_replay_refuses_a_bad_chapter(tmp_path, args):
    r = demo(*args, env={"VERIFICATION_FINAL": str(tmp_path)})
    assert r.returncode == 2, r.stderr


def test_replay_reads_verification_final_under_the_repo_by_default(tmp_path):
    repo = make_mini_repo(tmp_path)
    (repo / "verification/final/demo").mkdir(parents=True)
    (repo / "verification/final/demo/ch1.txt").write_text("nodes\n")
    r = subprocess.run(["bash", str(repo / "scripts/demo.sh"), "replay", "1"], capture_output=True, text=True,
                       cwd=tmp_path, env={k: v for k, v in os.environ.items() if k != "VERIFICATION_FINAL"})
    assert r.returncode == 0 and r.stdout == "nodes\n", r.stderr


def test_a_relative_verification_final_is_read_from_the_repo(tmp_path):
    repo = make_mini_repo(tmp_path)
    (repo / "elsewhere/demo").mkdir(parents=True)
    (repo / "elsewhere/demo/ch8.txt").write_text("bye\n")
    r = subprocess.run(["bash", str(repo / "scripts/demo.sh"), "replay", "8"], capture_output=True, text=True,
                       cwd=tmp_path, env={**os.environ, "VERIFICATION_FINAL": "elsewhere"})
    assert r.returncode == 0 and r.stdout == "bye\n", r.stderr


@pytest.mark.parametrize("args", [[], ["bogus"], ["list", "x"], ["reset", "x"], ["run", "x"]])
def test_usage_errors(args):
    r = demo(*args)
    assert r.returncode == 2 and "usage" in r.stderr, (r.stdout, r.stderr)


def test_the_scripts_are_executable_bash():
    for name in ("demo.sh", "preflight.sh"):
        path = ROOT / "scripts" / name
        assert os.access(path, os.X_OK), name
        assert path.read_text().startswith("#!/usr/bin/env bash\n"), name


def make_mini_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "scripts/lib").mkdir(parents=True)
    for rel in ("scripts/demo.sh", "scripts/lib/checks.sh", "scripts/lib/common.sh"):
        shutil.copy2(ROOT / rel, repo / rel)
    return repo


# --- the demo run, in a repo copy with fakes -----------------------------------------------------

CPU_POOL_STUB = r"""#!/usr/bin/env bash
# Test stub of scripts/cpu-pool.sh: logs "AUTO_APPROVE|args", renders N workers, "scales".
printf '%s|%s\n' "${AUTO_APPROVE:-}" "$*" >> "$CPU_POOL_LOG"
case "$1" in
  render)
    printf '{"cpu_workers": {'
    for ((i = 1; i <= $2; i++)); do
      ((i == 1)) || printf ', '
      printf '"k3s-cpu-%d": {"zone": "us-central1-a", "machine_type": "e2-small"}' "$i"
    done
    printf '}}\n' ;;
  scale)
    echo "cpu-pool stub: scale $2 $3 (token: scale-secret-$2)"
    [[ "$2" != 2 ]] || sleep "${CPU_POOL_SLEEP:-0}"
    echo "cpu-pool stub: scale $2 done"
    exit "${CPU_POOL_RC:-0}" ;;
esac
"""

DOWN_STUB = """#!/usr/bin/env bash
printf 'down.sh %s\\n' "$*" >> "$CPU_POOL_LOG"
read -r answer || true
printf 'down.sh read: %s\\n' "$answer" >> "$CPU_POOL_LOG"
echo "down.sh stub ran"
"""


class DemoEnv(Env):
    def __init__(self, tmp):
        super().__init__(tmp)
        for rel in ("demo.sh", "preflight.sh", "lib/checks.sh"):
            shutil.copy2(ROOT / "scripts" / rel, self.repo / "scripts" / rel)
        shutil.copy2(ROOT / "Makefile", self.repo / "Makefile")
        shutil.copy2(ROOT / ".gitleaks.toml", self.repo / ".gitleaks.toml")
        (self.repo / "lambda").mkdir()
        shutil.copy2(ROOT / "lambda/gpu-nodes.yaml", self.repo / "lambda/gpu-nodes.yaml")
        (self.repo / "scripts/cpu-pool.sh").write_text(CPU_POOL_STUB)
        (self.repo / "scripts/cpu-pool.sh").chmod(0o755)
        (self.repo / "scripts/down.sh").write_text(DOWN_STUB)
        (self.repo / "scripts/down.sh").chmod(0o755)
        (self.repo / ".venv/bin").mkdir(parents=True)
        (self.repo / ".venv/bin/python").symlink_to(sys.executable)
        self.pool_log = tmp / "cpu-pool.log"
        # No network: a curl that is called at all fails (OPERATOR_IP in .env means none should be).
        (self.bin / "curl").write_text('#!/bin/sh\n[ "$1" = --version ] && { echo "curl 8.0.0 (stub)"; exit 0; }\n'
                                       'echo "curl stub: $*" >&2\nexit 7\n')
        (self.bin / "curl").chmod(0o755)
        self.write_env(f"GCP_PROJECT=k3s-demo-123456\nLAMBDA_API_KEY=x\nUNDERLAY_MTU=1460\n"
                       f"SSH_PRIVATE_KEY_PATH={self.ssh_key}\nOPERATOR_IP={OPERATOR_IP}\n")
        make_kubeconfig(self)

    def environ(self, extra):
        env = super().environ(extra)
        env.setdefault("CPU_POOL_LOG", str(self.pool_log))
        env.setdefault("DEMO_POLL_S", "0.1")
        for k in ("CHAPTERS", "DEMO_NO_PAUSE", "VERIFICATION_FINAL"):
            if k not in extra:
                env.pop(k, None)
        return env

    def demo_fn(self, code, stdin="", **extra):
        """Source demo.sh --source-only, define CODE (stubs), then run demo_main ARGS from the extra 'args'."""
        args = extra.pop("args", "run")
        return self.run(["bash", "-c", f"source scripts/demo.sh --source-only\n{code}\ndemo_main {args}"],
                        stdin=stdin, **extra)

    def pool_calls(self):
        return self.pool_log.read_text().splitlines() if self.pool_log.exists() else []

    def demo_dirs(self):
        return sorted((self.repo / "verification").glob("demo-*"))

    def the_run(self):
        dirs = self.demo_dirs()
        assert len(dirs) == 1, dirs
        return dirs[0]


@pytest.fixture
def denv(tmp_path):
    e = DemoEnv(tmp_path)
    yield e
    if e.pidfile.exists():
        subprocess.run(["kill", "-KILL", "--", f"-{e.pidfile.read_text().strip()}"], capture_output=True)


def stubs(**rcs):
    """Bash stubs for every chapter check: each logs its name to $STUB_LOG, prints 'ran NAME', returns rcs[NAME]."""
    lines = []
    for f in ALL_CHECKS:
        lines.append(f'{f}() {{ echo "{f}" >> "$STUB_LOG"; echo "ran {f}"; return {rcs.get(f, 0)}; }}')
    return "\n".join(lines)


def stub_log(denv):
    path = denv.repo / "stub.log"
    return path.read_text().split() if path.exists() else []


def run_demo(denv, chapters, code="", stdin="", **extra):
    extra.setdefault("DEMO_NO_PAUSE", "1")
    return denv.demo_fn(stubs() + "\n" + code, stdin=stdin, CHAPTERS=chapters,
                        STUB_LOG=str(denv.repo / "stub.log"), **extra)


@pytest.mark.parametrize("chapter", [1, 2, 3, 4, 5])
def test_each_chapter_runs_the_checks_the_brief_names_in_order(denv, chapter):
    r = run_demo(denv, str(chapter))
    assert r.returncode == 0, r.stderr
    assert stub_log(denv) == CHECKS[chapter]


def test_the_default_chapters_run_1_to_6_and_8(denv):
    gpuctl_ok(denv)
    terraform_plans(denv)
    denv.scenario["ready"] = ["k3s-cpu-2"]
    r = run_demo(denv, "")
    assert r.returncode == 0, r.stderr
    assert stub_log(denv) == ALL_CHECKS
    run = denv.the_run()
    assert sorted(p.name for p in run.iterdir()) == [f"ch{n}.txt" for n in (1, 2, 3, 4, 5, 6, 8)] + ["timings.txt"]


def test_each_chapter_is_recorded_with_its_timing(denv):
    r = run_demo(denv, "1,3")
    assert r.returncode == 0, r.stderr
    run = denv.the_run()
    assert re.fullmatch(r"demo-\d{4}-\d{2}-\d{2}T\d{4}[+-]\d{4}", run.name)
    assert sorted(p.name for p in run.iterdir()) == ["ch1.txt", "ch3.txt", "timings.txt"]
    ch1, ch3 = (run / "ch1.txt").read_text(), (run / "ch3.txt").read_text()
    # (A2's output is shown as a summary, which drops the stub's line: see the A2 tests.)
    assert "ran check_a1_nodes" in ch1 and "ran check_a3_system_pods" in ch1 and "ran check_b1_matrix" not in ch1
    assert "ran check_b1_matrix" in ch3 and "check_a1_nodes" not in ch3
    timings = (run / "timings.txt").read_text().splitlines()
    assert [t.split()[0] for t in timings if t.startswith("ch")] == ["ch1", "ch3"]
    assert all(re.match(r"ch\d \d+s ", t) for t in timings if t.startswith("ch")), timings
    assert any(t.startswith("total") for t in timings), timings
    # Unreviewed evidence stays private (as verify.sh's runs are).
    assert (run.stat().st_mode & 0o777) == 0o700
    assert all((p.stat().st_mode & 0o777) == 0o600 for p in run.iterdir())


def test_what_is_on_screen_is_what_is_recorded(denv):
    r = run_demo(denv, "3")
    assert r.returncode == 0, r.stderr
    ch3 = (denv.the_run() / "ch3.txt").read_text()
    assert ch3 and ch3 in r.stdout


def test_each_step_prints_its_title_and_command_before_it_runs(denv):
    r = run_demo(denv, "3")
    out = r.stdout
    title = out.index("B1")
    command = out.index("$ check_b1_matrix")
    ran = out.index("ran check_b1_matrix")
    assert title < command < ran, out


def test_the_screen_and_the_recording_are_redacted(denv):
    code = (f'check_a1_nodes() {{ echo "token: {K3S_TOKEN}"; echo "from {OPERATOR_IP}/32" >&2; }}\n')
    r = run_demo(denv, "1", code=code)
    assert r.returncode == 0, r.stderr
    rec = (denv.the_run() / "ch1.txt").read_text()
    for text in (r.stdout, r.stderr, rec):
        assert K3S_TOKEN not in text and OPERATOR_IP not in text
    assert "token: <redacted>" in r.stdout and "token: <redacted>" in rec
    assert "from <operator_ip>/32" in r.stdout and "from <operator_ip>/32" in rec


def test_a_failed_step_names_the_replay_and_the_demo_goes_on(denv):
    r = run_demo(denv, "3,1", code=stubs(check_b1_matrix=1))
    assert r.returncode == 1, r.stderr
    assert "make demo-replay CHAPTER=3" in r.stdout
    assert stub_log(denv) == CHECKS[3] + CHECKS[1], "chapter 1 still ran"
    assert "make demo-replay CHAPTER=3" in (denv.the_run() / "ch3.txt").read_text()
    assert "\n  ch3: B1 " in r.stdout, "the failed step is listed at the end"


def test_ctrl_c_runs_the_check_s_cleanup_and_stops_the_demo(denv):
    """Task 21a note: a check's _vr_on_signal cleanup runs on Ctrl-C (in the step's subshell), its messages
    reach the operator, and the demo then stops by SIGINT with the demo-reset hint."""
    started, cleaned = denv.repo / "started", denv.repo / "cleaned"
    code = (f'check_a1_nodes() {{ _vr_on_signal --left "the test Pods" bash -c "echo yes > {cleaned}"; '
            f': > {started}; sleep 30; }}\n')
    proc = subprocess.Popen(["bash", "-c", f"source scripts/demo.sh --source-only\n{stubs()}\n{code}\ndemo_main run"],
                            cwd=denv.repo, env=denv.environ({"CHAPTERS": "1,3", "DEMO_NO_PAUSE": "1",
                                                             "STUB_LOG": str(denv.repo / "stub.log")}),
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            start_new_session=True)
    try:
        deadline = time.monotonic() + 30
        while not started.exists():
            assert proc.poll() is None and time.monotonic() < deadline, proc.communicate()
            time.sleep(0.05)
        os.killpg(proc.pid, signal.SIGINT)                 # Ctrl-C reaches the whole foreground group
        out, err = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
    assert proc.returncode in (-signal.SIGINT, 130), (proc.returncode, err)
    assert cleaned.read_text() == "yes\n"
    assert "cleanup done" in err and "make demo-reset" in err
    assert stub_log(denv) == [], "chapter 3 did not run"
    assert "ch1 interrupted after" in (denv.the_run() / "timings.txt").read_text()


def test_a_skipped_check_is_not_a_failure(denv):
    r = run_demo(denv, "5", code=stubs(check_d2_gpu_capacity=77, check_d4_gpu_smoke=77,
                                       check_d5_vectoradd=77, check_d6_overflow=77))
    assert r.returncode == 0, r.stderr
    assert "SKIP" in r.stdout and "make demo-replay CHAPTER=5" in r.stdout


def test_checks_run_in_a_subshell(denv):
    # _vr_on_signal sets INT/TERM traps and a check may set variables: none of it reaches the demo's shell.
    code = ('check_a1_nodes() { DEMO_LEAK=1; _vr_on_signal true; echo "a1"; }\n'
            'check_a3_system_pods() { echo "leak=${DEMO_LEAK:-none}"; }\n')
    r = run_demo(denv, "1", code=code)
    assert r.returncode == 0, r.stderr
    assert "leak=none" in r.stdout


def test_a_check_s_cleanup_messages_reach_the_operator_and_the_recording(denv):
    code = f'check_a1_nodes() {{ _vr_tell "cleanup done for {OPERATOR_IP}"; }}\n'
    r = run_demo(denv, "1", code=code)
    assert r.returncode == 0, r.stderr
    assert "cleanup done for <operator_ip>" in r.stderr
    assert "cleanup done for <operator_ip>" in (denv.the_run() / "ch1.txt").read_text()


def test_checks_get_no_stdin(denv):
    code = 'check_b1_matrix() { if read -r line; then echo "read: $line"; else echo "no stdin"; fi; }\n'
    r = run_demo(denv, "3", code=code, stdin="secret line\n\n\n", DEMO_NO_PAUSE="")
    assert "no stdin" in r.stdout and "read:" not in r.stdout


def test_it_pauses_after_every_step_unless_demo_no_pause(denv):
    r = run_demo(denv, "1", stdin="\n" * 10, DEMO_NO_PAUSE="")
    assert r.returncode == 0, r.stderr
    assert r.stderr.count("[Enter]") == 3
    (denv.repo / "stub.log").unlink()
    shutil.rmtree(denv.the_run())
    r = run_demo(denv, "1")
    assert r.returncode == 0 and "[Enter]" not in r.stderr


def test_the_a2_step_shows_a_summary_of_describe_node(denv):
    describe = ("$ kc describe node k3s-cp --request-timeout=60s\\nName:               k3s-cp\\n"
                "Roles:              control-plane\\nLabels:             beta.kubernetes.io/arch=amd64\\n"
                "Annotations:        flannel.alpha.coreos.com/backend-data: {}\\n"
                "Taints:             node-role.kubernetes.io/control-plane:NoSchedule\\n"
                "Addresses:\\n  InternalIP:  10.10.0.2\\n  ExternalIP:  34.82.10.10\\n"
                "Capacity:\\n  cpu:                2\\n  memory:             4005040Ki\\n  pods: 110\\n"
                "System Info:\\n  Machine ID:  abcdef\\n  Kernel Version:  6.8.0\\n"
                "  Container Runtime Version:  containerd://2.1\\nEvents:  <none>\\n")
    code = f'check_a2_describe_nodes() {{ printf "{describe}"; return 0; }}\n'
    r = run_demo(denv, "1", code=code)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    for kept in ("$ kc describe node k3s-cp", "Name:               k3s-cp", "Roles:", "Taints:", "ExternalIP:",
                 "  cpu:", "  memory:", "Kernel Version:", "Container Runtime Version:"):
        assert kept in out, kept
    for dropped in ("Labels:", "Annotations:", "Machine ID", "Events:"):
        assert dropped not in out, dropped


def test_a_failed_a2_still_fails_through_the_summary(denv):
    code = 'check_a2_describe_nodes() { echo "A2: describe node x failed"; return 1; }\n'
    r = run_demo(denv, "1", code=code)
    assert r.returncode == 1 and "make demo-replay CHAPTER=1" in r.stdout


def test_chapter_7_without_the_stretch_plan_says_so(denv):
    r = run_demo(denv, "7")
    assert r.returncode == 0, r.stderr
    assert "requires the stretch plan" in r.stdout
    assert "requires the stretch plan" in (denv.the_run() / "ch7.txt").read_text()


def test_the_default_chapters_include_7_once_scale_demo_exists(denv):
    (denv.repo / "k8s/scale-demo").mkdir()
    r = denv.run(["bash", "-c", 'source scripts/demo.sh --source-only; parse_chapters ""'])
    assert r.stdout.split() == ["1", "2", "3", "4", "5", "6", "7", "8"], r.stderr


def test_an_unknown_chapter_is_refused_before_anything_runs(denv):
    r = run_demo(denv, "1,9")
    assert r.returncode == 2 and "unknown chapter 9" in r.stderr
    assert denv.demo_dirs() == [] and denv.calls() == [] and stub_log(denv) == []


def test_a_second_run_in_the_same_minute_is_refused(denv):
    run_dir = denv.repo / "verification/demo-2026-10-22T1000-0700"
    run_dir.mkdir(parents=True)
    (run_dir / "ch1.txt").write_text("earlier\n")
    r = run_demo(denv, "1", DEMO_TS="2026-10-22T1000-0700")
    assert r.returncode == 2 and "already holds" in r.stderr
    assert (run_dir / "ch1.txt").read_text() == "earlier\n" and stub_log(denv) == []


def test_the_demo_scans_its_recordings_with_gitleaks(denv):
    if shutil.which("gitleaks") is None:
        pytest.skip("gitleaks is not installed")
    r = run_demo(denv, "3")
    assert r.returncode == 0, r.stderr
    assert "gitleaks found no leaks" in r.stdout + r.stderr


# --- chapter 6 (Ruling R63) and chapter 8 --------------------------------------------------------

def gpuctl_ok(denv):
    denv.scenario.setdefault("answers", []).append(
        {"tool": "gpuctl", "has": ["plan", "--set-count"], "stdout": "gpuctl plan: preview\n"})


def terraform_plans(denv, destroy_lines=3):
    denv.scenario.setdefault("answers", []).extend([
        {"tool": "terraform", "has": ["plan", "-destroy"],
         "stdout": "".join(f"destroy line {i}\n" for i in range(1, destroy_lines + 1))},
        {"tool": "terraform", "has": ["plan", "-var-file="],
         "stdout": "  # google_compute_instance.worker[\"k3s-cpu-2\"] will be created\nPlan: 4 to add, 0 to change, 0 to destroy.\n"},
    ])


def tf_calls(denv):
    return [c["argv"] for c in denv.calls("terraform")]


def test_chapter_6_plans_scales_up_in_the_background_back_down_and_previews_gpus(denv):
    gpuctl_ok(denv)
    terraform_plans(denv)
    denv.scenario["ready"] = ["k3s-cpu-2"]
    r = run_demo(denv, "6")
    assert r.returncode == 0, (r.stdout, r.stderr)
    # The plan of the inventory scaled to 2, before anything is scaled.
    plans = [a for a in tf_calls(denv) if a[1] == "plan"]
    assert len(plans) == 1
    assert plans[0][0] == f"-chdir={denv.repo.resolve()}/infra/workers" or plans[0][0] == "-chdir=infra/workers"
    var_file = next(a for a in plans[0] if a.startswith("-var-file="))
    assert "Plan: 4 to add" in r.stdout
    # The guarded scale to 2 (background), then back to 1, both under AUTO_APPROVE=1 (R63).
    calls = denv.pool_calls()
    assert [c for c in calls if "scale" in c] == ["1|scale 2 --guarded", "1|scale 1 --guarded"]
    assert set(calls[:calls.index("1|scale 2 --guarded")]) == {"|render 2"}, "rendered (read-only) first"
    assert not Path(var_file.split("=", 1)[1]).exists(), "the rendered inventory is removed"
    # The new node was watched with kubectl get nodes -w, and checked Ready.
    kubectl = [c["argv"][2:] for c in denv.calls("kubectl")]
    assert ["get", "nodes", "-w"] in kubectl
    assert any(a[:3] == ["get", "node", "k3s-cpu-2"] for a in kubectl)
    # gpu-plan with the inventory's pool set to 2.
    assert ["plan", "--set-count", "a6000=2"] in [c["argv"] for c in denv.calls("gpuctl")]
    ch6 = (denv.the_run() / "ch6.txt").read_text()
    # The background scale's log goes to ch6.txt (redacted), not to the screen.
    assert "cpu-pool stub: scale 2 --guarded (token: <redacted>" in ch6   # redact takes the value up to whitespace
    assert "cpu-pool stub: scale 2" not in r.stdout
    assert "scale-secret" not in ch6 + r.stdout + r.stderr
    # The foreground scale back to 1 is on the screen and in the recording.
    assert "cpu-pool stub: scale 1 --guarded" in r.stdout and "cpu-pool stub: scale 1 --guarded" in ch6
    assert not (denv.repo / ".run/demo-scale.log").exists() and not (denv.repo / ".run/demo-scale.rc").exists()


def test_chapter_6_reads_the_gpu_pool_from_the_inventory(denv):
    spec = denv.repo / "lambda/gpu-nodes.yaml"
    spec.write_text(spec.read_text().replace("  a6000:", "  rtx6000:"))
    gpuctl_ok(denv)
    terraform_plans(denv)
    denv.scenario["ready"] = ["k3s-cpu-2"]
    r = run_demo(denv, "6")
    assert r.returncode == 0, r.stderr
    assert ["plan", "--set-count", "rtx6000=2"] in [c["argv"] for c in denv.calls("gpuctl")]


def test_chapter_6_fails_the_step_when_the_scale_fails_and_still_scales_back(denv):
    gpuctl_ok(denv)
    terraform_plans(denv)
    r = run_demo(denv, "6", CPU_POOL_RC="1")
    assert r.returncode == 1
    assert "make demo-replay CHAPTER=6" in r.stdout
    assert denv.pool_calls()[-1] == "1|scale 1 --guarded"


def test_chapter_6_gives_up_waiting_but_leaves_the_scale_running(denv):
    gpuctl_ok(denv)
    terraform_plans(denv)
    r = run_demo(denv, "6", DEMO_SCALE_TIMEOUT_S="0", CPU_POOL_SLEEP="3")
    assert r.returncode == 1
    assert "has not finished" in r.stdout and "make demo-reset" in r.stdout
    # The background scale was not killed with the step: it finishes on its own and records its exit status.
    rc_file, log_file = denv.repo / ".run/demo-scale.rc", denv.repo / ".run/demo-scale.log"
    deadline = time.monotonic() + 30
    while not rc_file.exists():
        assert time.monotonic() < deadline, "the background scale did not finish"
        time.sleep(0.1)
    assert rc_file.read_text() == "0\n" and "cpu-pool stub: scale 2 done" in log_file.read_text()
    assert (log_file.stat().st_mode & 0o777) == 0o600


def test_chapter_6_fails_when_the_new_node_is_not_ready(denv):
    gpuctl_ok(denv)
    terraform_plans(denv)
    denv.scenario["registered"] = ["k3s-cpu-2"]          # NotReady
    r = run_demo(denv, "6")
    assert r.returncode == 1 and "k3s-cpu-2" in r.stdout


def test_chapter_8_previews_the_teardown_and_runs_make_down_only_on_yes(denv):
    gpuctl_ok(denv)
    terraform_plans(denv, destroy_lines=40)
    r = run_demo(denv, "8", stdin="\n\nn\n", DEMO_NO_PAUSE="")
    assert r.returncode == 0, r.stderr
    assert ["plan", "--set-count", "a6000=0"] in [c["argv"] for c in denv.calls("gpuctl")]
    destroy = [a for a in tf_calls(denv) if a[1] == "plan"]
    assert len(destroy) == 1 and "-destroy" in destroy[0]
    # Only the last lines of the destroy plan are shown.
    assert "destroy line 40" in r.stdout and "destroy line 26" in r.stdout and "destroy line 25\n" not in r.stdout
    assert "make down" in r.stderr
    assert not any(c.startswith("down.sh") for c in denv.pool_calls()), "declined: make down did not run"


def test_chapter_8_runs_make_down_on_yes_with_the_terminal_as_its_stdin(denv):
    gpuctl_ok(denv)
    terraform_plans(denv)
    r = run_demo(denv, "8", stdin="\n\ny\nk3s-demo-123456\n", DEMO_NO_PAUSE="")
    assert r.returncode == 0, r.stderr
    assert "down.sh " in denv.pool_calls() and "down.sh read: k3s-demo-123456" in denv.pool_calls()
    assert "down.sh stub ran" in r.stdout


@pytest.mark.parametrize("stdin, extra", [("", {"DEMO_NO_PAUSE": ""}), ("y\n", {"DEMO_NO_PAUSE": "1"})])
def test_chapter_8_never_runs_make_down_unasked(denv, stdin, extra):
    gpuctl_ok(denv)
    terraform_plans(denv)
    r = run_demo(denv, "8", stdin=stdin, **extra)
    assert r.returncode == 0, r.stderr
    assert not any(c.startswith("down.sh") for c in denv.pool_calls())


# --- reset -----------------------------------------------------------------------------------------

def kc_calls(denv):
    return [c["argv"][2:] for c in denv.calls("kubectl")]


RESET_DELETES = [
    ["-n", "limits-demo", "delete", "pod", "oom", "cpu-burn", "--ignore-not-found", "--timeout=120s", RT],
    ["delete", "namespace", "np-denied", "--ignore-not-found", "--timeout=120s", RT],
    ["-n", "gpu-test", "delete", "pod", "gpu-smoke", "vectoradd", "gpu-overflow", "--ignore-not-found",
     "--timeout=120s", RT],
    ["-n", "gpu-test", "delete", "pod", "-l", "app=gpu-hold", "--ignore-not-found", "--timeout=120s", RT],
]


def test_reset_deletes_the_demo_objects_then_scales_the_pool_to_one(denv):
    r = denv.script("demo.sh", "reset")
    assert r.returncode == 0, r.stderr
    assert kc_calls(denv) == RESET_DELETES
    assert denv.pool_calls() == ["1|scale 1 --guarded"]
    assert "$ kc -n gpu-test delete pod -l app=gpu-hold" in r.stdout
    assert "scale-secret" not in r.stdout + r.stderr


def test_reset_deletes_the_stretch_load_job_when_scale_demo_exists(denv):
    (denv.repo / "k8s/scale-demo").mkdir()
    r = denv.script("demo.sh", "reset")
    assert r.returncode == 0, r.stderr
    assert kc_calls(denv) == RESET_DELETES + [
        ["-n", "scale-demo", "delete", "job", "load-generator", "--ignore-not-found", "--timeout=120s", RT]]


def test_reset_goes_on_after_a_failed_delete_and_fails_at_the_end(denv):
    denv.scenario["answers"] = [{"tool": "kubectl", "has": ["namespace np-denied"], "rc": 1,
                                 "stderr": "Unable to connect to the server\n"}]
    r = denv.script("demo.sh", "reset")
    assert r.returncode == 1
    assert kc_calls(denv) == RESET_DELETES
    assert denv.pool_calls() == ["1|scale 1 --guarded"]
    assert "np-denied" in r.stderr


def test_reset_fails_when_the_scale_fails(denv):
    r = denv.script("demo.sh", "reset", CPU_POOL_RC="2")
    assert r.returncode == 1


# --- preflight ---------------------------------------------------------------------------------

def preflight_ok(denv):
    denv.scenario["answers"] = [
        {"tool": "terraform", "has": ["version"], "stdout": json.dumps({"terraform_version": "1.14.3"})},
        {"tool": "gcloud", "has": ["version"], "stdout": "Google Cloud SDK 545.0.0\n"},
        {"tool": "gcloud", "has": ["auth", "list"], "stdout": "operator@example.com\n"},
        {"tool": "gcloud", "has": ["auth", "application-default", "print-access-token"], "stdout": ADC_TOKEN + "\n"},
        {"tool": "kubectl", "has": ["version", "--client"], "stdout": "Client Version: v1.36.0\n"},
        {"tool": "gpuctl", "has": ["status"], "stdout": "GPU capacity: a6000 us-west-1\n"},
        {"tool": "gpuctl", "has": ["sync-peers"], "stdout": "sync-peers: operator 203.0.113.x\n"},
        {"tool": "kubectl", "has": ["get", "nodes"], "not": ["/readyz"],
         "stdout": "k3s-cp True\nk3s-cpu-1 True\n"},
    ]


def preflight(denv, **extra):
    return denv.script("preflight.sh", **extra)


def test_preflight_passes_and_prints_ok_for_every_item(denv):
    preflight_ok(denv)
    assert denv.script("tunnel.sh", "start").returncode == 0
    r = preflight(denv)
    assert r.returncode == 0, (r.stdout, r.stderr)
    out = r.stdout
    assert "FAIL" not in out
    for item in ("terraform 1.14.3", "gcloud", "kubectl", "jq", "gitleaks", "python", "flock",
                 "gcloud auth", "application-default", "gpuctl status", "sync-peers", "tunnel", "nodes Ready"):
        assert re.search(rf"^OK +.*{re.escape(item)}", out, re.M), (item, out)
    gpuctl = [c["argv"] for c in denv.calls("gpuctl")]
    assert gpuctl == [["status"], ["sync-peers"]]
    assert ["auth", "list", "--filter=status:ACTIVE", "--format=value(account)"] in denv.argvs("gcloud")


def test_preflight_never_prints_the_adc_token(denv):
    preflight_ok(denv)
    r = preflight(denv)
    assert ADC_TOKEN not in r.stdout + r.stderr


def test_preflight_starts_the_tunnel_when_it_is_down(denv):
    preflight_ok(denv)
    r = preflight(denv)
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert re.search(r"^OK +tunnel.*started", r.stdout, re.M), r.stdout
    assert denv.pidfile.exists()


@pytest.mark.parametrize("rule, item", [
    ({"tool": "gcloud", "has": ["auth", "list"], "stdout": ""}, "gcloud auth"),
    ({"tool": "gcloud", "has": ["auth", "application-default"], "rc": 1, "stderr": "reauth needed\n"},
     "application-default"),
    ({"tool": "gpuctl", "has": ["status"], "rc": 1}, "gpuctl status"),
    ({"tool": "gpuctl", "has": ["sync-peers"], "rc": 1}, "sync-peers"),
    ({"tool": "kubectl", "has": ["get", "nodes"], "not": ["/readyz"], "stdout": "k3s-cp True\nk3s-cpu-1 False\n"},
     "nodes"),
    ({"tool": "kubectl", "has": ["get", "nodes"], "not": ["/readyz"], "stdout": ""}, "nodes"),
    ({"tool": "terraform", "has": ["version"], "stdout": json.dumps({"terraform_version": "1.8.5"})}, "terraform"),
])
def test_preflight_fails_one_item_and_still_checks_the_rest(denv, rule, item):
    preflight_ok(denv)
    denv.scenario["answers"].insert(0, rule)
    assert denv.script("tunnel.sh", "start").returncode == 0
    r = preflight(denv)
    assert r.returncode == 1, (r.stdout, r.stderr)
    fails = [line for line in r.stdout.splitlines() if line.startswith("FAIL")]
    assert len(fails) == 1 and item in fails[0], r.stdout
    assert re.search(r"^OK +.*nodes Ready", r.stdout, re.M) or item == "nodes", "later items still ran"


def test_preflight_takes_gpuctl_drift_as_ok(denv):
    preflight_ok(denv)
    denv.scenario["answers"].insert(0, {"tool": "gpuctl", "has": ["status"], "rc": 3,
                                        "stdout": "drift: pool a6000 has 0, wants 1\n"})
    assert denv.script("tunnel.sh", "start").returncode == 0
    r = preflight(denv)
    assert r.returncode == 0, r.stdout
    assert re.search(r"^OK +gpuctl status.*drift.*make gpu-apply", r.stdout, re.M), r.stdout


def test_preflight_fails_when_the_tunnel_cannot_start(denv):
    preflight_ok(denv)
    denv.scenario["tunnel"] = "exit"
    r = preflight(denv, TUNNEL_TIMEOUT_S="1")
    assert r.returncode == 1
    assert re.search(r"^FAIL +tunnel", r.stdout, re.M), r.stdout


def test_preflight_fails_on_a_missing_tool(denv, tmp_path):
    preflight_ok(denv)
    assert denv.script("tunnel.sh", "start").returncode == 0
    # A PATH without flock: only the fakes and links to what the script needs besides it.
    bare = tmp_path / "bare"
    bare.mkdir()
    for tool in ("bash", "sh", "env", "jq", "gitleaks", "sed", "grep", "cat", "date", "mkdir", "chmod", "rm",
                 "head", "tail", "tr", "sort", "awk", "stat", "dirname", "kill", "ps", "sleep", "mktemp", "timeout",
                 "cksum", "wc", "id", "setsid", "uname", "cut", "ls", "nohup"):
        path = shutil.which(tool)
        if path:
            (bare / tool).symlink_to(path)
    r = preflight(denv, PATH=f"{denv.bin}:{bare}")
    assert r.returncode == 1
    assert re.search(r"^FAIL +flock", r.stdout, re.M), r.stdout


# --- Makefile -----------------------------------------------------------------------------------

def make(denv, *args, **extra):
    return denv.run(["make", "--no-print-directory", "-f", str(denv.repo / "Makefile"), *args], **extra)


def test_make_help_lists_the_demo_targets():
    r = subprocess.run(["make", "--no-print-directory", "-f", str(ROOT / "Makefile"), "help"],
                       capture_output=True, text=True)
    for target in ("preflight", "demo", "demo-replay", "demo-reset", "promote-demo"):
        assert re.search(rf"^  {re.escape(target)} ", r.stdout, re.M), target


def test_make_demo_passes_chapters(denv):
    r = make(denv, "demo", "CHAPTERS=9")
    assert r.returncode == 2 and "unknown chapter 9" in r.stderr


def test_make_demo_runs_the_chosen_chapters(denv):
    r = make(denv, "demo", "CHAPTERS=7", DEMO_NO_PAUSE="1")
    assert r.returncode == 0, r.stderr
    assert "requires the stretch plan" in r.stdout


def test_make_demo_replay_requires_chapter(denv):
    r = make(denv, "demo-replay")
    assert r.returncode == 2 and "CHAPTER" in r.stderr


def test_make_demo_replay_prints_the_recording(denv):
    (denv.repo / "verification/final/demo").mkdir(parents=True)
    (denv.repo / "verification/final/demo/ch3.txt").write_text("matrix ok\n")
    r = make(denv, "demo-replay", "CHAPTER=3")
    assert r.returncode == 0 and r.stdout == "matrix ok\n", r.stderr


def test_make_demo_reset_runs_the_reset(denv):
    r = make(denv, "demo-reset")
    assert r.returncode == 0, r.stderr
    assert kc_calls(denv) == RESET_DELETES and denv.pool_calls() == ["1|scale 1 --guarded"]


def test_make_preflight_runs_the_preflight(denv):
    preflight_ok(denv)
    assert denv.script("tunnel.sh", "start").returncode == 0
    r = make(denv, "preflight")
    assert r.returncode == 0, r.stdout
    assert "OK" in r.stdout


def test_make_promote_demo_requires_run(denv):
    r = make(denv, "promote-demo")
    assert r.returncode == 2 and "RUN" in r.stderr
