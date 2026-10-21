"""scripts/lib/checks.sh (run_check, vr_cmd, redact, SUMMARY.md) and the scripts/verify.sh skeleton.

Fake secrets are assembled from pieces ("K10" + "f" * 64, ...) so that this file itself never
matches the repo's gitleaks rules; the strings the scripts see at run time are the full shapes.
"""
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LA_ENV = {**os.environ, "TZ": "America/Los_Angeles"}
HEX20 = "01234567" + "89abcdef" + "0123"


def bash(script):
    return subprocess.run(["bash", "-c", f"set -euo pipefail; source {ROOT}/scripts/lib/checks.sh; {script}"],
                          capture_output=True, text=True, cwd=ROOT, env=LA_ENV)


def redact(text):
    proc = subprocess.run(["bash", "-c", f"source {ROOT}/scripts/lib/checks.sh; redact"], input=text,
                          capture_output=True, text=True, check=True)
    return proc.stdout


# --- run_check / vr_finish ----------------------------------------------------------------------

def test_pass_fail_skip_recorded(tmp_path):
    r = bash(f'vr_init {tmp_path}; ok(){{ echo fine; }}; bad(){{ echo broken; return 1; }}; sk(){{ echo "no gpu"; return 77; }}; '
             'run_check A1 "ok check" ok; run_check B1 "bad check" bad; run_check D1 "gpu check" sk; vr_finish')
    s = (tmp_path / "SUMMARY.md").read_text()
    assert r.returncode == 1
    assert "| 1 | A1 | ok check | PASS |" in s and "| 2 | B1 | bad check | FAIL |" in s and "| 3 | D1 | gpu check | SKIP |" in s
    assert (tmp_path / "01-A1.txt").read_text().splitlines()[0] == "# A1 — ok check"
    assert "broken" in (tmp_path / "02-B1.txt").read_text()


def test_only_skips_is_success(tmp_path):                       # Review Focus #5
    assert bash(f"vr_init {tmp_path}; sk(){{ return 77; }}; run_check D1 gpu sk; vr_finish").returncode == 0


def test_summary_table_and_result_line(tmp_path):
    r = bash(f"vr_init {tmp_path}; ok(){{ :; }}; bad(){{ return 3; }}; sk(){{ return 77; }}; "
             "run_check A1 a ok; run_check A2 b ok; run_check B1 c bad; run_check D1 d sk; vr_finish")
    s = (tmp_path / "SUMMARY.md").read_text()
    assert r.returncode == 1
    assert "| # | ID | Check | Result |" in s
    assert "**Result:** 2 passed, 1 failed, 1 skipped" in s
    assert sorted(p.name for p in tmp_path.glob("0*.txt")) == ["01-A1.txt", "02-A2.txt", "03-B1.txt", "04-D1.txt"]


def test_all_pass_is_success_and_zero_checks_is_success(tmp_path):
    (tmp_path / "a").mkdir()
    assert bash(f"vr_init {tmp_path}/a; ok(){{ :; }}; run_check A1 a ok; vr_finish").returncode == 0
    r = bash(f"vr_init {tmp_path}/b; vr_finish")
    assert r.returncode == 0, r.stderr
    assert "**Result:** 0 passed, 0 failed, 0 skipped" in (tmp_path / "b/SUMMARY.md").read_text()


def test_check_file_header_has_started_timestamp_in_los_angeles(tmp_path):
    bash(f"vr_init {tmp_path}; ok(){{ :; }}; run_check A1 'ok check' ok")
    lines = (tmp_path / "01-A1.txt").read_text().splitlines()
    m = re.fullmatch(r"# started: \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}([+-]\d{2}:\d{2})", lines[1])
    assert m, lines[1]
    assert m.group(1) in ("-07:00", "-08:00")                  # PDT / PST


def test_run_check_passes_args_and_captures_stderr(tmp_path):
    bash(f'vr_init {tmp_path}; f(){{ echo "args=$*"; echo "to-stderr" >&2; }}; run_check A1 t f x "y z"')
    text = (tmp_path / "01-A1.txt").read_text()
    assert "args=x y z" in text and "to-stderr" in text


def test_run_check_survives_exit_and_errexit_in_the_check(tmp_path):
    # A check that calls exit, or fails under the caller's set -e, is recorded as FAIL and the run goes on.
    r = bash(f"vr_init {tmp_path}; ex(){{ exit 5; }}; ee(){{ false; echo not-reached; }}; ok(){{ :; }}; "
             "run_check A1 exits ex; run_check A2 errexit ee; run_check A3 ok ok; vr_finish")
    s = (tmp_path / "SUMMARY.md").read_text()
    assert r.returncode == 1
    assert "| 1 | A1 | exits | FAIL |" in s and "| 2 | A2 | errexit | FAIL |" in s and "| 3 | A3 | ok | PASS |" in s
    assert "not-reached" not in (tmp_path / "02-A2.txt").read_text()


def test_run_check_does_not_let_the_check_read_stdin(tmp_path):
    r = subprocess.run(["bash", "-c", f"set -euo pipefail; source {ROOT}/scripts/lib/checks.sh; vr_init {tmp_path}; "
                        "rd(){ cat; }; run_check A1 rd rd; vr_finish"],
                       input="stdin-data\n", capture_output=True, text=True, env=LA_ENV)
    assert r.returncode == 0
    assert "stdin-data" not in (tmp_path / "01-A1.txt").read_text()


def test_run_check_prints_a_status_line(tmp_path):
    r = bash(f"vr_init {tmp_path}; bad(){{ return 1; }}; run_check B1 'bad check' bad")
    assert r.returncode == 0
    assert re.search(r"B1\s+FAIL\s+bad check", r.stdout), r.stdout


def test_summary_escapes_pipes_in_titles(tmp_path):
    bash(f"vr_init {tmp_path}; ok(){{ :; }}; run_check A1 'a | b' ok")
    assert r"| 1 | A1 | a \| b | PASS |" in (tmp_path / "SUMMARY.md").read_text()


def test_vr_init_refuses_a_directory_that_already_holds_files(tmp_path):
    (tmp_path / "old.txt").write_text("previous run\n")
    r = bash(f"vr_init {tmp_path}")
    assert r.returncode != 0
    assert not (tmp_path / "SUMMARY.md").exists()


def test_vr_finish_fails_when_gitleaks_finds_a_secret(tmp_path):
    if not shutil.which("gitleaks"):
        pytest.skip("gitleaks not installed")
    token = "K10" + "e" * 64 + "::server:" + "0123456789abcdef"
    # Written straight into the run directory, bypassing redact, as a buggy check could.
    r = bash(f"vr_init {tmp_path}; leak(){{ echo '{token}' > {tmp_path}/raw.txt; }}; run_check A1 leak leak; vr_finish")
    assert r.returncode == 1
    assert "| 1 | A1 | leak | PASS |" in (tmp_path / "SUMMARY.md").read_text()
    assert token not in r.stdout + r.stderr


# --- redact -------------------------------------------------------------------------------------

def test_redaction_covers_all_secret_shapes():
    leak = ("K10" + "f" * 64 + "::server:abc123def456\nclient-key-data: LS0tLS1CRUdJTg==\n"
            "LAMBDA_API_KEY=" "secret_x_" + "0123456789abcdef\nAuthorization: Bearer abc.def.ghi\n")
    out = redact(leak)
    for s in ("f" * 64, "LS0tLS1CRUdJTg", "0123456789abcdef", "abc.def.ghi"):
        assert s not in out
    assert out.count("<redacted>") == 4


@pytest.mark.parametrize("line, secret", [
    ("    client-certificate-data: " + "Q0VSVA" + "==", "Q0VSVA"),
    ("    certificate-authority-data: " + "Q0FDQQ" + "==", "Q0FDQQ"),
    ("    token: " + "eyJhbGciOi" + "Jz.payload.sig", "eyJhbGciOi"),
    ('  "token": "' + "dG9rZW4t" + 'dmFsdWU=",', "dG9rZW4t"),
    ("> authorization: " + "Bearer " + "low.er.case", "low.er.case"),
    ("  private key: " + "aGlkZGVu" + "S2V5", "aGlkZGVu"),
    ("  Private Key: " + "UFJJVkFU" + "RQ==", "UFJJVkFU"),
    ("PRIVATE KEY: " + "VVBQRVI" + "=", "VVBQRVI"),
    ("curl -u " + "secret_ops_" + HEX20 + ": https://cloud.lambda.ai/", HEX20),
])
def test_redaction_variants(line, secret):
    out = redact(line + "\n")
    assert secret not in out
    assert "<redacted>" in out


def test_redaction_keeps_the_key_name_and_ordinary_text():
    out = redact("client-key-data: " + "LS0t" + "LS1C\nnode k3s-cp Ready token count 3\n")
    assert out == "client-key-data: <redacted>\nnode k3s-cp Ready token count 3\n"


def test_check_output_is_redacted_in_file(tmp_path):
    token = "K10" + "a" * 64 + "::server:" + "s3cretpassw0rd"
    bash(f"vr_init {tmp_path}; leak(){{ echo '{token}'; echo 'token: {token}' >&2; }}; run_check B3 leak leak")
    text = (tmp_path / "01-B3.txt").read_text()
    assert "a" * 64 not in text and "s3cretpassw0rd" not in text
    assert text.count("<redacted>") == 2


# --- vr_cmd -------------------------------------------------------------------------------------

def test_vr_cmd_echoes_command(tmp_path):
    bash(f"vr_init {tmp_path}; c(){{ vr_cmd echo hi; }}; run_check A1 cmd c")
    lines = (tmp_path / "01-A1.txt").read_text().splitlines()
    assert "$ echo hi" in lines and "hi" in lines
    assert lines.index("$ echo hi") < lines.index("hi")


def test_vr_cmd_returns_the_command_status():
    r = bash("rc=0; vr_cmd sh -c 'exit 7' || rc=$?; echo rc=$rc")
    assert r.stdout.splitlines()[-1] == "rc=7"


def test_vr_cmd_redacts_the_echoed_command():
    r = bash('vr_cmd true "LAMBDA_API_KEY=' + 'abcdefgh' + 'ijklmnop1234"')
    assert "abcdefgh" not in r.stdout and "<redacted>" in r.stdout


def test_vr_cmd_redacts_arguments_that_contain_spaces():
    # printf %q would print Authorization:\ Bearer\ <value>, which no redact pattern matches.
    r = bash("vr_cmd true -H 'Authorization: Bearer " + "hdr.sec" + ".ret' --data \"it's\"")
    assert r.returncode == 0
    assert r.stdout.splitlines()[0] == "$ true -H 'Authorization: Bearer <redacted>' --data 'it'\\''s'"


# --- verify.sh ----------------------------------------------------------------------------------

REGISTRATIONS_MARKER = "# --- registered checks ---"


@pytest.fixture
def repo(tmp_path):
    """A throwaway repo root holding copies of verify.sh, its libraries and the gitleaks config."""
    (tmp_path / "scripts/lib").mkdir(parents=True)
    for rel in ("scripts/verify.sh", "scripts/lib/checks.sh", "scripts/lib/common.sh", ".gitleaks.toml"):
        shutil.copy2(ROOT / rel, tmp_path / rel)
    return tmp_path


def register(repo, lines):
    """Insert check definitions and registrations below verify.sh's registration marker."""
    verify = repo / "scripts/verify.sh"
    text = verify.read_text()
    assert text.count(REGISTRATIONS_MARKER) == 1
    verify.write_text(text.replace(REGISTRATIONS_MARKER, REGISTRATIONS_MARKER + "\n" + "\n".join(lines)))


def verify(repo, *groups, **env):
    clean = {k: v for k, v in LA_ENV.items() if k != "RESILIENCE"}
    return subprocess.run(["bash", str(repo / "scripts/verify.sh"), *groups], capture_output=True, text=True,
                          cwd=repo, env={**clean, **env})


def run_dir(repo):
    dirs = list((repo / "verification").iterdir())
    assert len(dirs) == 1, dirs
    return dirs[0]


def summary_ids(repo):
    return re.findall(r"^\| \d+ \| (\w+) \|", (run_dir(repo) / "SUMMARY.md").read_text(), re.M)


def test_verify_with_no_registered_checks_succeeds(repo):
    r = verify(repo)
    assert r.returncode == 0, r.stderr
    d = run_dir(repo)
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{4}-0[78]00", d.name), d.name
    assert "**Result:** 0 passed, 0 failed, 0 skipped" in (d / "SUMMARY.md").read_text()


def test_verify_runs_default_groups_in_order_and_e_only_with_resilience(repo):
    register(repo, [
        "ok() { :; }",
        "register_check E E1 'resilience' ok",
        "register_check B B1 'net' ok",
        "register_check A A1 'nodes' ok",
        "register_check D D1 'gpu' ok",
        "register_check C C1 'app' ok",
        "register_check A A2 'more nodes' ok",
    ])
    assert verify(repo).returncode == 0
    assert summary_ids(repo) == ["A1", "A2", "B1", "C1", "D1"]
    shutil.rmtree(repo / "verification")
    assert verify(repo, RESILIENCE="1").returncode == 0
    assert summary_ids(repo) == ["A1", "A2", "B1", "C1", "D1", "E1"]


def test_verify_runs_only_the_requested_groups_in_the_given_order(repo):
    register(repo, ["ok() { :; }", "register_check A A1 x ok", "register_check C C1 y ok", "register_check B B1 z ok"])
    assert verify(repo, "C", "A").returncode == 0
    assert summary_ids(repo) == ["C1", "A1"]


def test_verify_exit_code_follows_the_results(repo):
    register(repo, ["sk() { return 77; }", "bad() { return 1; }",
                    "register_check D D1 gpu sk", "register_check B B1 net bad"])
    assert verify(repo, "D").returncode == 0                   # Review Focus #5: SKIP-only is success
    shutil.rmtree(repo / "verification")
    assert verify(repo).returncode == 1


def test_verify_rejects_an_unknown_group(repo):
    r = verify(repo, "A", "X")
    assert r.returncode == 2
    assert "X" in r.stderr
    assert not (repo / "verification").exists()
