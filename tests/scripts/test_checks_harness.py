"""scripts/lib/checks.sh (run_check, vr_cmd, redact, SUMMARY.md) and the scripts/verify.sh skeleton.

Fake secrets are assembled from pieces ("K10" + "f" * 64, ...) so that this file itself never
matches the repo's gitleaks rules; the strings the scripts see at run time are the full shapes.
"""
import base64
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LA_ENV = {**os.environ, "TZ": "America/Los_Angeles"}
HEX20 = "01234567" + "89abcdef" + "0123"


UTC_ENV = {**os.environ, "TZ": "UTC"}
# What the throwaway repo's .env says (TEST-NET-1), and what its stubbed checkip.amazonaws.com answers.
ENV_OPERATOR_IP = "192.0.2.77"
CHECKIP_IP = "198.51.100.9"
CHECKIP_CALL = "-fsS --max-time 5 https://checkip.amazonaws.com"


class Hermetic:
    """A throwaway repo root for these tests (review F3-1): copies of checks.sh, common.sh and the gitleaks
    config, a private .env with OPERATOR_IP set (and a sentinel), and a curl stub first on PATH that logs
    every call; checkip.amazonaws.com answers CHECKIP_IP, anything else fails. Sourcing the copy makes it
    REPO_ROOT, so vr_init never reads the operator's real .env and never reaches the network."""

    def __init__(self, root):
        self.root = root
        (root / "scripts/lib").mkdir(parents=True)
        for rel in ("scripts/lib/checks.sh", "scripts/lib/common.sh", ".gitleaks.toml"):
            shutil.copy2(ROOT / rel, root / rel)
        self.write_env(f"VR_SENTINEL=hermetic-env\nOPERATOR_IP={ENV_OPERATOR_IP}\n")
        self.bin = root / "bin"
        self.bin.mkdir()
        self.curl_log = root / "curl.log"
        (self.bin / "curl").write_text(
            f'#!/bin/sh\nprintf "%s\\n" "$*" >> "{self.curl_log}"\n'
            f'case "$*" in *checkip.amazonaws.com*) echo {CHECKIP_IP}; exit 0 ;; esac\n'
            'echo "curl stub: no network in these tests" >&2\nexit 7\n')
        (self.bin / "curl").chmod(0o755)
        self.checks = root / "scripts/lib/checks.sh"

    def write_env(self, text):
        env = self.root / ".env"
        env.write_text(text)
        env.chmod(0o600)

    def environ(self, env):
        return {**env, "PATH": f"{self.bin}:{env['PATH']}"}

    def curl_calls(self):
        """The curl calls so far; the log is emptied, so the teardown check sees only later ones."""
        calls = self.curl_log.read_text().splitlines() if self.curl_log.exists() else []
        self.curl_log.unlink(missing_ok=True)
        return calls


H = None


@pytest.fixture(autouse=True)
def hermetic(tmp_path_factory):
    global H
    H = Hermetic(tmp_path_factory.mktemp("repo"))
    yield H
    # OPERATOR_IP is set, so nothing may call curl (a test that empties it reads curl_calls() itself)
    assert H.curl_calls() == []


def bash(script, env=LA_ENV, **kw):
    return subprocess.run(["bash", "-c", f"set -euo pipefail; source {H.checks}; {script}"],
                          capture_output=True, text=True, cwd=H.root, env=H.environ(env), **kw)


def redact(text):
    proc = subprocess.run(["bash", "-c", f"source {H.checks}; redact"], input=text,
                          capture_output=True, text=True, check=True, cwd=H.root, env=H.environ(LA_ENV))
    return proc.stdout


# --- hermetic setup (F3-1) ----------------------------------------------------------------------

def test_the_harness_tests_read_their_own_env_and_never_the_operators(tmp_path):
    r = bash('load_env; printf "%s\\n" "$REPO_ROOT" "$VR_SENTINEL"')
    assert r.stdout.splitlines() == [str(H.root.resolve()), "hermetic-env"] and H.root != ROOT
    assert bash(f"vr_init {tmp_path}/run").returncode == 0
    assert "Operator IP: masked as <operator_ip> (OPERATOR_IP in .env)" in (tmp_path / "run/SUMMARY.md").read_text()
    assert H.curl_calls() == []                               # no checkip lookup with OPERATOR_IP set


def test_vr_init_without_operator_ip_asks_checkip_through_the_stub(tmp_path):
    H.write_env("VR_SENTINEL=hermetic-env\nOPERATOR_IP=\n")
    r = bash(f'vr_init {tmp_path}/run; echo "$VR_OPERATOR_IP"; echo "from {CHECKIP_IP}" | redact')
    assert r.stdout.splitlines() == [CHECKIP_IP, "from <operator_ip>"], r.stderr
    assert H.curl_calls() == [CHECKIP_CALL]


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
    # The operator's shell may run in UTC; the harness forces America/Los_Angeles itself.
    bash(f"vr_init {tmp_path}; ok(){{ :; }}; run_check A1 'ok check' ok", env=UTC_ENV)
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
    r = bash(f"vr_init {tmp_path}; rd(){{ cat; }}; run_check A1 rd rd; vr_finish", input="stdin-data\n")
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
    assert r.returncode == 2
    assert "already holds files" in r.stderr
    assert not (tmp_path / "SUMMARY.md").exists()
    assert (tmp_path / "old.txt").read_text() == "previous run\n"


def test_run_directory_and_evidence_are_private_to_the_operator(tmp_path):
    d = tmp_path / "run"
    bash(f"umask 022; vr_init {d}; ok(){{ echo hi; }}; run_check A1 ok ok; vr_finish")
    assert d.stat().st_mode & 0o777 == 0o700
    for f in ("SUMMARY.md", "01-A1.txt"):
        assert (d / f).stat().st_mode & 0o077 == 0, f


def test_vr_init_does_not_change_the_callers_umask(tmp_path):
    r = bash(f"umask 022; vr_init {tmp_path}/run; umask")
    assert r.stdout.split()[-1] == "0022"


def test_footer_is_on_its_own_line_after_output_without_newline(tmp_path):
    bash(f"vr_init {tmp_path}; ok(){{ printf abc; }}; run_check A1 ok ok")
    lines = (tmp_path / "01-A1.txt").read_text().splitlines()
    assert lines[2] == "abc"
    assert lines[3].startswith("# finished: ") and lines[3].endswith("PASS (exit 0)")


def test_failed_redact_filter_records_fail_not_pass(tmp_path):
    # A broken filter loses the check's evidence; that must never count as a PASS.
    r = bash(f"vr_init {tmp_path}; _VR_SED=(sh -c 'cat >/dev/null; exit 3'); ok(){{ echo fine; }}; "
             "run_check A1 ok ok; vr_finish")
    assert r.returncode == 1
    assert "| 1 | A1 | ok | FAIL |" in (tmp_path / "SUMMARY.md").read_text()
    assert "# redact failed (exit 3); output incomplete" in (tmp_path / "01-A1.txt").read_text()


def test_vr_cmd_does_not_run_the_command_when_redact_fails(tmp_path):
    r = bash(f"_VR_SED=(sh -c 'cat >/dev/null; exit 3'); rc=0; vr_cmd touch {tmp_path}/ran || rc=$?; echo rc=$rc")
    assert r.stdout.splitlines()[-1] == "rc=3"
    assert not (tmp_path / "ran").exists()


def test_background_jobs_of_a_check_do_not_block_the_run(tmp_path):
    # A check that leaves e.g. `kubectl port-forward &` running must not hang make verify.
    r = bash(f"vr_init {tmp_path}; bg(){{ sleep 30 & echo started; }}; run_check A1 bg bg; vr_finish", timeout=15)
    assert r.returncode == 0, r.stderr
    assert "started" in (tmp_path / "01-A1.txt").read_text()


def wg_key(seed):
    """A WireGuard-shaped key (44 base64 chars); built at run time so this file holds no key literal."""
    return base64.b64encode(bytes((seed * 37 + i * 11) % 256 for i in range(32))).decode()


WG_SHOW = """interface: flannel-wg
  public key: {pub}
  private key: (hidden)
  listening port: 51820

peer: {peer}
  endpoint: 34.120.1.2:51820
  allowed ips: 10.42.1.0/24
  latest handshake: 12 seconds ago
  transfer: 1.20 KiB received, 3.40 KiB sent
  persistent keepalive: every 25 seconds
"""


def test_vr_finish_accepts_wg_show_output(tmp_path):
    # Spec 9 B3 runs `wg show flannel-wg`; its public keys are not secrets and must not fail the run.
    if not shutil.which("gitleaks"):
        pytest.skip("gitleaks not installed")
    (tmp_path / "wg.txt").write_text(WG_SHOW.format(pub=wg_key(1), peer=wg_key(2)))
    r = bash(f"vr_init {tmp_path}/run; b3(){{ cat {tmp_path}/wg.txt; }}; run_check B3 wg b3; vr_finish")
    assert r.returncode == 0, r.stdout + r.stderr
    assert wg_key(2) in (tmp_path / "run/01-B3.txt").read_text()


def test_vr_finish_still_flags_a_wireguard_private_key(tmp_path):
    if not shutil.which("gitleaks"):
        pytest.skip("gitleaks not installed")
    # Written straight into the run directory, bypassing redact, as a buggy check could.
    r = bash(f"vr_init {tmp_path}; leak(){{ echo 'private key: {wg_key(3)}' > {tmp_path}/raw.txt; }}; "
             "run_check A1 leak leak; vr_finish")
    assert r.returncode == 1


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
    ("token: '" + "c2luZ2xl" + "UXVvdGVk'", "c2luZ2xl"),
    ("  client-key-data: '" + "TFMwdExT" + "MUNSVQ=='", "TFMwdExT"),
    ("  'token': '" + "cXVvdGVk" + "S2V5'", "cXVvdGVk"),
    ('Authorization: Bearer "' + "cXVvdGVk" + 'QmVhcmVy"', "cXVvdGVk"),
    ("private key: '" + "c3FQcml2" + "S2V5'", "c3FQcml2"),
    ('  "LAMBDA_API_KEY": "' + "anNvbkxh" + 'bWJkYUtleQ",', "anNvbkxh"),
    ("LAMBDA_API_KEY: " + "eWFtbExh" + "bWJkYQ", "eWFtbExh"),
    ("  tls.key: " + "TFMwdExT" + "MUNSVUdK", "TFMwdExT"),
    ("    password: " + "aHVudGVy" + "Mg==", "aHVudGVy"),
    ("kubectl --token=" + "YXJndkVx" + "dWFscw get pods", "YXJndkVx"),
    ("kubectl --token " + "YXJndlNw" + "YWNl get pods", "YXJndlNw"),
    ('{"t": "' + "K10" + "c" * 64 + "::server:abc" + ',def"}', ",def"),
])
def test_redaction_variants(line, secret):
    out = redact(line + "\n")
    assert secret not in out
    assert "<redacted>" in out


def test_redaction_replaces_pem_private_key_bodies():
    body = ["TUlJRXZn" + "SUJBREFO", "QmdrcWhr" + "aUc5dzBC"]
    pem = ("  private key: -----BEGIN " + "PRIVATE KEY-----\n" + "".join(f"    {b}\n" for b in body)
           + "    -----END " + "PRIVATE KEY-----\nafter\n")
    out = redact(pem)
    for b in body:
        assert b not in out
    assert out.splitlines()[1:] == ["    <redacted>", "    <redacted>", "    -----END " + "PRIVATE KEY-----", "after"]


def test_redaction_survives_non_utf8_bytes():
    raw = b"\xff\xfe payload\ntoken: " + b"bm9uVXRm" + b"OA\n"
    proc = subprocess.run(["bash", "-c", f"source {H.checks}; redact"], input=raw, capture_output=True, check=True,
                          cwd=H.root, env=H.environ({**os.environ, "LC_ALL": "C.UTF-8"}))
    assert proc.stdout == b"\xff\xfe payload\ntoken: <redacted>\n"


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
REGISTRATIONS_END = "# --- end of registered checks ---"


@pytest.fixture
def repo(tmp_path):
    """A throwaway repo root holding copies of verify.sh, its libraries and the gitleaks config. The copy's
    real registrations (Task 18 on) are removed: they would run kubectl, gcloud and ssh against the
    operator's cluster. The tests register their own stand-in checks."""
    (tmp_path / "scripts/lib").mkdir(parents=True)
    for rel in ("scripts/verify.sh", "scripts/lib/checks.sh", "scripts/lib/common.sh", ".gitleaks.toml"):
        shutil.copy2(ROOT / rel, tmp_path / rel)
    verify = tmp_path / "scripts/verify.sh"
    text = verify.read_text()
    assert text.count(REGISTRATIONS_MARKER) == 1 and text.count(REGISTRATIONS_END) == 1
    head, rest = text.split(REGISTRATIONS_MARKER)
    verify.write_text(head + REGISTRATIONS_MARKER + "\n" + REGISTRATIONS_END + rest.split(REGISTRATIONS_END)[1])
    return tmp_path


def register(repo, lines):
    """Insert check definitions and registrations below verify.sh's registration marker."""
    verify = repo / "scripts/verify.sh"
    text = verify.read_text()
    assert text.count(REGISTRATIONS_MARKER) == 1
    verify.write_text(text.replace(REGISTRATIONS_MARKER, REGISTRATIONS_MARKER + "\n" + "\n".join(lines)))


def verify(repo, *groups, **env):
    clean = {k: v for k, v in LA_ENV.items() if k != "RESILIENCE"}
    # The copy has no .env, so vr_init does not look the operator IP up; the curl stub would catch it if it did.
    return subprocess.run(["bash", str(repo / "scripts/verify.sh"), *groups], capture_output=True, text=True,
                          cwd=repo, env=H.environ({**clean, **env}))


def run_dir(repo):
    dirs = list((repo / "verification").iterdir())
    assert len(dirs) == 1, dirs
    return dirs[0]


def summary_ids(repo):
    return re.findall(r"^\| \d+ \| (\w+) \|", (run_dir(repo) / "SUMMARY.md").read_text(), re.M)


def test_verify_with_no_registered_checks_succeeds(repo):
    r = verify(repo, TZ="UTC")                                 # the directory name is LA time regardless
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
