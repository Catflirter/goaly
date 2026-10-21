import json, shutil, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def scan(tmp_path, text):
    """Scan `text` with the repo's gitleaks config; return (exit code, rule IDs that fired).

    Exit code 1 alone is ambiguous (gitleaks also exits 1 when the config fails to load),
    so the tests also check which rule produced the finding.
    """
    assert shutil.which("gitleaks"), "install gitleaks >= 8.19 (Task 1)"
    (tmp_path / "sample.txt").write_text(text)
    proc = subprocess.run(
        ["gitleaks", "dir", str(tmp_path), "--config", str(ROOT / ".gitleaks.toml"),
         "--no-banner", "--redact", "--exit-code", "1",
         "--report-format", "json", "--report-path", "-"],
        capture_output=True, text=True)
    rules = {f["RuleID"] for f in json.loads(proc.stdout)} if proc.stdout.strip() else set()
    return proc.returncode, rules

def test_k3s_join_token_is_flagged(tmp_path):
    assert scan(tmp_path, "K10" + "a" * 64 + "::server:0123456789abcdef\n") == (1, {"k3s-join-token"})

def test_lambda_key_assignment_is_flagged(tmp_path):
    # Split so this source file does not itself trip the pre-commit gitleaks hook.
    assert scan(tmp_path, "LAMBDA_API_KEY=" + "secret_demo_0123456789abcdef0123\n") == (1, {"lambda-api-key-assignment"})

def test_env_example_is_clean(tmp_path):
    assert scan(tmp_path, (ROOT / ".env.example").read_text()) == (0, set())
