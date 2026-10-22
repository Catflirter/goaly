"""scripts/promote.sh verify and `make promote-verify` (Task 24a, Ruling R44): copy one verify run to
verification/final/ and scan the copy with the repo's gitleaks rules before it replaces the old one.

Fake secrets are assembled from pieces so this file itself never matches the gitleaks rules (see
tests/test_gitleaks_rules.py); the files the script sees at run time hold the full shapes.
"""
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TS = "2026-10-21T1430-0700"
K3S_TOKEN = "K10" + "c" * 64 + "::server:" + "0123456789abcdef"
FAKE_LAMBDA_KEY = "secret" + "_demo_" + "0123456789abcdef" * 2 + "." + "Q7xZ" * 8
SUMMARY = ("# Verification run\n\nStarted: 2026-10-21T14:30:05-07:00 (America/Los_Angeles)\n\n"
           "| # | ID | Check | Result |\n|---|---|---|---|\n| 1 | A1 | nodes | PASS |\n\n"
           "**Result:** 1 passed, 0 failed, 0 skipped\n\n**Secret scan:** gitleaks found no leaks\n")


def make_repo(root):
    """A throwaway repo root holding copies of promote.sh, common.sh, the gitleaks config and the Makefile."""
    (root / "scripts/lib").mkdir(parents=True)
    for rel in ("scripts/promote.sh", "scripts/lib/common.sh", ".gitleaks.toml", "Makefile"):
        shutil.copy2(ROOT / rel, root / rel)
    (root / "verification").mkdir()
    (root / "verification/.gitkeep").write_text("")
    return root.resolve()


@pytest.fixture
def repo(tmp_path):
    return make_repo(tmp_path / "repo")


def make_run(repo, name=TS, files=None, summary=SUMMARY):
    """A verification/<name>/ run as scripts/verify.sh leaves it (0700 directory, 0600 files)."""
    run = repo / "verification" / name
    run.mkdir(mode=0o700)
    if summary is not None:
        (run / "SUMMARY.md").write_text(summary)
    for rel, text in (files or {"01-A1.txt": "# A1 — nodes\n$ kubectl get nodes -o wide\nk3s-cp Ready\n",
                                 "02-B1.txt": "# B1 — matrix\nok\n"}).items():
        (run / rel).parent.mkdir(parents=True, exist_ok=True)
        (run / rel).write_text(text)
    for p in [run, *run.rglob("*")]:
        p.chmod(0o700 if p.is_dir() else 0o600)
    return run


def snapshot(path):
    """{relative path: bytes, or None for a directory}: what a copy must reproduce exactly."""
    return {str(p.relative_to(path)): (None if p.is_dir() else p.read_bytes()) for p in path.rglob("*")}


def promote(repo, *args, env=None, cwd=None):
    return subprocess.run(["bash", str(repo / "scripts/promote.sh"), *args], capture_output=True, text=True,
                          cwd=cwd or repo, env=env or os.environ.copy(), timeout=120)


def make_target(repo, *args):
    return subprocess.run(["make", "--no-print-directory", "-f", str(repo / "Makefile"), "promote-verify", *args],
                          capture_output=True, text=True, cwd=repo)


def verification_entries(repo):
    return sorted(p.name for p in (repo / "verification").iterdir())


def previous_final(repo):
    final = repo / "verification/final"
    final.mkdir()
    (final / "SUMMARY.md").write_text("# Verification run\n\nold\n")
    (final / "99-old.txt").write_text("from the previous final\n")
    return snapshot(final)


# --- happy path ---------------------------------------------------------------------------------

def test_promote_copies_the_run_exactly(repo):
    run = make_run(repo, files={"01-A1.txt": "a\n", "02-B1.txt": "b\n", "03-C1.txt": "c\n", "autoscaler.log": "d\n"})
    before = snapshot(run)
    r = promote(repo, "verify", f"verification/{TS}")
    assert r.returncode == 0, r.stderr
    assert snapshot(repo / "verification/final") == before
    assert snapshot(run) == before, "the run itself is left as it was"
    assert verification_entries(repo) == [".gitkeep", TS, "final"], "no staging directory is left behind"
    assert "verification/final" in r.stderr and "no leaks" in r.stderr


@pytest.mark.parametrize("arg", ["abs", f"verification/{TS}/", f"./verification/{TS}", f"verification/../verification/{TS}"])
def test_promote_accepts_the_run_in_any_spelling(repo, arg):
    run = make_run(repo)
    r = promote(repo, "verify", str(run) if arg == "abs" else arg)
    assert r.returncode == 0, r.stderr
    assert snapshot(repo / "verification/final") == snapshot(run)


def test_promote_resolves_a_relative_run_from_the_current_directory(repo):
    run = make_run(repo)
    r = promote(repo, "verify", TS, cwd=repo / "verification")
    assert r.returncode == 0, r.stderr
    assert snapshot(repo / "verification/final") == snapshot(run)


def test_promote_replaces_the_previous_final_instead_of_merging(repo):
    previous_final(repo)
    run = make_run(repo)
    r = promote(repo, "verify", f"verification/{TS}")
    assert r.returncode == 0, r.stderr
    assert snapshot(repo / "verification/final") == snapshot(run)
    assert not (repo / "verification/final/99-old.txt").exists()


def test_promote_replaces_a_final_symlink_without_touching_its_target(repo, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "keep.txt").write_text("not the repo's\n")
    (repo / "verification/final").symlink_to(elsewhere)
    run = make_run(repo)
    r = promote(repo, "verify", f"verification/{TS}")
    assert r.returncode == 0, r.stderr
    final = repo / "verification/final"
    assert not final.is_symlink() and snapshot(final) == snapshot(run)
    assert (elsewhere / "keep.txt").read_text() == "not the repo's\n"


def test_promote_accepts_a_run_that_scripts_verify_sh_wrote(repo):
    # The tests above use a hand-made run; this one is the real harness's output (no checks registered).
    for rel in ("scripts/verify.sh", "scripts/lib/checks.sh"):
        shutil.copy2(ROOT / rel, repo / rel)
    verify_sh = repo / "scripts/verify.sh"
    # Replace the whole registration block: the real checks need a live cluster.
    start, end = "# --- registered checks ---\n", "# --- end of registered checks ---\n"
    text = verify_sh.read_text()
    assert start in text and end in text
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    verify_sh.write_text(head + start + "check_ok() { echo ok; }\nregister_check A A1 'always passes' check_ok\n"
                         + end + tail)
    v = subprocess.run(["bash", str(repo / "scripts/verify.sh")], capture_output=True, text=True, cwd=repo,
                       env={k: val for k, val in os.environ.items() if k != "RESILIENCE"})
    assert v.returncode == 0, v.stderr
    (run,) = [p for p in (repo / "verification").iterdir() if p.is_dir()]
    r = promote(repo, "verify", f"verification/{run.name}")
    assert r.returncode == 0, r.stderr
    assert snapshot(repo / "verification/final") == snapshot(run)
    assert "1 passed, 0 failed, 0 skipped" in r.stderr


# --- secret scan --------------------------------------------------------------------------------

# Both shapes are the repo's own rules in .gitleaks.toml (not gitleaks' defaults), so a pass here also
# proves the scan used the repo config.
@pytest.mark.parametrize("secret", [K3S_TOKEN, "Authorization: Bearer " + FAKE_LAMBDA_KEY])
def test_a_leak_fails_and_leaves_no_final(repo, secret):
    make_run(repo, files={"01-A1.txt": "fine\n", "02-B1.txt": "oops " + secret + "\n"})
    r = promote(repo, "verify", f"verification/{TS}")
    assert r.returncode == 1, r.stderr
    assert not (repo / "verification/final").exists()
    assert verification_entries(repo) == [".gitkeep", TS], "the scanned copy is removed"
    out = r.stdout + r.stderr
    assert secret.split()[-1] not in out, "gitleaks output is redacted"
    assert "leak" in out and "02-B1.txt" in out and "01-A1.txt" not in out


def test_a_leak_keeps_the_previous_final_unchanged(repo):
    before = previous_final(repo)
    make_run(repo, files={"01-A1.txt": K3S_TOKEN + "\n"})
    r = promote(repo, "verify", f"verification/{TS}")
    assert r.returncode == 1, r.stderr
    assert snapshot(repo / "verification/final") == before
    assert verification_entries(repo) == [".gitkeep", TS, "final"]


def test_a_failing_scanner_counts_as_a_leak(repo, tmp_path):
    # gitleaks also exits non-zero when it cannot load its config or read the files: no clean scan, no final.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "gitleaks").write_text("#!/bin/sh\necho 'unable to load gitleaks config' >&2\nexit 2\n")
    (bin_dir / "gitleaks").chmod(0o755)
    before = previous_final(repo)
    make_run(repo)
    r = promote(repo, "verify", f"verification/{TS}", env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"})
    assert r.returncode == 1, r.stderr
    assert snapshot(repo / "verification/final") == before
    assert verification_entries(repo) == [".gitkeep", TS, "final"]


def test_a_missing_gitleaks_refuses_before_copying(repo, tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("bash", "cat", "chmod", "cp", "dirname", "find", "grep", "head", "ls", "mkdir", "mktemp",
                 "mv", "rm", "rmdir", "sed", "stat", "wc"):
        if shutil.which(tool):
            (bin_dir / tool).symlink_to(shutil.which(tool))
    before = previous_final(repo)
    make_run(repo)
    r = promote(repo, "verify", f"verification/{TS}", env={**os.environ, "PATH": str(bin_dir)})
    assert r.returncode == 1, r.stderr
    assert "gitleaks" in r.stderr
    assert snapshot(repo / "verification/final") == before
    assert verification_entries(repo) == [".gitkeep", TS, "final"]


# --- refused runs (exit 2, nothing changed) ---------------------------------------------------

def assert_refused(repo, r, before, *words):
    assert r.returncode == 2, (r.returncode, r.stderr)
    final = repo / "verification/final"
    assert snapshot(final) == before and not final.is_symlink()
    assert not [n for n in verification_entries(repo) if n.startswith(".promote")]
    for word in words:
        assert word in r.stderr, r.stderr


@pytest.mark.parametrize("args", [[], ["verify"], ["verify", f"verification/{TS}", "extra"], ["bogus"],
                                  ["demo", f"verification/{TS}"]])
def test_usage_errors(repo, args):
    before = previous_final(repo)
    make_run(repo)
    assert_refused(repo, promote(repo, *args), before, "usage")


def test_help_prints_usage(repo):
    r = promote(repo, "--help")
    assert r.returncode == 0 and "verify" in r.stdout


@pytest.mark.parametrize("arg", ["verification/2026-01-01T0000-0800", "README", "verification/.gitkeep"])
def test_refuses_a_missing_run_or_a_file(repo, arg):
    (repo / "README").write_text("x\n")
    before = previous_final(repo)
    assert_refused(repo, promote(repo, "verify", arg), before, arg)


@pytest.mark.parametrize("arg", ["verification", "verification/final", "verification/final/",
                                 "verification/./final", f"verification/{TS}/../final", "."])
def test_refuses_verification_itself_and_final(repo, arg):
    before = previous_final(repo)
    make_run(repo)
    r = promote(repo, "verify", arg)
    assert_refused(repo, r, before)
    if "final" in arg:
        assert "verification/final itself" in r.stderr, r.stderr


def test_refuses_a_run_outside_verification(repo, tmp_path):
    before = previous_final(repo)
    outside = tmp_path / TS                                         # right name, wrong place
    outside.mkdir()
    (outside / "SUMMARY.md").write_text(SUMMARY)
    in_repo = repo / TS
    in_repo.mkdir()
    (in_repo / "SUMMARY.md").write_text(SUMMARY)
    for arg in (str(outside), TS):
        assert_refused(repo, promote(repo, "verify", arg), before, "verification/")


def test_refuses_a_directory_nested_in_a_run(repo):
    before = previous_final(repo)
    make_run(repo, files={"sub/SUMMARY.md": SUMMARY})
    assert_refused(repo, promote(repo, "verify", f"verification/{TS}/sub"), before)


@pytest.mark.parametrize("name", ["scratch", "demo-2026-10-21T1430-0700", "2026-10-21T1430", "final.old"])
def test_refuses_a_directory_not_named_like_a_verify_run(repo, name):
    before = previous_final(repo)
    make_run(repo, name=name)
    assert_refused(repo, promote(repo, "verify", f"verification/{name}"), before, name)


def test_refuses_a_run_without_summary(repo):
    before = previous_final(repo)
    make_run(repo, summary=None)
    assert_refused(repo, promote(repo, "verify", f"verification/{TS}"), before, "SUMMARY.md")


@pytest.mark.parametrize("summary", ["notes\n", SUMMARY.split("\n**Result:**")[0]])
def test_refuses_a_summary_that_is_not_a_finished_verify_run(repo, summary):
    before = previous_final(repo)
    make_run(repo, summary=summary)
    assert_refused(repo, promote(repo, "verify", f"verification/{TS}"), before, "SUMMARY.md")


# --- symlink tricks -----------------------------------------------------------------------------

def test_refuses_a_run_that_is_a_symlink_to_outside(repo, tmp_path):
    before = previous_final(repo)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "SUMMARY.md").write_text(SUMMARY)
    (repo / "verification" / TS).symlink_to(outside)
    assert_refused(repo, promote(repo, "verify", f"verification/{TS}"), before)


def test_refuses_a_run_that_is_a_symlink_to_final(repo):
    previous_final(repo)
    (repo / "verification/final/SUMMARY.md").write_text(SUMMARY)
    before = snapshot(repo / "verification/final")
    (repo / "verification" / TS).symlink_to(repo / "verification/final")
    assert_refused(repo, promote(repo, "verify", f"verification/{TS}"), before)


def test_refuses_a_summary_that_is_a_symlink(repo, tmp_path):
    before = previous_final(repo)
    run = make_run(repo, summary=None)
    (tmp_path / "s.md").write_text(SUMMARY)
    (run / "SUMMARY.md").symlink_to(tmp_path / "s.md")
    assert_refused(repo, promote(repo, "verify", f"verification/{TS}"), before, "has no SUMMARY.md")


@pytest.mark.parametrize("kind", ["file", "dir", "dangling"])
def test_refuses_a_run_holding_a_symlink(repo, tmp_path, kind):
    # gitleaks dir does not follow symlinks, so a linked file would reach final/ unscanned.
    before = previous_final(repo)
    run = make_run(repo)
    target = tmp_path / "secret.env"
    if kind == "file":
        target.write_text("LAMBDA_API_KEY=" + "plainvalue0123456789abcdef\n")
    elif kind == "dir":
        target.mkdir()
    (run / "03-link.txt").symlink_to(target)
    assert_refused(repo, promote(repo, "verify", f"verification/{TS}"), before, "03-link.txt")


def test_refuses_when_verification_is_a_symlink(repo, tmp_path):
    real = tmp_path / "real-verification"
    shutil.move(repo / "verification", real)
    (repo / "verification").symlink_to(real)
    make_run(repo)
    r = promote(repo, "verify", f"verification/{TS}")
    assert r.returncode == 2, r.stderr
    assert not (real / "final").exists()


# --- make promote-verify ------------------------------------------------------------------------

def test_make_promote_verify_requires_run(repo):
    before = previous_final(repo)
    make_run(repo)
    r = make_target(repo)
    assert r.returncode != 0
    assert "RUN=verification/" in r.stderr, r.stderr
    assert snapshot(repo / "verification/final") == before


def test_make_promote_verify_promotes_the_run(repo):
    run = make_run(repo)
    r = make_target(repo, f"RUN=verification/{TS}")
    assert r.returncode == 0, r.stderr
    assert snapshot(repo / "verification/final") == snapshot(run)


def test_make_promote_verify_fails_on_a_leak(repo):
    make_run(repo, files={"01-A1.txt": K3S_TOKEN + "\n"})
    r = make_target(repo, f"RUN=verification/{TS}")
    assert r.returncode != 0
    assert not (repo / "verification/final").exists()


def test_make_help_lists_promote_verify():
    r = subprocess.run(["make", "--no-print-directory", "-f", str(ROOT / "Makefile"), "help"],
                       capture_output=True, text=True, cwd=ROOT)
    assert "promote-verify" in r.stdout


# --- review round 0 (PR #92) --------------------------------------------------------------------

# C0-1/X0-1: gitleaks' default path allowlist (.gitleaks.toml has useDefault = true) skips images,
# .git, vendor/, node_modules/ and more, so the run may hold only the flat files verify.sh writes.
@pytest.mark.parametrize("name", ["02-B1.png", "notes.md", ".hidden.txt", "01-gitleaks.toml.txt", "README",
                                  ".git/03-x.txt", "vendor/03-x.txt", "node_modules/03-x.txt",
                                  "nested/03-C1.txt", "03-x.txt/"])
def test_refuses_a_run_holding_a_file_verify_sh_does_not_write(repo, name):
    before = previous_final(repo)
    run = make_run(repo)
    if name.endswith("/"):
        (run / name).mkdir()
    else:
        (run / name).parent.mkdir(parents=True, exist_ok=True)
        (run / name).write_text("oops " + K3S_TOKEN + "\n")
    assert_refused(repo, promote(repo, "verify", f"verification/{TS}"), before, name.split("/")[0])


def test_an_autoscaler_log_is_promoted_and_scanned(repo):
    # spec section 10.5: the autoscaler writes verification/<ts>/autoscaler.log.
    run = make_run(repo, files={"01-A1.txt": "a\n", "autoscaler.log": "scale up: pending GPU pod\n"})
    r = promote(repo, "verify", f"verification/{TS}")
    assert r.returncode == 0, r.stderr
    assert snapshot(repo / "verification/final") == snapshot(run)
    shutil.rmtree(repo / "verification/final")
    (run / "autoscaler.log").write_text("join " + K3S_TOKEN + "\n")
    r = promote(repo, "verify", f"verification/{TS}")
    assert r.returncode == 1 and "autoscaler.log" in r.stderr, r.stderr
    assert not (repo / "verification/final").exists()


def test_a_gitleaks_allow_marker_does_not_hide_a_leak(repo):
    before = previous_final(repo)
    make_run(repo, files={"01-A1.txt": "token " + K3S_TOKEN + "  # gitleaks" + ":allow\n"})
    r = promote(repo, "verify", f"verification/{TS}")
    assert r.returncode == 1, r.stderr
    assert snapshot(repo / "verification/final") == before


def test_a_repo_under_an_allowlisted_directory_is_still_scanned(tmp_path):
    # The default allowlist matches (^|/)node_modules(/.*)?$ against the scanned path: the repo's own
    # location must not exempt the run.
    repo = make_repo(tmp_path / "node_modules" / "repo")
    make_run(repo, files={"01-A1.txt": K3S_TOKEN + "\n"})
    r = promote(repo, "verify", f"verification/{TS}")
    assert r.returncode == 1, r.stderr
    assert not (repo / "verification/final").exists()


# C0-3/X0-3, F2-2: final is the run where every check passed (spec section 9, Task 24 Step 3: A-E all
# PASS). A failed check is always refused; a skipped one only with PROMOTE_ALLOW_SKIP=1, recorded in final.
def summary_of(*results):
    """A finished SUMMARY.md whose table rows and Result line hold these check results, in order."""
    rows = "".join(f"| {i} | A{i} | check {i} | {res} |\n" for i, res in enumerate(results, 1))
    return ("# Verification run\n\nStarted: 2026-10-21T14:30:05-07:00 (America/Los_Angeles)\n\n"
            "| # | ID | Check | Result |\n|---|---|---|---|\n" + rows +
            f"\n**Result:** {results.count('PASS')} passed, {results.count('FAIL')} failed, "
            f"{results.count('SKIP')} skipped\n\n**Secret scan:** gitleaks found no leaks\n")


FAILED_SUMMARY = summary_of("PASS", "FAIL", "PASS", "FAIL", "PASS")
SKIPPED_SUMMARY = summary_of("PASS", "SKIP", "SKIP")


def test_refuses_a_run_with_failed_checks(repo):
    before = previous_final(repo)
    make_run(repo, summary=FAILED_SUMMARY)
    r = promote(repo, "verify", f"verification/{TS}")
    assert_refused(repo, r, before, "2 failed check(s)")
    assert "PROMOTE_ALLOW" not in r.stderr, "no override is offered for a failed check"


@pytest.mark.parametrize("via_make", [False, True])
@pytest.mark.parametrize("summary", [FAILED_SUMMARY, summary_of("PASS", "FAIL", "SKIP")], ids=["fail", "fail+skip"])
def test_no_opt_in_promotes_a_failed_check(repo, via_make, summary):
    before = previous_final(repo)
    make_run(repo, summary=summary)
    opt_in = {"PROMOTE_ALLOW_SKIP": "1", "PROMOTE_ALLOW_FAIL": "1"}
    if via_make:
        r = make_target(repo, f"RUN=verification/{TS}", *(f"{k}={v}" for k, v in opt_in.items()))
    else:
        r = promote(repo, "verify", f"verification/{TS}", env={**os.environ, **opt_in})
    assert_refused(repo, r, before, "failed check(s)")


@pytest.mark.parametrize("allow", [None, "0", "yes", ""])
def test_refuses_a_run_with_skipped_checks(repo, allow):
    before = previous_final(repo)
    make_run(repo, summary=SKIPPED_SUMMARY)
    env = {k: v for k, v in os.environ.items() if k != "PROMOTE_ALLOW_SKIP"}
    if allow is not None:
        env["PROMOTE_ALLOW_SKIP"] = allow
    assert_refused(repo, promote(repo, "verify", f"verification/{TS}", env=env), before,
                   "2 skipped", "PROMOTE_ALLOW_SKIP=1")


@pytest.mark.parametrize("via_make", [False, True])
def test_promote_allow_skip_promotes_a_skipped_run_and_records_it(repo, via_make):
    run = make_run(repo, summary=SKIPPED_SUMMARY)
    if via_make:
        r = make_target(repo, f"RUN=verification/{TS}", "PROMOTE_ALLOW_SKIP=1")
    else:
        r = promote(repo, "verify", f"verification/{TS}", env={**os.environ, "PROMOTE_ALLOW_SKIP": "1"})
    assert r.returncode == 0, r.stderr
    final = repo / "verification/final"
    got, want = snapshot(final), snapshot(run)
    assert got.pop("SUMMARY.md") != want.pop("SUMMARY.md") and got == want, "only SUMMARY.md gains the note"
    text = (final / "SUMMARY.md").read_text()
    assert text.startswith(SKIPPED_SUMMARY), "the run's summary is kept as it was"
    note = text[len(SKIPPED_SUMMARY):]
    assert "2 skipped" in note and "PROMOTE_ALLOW_SKIP=1" in note, note
    assert (run / "SUMMARY.md").read_text() == SKIPPED_SUMMARY, "the run itself is not changed"
    assert "WARNING" in r.stderr and "2 skipped" in r.stderr


def test_a_passing_run_gets_no_note(repo):
    run = make_run(repo)
    r = promote(repo, "verify", f"verification/{TS}", env={**os.environ, "PROMOTE_ALLOW_SKIP": "1"})
    assert r.returncode == 0, r.stderr
    assert snapshot(repo / "verification/final") == snapshot(run)


def test_refuses_a_run_where_no_check_ran(repo):
    before = previous_final(repo)
    make_run(repo, summary=summary_of())
    assert_refused(repo, promote(repo, "verify", f"verification/{TS}"), before, "no check")


@pytest.mark.parametrize("summary", [
    SUMMARY.replace("| PASS |", "| FAIL |"),                              # a FAIL row under a passing total
    SUMMARY.replace("| PASS |", "| SKIP |"),
    SUMMARY.replace("1 passed", "2 passed"),                              # more checks counted than listed
    summary_of("PASS", "PASS").replace("2 passed", "1 passed"),          # a row the total leaves out
], ids=["fail-row", "skip-row", "missing-row", "extra-row"])
def test_refuses_a_summary_whose_table_disagrees_with_its_result(repo, summary):
    before = previous_final(repo)
    make_run(repo, summary=summary)
    assert_refused(repo, promote(repo, "verify", f"verification/{TS}"), before, "Result")


@pytest.mark.parametrize("result", ["**Result:** all good\n", "**Result:** 1 passed, x failed, 0 skipped\n"])
def test_refuses_a_result_line_it_cannot_read(repo, result):
    before = previous_final(repo)
    make_run(repo, summary=SUMMARY.split("**Result:**")[0] + result)
    assert_refused(repo, promote(repo, "verify", f"verification/{TS}"), before, "Result")


# X0-2: the operator IP (Task 24 Step 4) must not reach final.
OPERATOR_IP = "73.15.40.40"


@pytest.mark.parametrize("via_make", [False, True])
def test_the_operator_ip_fails_and_keeps_the_previous_final(repo, via_make):
    before = previous_final(repo)
    make_run(repo, files={"01-A1.txt": "fine\n", "06-B6.txt": f"fw-ssh-operator  {OPERATOR_IP}/32\n"})
    if via_make:
        r = make_target(repo, f"RUN=verification/{TS}", f"OPERATOR_IP={OPERATOR_IP}")
    else:
        r = promote(repo, "verify", f"verification/{TS}", env={**os.environ, "OPERATOR_IP": OPERATOR_IP})
    assert r.returncode == (2 if via_make else 1), r.stderr     # make reports a failed recipe as 2
    assert snapshot(repo / "verification/final") == before
    assert verification_entries(repo) == [".gitkeep", TS, "final"]
    assert "06-B6.txt" in r.stderr and "01-A1.txt" not in r.stderr and "OPERATOR_IP" in r.stderr


# F3-2: the address is matched as a whole number token, not as a substring of a longer one.
@pytest.mark.parametrize("ip, line", [
    ("73.15.40.4", "allowed 73.15.40.4\n"),                   # end of line
    ("73.15.40.4", "73.15.40.4 allowed\n"),                   # start of line
    ("73.15.40.4", "73.15.40.4"),                             # end of file, no newline
    ("73.15.40.4", "src ip=73.15.40.4, port 22\n"),
    ("73.15.40.4", "server: https://73.15.40.4:6443\n"),
    ("73.15.40.4", "fw-ssh-operator  73.15.40.4/32\n"),
    ("73.15.40.4", "reached 73.15.40.4.\n"),                  # end of a sentence
    ("73.15.40.4", "connecting...73.15.40.4\n"),
])
def test_the_operator_ip_is_found_between_any_non_digits(repo, ip, line):
    before = previous_final(repo)
    make_run(repo, files={"01-A1.txt": "fine\n", "06-B6.txt": line})
    r = promote(repo, "verify", f"verification/{TS}", env={**os.environ, "OPERATOR_IP": ip})
    assert r.returncode == 1, r.stderr
    assert snapshot(repo / "verification/final") == before
    assert "06-B6.txt" in r.stderr and "01-A1.txt" not in r.stderr


def test_the_operator_ip_is_found_after_a_byte_that_is_not_utf8(repo):
    # In a UTF-8 locale grep's [^0-9] does not match an invalid byte such as \xff, so the address
    # right after it would be missed; promote.sh runs its grep under LC_ALL=C for this reason.
    before = previous_final(repo)
    run = make_run(repo, files={"01-A1.txt": "fine\n", "06-B6.txt": "placeholder\n"})
    (run / "06-B6.txt").write_bytes(b"x\xff73.15.40.4\n")
    env = {**os.environ, "LC_ALL": "C.UTF-8", "LANG": "C.UTF-8", "OPERATOR_IP": "73.15.40.4"}
    r = promote(repo, "verify", f"verification/{TS}", env=env)
    assert r.returncode == 1, r.stderr
    assert snapshot(repo / "verification/final") == before
    assert "06-B6.txt" in r.stderr


@pytest.mark.parametrize("ip, line", [
    ("73.15.40.4", "fw-ssh-operator  73.15.40.40/32\n"),     # a longer last octet
    ("73.15.40.4", "peer 73.15.40.41\n"),
    ("3.15.40.4", "peer 73.15.40.4\n"),                      # a longer first octet
    ("73.15.40.4", "version 173.15.40.49\n"),
    ("73.15.40.4", "id 73x15-40/4\n"),                       # a dot matches only a dot
])
def test_a_longer_address_is_not_the_operator_ip(repo, ip, line):
    run = make_run(repo, files={"01-A1.txt": "fine\n", "06-B6.txt": line})
    r = promote(repo, "verify", f"verification/{TS}", env={**os.environ, "OPERATOR_IP": ip})
    assert r.returncode == 0, r.stderr
    assert snapshot(repo / "verification/final") == snapshot(run)


def test_success_reminds_the_operator_ip_check(repo):
    make_run(repo)
    r = promote(repo, "verify", f"verification/{TS}", env={**os.environ, "OPERATOR_IP": OPERATOR_IP})
    assert r.returncode == 0, r.stderr
    assert "checkip.amazonaws.com" in r.stderr


# F2-4: OPERATOR_IP is read as gpuctl reads it (surrounding whitespace dropped) and must be one IPv4
# address; anything else would make the check match nothing.
@pytest.mark.parametrize("value", [f" {OPERATOR_IP}", f"{OPERATOR_IP}\n", f"\t{OPERATOR_IP}  "])
def test_the_operator_ip_is_checked_despite_surrounding_whitespace(repo, value):
    before = previous_final(repo)
    make_run(repo, files={"01-A1.txt": "fine\n", "06-B6.txt": f"fw-ssh-operator  {OPERATOR_IP}/32\n"})
    r = promote(repo, "verify", f"verification/{TS}", env={**os.environ, "OPERATOR_IP": value})
    assert r.returncode == 1, r.stderr
    assert snapshot(repo / "verification/final") == before
    assert "06-B6.txt" in r.stderr


@pytest.mark.parametrize("value", ["73.15.40", "73.15.40.40.1", "73.15.40.400", "073.15.40.40", "not-an-ip",
                                   f"{OPERATOR_IP}/32", f"{OPERATOR_IP} 10.0.0.1", "::1", "73.15.40.4x"])
def test_refuses_an_operator_ip_that_is_not_one_ipv4_address(repo, value):
    before = previous_final(repo)
    make_run(repo, files={"01-A1.txt": "fine\n", "06-B6.txt": f"fw-ssh-operator  {OPERATOR_IP}/32\n"})
    r = promote(repo, "verify", f"verification/{TS}", env={**os.environ, "OPERATOR_IP": value})
    assert_refused(repo, r, before, "OPERATOR_IP")


@pytest.mark.parametrize("value", ["", "  ", "\n"])
def test_a_blank_operator_ip_is_unset(repo, value):
    # As in gpuctl (detect_operator_ip): blank means "not given"; the success message then prints the
    # Task 24 Step 4 grep for the operator to run.
    run = make_run(repo)
    r = promote(repo, "verify", f"verification/{TS}", env={**os.environ, "OPERATOR_IP": value})
    assert r.returncode == 0, r.stderr
    assert snapshot(repo / "verification/final") == snapshot(run)
    assert "checkip.amazonaws.com" in r.stderr


# F2-1, F3-1: what is scanned and promoted is the staged copy, so the copy itself must be a flat,
# finished, passing run of regular files with the run's skip count. An entry that reaches the run after it
# was checked (here: added by a cp wrapper just before the copy) fails the promotion at the copy's check,
# before gitleaks runs. Every injected file is clean, so only that check (not gitleaks) can stop it.
def fake_cp(tmp_path, kind):
    bin_dir = tmp_path / "fakecp"
    bin_dir.mkdir()
    clean = tmp_path / "clean.txt"
    clean.write_text("# C1 - nothing secret here\nok\n")
    for name, text in {"failed.md": FAILED_SUMMARY, "skipped.md": SKIPPED_SUMMARY, "passed.md": SUMMARY,
                       "one-skip.md": summary_of("PASS", "PASS", "SKIP")}.items():
        (tmp_path / name).write_text(text)
    add = {
        "none": ":",
        "dir": f'mkdir "$run/03-C1.txt" && cp {clean} "$run/03-C1.txt/04-x.txt"',
        "symlink": f'ln -s {clean} "$run/03-C1.txt"',
        "fifo": 'mkfifo "$run/03-C1.txt"',
        "badname": f'cp {clean} "$run/03-C1.png"',
        "no-summary": 'rm "$run/SUMMARY.md"',
        "failed": f'cp {tmp_path / "failed.md"} "$run/SUMMARY.md"',
        "skip-added": f'cp {tmp_path / "skipped.md"} "$run/SUMMARY.md"',
        "skip-removed": f'cp {tmp_path / "passed.md"} "$run/SUMMARY.md"',
        "skip-changed": f'cp {tmp_path / "one-skip.md"} "$run/SUMMARY.md"',
    }[kind]
    real_cp = shutil.which("cp")
    # promote.sh runs `cp -R -- RUN COPY`; the wrapper changes RUN first. Other cp calls pass through.
    (bin_dir / "cp").write_text(
        "#!/bin/sh\n"
        f'if [ "$1" = -R ] && [ "$2" = -- ]; then run="$3"; cp() {{ {real_cp} "$@"; }}; {add}; fi\n'
        f'exec {real_cp} "$@"\n')
    (bin_dir / "cp").chmod(0o755)
    return {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}


COPY_OF = f"the copy of verification/{TS}"
NOT_WRITTEN = f"{COPY_OF} holds entries verify.sh does not write (only regular files named SUMMARY.md, " \
              "NN-<ID>.txt or <name>.log):"


@pytest.mark.parametrize("kind, summary, error", [
    ("dir", SUMMARY, f"{NOT_WRITTEN} 03-C1.txt"),
    ("symlink", SUMMARY, f"{NOT_WRITTEN} 03-C1.txt"),
    ("fifo", SUMMARY, f"{NOT_WRITTEN} 03-C1.txt"),
    ("badname", SUMMARY, f"{NOT_WRITTEN} 03-C1.png"),
    ("no-summary", SUMMARY, f"{COPY_OF} has no SUMMARY.md"),
    ("failed", SUMMARY, f"{COPY_OF} has 2 failed check(s)"),
    # PROMOTE_ALLOW_SKIP=1 lets both pass their check; the copy must still have the run's skip count.
    ("skip-added", SUMMARY, f"{COPY_OF} has 2 skipped check(s), the run had 0"),
    ("skip-removed", SKIPPED_SUMMARY, f"{COPY_OF} has 0 skipped check(s), the run had 2"),
    ("skip-changed", SKIPPED_SUMMARY, f"{COPY_OF} has 1 skipped check(s), the run had 2"),
], ids=["dir", "symlink", "fifo", "badname", "no-summary", "failed", "skip-added", "skip-removed",
        "skip-changed"])
def test_an_entry_that_reaches_the_run_after_its_check_fails_the_promotion(repo, tmp_path, kind, summary, error):
    before = previous_final(repo)
    make_run(repo, summary=summary)
    r = promote(repo, "verify", f"verification/{TS}", env={**fake_cp(tmp_path, kind), "PROMOTE_ALLOW_SKIP": "1"})
    assert r.returncode == 1, (r.returncode, r.stderr)
    assert snapshot(repo / "verification/final") == before
    assert verification_entries(repo) == [".gitkeep", TS, "final"], "the copy is removed"
    assert error in r.stderr, r.stderr
    assert "scanning each file" not in r.stderr and "gitleaks:" not in r.stderr, "stopped before the scan"


@pytest.mark.parametrize("summary", [SUMMARY, SKIPPED_SUMMARY], ids=["passed", "skipped"])
def test_the_same_cp_wrapper_injecting_nothing_promotes(repo, tmp_path, summary):
    # Positive control of the test above.
    run = make_run(repo, summary=summary)
    r = promote(repo, "verify", f"verification/{TS}", env={**fake_cp(tmp_path, "none"), "PROMOTE_ALLOW_SKIP": "1"})
    assert r.returncode == 0, r.stderr
    got, want = snapshot(repo / "verification/final"), snapshot(run)
    assert got.pop("SUMMARY.md").startswith(want.pop("SUMMARY.md")) and got == want


def fake_gitleaks(tmp_path, output):
    """A PATH dir whose gitleaks reads stdin, logs OUTPUT and exits 0."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (tmp_path / "gitleaks.out").write_text(output)
    (bin_dir / "gitleaks").write_text(f"#!/bin/sh\ncat >/dev/null\ncat {tmp_path / 'gitleaks.out'} >&2\nexit 0\n")
    (bin_dir / "gitleaks").chmod(0o755)
    return {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}


@pytest.mark.parametrize("output", [
    # What gitleaks stdin logs for a directory on stdin; it still exits 0 (measured, gitleaks 8.x).
    '6:00AM ERR error="could not read file: read /dev/stdin: is a directory"\n'
    "6:00AM INF scanned ~0 bytes (0) in 35ms\n6:00AM INF no leaks found\n",
    "6:00AM WRN something unexpected\n6:00AM INF no leaks found\n",
    "",                                                                   # no "no leaks found" at all
    "6:00AM INF scanned ~0 bytes (0) in 35ms\n",
], ids=["read-error", "warning", "silent", "no-verdict"])
def test_the_scan_fails_closed_on_anything_but_a_clean_report(repo, tmp_path, output):
    before = previous_final(repo)
    make_run(repo)
    r = promote(repo, "verify", f"verification/{TS}", env=fake_gitleaks(tmp_path, output))
    assert r.returncode == 1, r.stderr
    assert snapshot(repo / "verification/final") == before
    assert verification_entries(repo) == [".gitkeep", TS, "final"]


def test_a_clean_report_from_the_fake_scanner_promotes(repo, tmp_path):
    # The positive control of the test above: the same fake with gitleaks' clean output promotes.
    run = make_run(repo)
    env = fake_gitleaks(tmp_path, "6:00AM INF scanned ~6 bytes (6 bytes) in 37ms\n6:00AM INF no leaks found\n")
    r = promote(repo, "verify", f"verification/{TS}", env=env)
    assert r.returncode == 0, r.stderr
    assert snapshot(repo / "verification/final") == snapshot(run)


# S0-2/C0-2/X0-4, F2-3: replacing final survives a failed rename, Ctrl-C, and a second promotion. The fake
# mv blocks until the test releases it, so no timing window decides the outcome.

def fake_mv(tmp_path, mode):
    """A PATH dir whose mv fails ("fail") or blocks ("block") when renaming the scanned copy (step 2), or
    blocks when moving the old final aside ("block-old"). On reaching that point it creates the marker
    file; a blocked mv goes on once the release file exists (and fails after about 60 s, so nothing hangs)."""
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir()
    marker, release = tmp_path / "mv-reached", tmp_path / "mv-release"
    src = "*/.promote-*/final" if mode in ("fail", "block") else "*/verification/final"
    action = "exit 1" if mode == "fail" else (
        f'i=0; while [ ! -e "{release}" ]; do i=$((i + 1)); [ "$i" -le 1200 ] || exit 1; sleep 0.05; done')
    (bin_dir / "mv").write_text(
        "#!/bin/sh\n"
        f'case "$2" in {src}) : >"{marker}"; {action} ;; esac\n'
        f'exec {shutil.which("mv")} "$@"\n')
    (bin_dir / "mv").chmod(0o755)
    return {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}, marker, release


def wait_for(path, proc, seconds=60):
    deadline = time.monotonic() + seconds
    while not path.exists():
        assert proc.poll() is None, proc.communicate()
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.05)


def test_a_failed_rename_restores_the_previous_final(repo, tmp_path):
    before = previous_final(repo)
    make_run(repo)
    env, _, _ = fake_mv(tmp_path, "fail")
    r = promote(repo, "verify", f"verification/{TS}", env=env)
    assert r.returncode == 1, r.stderr
    assert snapshot(repo / "verification/final") == before
    assert verification_entries(repo) == [".gitkeep", TS, "final"]


@pytest.mark.parametrize("mode", ["block-old", "block"])
def test_ctrl_c_while_replacing_final_leaves_a_whole_final(repo, tmp_path, mode):
    previous_final(repo)
    run = make_run(repo)
    env, marker, release = fake_mv(tmp_path, mode)
    proc = subprocess.Popen(["bash", str(repo / "scripts/promote.sh"), "verify", f"verification/{TS}"],
                            cwd=repo, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            start_new_session=True)
    try:
        wait_for(marker, proc)                             # blocked inside the swap
        os.killpg(proc.pid, signal.SIGINT)                 # Ctrl-C reaches the whole foreground group
    finally:
        release.touch()                                    # only now may the swap go on
    _, err = proc.communicate(timeout=60)
    assert snapshot(repo / "verification/final") == snapshot(run), ("the swap ran to completion", err)
    assert verification_entries(repo) == [".gitkeep", TS, "final"]


def test_a_second_promotion_is_refused_while_the_first_holds_the_lock(repo, tmp_path):
    previous_final(repo)
    run = make_run(repo)
    other = make_run(repo, name="2026-10-21T1500-0700")
    env, marker, release = fake_mv(tmp_path, "block")
    first = subprocess.Popen(["bash", str(repo / "scripts/promote.sh"), "verify", f"verification/{TS}"],
                             cwd=repo, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        wait_for(marker, first)                            # the first holds the lock, blocked inside the swap
        second = promote(repo, "verify", f"verification/{other.name}")
    finally:
        release.touch()
    _, err = first.communicate(timeout=60)
    assert first.returncode == 0, err
    assert second.returncode == 1 and "another promotion" in second.stderr, second.stderr
    assert snapshot(repo / "verification/final") == snapshot(run)
    assert not (repo / "verification/final/final").exists()
    assert verification_entries(repo) == [".gitkeep", TS, other.name, "final"]


def test_a_stale_lock_is_reported_not_broken(repo):
    before = previous_final(repo)
    make_run(repo)
    (repo / "verification/.promote.lock").write_text("999999999\n")
    r = promote(repo, "verify", f"verification/{TS}")
    assert r.returncode == 1 and ".promote.lock" in r.stderr, r.stderr
    assert snapshot(repo / "verification/final") == before
    assert (repo / "verification/.promote.lock").read_text() == "999999999\n", "not ours to remove"
