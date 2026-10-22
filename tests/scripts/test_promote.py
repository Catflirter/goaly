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
DEMO_TS = "demo-2026-10-21T1600-0700"
K3S_TOKEN = "K10" + "c" * 64 + "::server:" + "0123456789abcdef"
FAKE_LAMBDA_KEY = "secret" + "_demo_" + "0123456789abcdef" * 2 + "." + "Q7xZ" * 8
SUMMARY = ("# Verification run\n\nStarted: 2026-10-21T14:30:05-07:00 (America/Los_Angeles)\n\n"
           "| # | ID | Check | Result |\n|---|---|---|---|\n| 1 | A1 | nodes | PASS |\n\n"
           "**Result:** 1 passed, 0 failed, 0 skipped\n\n**Secret scan:** gitleaks found no leaks\n")
# The caller's environment without GCP_PROJECT: an operator's shell may export one, and promote.sh checks the copy
# for it (B1-1). A test that wants a project id passes it, or writes the repo copy's .env.
BASE_ENV = {k: v for k, v in os.environ.items() if k != "GCP_PROJECT"}


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
                          cwd=cwd or repo, env=env or dict(BASE_ENV), timeout=120)


def make_target(repo, *args):
    return subprocess.run(["make", "--no-print-directory", "-f", str(repo / "Makefile"), "promote-verify", *args],
                          capture_output=True, text=True, cwd=repo, env=dict(BASE_ENV))


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
    # The tests above use a hand-made run; this one is what the real harness writes, with one stub check (A1,
    # always passes) registered in place of the real ones.
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
    r = promote(repo, "verify", f"verification/{TS}", env={**BASE_ENV, "PATH": f"{bin_dir}:{os.environ['PATH']}"})
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
    r = promote(repo, "verify", f"verification/{TS}", env={**BASE_ENV, "PATH": str(bin_dir)})
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
                                  ["demo"], ["demo", f"verification/{DEMO_TS}", "extra"]])
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
        r = promote(repo, "verify", f"verification/{TS}", env={**BASE_ENV, **opt_in})
    assert_refused(repo, r, before, "failed check(s)")


@pytest.mark.parametrize("allow", [None, "0", "yes", ""])
def test_refuses_a_run_with_skipped_checks(repo, allow):
    before = previous_final(repo)
    make_run(repo, summary=SKIPPED_SUMMARY)
    env = {k: v for k, v in BASE_ENV.items() if k != "PROMOTE_ALLOW_SKIP"}
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
        r = promote(repo, "verify", f"verification/{TS}", env={**BASE_ENV, "PROMOTE_ALLOW_SKIP": "1"})
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
    r = promote(repo, "verify", f"verification/{TS}", env={**BASE_ENV, "PROMOTE_ALLOW_SKIP": "1"})
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
        r = promote(repo, "verify", f"verification/{TS}", env={**BASE_ENV, "OPERATOR_IP": OPERATOR_IP})
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
    r = promote(repo, "verify", f"verification/{TS}", env={**BASE_ENV, "OPERATOR_IP": ip})
    assert r.returncode == 1, r.stderr
    assert snapshot(repo / "verification/final") == before
    assert "06-B6.txt" in r.stderr and "01-A1.txt" not in r.stderr


def test_the_operator_ip_is_found_after_a_byte_that_is_not_utf8(repo):
    # In a UTF-8 locale grep's [^0-9] does not match an invalid byte such as \xff, so the address
    # right after it would be missed; promote.sh runs its grep under LC_ALL=C for this reason.
    before = previous_final(repo)
    run = make_run(repo, files={"01-A1.txt": "fine\n", "06-B6.txt": "placeholder\n"})
    (run / "06-B6.txt").write_bytes(b"x\xff73.15.40.4\n")
    env = {**BASE_ENV, "LC_ALL": "C.UTF-8", "LANG": "C.UTF-8", "OPERATOR_IP": "73.15.40.4"}
    r = promote(repo, "verify", f"verification/{TS}", env=env)
    assert r.returncode == 1, r.stderr
    assert snapshot(repo / "verification/final") == before
    assert "06-B6.txt" in r.stderr and "OPERATOR_IP" in r.stderr


@pytest.mark.parametrize("ip, line", [
    ("73.15.40.4", "fw-ssh-operator  73.15.40.40/32\n"),     # a longer last octet
    ("73.15.40.4", "peer 73.15.40.41\n"),
    ("3.15.40.4", "peer 73.15.40.4\n"),                      # a longer first octet
    ("73.15.40.4", "version 173.15.40.49\n"),
    ("73.15.40.4", "id 73x15-40/4\n"),                       # a dot matches only a dot
])
def test_a_longer_address_is_not_the_operator_ip(repo, ip, line):
    run = make_run(repo, files={"01-A1.txt": "fine\n", "06-B6.txt": line})
    r = promote(repo, "verify", f"verification/{TS}", env={**BASE_ENV, "OPERATOR_IP": ip})
    assert r.returncode == 0, r.stderr
    assert snapshot(repo / "verification/final") == snapshot(run)


def test_success_reminds_the_operator_ip_check(repo):
    make_run(repo)
    r = promote(repo, "verify", f"verification/{TS}", env={**BASE_ENV, "OPERATOR_IP": OPERATOR_IP})
    assert r.returncode == 0, r.stderr
    assert "checkip.amazonaws.com" in r.stderr


# F2-4: OPERATOR_IP is read as gpuctl reads it (surrounding whitespace dropped) and must be one IPv4
# address; anything else would make the check match nothing.
@pytest.mark.parametrize("value", [f" {OPERATOR_IP}", f"{OPERATOR_IP}\n", f"\t{OPERATOR_IP}  "])
def test_the_operator_ip_is_checked_despite_surrounding_whitespace(repo, value):
    before = previous_final(repo)
    make_run(repo, files={"01-A1.txt": "fine\n", "06-B6.txt": f"fw-ssh-operator  {OPERATOR_IP}/32\n"})
    r = promote(repo, "verify", f"verification/{TS}", env={**BASE_ENV, "OPERATOR_IP": value})
    assert r.returncode == 1, r.stderr
    assert snapshot(repo / "verification/final") == before
    assert "06-B6.txt" in r.stderr


@pytest.mark.parametrize("value", ["73.15.40", "73.15.40.40.1", "73.15.40.400", "073.15.40.40", "not-an-ip",
                                   f"{OPERATOR_IP}/32", f"{OPERATOR_IP} 10.0.0.1", "::1", "73.15.40.4x"])
def test_refuses_an_operator_ip_that_is_not_one_ipv4_address(repo, value):
    before = previous_final(repo)
    make_run(repo, files={"01-A1.txt": "fine\n", "06-B6.txt": f"fw-ssh-operator  {OPERATOR_IP}/32\n"})
    r = promote(repo, "verify", f"verification/{TS}", env={**BASE_ENV, "OPERATOR_IP": value})
    assert_refused(repo, r, before, "OPERATOR_IP")


@pytest.mark.parametrize("value", ["", "  ", "\n"])
def test_a_blank_operator_ip_is_unset(repo, value):
    # As in gpuctl (detect_operator_ip): blank means "not given"; the success message then prints the
    # Task 24 Step 4 grep for the operator to run.
    run = make_run(repo)
    r = promote(repo, "verify", f"verification/{TS}", env={**BASE_ENV, "OPERATOR_IP": value})
    assert r.returncode == 0, r.stderr
    assert snapshot(repo / "verification/final") == snapshot(run)
    assert "checkip.amazonaws.com" in r.stderr


# B1-1: the GCP project id must not reach final either. verify evidence carries it in the GCE FQDN hostname
# (k3s-cp.<zone>.c.<project-id>.internal, A2's kubectl describe node) and in the state bucket's name. Both
# GCP_PROJECT from the environment and the one the repo's .env gives (load_env) are looked for (X0-1); neither is
# ever printed.
PROJECT_ID = "acme-gpu-lab-7731"
SKIPPED = "WARNING: no GCP project id (GCP_PROJECT in the environment or .env): the project-id check was skipped"


def write_dotenv(repo, text, mode=0o600):
    env = repo / ".env"
    env.write_text(text)
    env.chmod(mode)


def assert_no_id(r, *ids):
    out = (r.stdout + r.stderr).lower()
    for project_id in ids or (PROJECT_ID,):
        assert project_id.lower() not in out, "the project id is never printed"


def assert_failed_on(repo, r, before, *names):
    assert r.returncode == 1, r.stderr
    assert snapshot(repo / "verification/final") == before, "the previous final is untouched"
    assert verification_entries(repo) == [".gitkeep", TS, "final"], "the scanned copy is removed"
    assert "GCP_PROJECT" in r.stderr and "verification/final was not changed" in r.stderr
    for name in names:
        assert name in r.stderr
    assert_no_id(r)


@pytest.mark.parametrize("via_make", [False, True])
@pytest.mark.parametrize("line", [
    f"Hostname:  k3s-cp.us-west1-b.c.{PROJECT_ID}.internal\n",       # the GCE FQDN in kubectl describe node
    f"state in gs://{PROJECT_ID}-k3s-tfstate/base\n",                # the state bucket
    f"project: {PROJECT_ID.upper()}\n",                               # any case
    f"x{PROJECT_ID}y",                                                # anywhere, even inside a word, no newline
], ids=["fqdn", "bucket", "uppercase", "substring"])
def test_the_project_id_fails_and_keeps_the_previous_final(repo, line, via_make):
    before = previous_final(repo)
    make_run(repo, files={"01-A1.txt": "fine\n", "02-A2.txt": line, "03-B1.txt": "fine too\n"})
    if via_make:
        r = subprocess.run(["make", "--no-print-directory", "-f", str(repo / "Makefile"), "promote-verify",
                            f"RUN=verification/{TS}"], capture_output=True, text=True, cwd=repo,
                           env={**BASE_ENV, "GCP_PROJECT": PROJECT_ID})
        assert r.returncode == 2, r.stderr  # make reports a failed recipe as 2
        r.returncode = 1
    else:
        r = promote(repo, "verify", f"verification/{TS}", env={**BASE_ENV, "GCP_PROJECT": PROJECT_ID})
    assert_failed_on(repo, r, before, "02-A2.txt")
    assert "01-A1.txt" not in r.stderr and "03-B1.txt" not in r.stderr, "only the files that hold it are named"


def test_every_file_that_holds_the_project_id_is_named(repo):
    before = previous_final(repo)
    make_run(repo, files={"01-A1.txt": f"node k3s-cp.us-west1-b.c.{PROJECT_ID}.internal\n", "02-A2.txt": "fine\n",
                          "autoscaler.log": f"bucket {PROJECT_ID}-k3s-tfstate\n"})
    r = promote(repo, "verify", f"verification/{TS}", env={**BASE_ENV, "GCP_PROJECT": PROJECT_ID})
    assert_failed_on(repo, r, before, "01-A1.txt", "autoscaler.log")
    assert "02-A2.txt" not in r.stderr


def test_the_project_id_is_found_after_a_byte_that_is_not_utf8(repo):
    before = previous_final(repo)
    run = make_run(repo, files={"01-A1.txt": "fine\n", "02-A2.txt": "placeholder\n"})
    (run / "02-A2.txt").write_bytes(b"x\xff" + PROJECT_ID.upper().encode() + b"\n")
    env = {**BASE_ENV, "LC_ALL": "C.UTF-8", "LANG": "C.UTF-8", "GCP_PROJECT": PROJECT_ID}
    assert_failed_on(repo, promote(repo, "verify", f"verification/{TS}", env=env), before, "02-A2.txt")


def test_the_project_id_is_read_from_the_dotenv(repo):
    before = previous_final(repo)
    write_dotenv(repo, f"GCP_PROJECT={PROJECT_ID}\nLAMBDA_API_KEY=x\n")
    make_run(repo, files={"01-A1.txt": "fine\n", "02-A2.txt": f"k3s-cp.us-west1-b.c.{PROJECT_ID}.internal\n"})
    r = promote(repo, "verify", f"verification/{TS}")
    assert_failed_on(repo, r, before, "02-A2.txt")


OTHER_ID = "other-project-42"


@pytest.mark.parametrize("leaked", [PROJECT_ID, OTHER_ID], ids=["environment-id", "dotenv-id"])
def test_both_the_environment_and_the_dotenv_ids_are_looked_for(repo, leaked):
    """X0-1: verify.sh's load_env sources .env over the caller's environment, so the evidence holds .env's id even
    when the operator's shell exports another GCP_PROJECT. Neither id may reach final, whichever the copy holds."""
    write_dotenv(repo, f"GCP_PROJECT={OTHER_ID}\n")
    before = previous_final(repo)
    make_run(repo, files={"01-A1.txt": "fine\n", "02-A2.txt": f"k3s-cp.us-west1-b.c.{leaked}.internal\n"})
    r = promote(repo, "verify", f"verification/{TS}", env={**BASE_ENV, "GCP_PROJECT": PROJECT_ID})
    assert_failed_on(repo, r, before, "02-A2.txt")
    assert "01-A1.txt" not in r.stderr
    assert_no_id(r, PROJECT_ID, OTHER_ID)


def test_with_both_ids_a_clean_copy_promotes(repo):
    write_dotenv(repo, f"GCP_PROJECT={OTHER_ID}\n")
    run = make_run(repo, files={"01-A1.txt": "k3s-cp.us-west1-b.c.another-project-1.internal\n"})
    r = promote(repo, "verify", f"verification/{TS}", env={**BASE_ENV, "GCP_PROJECT": PROJECT_ID})
    assert r.returncode == 0, r.stderr
    assert snapshot(repo / "verification/final") == snapshot(run)
    assert "WARNING" not in r.stderr
    assert_no_id(r, PROJECT_ID, OTHER_ID)


@pytest.mark.parametrize("bad", ["env", "dotenv"])
def test_refuses_when_either_id_is_not_a_project_id(repo, bad):
    """Both ids are looked for, so both must be project ids: one the check cannot match must not pass silently."""
    value = "Not_A_Project-1"
    write_dotenv(repo, f"GCP_PROJECT={value if bad == 'dotenv' else OTHER_ID}\n")
    before = previous_final(repo)
    make_run(repo)
    r = promote(repo, "verify", f"verification/{TS}",
                env={**BASE_ENV, "GCP_PROJECT": value if bad == "env" else PROJECT_ID})
    assert_refused(repo, r, before, "GCP_PROJECT", "not a GCP project id")
    assert_no_id(r, value, PROJECT_ID, OTHER_ID)


@pytest.mark.parametrize("source", ["env", "dotenv", "both"])
def test_the_project_id_never_reaches_a_command_line(repo, tmp_path, source):
    """X0-2: argv is visible to every local user (ps, /proc/PID/cmdline) and to execve audit logs, so grep gets the
    id(s) as a pattern file, never as an argument. A grep wrapper records every argument list it is given."""
    bin_dir = tmp_path / "grepbin"
    bin_dir.mkdir()
    record = tmp_path / "grep.argv"
    (bin_dir / "grep").write_text(f'#!/bin/sh\nprintf \'%s\\n\' "$*" >> "{record}"\nexec "{shutil.which("grep")}" "$@"\n')
    (bin_dir / "grep").chmod(0o755)
    env = {**BASE_ENV, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    if source in ("env", "both"):
        env["GCP_PROJECT"] = PROJECT_ID
    if source in ("dotenv", "both"):
        write_dotenv(repo, f"GCP_PROJECT={OTHER_ID if source == 'both' else PROJECT_ID}\n")
    before = previous_final(repo)
    make_run(repo, files={"01-A1.txt": "fine\n", "02-A2.txt": f"k3s-cp.us-west1-b.c.{PROJECT_ID}.internal\n"})
    r = promote(repo, "verify", f"verification/{TS}", env=env)
    assert_failed_on(repo, r, before, "02-A2.txt")
    text = record.read_text().lower()
    assert PROJECT_ID not in text and OTHER_ID not in text
    argvs = record.read_text().splitlines()
    assert any(" -f " in f" {a} " for a in argvs if "-liF" in a), "the project-id grep ran, with a pattern file"


@pytest.mark.parametrize("value", ["Acme-Gpu-Lab-7731", "short", "1starts-with-digit", "ends-with-hyphen-",
                                   "has space 123", "x" * 31, "acme-gpu;rm -rf x", "acme-gpu-lab-7731 "])
@pytest.mark.parametrize("source", ["env", "dotenv"])
def test_refuses_a_project_id_that_is_not_one(repo, value, source):
    before = previous_final(repo)
    make_run(repo, files={"01-A1.txt": "fine\n", "02-A2.txt": f"{value}\n"})
    if source == "env":
        r = promote(repo, "verify", f"verification/{TS}", env={**BASE_ENV, "GCP_PROJECT": value})
    else:
        write_dotenv(repo, f"GCP_PROJECT='{value}'\n")
        r = promote(repo, "verify", f"verification/{TS}")
    assert_refused(repo, r, before, "GCP_PROJECT", "not a GCP project id")
    assert_no_id(r, value.strip())


def test_a_dotenv_that_is_not_private_is_refused_as_load_env_refuses_it(repo):
    before = previous_final(repo)
    write_dotenv(repo, f"GCP_PROJECT={PROJECT_ID}\n", mode=0o644)
    make_run(repo)
    r = promote(repo, "verify", f"verification/{TS}")
    assert_refused(repo, r, before, "chmod 600")
    assert_no_id(r)


@pytest.mark.parametrize("dotenv", [None, "LAMBDA_API_KEY=x\n", "GCP_PROJECT=\n"], ids=["no-dotenv", "unset", "empty"])
def test_without_a_project_id_the_check_is_skipped_with_one_warning(repo, dotenv):
    if dotenv is not None:
        write_dotenv(repo, dotenv)
    run = make_run(repo, files={"01-A1.txt": f"k3s-cp.us-west1-b.c.{PROJECT_ID}.internal\n"})
    r = promote(repo, "verify", f"verification/{TS}", env={**BASE_ENV, "GCP_PROJECT": ""})
    assert r.returncode == 0, r.stderr
    assert snapshot(repo / "verification/final") == snapshot(run)
    assert r.stderr.count(SKIPPED) == 1 and r.stderr.count("WARNING") == 1


def test_a_clean_copy_with_a_project_id_promotes_without_a_warning(repo):
    run = make_run(repo, files={"01-A1.txt": "k3s-cp.us-west1-b.c.another-project-1.internal\n"})
    r = promote(repo, "verify", f"verification/{TS}", env={**BASE_ENV, "GCP_PROJECT": PROJECT_ID})
    assert r.returncode == 0, r.stderr
    assert snapshot(repo / "verification/final") == snapshot(run)
    assert "WARNING" not in r.stderr
    assert_no_id(r)


def test_the_dotenv_secrets_stay_out_of_the_scan(repo, tmp_path):
    """.env is read in a subshell for GCP_PROJECT only: its other values (the Lambda API key) do not reach
    gitleaks or any other command promote.sh runs."""
    bin_dir = tmp_path / "envbin"
    bin_dir.mkdir()
    real = shutil.which("gitleaks")
    seen = tmp_path / "gitleaks.env"
    (bin_dir / "gitleaks").write_text(f'#!/bin/sh\nenv >> "{seen}"\nexec "{real}" "$@"\n')
    (bin_dir / "gitleaks").chmod(0o755)
    dotenv_key = "lambda-key-" + "from-dotenv"  # assembled: the file never holds a key=value secret shape
    write_dotenv(repo, f"GCP_PROJECT={PROJECT_ID}\nLAMBDA_API_KEY" + "=" + dotenv_key + "\n")
    make_run(repo)
    r = promote(repo, "verify", f"verification/{TS}",
                env={**BASE_ENV, "PATH": f"{bin_dir}:{os.environ['PATH']}"})
    assert r.returncode == 0, r.stderr
    text = seen.read_text()
    assert "GCP_PROJECT=" not in text and dotenv_key not in text


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
    return {**BASE_ENV, "PATH": f"{bin_dir}:{os.environ['PATH']}"}


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
    return {**BASE_ENV, "PATH": f"{bin_dir}:{os.environ['PATH']}"}


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
    return {**BASE_ENV, "PATH": f"{bin_dir}:{os.environ['PATH']}"}, marker, release


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


# --- promote.sh demo (Task 25) and verify keeping final/demo/ (progress.md note at line 322) -------

DEMO_FILES = {"ch1.txt": "--- [ch1.1] A1\n$ check_a1_nodes\nA1: 2 of 2 nodes Ready\n",
              "ch3.txt": "--- [ch3.1] B1\nmatrix ok\n",
              "timings.txt": "ch1 120s (commands 40s; 3 step(s), 0 failed, 0 skipped) Architecture\n"}


def make_demo(repo, name=DEMO_TS, files=None):
    """A verification/demo-<ts>/ run as scripts/demo.sh leaves it (0700 directory, 0600 files)."""
    run = repo / "verification" / name
    run.mkdir(mode=0o700)
    for rel, text in (DEMO_FILES if files is None else files).items():
        (run / rel).parent.mkdir(parents=True, exist_ok=True)
        (run / rel).write_text(text)
    for p in [run, *run.rglob("*")]:
        p.chmod(0o700 if p.is_dir() else 0o600)
    return run


def promote_demo(repo, arg=f"verification/{DEMO_TS}", env=None):
    return promote(repo, "demo", arg, env=env)


def test_promote_demo_copies_the_run_to_final_demo_and_keeps_the_rest_of_final(repo):
    before = previous_final(repo)
    run = make_demo(repo)
    recorded = snapshot(run)
    r = promote_demo(repo)
    assert r.returncode == 0, r.stderr
    final = repo / "verification/final"
    assert snapshot(final / "demo") == recorded
    assert {k: v for k, v in snapshot(final).items() if not k.startswith("demo")} == before
    assert snapshot(run) == recorded, "the run itself is left as it was"
    assert verification_entries(repo) == [".gitkeep", DEMO_TS, "final"], "no staging directory is left behind"
    assert "verification/final/demo" in r.stderr and "no leaks" in r.stderr


@pytest.mark.parametrize("arg", ["abs", f"verification/{DEMO_TS}/", f"./verification/{DEMO_TS}"])
def test_promote_demo_accepts_the_run_in_any_spelling(repo, arg):
    run = make_demo(repo)
    r = promote_demo(repo, str(run) if arg == "abs" else arg)
    assert r.returncode == 0, r.stderr
    assert snapshot(repo / "verification/final/demo") == snapshot(run)


def test_promote_demo_creates_final_when_there_is_none(repo):
    run = make_demo(repo)
    r = promote_demo(repo)
    assert r.returncode == 0, r.stderr
    assert snapshot(repo / "verification/final/demo") == snapshot(run)
    assert sorted(p.name for p in (repo / "verification/final").iterdir()) == ["demo"]


def test_promote_demo_replaces_the_previous_demo_instead_of_merging(repo):
    previous_final(repo)
    old = repo / "verification/final/demo"
    old.mkdir()
    (old / "ch8.txt").write_text("old recording\n")
    run = make_demo(repo)
    r = promote_demo(repo)
    assert r.returncode == 0, r.stderr
    assert snapshot(old) == snapshot(run) and not (old / "ch8.txt").exists()


@pytest.mark.parametrize("secret", [K3S_TOKEN, "Authorization: Bearer " + FAKE_LAMBDA_KEY])
def test_promote_demo_fails_on_a_leak_and_changes_nothing(repo, secret):
    before = previous_final(repo)
    make_demo(repo, files={**DEMO_FILES, "ch4.txt": "oops " + secret + "\n"})
    r = promote_demo(repo)
    assert r.returncode == 1, r.stderr
    assert snapshot(repo / "verification/final") == before
    assert verification_entries(repo) == [".gitkeep", DEMO_TS, "final"]
    assert "ch4.txt" in r.stderr
    assert secret.split()[-1] not in r.stdout + r.stderr


def test_promote_demo_fails_on_the_operator_ip(repo):
    before = previous_final(repo)
    make_demo(repo, files={**DEMO_FILES, "ch2.txt": f"fw-ssh-operator  {OPERATOR_IP}/32\n"})
    r = promote_demo(repo, env={**os.environ, "OPERATOR_IP": OPERATOR_IP})
    assert r.returncode == 1, r.stderr
    assert snapshot(repo / "verification/final") == before
    assert "ch2.txt" in r.stderr and "ch1.txt" not in r.stderr and "OPERATOR_IP" in r.stderr


@pytest.mark.parametrize("name", ["scratch", TS, "demo-2026-10-21T1430", "demo-final", "final"])
def test_promote_demo_refuses_a_directory_not_named_like_a_demo_run(repo, name):
    before = previous_final(repo)
    if name not in ("final", TS):
        make_demo(repo, name=name)
    if name == TS:
        make_run(repo)
    assert_refused(repo, promote_demo(repo, f"verification/{name}"), before)


def test_promote_demo_refuses_a_demo_run_outside_verification(repo, tmp_path):
    before = previous_final(repo)
    outside = tmp_path / DEMO_TS
    outside.mkdir()
    (outside / "ch1.txt").write_text("x\n")
    (outside / "timings.txt").write_text("x\n")
    assert_refused(repo, promote_demo(repo, str(outside)), before)


@pytest.mark.parametrize("name", ["ch9.txt", "ch0.txt", "notes.md", ".hidden", "ch1.txt.png", "SUMMARY.md",
                                  "sub/ch1.txt"])
def test_promote_demo_refuses_a_file_demo_sh_does_not_write(repo, name):
    before = previous_final(repo)
    make_demo(repo, files={**DEMO_FILES, name: "x\n"})
    assert_refused(repo, promote_demo(repo), before, name.split("/")[0])


def test_promote_demo_refuses_a_symlink_in_the_run(repo, tmp_path):
    before = previous_final(repo)
    run = make_demo(repo)
    (tmp_path / "x.txt").write_text("x\n")
    (run / "ch5.txt").symlink_to(tmp_path / "x.txt")
    assert_refused(repo, promote_demo(repo), before, "ch5.txt")


@pytest.mark.parametrize("files", [{}, {"timings.txt": "x\n"}, {"ch1.txt": "x\n"}],
                         ids=["empty", "no-chapter", "no-timings"])
def test_promote_demo_refuses_a_run_without_chapters_or_timings(repo, files):
    before = previous_final(repo)
    make_demo(repo, files=files)
    assert_refused(repo, promote_demo(repo), before)


def test_promote_demo_refuses_a_final_that_is_a_symlink(repo, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (repo / "verification/final").symlink_to(elsewhere)
    make_demo(repo)
    r = promote_demo(repo)
    assert r.returncode == 2, r.stderr
    assert list(elsewhere.iterdir()) == []


def test_a_failed_rename_restores_the_previous_demo(repo, tmp_path):
    previous_final(repo)
    old = repo / "verification/final/demo"
    old.mkdir()
    (old / "ch8.txt").write_text("old recording\n")
    before = snapshot(repo / "verification/final")
    make_demo(repo)
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir()
    (bin_dir / "mv").write_text("#!/bin/sh\n"
                                # mv -- SRC DST: $2 is the source; the scanned copy's rename fails
                                'case "$2" in */.promote-*/demo) exit 1 ;; esac\n'
                                f'exec {shutil.which("mv")} "$@"\n')
    (bin_dir / "mv").chmod(0o755)
    r = promote_demo(repo, env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"})
    assert r.returncode == 1, r.stderr
    assert snapshot(repo / "verification/final") == before
    assert verification_entries(repo) == [".gitkeep", DEMO_TS, "final"]


def test_make_promote_demo_promotes_the_run(repo):
    run = make_demo(repo)
    r = subprocess.run(["make", "--no-print-directory", "-f", str(repo / "Makefile"), "promote-demo",
                        f"RUN=verification/{DEMO_TS}"], capture_output=True, text=True, cwd=repo)
    assert r.returncode == 0, r.stderr
    assert snapshot(repo / "verification/final/demo") == snapshot(run)


def test_make_promote_demo_requires_run(repo):
    r = subprocess.run(["make", "--no-print-directory", "-f", str(repo / "Makefile"), "promote-demo"],
                       capture_output=True, text=True, cwd=repo)
    assert r.returncode == 2 and "RUN=verification/demo-" in r.stderr


def demo_in_final(repo):
    """A final/ with a promoted demo/ in it; returns the demo's snapshot."""
    previous_final(repo)
    make_demo(repo)
    assert promote_demo(repo).returncode == 0
    shutil.rmtree(repo / "verification" / DEMO_TS)
    return snapshot(repo / "verification/final/demo")


def test_a_verify_promotion_keeps_the_promoted_demo(repo):
    demo = demo_in_final(repo)
    run = make_run(repo)
    r = promote(repo, "verify", f"verification/{TS}")
    assert r.returncode == 0, r.stderr
    final = repo / "verification/final"
    assert snapshot(final / "demo") == demo
    assert {k: v for k, v in snapshot(final).items() if not k.startswith("demo")} == snapshot(run)
    assert not (final / "99-old.txt").exists(), "the rest of the old final is still replaced, not merged"
    assert verification_entries(repo) == [".gitkeep", TS, "final"]


def test_a_verify_promotion_without_a_demo_adds_none(repo):
    previous_final(repo)
    run = make_run(repo)
    assert promote(repo, "verify", f"verification/{TS}").returncode == 0
    assert snapshot(repo / "verification/final") == snapshot(run)


def test_a_demo_promoted_after_verify_stays_with_it(repo):
    run = make_run(repo)
    assert promote(repo, "verify", f"verification/{TS}").returncode == 0
    demo = make_demo(repo)
    assert promote_demo(repo).returncode == 0
    final = repo / "verification/final"
    assert snapshot(final / "demo") == snapshot(demo)
    assert {k: v for k, v in snapshot(final).items() if not k.startswith("demo")} == snapshot(run)


@pytest.mark.parametrize("change", ["leak", "bad-name", "symlink", "operator-ip", "no-chapter"])
def test_a_verify_promotion_refuses_a_kept_demo_it_cannot_vouch_for(repo, tmp_path, change):
    demo_in_final(repo)
    demo = repo / "verification/final/demo"
    if change == "leak":
        (demo / "ch4.txt").write_text(K3S_TOKEN + "\n")
    elif change == "bad-name":
        (demo / "notes.md").write_text("x\n")
    elif change == "symlink":
        (tmp_path / "x.txt").write_text("x\n")
        (demo / "ch4.txt").symlink_to(tmp_path / "x.txt")
    elif change == "operator-ip":
        (demo / "ch4.txt").write_text(f"rule {OPERATOR_IP}/32\n")
    else:
        for p in demo.glob("ch*.txt"):
            p.unlink()
    before = snapshot(repo / "verification/final")
    make_run(repo)
    r = promote(repo, "verify", f"verification/{TS}", env={**os.environ, "OPERATOR_IP": OPERATOR_IP})
    assert r.returncode == 1, r.stderr
    assert snapshot(repo / "verification/final") == before
    assert verification_entries(repo) == [".gitkeep", TS, "final"]
    assert "final/demo" in r.stderr


def test_a_verify_promotion_refuses_a_kept_demo_that_is_a_symlink(repo, tmp_path):
    previous_final(repo)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "ch1.txt").write_text("x\n")
    (repo / "verification/final/demo").symlink_to(elsewhere)
    before = snapshot(repo / "verification/final")
    make_run(repo)
    r = promote(repo, "verify", f"verification/{TS}")
    assert r.returncode == 1, r.stderr
    assert snapshot(repo / "verification/final") == before
    assert "final/demo" in r.stderr


def test_a_failed_rename_in_a_verify_promotion_keeps_the_demo(repo, tmp_path):
    demo = demo_in_final(repo)
    make_run(repo)
    env, _, _ = fake_mv(tmp_path, "fail")
    r = promote(repo, "verify", f"verification/{TS}", env=env)
    assert r.returncode == 1, r.stderr
    assert snapshot(repo / "verification/final/demo") == demo
    assert (repo / "verification/final/99-old.txt").exists()


# --- promote.sh demo: the GCP project id and failed chapters (Task 25 fix round 1) --------------

DEMO_PROJECT = "k3s-demo-123456"


def no_project_env(**extra):
    return {**{k: v for k, v in os.environ.items() if k not in ("GCP_PROJECT", "PROMOTE_ALLOW_FAIL")}, **extra}


@pytest.mark.parametrize("text", [f"state in gs://{DEMO_PROJECT}-k3s-tfstate/workers\n",
                                  f"k3s-worker@{DEMO_PROJECT.upper()}.iam.gserviceaccount.com\n"])
def test_promote_demo_fails_on_the_project_id_and_changes_nothing(repo, text):
    before = previous_final(repo)
    make_demo(repo, files={**DEMO_FILES, "ch6.txt": "--- [ch6.1] plan\n" + text})
    r = promote_demo(repo, env=no_project_env(GCP_PROJECT=DEMO_PROJECT))
    assert r.returncode == 1, r.stderr
    assert snapshot(repo / "verification/final") == before
    assert verification_entries(repo) == [".gitkeep", DEMO_TS, "final"]
    assert "ch6.txt" in r.stderr and "ch1.txt" not in r.stderr and "GCP_PROJECT" in r.stderr
    assert DEMO_PROJECT not in (r.stdout + r.stderr).lower()


def test_promote_demo_reads_the_project_id_from_env_file(repo):
    before = previous_final(repo)
    (repo / ".env").write_text(f"GCP_PROJECT={DEMO_PROJECT}\n")
    (repo / ".env").chmod(0o600)
    make_demo(repo, files={**DEMO_FILES, "ch8.txt": f"make down destroys, in GCP project {DEMO_PROJECT}:\n"})
    r = promote_demo(repo, env=no_project_env())
    assert r.returncode == 1, r.stderr
    assert snapshot(repo / "verification/final") == before
    assert "ch8.txt" in r.stderr


def test_promote_demo_without_a_project_id_warns_and_promotes(repo):
    run = make_demo(repo)
    r = promote_demo(repo, env=no_project_env())
    assert r.returncode == 0, r.stderr
    assert snapshot(repo / "verification/final/demo") == snapshot(run)
    assert "WARNING" in r.stderr and "project" in r.stderr


def test_promote_demo_with_a_masked_project_id_promotes(repo):
    run = make_demo(repo, files={**DEMO_FILES, "ch6.txt": "state in gs://<GCP_PROJECT>-k3s-tfstate\n"})
    r = promote_demo(repo, env=no_project_env(GCP_PROJECT=DEMO_PROJECT))
    assert r.returncode == 0, r.stderr
    assert snapshot(repo / "verification/final/demo") == snapshot(run)


def test_promote_demo_refuses_a_project_id_that_is_not_one(repo):
    before = previous_final(repo)
    make_demo(repo)
    r = promote_demo(repo, env=no_project_env(GCP_PROJECT="Not A Project"))
    assert r.returncode == 2, r.stderr
    assert snapshot(repo / "verification/final") == before


def test_a_verify_promotion_refuses_a_kept_demo_with_the_project_id(repo):
    demo_in_final(repo)
    (repo / "verification/final/demo/ch6.txt").write_text(f"gs://{DEMO_PROJECT}-k3s-tfstate\n")
    before = snapshot(repo / "verification/final")
    make_run(repo)
    r = promote(repo, "verify", f"verification/{TS}", env=no_project_env(GCP_PROJECT=DEMO_PROJECT))
    assert r.returncode == 1, r.stderr
    assert snapshot(repo / "verification/final") == before
    assert "final/demo" in r.stderr and "GCP_PROJECT" in r.stderr
    assert DEMO_PROJECT not in r.stdout + r.stderr


FAILED_TIMINGS = ("ch1 120s (commands 40s; 3 step(s), 0 failed, 0 skipped) Architecture\n"
                  "ch3 60s (commands 30s; 1 step(s), 1 failed, 0 skipped) Cross-node connectivity matrix (B1)\n"
                  "total 180s; chapters 1-6: 3m00s (target 25 min)\n")
INTERRUPTED_TIMINGS = ("ch1 120s (commands 40s; 3 step(s), 0 failed, 0 skipped) Architecture\n"
                       "ch3 interrupted after 12s\n")


@pytest.mark.parametrize("timings", [FAILED_TIMINGS, INTERRUPTED_TIMINGS], ids=["failed", "interrupted"])
def test_promote_demo_refuses_a_run_with_a_failed_chapter(repo, timings):
    before = previous_final(repo)
    make_demo(repo, files={**DEMO_FILES, "timings.txt": timings})
    r = promote_demo(repo, env=no_project_env())
    assert r.returncode == 2, r.stderr
    assert snapshot(repo / "verification/final") == before
    assert "ch3" in r.stderr and "ch1" not in r.stderr and "PROMOTE_ALLOW_FAIL=1" in r.stderr
    assert verification_entries(repo) == [".gitkeep", DEMO_TS, "final"]


def test_promote_demo_promotes_a_failed_chapter_with_promote_allow_fail(repo):
    run = make_demo(repo, files={**DEMO_FILES, "timings.txt": FAILED_TIMINGS})
    r = promote_demo(repo, env=no_project_env(PROMOTE_ALLOW_FAIL="1"))
    assert r.returncode == 0, r.stderr
    assert snapshot(repo / "verification/final/demo") == snapshot(run)
    assert "WARNING" in r.stderr and "ch3" in r.stderr


def test_promote_demo_takes_a_run_whose_chapters_all_passed(repo):
    timings = ("ch1 120s (commands 40s; 3 step(s), 0 failed, 1 skipped) Architecture\n"
               "total 120s; chapters 1-6: 2m00s (target 25 min)\n")
    run = make_demo(repo, files={**DEMO_FILES, "timings.txt": timings})
    r = promote_demo(repo, env=no_project_env())
    assert r.returncode == 0, r.stderr
    assert snapshot(repo / "verification/final/demo") == snapshot(run)
