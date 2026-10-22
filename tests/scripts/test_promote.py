"""scripts/promote.sh verify and `make promote-verify` (Task 24a, Ruling R44): copy one verify run to
verification/final/ and scan the copy with the repo's gitleaks rules before it replaces the old one.

Fake secrets are assembled from pieces so this file itself never matches the gitleaks rules (see
tests/test_gitleaks_rules.py); the files the script sees at run time hold the full shapes.
"""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TS = "2026-10-21T1430-0700"
K3S_TOKEN = "K10" + "c" * 64 + "::server:" + "0123456789abcdef"
FAKE_LAMBDA_KEY = "secret" + "_demo_" + "0123456789abcdef" * 2 + "." + "Q7xZ" * 8
SUMMARY = ("# Verification run\n\nStarted: 2026-10-21T14:30:05-07:00 (America/Los_Angeles)\n\n"
           "| # | ID | Check | Result |\n|---|---|---|---|\n| 1 | A1 | nodes | PASS |\n\n"
           "**Result:** 1 passed, 0 failed, 0 skipped\n\n**Secret scan:** gitleaks found no leaks\n")


@pytest.fixture
def repo(tmp_path):
    """A throwaway repo root holding copies of promote.sh, common.sh, the gitleaks config and the Makefile."""
    root = tmp_path / "repo"
    (root / "scripts/lib").mkdir(parents=True)
    for rel in ("scripts/promote.sh", "scripts/lib/common.sh", ".gitleaks.toml", "Makefile"):
        shutil.copy2(ROOT / rel, root / rel)
    (root / "verification").mkdir()
    (root / "verification/.gitkeep").write_text("")
    return root.resolve()


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
                          cwd=cwd or repo, env=env or os.environ.copy())


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
    run = make_run(repo, files={"01-A1.txt": "a\n", "02-B1.txt": "b\n", "nested/03-C1.txt": "c\n"})
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
    v = subprocess.run(["bash", str(repo / "scripts/verify.sh")], capture_output=True, text=True, cwd=repo,
                       env={k: val for k, val in os.environ.items() if k != "RESILIENCE"})
    assert v.returncode == 0, v.stderr
    (run,) = [p for p in (repo / "verification").iterdir() if p.is_dir()]
    r = promote(repo, "verify", f"verification/{run.name}")
    assert r.returncode == 0, r.stderr
    assert snapshot(repo / "verification/final") == snapshot(run)
    assert "0 passed, 0 failed, 0 skipped" in r.stderr


# --- secret scan --------------------------------------------------------------------------------

# Both shapes are the repo's own rules in .gitleaks.toml (not gitleaks' defaults), so a pass here also
# proves the scan used the repo config.
@pytest.mark.parametrize("secret", [K3S_TOKEN, "Authorization: Bearer " + FAKE_LAMBDA_KEY])
def test_a_leak_fails_and_leaves_no_final(repo, secret):
    make_run(repo, files={"01-A1.txt": "fine\n", "nested/02-B1.txt": "oops " + secret + "\n"})
    r = promote(repo, "verify", f"verification/{TS}")
    assert r.returncode == 1, r.stderr
    assert not (repo / "verification/final").exists()
    assert verification_entries(repo) == [".gitkeep", TS], "the scanned copy is removed"
    out = r.stdout + r.stderr
    assert secret.split()[-1] not in out, "gitleaks output is redacted"
    assert "leak" in out and "nested/02-B1.txt" in out


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
