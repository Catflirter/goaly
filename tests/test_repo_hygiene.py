"""Repo-level guards: what .gitignore keeps out, that `make secrets` scans tracked content, and where tf-test
keeps its offline Terraform data."""
import os, shutil, subprocess
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]

def ignored(path):
    proc = subprocess.run(["git", "check-ignore", "-q", "--no-index", path], cwd=ROOT)
    assert proc.returncode in (0, 1), proc
    return proc.returncode == 0

@pytest.mark.parametrize("path", [
    ".env", ".env.local", ".env.bak", "kubeconfig", "kubeconfig.yaml", "x.tfvars",
    "infra/base/terraform.tfstate", "verification/run1.txt",
    # Task 24a: every verify run but final/, and promote.sh's staging copy, stay out of git.
    "verification/2026-10-21T1430-0700/SUMMARY.md", "verification/2026-10-21T1430-0700/01-A1.txt",
    "verification/.promote-a1B2c3/final/SUMMARY.md", "verification/.promote.lock",
    # The workflow makes .venv a symlink to a shared venv; ".venv/" would match only a directory.
    ".venv"])
def test_secret_and_state_files_are_ignored(path):
    assert ignored(path)

@pytest.mark.parametrize("path", [
    ".env.example", "scripts/kubeconfig.sh", "x.auto.tfvars.json",
    "verification/.gitkeep", "verification/final/a.txt",
    "verification/final/SUMMARY.md", "verification/final/01-A1.txt", "verification/final/nested/02-B1.txt"])
def test_deliverable_files_are_not_ignored(path):
    assert not ignored(path)

# --- make secrets: scans committed history and uncommitted tracked changes --------------------

K3S_TOKEN = "K10" + "b" * 64 + "::server:" + "0123456789abcdef"

def git(repo, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                    "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false", *args],
                   cwd=repo, check=True, capture_output=True)

def make_secrets(repo):
    assert shutil.which("gitleaks"), "install gitleaks >= 8.19 (Task 1)"
    return subprocess.run(["make", "--no-print-directory", "-f", str(ROOT / "Makefile"), "secrets"],
                          cwd=repo, capture_output=True, text=True)

def assert_leak_found(proc):
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0 and "leaks found" in out, out

@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / "README").write_text("clean\n")
    git(tmp_path, "add", "README")
    git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path

def test_make_secrets_passes_on_clean_repo(repo):
    proc = make_secrets(repo)
    assert proc.returncode == 0, proc.stdout + proc.stderr

def test_make_secrets_flags_committed_token(repo):
    (repo / "leak.txt").write_text(K3S_TOKEN + "\n")
    git(repo, "add", "leak.txt")
    git(repo, "commit", "-q", "-m", "leak")
    assert_leak_found(make_secrets(repo))

def test_make_secrets_flags_token_removed_in_later_commit(repo):
    (repo / "leak.txt").write_text(K3S_TOKEN + "\n")
    git(repo, "add", "leak.txt")
    git(repo, "commit", "-q", "-m", "leak")
    git(repo, "rm", "-q", "leak.txt")
    git(repo, "commit", "-q", "-m", "remove leak")
    assert_leak_found(make_secrets(repo))

def test_make_secrets_flags_unstaged_change_to_tracked_file(repo):
    (repo / "README").write_text("clean\n" + K3S_TOKEN + "\n")
    assert_leak_found(make_secrets(repo))

def test_make_secrets_flags_staged_change(repo):
    (repo / "new.txt").write_text(K3S_TOKEN + "\n")
    git(repo, "add", "new.txt")
    assert_leak_found(make_secrets(repo))

# --- make tf-test: the offline init never touches infra/<stack>/.terraform (B-1) ---------------

def test_tf_test_keeps_its_offline_init_out_of_the_stack_directories(tmp_path):
    # A -backend=false init in infra/<stack> left .terraform/providers without a backend, which gpuctl took for an
    # initialised stack. Every terraform call of tf-test must use a TF_DATA_DIR under .run/tf-test/<stack>.
    shutil.copy(ROOT / "Makefile", tmp_path / "Makefile")
    for stack in ("base", "workers"):
        (tmp_path / "infra" / stack).mkdir(parents=True)
    (tmp_path / "infra/workers/tests").mkdir()
    bin_dir, log = tmp_path / "bin", tmp_path / "terraform.log"
    bin_dir.mkdir()
    (bin_dir / "terraform").write_text(f'#!/bin/bash\nprintf "%s %s\\n" "${{TF_DATA_DIR-unset}}" "$*" >> {log}\n')
    (bin_dir / "terraform").chmod(0o755)
    proc = subprocess.run(["make", "--no-print-directory", "-f", str(tmp_path / "Makefile"), "tf-test"],
                          cwd=tmp_path, capture_output=True, text=True,
                          env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    calls = [line.split(" ", 2) for line in log.read_text().splitlines()]
    assert [c[2].split()[0] for c in calls] == ["init", "validate", "init", "validate", "test"]
    for data_dir, chdir, _ in calls:
        stack = chdir.removeprefix(f"-chdir={tmp_path}/infra/")
        assert data_dir == f"{tmp_path}/.run/tf-test/{stack}", calls
