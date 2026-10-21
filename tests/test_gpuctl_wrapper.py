"""scripts/lib/common.sh (load_env, die, log) and the scripts/gpuctl wrapper, run against a copy of the repo layout."""
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FAKE_PYTHON = """#!/usr/bin/env bash
printf 'PYTHONPATH=%s\\n' "$PYTHONPATH"
printf 'KEY=%s\\n' "${LAMBDA_API_KEY-unset}"
printf 'ARG=%s\\n' "$@"
"""


@pytest.fixture
def repo(tmp_path):
    """A throwaway repo root holding copies of the two scripts and a fake .venv/bin/python."""
    (tmp_path / "scripts/lib").mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/gpuctl", tmp_path / "scripts/gpuctl")
    shutil.copy2(ROOT / "scripts/lib/common.sh", tmp_path / "scripts/lib/common.sh")
    python = tmp_path / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text(FAKE_PYTHON)
    python.chmod(0o755)
    return tmp_path


def write_env(repo, text="LAMBDA_API_KEY=k-123\n", mode=0o600):
    env = repo / ".env"
    env.write_text(text)
    env.chmod(mode)
    return env


def run(cmd, repo, cwd=None):
    env = {k: v for k, v in os.environ.items() if k not in ("LAMBDA_API_KEY", "REPO_ROOT", "PYTHONPATH")}
    return subprocess.run(cmd, cwd=cwd or repo, env=env, capture_output=True, text=True)


def lib(repo, script):
    return run(["bash", "-c", f'source "{repo}/scripts/lib/common.sh"; {script}'], repo, cwd="/")


def test_wrapper_is_executable_bash():
    path = ROOT / "scripts/gpuctl"
    assert path.stat().st_mode & stat.S_IXUSR
    assert path.read_text().startswith("#!/usr/bin/env bash\n")


def test_repo_root_is_the_checkout_whatever_the_cwd(repo):
    proc = lib(repo, 'printf "%s" "$REPO_ROOT"')
    assert proc.returncode == 0 and proc.stdout == str(repo.resolve())


def test_load_env_exports_the_file(repo):
    write_env(repo, "LAMBDA_API_KEY=k-123\nGCP_PROJECT=demo\n")
    proc = lib(repo, 'load_env; bash -c \'printf "%s %s" "$LAMBDA_API_KEY" "$GCP_PROJECT"\'')
    assert proc.returncode == 0 and proc.stdout == "k-123 demo"


def test_load_env_without_file_fails(repo):
    proc = lib(repo, "load_env; echo reached")
    # 2 = refused, the same code `gpuctl` itself uses when LAMBDA_API_KEY is missing
    assert proc.returncode == 2 and "reached" not in proc.stdout
    assert ".env" in proc.stderr and "chmod 600" in proc.stderr


@pytest.mark.parametrize("mode", [0o644, 0o640, 0o400, 0o700])
def test_load_env_refuses_any_mode_but_600(repo, mode):
    write_env(repo, mode=mode)
    proc = lib(repo, "load_env; echo reached")
    assert proc.returncode == 2 and "reached" not in proc.stdout
    assert "600" in proc.stderr and "k-123" not in proc.stdout + proc.stderr


def test_load_env_checks_the_target_of_a_symlink(repo, tmp_path_factory):
    real = tmp_path_factory.mktemp("secrets") / "env"
    real.write_text("LAMBDA_API_KEY=k-123\n")
    real.chmod(0o644)
    (repo / ".env").symlink_to(real)
    assert lib(repo, "load_env").returncode == 2


def test_die_exits_1_and_log_writes_stderr(repo):
    proc = lib(repo, 'log "hello"; die "boom"; echo reached')
    assert proc.returncode == 1 and proc.stdout == ""
    assert "hello" in proc.stderr and "boom" in proc.stderr


def test_die_takes_an_optional_exit_code(repo):
    proc = lib(repo, 'die "refused" 2; echo reached')
    assert proc.returncode == 2 and proc.stdout == "" and "error: refused" in proc.stderr


def test_wrapper_runs_the_package_from_the_venv_with_env_loaded(repo, tmp_path_factory):
    write_env(repo)
    elsewhere = tmp_path_factory.mktemp("cwd")
    proc = run([str(repo / "scripts/gpuctl"), "plan", "--set-count", "a6000=2", "two words"], repo, cwd=elsewhere)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines() == [
        f"PYTHONPATH={repo.resolve()}/lambda", "KEY=k-123",
        "ARG=-P", "ARG=-m", "ARG=gpuctl", "ARG=plan", "ARG=--set-count", "ARG=a6000=2", "ARG=two words",
    ]


KUBECONFIG_PYTHON = """#!/usr/bin/env bash
printf 'KUBECONFIG=%s\\n' "${KUBECONFIG-unset}"
"""


@pytest.mark.parametrize("env_text, ambient, expected", [
    ("LAMBDA_API_KEY=k-123\n", None, "{home}/.kube/hybrid-gpu.yaml"),
    ("LAMBDA_API_KEY=k-123\n", "/home/op/.kube/work-cluster", "{home}/.kube/hybrid-gpu.yaml"),
    ("LAMBDA_API_KEY=k-123\nKUBECONFIG=/srv/kc.yaml\n", "/home/op/.kube/work-cluster", "/srv/kc.yaml"),
])
def test_wrapper_pins_the_cluster_kubeconfig(repo, env_text, ambient, expected):
    """Spec section 11: gpuctl's kubectl reaches ~/.kube/hybrid-gpu.yaml, not an ambient KUBECONFIG; .env overrides."""
    write_env(repo, env_text)
    (repo / ".venv/bin/python").write_text(KUBECONFIG_PYTHON)
    env = {k: v for k, v in os.environ.items() if k not in ("LAMBDA_API_KEY", "REPO_ROOT", "KUBECONFIG")}
    env["HOME"] = "/home/op"
    if ambient is not None:
        env["KUBECONFIG"] = ambient
    proc = subprocess.run([str(repo / "scripts/gpuctl"), "status"], cwd=repo, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == f"KUBECONFIG={expected.format(home='/home/op')}\n"


def test_wrapper_refuses_without_a_private_env(repo):
    write_env(repo, mode=0o644)
    proc = run([str(repo / "scripts/gpuctl"), "status"], repo)
    assert proc.returncode == 2 and "PYTHONPATH" not in proc.stdout


def test_wrapper_without_venv_says_how_to_fix(repo):
    write_env(repo)
    shutil.rmtree(repo / ".venv")
    proc = run([str(repo / "scripts/gpuctl"), "status"], repo)
    assert proc.returncode != 0 and "make venv" in proc.stderr


def test_wrapper_never_imports_modules_from_the_callers_cwd(repo, tmp_path_factory):
    """`python -m` would put the cwd first on sys.path: a stray argparse.py there must not run with .env loaded."""
    write_env(repo)
    shutil.copytree(ROOT / "lambda/gpuctl", repo / "lambda/gpuctl")
    (repo / ".venv/bin/python").write_text(f'#!/usr/bin/env bash\nexec "{sys.executable}" "$@"\n')
    hostile = tmp_path_factory.mktemp("hostile")
    marker = hostile / "imported"
    for module in ("argparse.py", "yaml.py", "gpuctl.py"):
        (hostile / module).write_text(f"open({str(marker)!r}, 'a').write({module!r})\n")
    proc = run([str(repo / "scripts/gpuctl"), "--help"], repo, cwd=hostile)
    assert proc.returncode == 0, proc.stderr
    assert "list-nodes" in proc.stdout and not marker.exists()
