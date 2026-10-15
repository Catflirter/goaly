"""Throwaway git repositories laid out like goaly's, for the workspace, commit and guard tests."""

import subprocess
from pathlib import Path

# The files of a repository's first commit: the frozen layers, and a v0 Effi-SQL pack.
BASE_FILES = {
    ".gitignore": "__pycache__/\n*.py[cod]\n",
    "README.md": "# goaly\n",
    "src/goaly/__init__.py": "",
    "src/goaly/bench/__init__.py": "",
    "src/goaly/bench/runner.py": "def run() -> None:\n    pass\n",
    "src/goaly/contracts/__init__.py": "",
    "src/goaly/harness/__init__.py": "",
    "src/goaly/harness/core/__init__.py": "",
    "src/goaly/harness/packs/effisql/__init__.py": "",
    "src/goaly/harness/packs/effisql/prompt.md": "Rewrite the query to run faster.\n",
    "src/goaly/harness/packs/effisql/policy.py": "STEPS = 3\n",
}


def git(cwd: Path, *args: str) -> str:
    """Run git in ``cwd``, fail on an error, and return its stdout."""
    return subprocess.run(  # noqa: S603  (git with arguments built by the tests)
        ["git", *args],  # noqa: S607  (git from PATH, as goaly.rsi runs it)
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def write(root: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def make_repo(path: Path) -> Path:
    """Create a repository at ``path`` whose one commit holds BASE_FILES, tagged best/effisql."""
    path.mkdir(parents=True)
    git(path, "init", "--quiet", "--initial-branch=main")
    write(path, BASE_FILES)
    git(path, "add", "--all")
    git(path, "commit", "--quiet", "--message=v0")
    git(path, "tag", "best/effisql")
    return path
