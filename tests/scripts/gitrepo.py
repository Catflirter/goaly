"""A throwaway git repository with a bare origin, for the freeze and final-run scripts' tests."""

import subprocess
from pathlib import Path

PACK = Path("src/goaly/harness/packs/effisql")


def git(repo: Path, *args: str) -> str:
    completed = subprocess.run(  # noqa: S603 - git, with arguments built here.
        ["git", "-C", str(repo), "-c", "user.email=t@example.com", "-c", "user.name=T", *args],  # noqa: S607 - git from PATH.
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def commit_pack(repo: Path, prompt: str) -> str:
    """Commit the Effi-SQL pack's prompt as ``prompt``, and return the commit."""
    (repo / PACK).mkdir(parents=True, exist_ok=True)
    (repo / PACK / "prompt.md").write_text(prompt, encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "--quiet", "--message", prompt)
    return git(repo, "rev-parse", "HEAD")


def repo_with_origin(tmp_path: Path) -> tuple[Path, Path]:
    """A clone of a bare origin whose main holds a v0 pack."""
    origin, repo = tmp_path / "origin.git", tmp_path / "repo"
    git(tmp_path, "init", "--quiet", "--bare", "-b", "main", str(origin))
    git(tmp_path, "clone", "--quiet", str(origin), str(repo))
    git(repo, "checkout", "--quiet", "-b", "main")
    commit_pack(repo, "v0")
    git(repo, "push", "--quiet", "origin", "main")
    return repo, origin
