"""Freeze the final harness: tag main, with best/effisql's pack merged into it, as ``final``.

    uv run --locked python scripts/freeze.py          # print the plan, change nothing
    uv run --locked python scripts/freeze.py --yes    # tag, then push the tag alone

Spec §10: core and ``best/effisql`` are merged and tagged ``final``. The merge is a reviewed PR,
never this script: it fetches main and checks that main's Effi-SQL pack is exactly
``best/effisql``'s, then tags main's commit. If ``best/effisql`` does not exist, no round
accepted a change, and the final harness is main's pack, v0. The script never merges, never
moves a branch and never forces anything: with --yes it creates the annotated tag ``final`` and
pushes ``refs/tags/final`` alone, and it refuses if a ``final`` tag already marks another
commit.
"""

import argparse
import shlex
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PACK = "src/goaly/harness/packs/effisql"
TAG = "final"
BEST = "best/effisql"


class FreezeError(Exception):
    """The freeze cannot go ahead; the message says what to do first."""


@dataclass(frozen=True)
class Freeze:
    """What the freeze tags, and why that commit is the final harness."""

    remote: str
    commit: str  # The commit to tag: main's.
    message: str  # The annotated tag's message.
    done: bool  # The tag already marks the commit.

    def steps(self) -> list[str]:
        """The git commands :meth:`apply` runs, for the dry run to show."""
        if self.done:
            return [f"# {TAG} already marks {self.commit[:12]}: nothing to do"]
        return [
            f"git tag --annotate {TAG} {self.commit} --message {shlex.quote(self.message)}",
            f"git push {self.remote} refs/tags/{TAG}:refs/tags/{TAG}",
        ]

    def apply(self, repo: Path) -> None:
        """Create the tag, then push it alone; never a branch, never with --force."""
        if self.done:
            return
        _git(repo, "tag", "--annotate", TAG, self.commit, "--message", self.message)
        _git(repo, "push", self.remote, f"refs/tags/{TAG}:refs/tags/{TAG}")


def freeze_plan(repo: Path, *, remote: str = "origin", branch: str = "main") -> Freeze:
    """Fetch ``branch`` and check that it can be frozen; raise FreezeError if not."""
    _git(repo, "fetch", "--quiet", remote, f"+refs/heads/{branch}:refs/remotes/{remote}/{branch}")
    commit = _resolve(repo, f"refs/remotes/{remote}/{branch}^{{commit}}")
    if commit is None:
        raise FreezeError(f"{remote}/{branch} does not exist")
    pack = _resolve(repo, f"{commit}:{PACK}")
    best = _resolve(repo, f"refs/tags/{BEST}^{{commit}}")
    if best is None:
        why = f"{BEST} does not exist: no round accepted a change, so the final harness is v0"
    elif _resolve(repo, f"{best}:{PACK}") != pack:
        raise FreezeError(
            f"{remote}/{branch} ({commit[:12]}) does not hold {BEST}'s pack ({best[:12]}):"
            f" merge its results PR into {branch} first; this script never merges"
        )
    else:
        why = f"its Effi-SQL pack is {BEST}'s, {best}"
    existing = _resolve(repo, f"refs/tags/{TAG}^{{commit}}")
    if existing not in (None, commit):
        raise FreezeError(f"the tag {TAG} already marks {existing[:12]}, not {commit[:12]}")
    message = f"The final harness: {branch} at {commit}, with pack tree {pack}; {why}."
    return Freeze(remote, commit, message, done=existing == commit)


def _resolve(repo: Path, revision: str) -> str | None:
    try:
        return _git(repo, "rev-parse", "--verify", "--quiet", revision)
    except subprocess.CalledProcessError:
        return None


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(  # noqa: S603 - git, with arguments built here.
        ["git", "-C", str(repo), *args],  # noqa: S607 - git from PATH, as the developer runs it.
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    return completed.stdout.strip()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tag the final harness.")
    parser.add_argument("--yes", action="store_true", help="tag and push; otherwise only print")
    parser.add_argument("--remote", default="origin")
    args = parser.parse_args(argv)
    try:
        freeze = freeze_plan(REPO, remote=args.remote)
        print(f"# {freeze.message}")  # noqa: T201 - the plan is the output.
        for step in freeze.steps():
            print(step)  # noqa: T201
        if not args.yes:
            print("# dry run: pass --yes to tag and push the tag")  # noqa: T201
            return 0
        freeze.apply(REPO)
    except FreezeError as error:
        sys.stderr.write(f"error: {error}\n")
        return 1
    except subprocess.CalledProcessError as error:
        sys.stderr.write(f"error: {' '.join(error.cmd)} failed: {error.stderr.strip()}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
