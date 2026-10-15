"""Freeze the final harness: tag main, with best/effisql's pack merged into it, as ``final``.

    uv run --locked python scripts/freeze.py          # print the plan, change nothing
    uv run --locked python scripts/freeze.py --yes    # tag, then push the tag alone

Spec §10: core and ``best/effisql`` are merged and tagged ``final``. The merge is a reviewed PR,
never this script: it fetches main and checks that main's Effi-SQL pack is exactly
``best/effisql``'s, the pack and its tests, then tags main's commit. If ``best/effisql`` does
not exist, here or on the remote, no round accepted a change, and the final harness is main's
pack, v0. The script never merges, never moves a branch and never forces anything: with --yes
it creates the annotated tag ``final`` and pushes ``refs/tags/final`` alone. It refuses if a
``final`` tag already marks another commit, here or on the remote; a push that failed is retried
by the next run; and it exits non-zero unless the remote ends up with the tag at the commit.
"""

import argparse
import shlex
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# The lane's pack directories (goaly.rsi.lanes): the pack and its tests.
PACKS = ("src/goaly/harness/packs/effisql", "tests/harness/packs/effisql")
TAG = "final"
BEST = "best/effisql"


class FreezeError(Exception):
    """The freeze cannot go ahead; the message says what to do first."""


@dataclass(frozen=True)
class Freeze:
    """What the freeze tags, why that commit is the final harness, and what is left to do."""

    remote: str
    commit: str  # The commit to tag: main's.
    message: str  # The annotated tag's message.
    tagged: bool  # The local tag already marks the commit.
    pushed: bool  # The remote's tag already marks the commit.

    @property
    def done(self) -> bool:
        return self.tagged and self.pushed

    def steps(self) -> list[str]:
        """The git commands :meth:`apply` runs, for the dry run to show."""
        if self.done:
            return [f"# {TAG} marks {self.commit[:12]} here and on {self.remote}: nothing to do"]
        steps = [] if not self.tagged else [f"# {TAG} already marks {self.commit[:12]} here"]
        if not self.tagged:
            steps.append(
                f"git tag --annotate {TAG} {self.commit} --message {shlex.quote(self.message)}"
            )
        return [*steps, f"git push {self.remote} refs/tags/{TAG}:refs/tags/{TAG}"]

    def apply(self, repo: Path) -> None:
        """Create the tag unless it exists, then push it alone, never with --force, and check
        that the remote has it. Raises FreezeError if it does not."""
        if not self.tagged:
            _git(repo, "tag", "--annotate", TAG, self.commit, "--message", self.message)
        if not self.pushed:
            _git(repo, "push", self.remote, f"refs/tags/{TAG}:refs/tags/{TAG}")
        if _remote_tags(repo, self.remote).get(TAG) != self.commit:
            raise FreezeError(
                f"{self.remote} does not have {TAG} at {self.commit[:12]} after the push"
            )


def freeze_plan(repo: Path, *, remote: str = "origin", branch: str = "main") -> Freeze:
    """Fetch ``branch`` and check that it can be frozen; raise FreezeError if not.

    The remote's tags count as much as the local ones: a ``final`` or a ``best/effisql`` there
    that the local repository lacks or contradicts stops the freeze, and a local ``final`` that
    never reached the remote, after a failed push, is pushed again.
    """
    refspec = f"+refs/heads/{branch}:refs/remotes/{remote}/{branch}"
    _git(repo, "fetch", "--quiet", "--no-tags", remote, refspec)  # Tags come from ls-remote.
    commit = _resolve(repo, f"refs/remotes/{remote}/{branch}^{{commit}}")
    if commit is None:
        raise FreezeError(f"{remote}/{branch} does not exist")
    remote_tags = _remote_tags(repo, remote)
    best = _resolve(repo, f"refs/tags/{BEST}^{{commit}}")
    if remote_tags.get(BEST, best) != best:
        here = "lacks it" if best is None else f"has {best[:12]}"
        raise FreezeError(
            f"{remote} has {BEST} at {remote_tags[BEST][:12]}, but this repository {here}:"
            f" fetch the lane's tag first (git fetch {remote} tag {BEST})"
        )
    if best is None:
        why = f"{BEST} does not exist: no round accepted a change, so the final harness is v0"
    elif differ := [pack for pack in PACKS if _tree(repo, best, pack) != _tree(repo, commit, pack)]:
        raise FreezeError(
            f"{remote}/{branch} ({commit[:12]}) does not hold {BEST}'s {', '.join(differ)}"
            f" ({best[:12]}): merge its results PR into {branch} first; this script never merges"
        )
    else:
        why = f"its Effi-SQL pack is {BEST}'s, {best}"
    local = _resolve(repo, f"refs/tags/{TAG}^{{commit}}")
    places: list[tuple[str, str | None]] = [("here", local), (remote, remote_tags.get(TAG))]
    for where, found in places:
        if found not in (None, commit):
            raise FreezeError(
                f"the tag {TAG} already marks {found[:12]} {where}, not {commit[:12]}"
            )
    pack = _tree(repo, commit, PACKS[0])
    message = f"The final harness: {branch} at {commit}, with pack tree {pack}; {why}."
    return Freeze(remote, commit, message, local == commit, remote_tags.get(TAG) == commit)


def _remote_tags(repo: Path, remote: str) -> dict[str, str]:
    """The commit that each of the remote's tags final and best/effisql marks, peeled."""
    refs = [f"refs/tags/{name}{peel}" for name in (TAG, BEST) for peel in ("", "^{}")]
    tags: dict[str, str] = {}
    for line in _git(repo, "ls-remote", remote, *refs).splitlines():
        sha, ref = line.split()
        name = ref.removeprefix("refs/tags/")
        if name.endswith("^{}"):  # An annotated tag's commit wins over the tag object.
            tags[name.removesuffix("^{}")] = sha
        else:
            tags.setdefault(name, sha)
    return tags


def _tree(repo: Path, commit: str, path: str) -> str | None:
    return _resolve(repo, f"{commit}:{path}")


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
