"""The "RSI results" PRs a lane opens for the project lead to review (spec §8.7).

A lane opens one when it has 3 accepted changes since its last, or when 3 hours have passed
since then with something new decided, and a final one when it stops. goaly.rsi does this by
itself:

1. it builds ``rsi/<lane>/results-<n>`` from ``main``: one commit per accepted experiment, in
   the order they were accepted, setting the lane's pack directories to that candidate's, then
   one commit that copies the lane's ledger entries and lessons to ``rsi/results/<lane>/``. The
   live ledger, ``rsi/ledger/``, is git-ignored, so its files are copied, never force-added.
   The commits are built in a temporary index, so no checkout or working tree changes;
2. it pushes only that branch, and never over an existing one;
3. it opens the PR through the GitHub REST API, with a body generated from the ledger, and
   records a RESULTS_PR LaneEvent holding its number.

**Idempotent.** The next ``n`` comes from the RESULTS_PR events. Before building, the publisher
asks GitHub for a PR on ``results-<n>``, and for the branch itself: a crash after the push or
after the PR was opened is completed from what is there, never duplicated. Once a final PR is
recorded, the lane opens no other.

**The token** is read at run time, for each publication (goaly.harness.core.settings, then the
Coder workspace's), and goes only in HTTP headers: to git through ``GIT_CONFIG_*`` variables,
never in a URL, an argument or a log. The code never merges or approves, and pushes nothing
outside ``rsi/*``. A failed publication is logged and never stops the lane; it is tried again
the next time the lane asks.
"""

import base64
import logging
import os
import re
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import httpx
from google.protobuf.message import Message
from google.protobuf.timestamp_pb2 import Timestamp

from goaly.contracts.gen.experiment_pb2 import (
    DECISION_ACCEPTED,
    DECISION_PENDING,
    LANE_EVENT_KIND_NOTIFIED,
    LANE_EVENT_KIND_NOTIFY_FAILED,
    LANE_EVENT_KIND_RESULTS_PR,
    LANE_EVENT_KIND_STARTED,
    LANE_EVENT_KIND_STOPPED,
    Decision,
    ExperimentRecord,
    Lane,
    LaneEvent,
    Lesson,
)
from goaly.contracts.jsonl import to_json_line
from goaly.harness.core.settings import load_settings
from goaly.rsi.git import _KEPT as _KEPT_GIT  # Who commits: the only GIT_* kept.
from goaly.rsi.git import _SETTINGS as GIT_SETTINGS  # The same hardening as run_git.
from goaly.rsi.git import GitError, run_git
from goaly.rsi.lanes import LaneInfo, lane_info
from goaly.rsi.ledger import EXPERIMENTS, LANE_EVENTS, LESSONS, LESSONS_MARKDOWN, Ledger
from goaly.rsi.notify import NotificationKind, Notifier

__all__ = [
    "RESULTS_DIR",
    "GitHubApi",
    "GitHubRepo",
    "HttpsRemote",
    "PublishedPr",
    "Remote",
    "ResultsPrError",
    "ResultsPublisher",
    "ResultsPublisherImpl",
    "pr_body",
    "runtime_github_token",
]

RESULTS_DIR = PurePosixPath("rsi/results")
_RESULTS_BRANCH = re.compile(r"rsi/[a-z]+/results-([1-9][0-9]*)")
# Who the results commits are by, unless GIT_AUTHOR_* and GIT_COMMITTER_* say otherwise.
_IDENTITY = ("goaly rsi", "rsi@goaly.invalid")
_TRAILER = "Goaly-Experiments:"
_MAX_BODY = 60_000  # GitHub refuses bodies over 65536 characters.

_logger = logging.getLogger(__name__)

type TokenSource = Callable[[], str]


class ResultsPrError(Exception):
    """A results PR could not be built, pushed or opened; the message holds no secret."""


@dataclass(frozen=True)
class PublishedPr:
    """A results PR, as the ledger records it."""

    number: int
    url: str
    branch: str
    head_commit: str


class ResultsPublisher(Protocol):
    """What a lane calls to open its results PRs."""

    def due(self, lane: Lane.ValueType, now: datetime) -> bool:
        """Say whether a periodic results PR is due."""
        ...

    def publish(self, lane: Lane.ValueType, *, final: bool, now: datetime) -> PublishedPr | None:
        """Open the next results PR, or return the one already open; None if it failed."""
        ...


def runtime_github_token(env_file: Path | None = None, *, timeout_s: float = 30.0) -> str:
    """Return the GitHub token now: GOALY_GITHUB_TOKEN, else the Coder workspace's token.

    Raises ResultsPrError, quoting nothing the command printed, if neither gives one.
    """
    token = load_settings(env_file).github_token
    if token is not None:
        return token.get_secret_value()
    try:
        done = subprocess.run(
            ["coder", "external-auth", "access-token", "github"],  # noqa: S607  (from PATH)
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ResultsPrError(f"no GitHub token: coder failed ({type(error).__name__})") from None
    value = done.stdout.decode("utf-8", "replace").strip()
    if done.returncode != 0 or not value or not all("!" <= char <= "~" for char in value):
        raise ResultsPrError(f"no GitHub token: coder exited with {done.returncode}")
    return value


@dataclass(frozen=True)
class GitHubRepo:
    """The GitHub repository results PRs go to, such as Catflirter/goaly."""

    owner: str
    name: str

    @property
    def git_url(self) -> str:
        return f"https://github.com/{self.owner}/{self.name}.git"


class GitHubApi:
    """The few REST calls results PRs need: no merge, no review, nothing that writes but a PR."""

    def __init__(
        self,
        repo: GitHubRepo,
        token: TokenSource,
        *,
        api_url: str = "https://api.github.com",
        timeout_s: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._repo = repo
        self._token = token
        self._client = httpx.Client(
            base_url=f"{api_url}/repos/{repo.owner}/{repo.name}",
            timeout=timeout_s,
            transport=transport,
            trust_env=False,
            headers={"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"},
        )

    def find_pr(self, branch: str) -> tuple[int, str] | None:
        """Return the number and URL of a PR from ``branch``, open or closed, if there is one."""
        found = self._call(
            "GET", "/pulls", params={"head": f"{self._repo.owner}:{branch}", "state": "all"}
        )
        if not isinstance(found, list) or not found:
            return None
        return int(found[0]["number"]), str(found[0]["html_url"])

    def branch_head(self, branch: str) -> str | None:
        """Return the commit at the head of ``branch``, or None if it does not exist."""
        found = self._call("GET", f"/branches/{branch}", missing_ok=True)
        return None if found is None else str(found["commit"]["sha"])

    def commit_message(self, commit: str) -> str:
        return str(self._call("GET", f"/commits/{commit}")["commit"]["message"])

    def open_pr(self, *, branch: str, base: str, title: str, body: str) -> tuple[int, str]:
        made = self._call(
            "POST", "/pulls", json={"title": title, "head": branch, "base": base, "body": body}
        )
        return int(made["number"]), str(made["html_url"])

    def close(self) -> None:
        self._client.close()

    def _call(
        self,
        method: str,
        path: str,
        *,
        missing_ok: bool = False,
        params: Mapping[str, str] | None = None,
        json: object = None,
    ) -> Any:
        headers = {"Authorization": f"Bearer {self._token()}"}
        try:
            response = self._client.request(method, path, params=params, json=json, headers=headers)
        except httpx.HTTPError as error:
            raise ResultsPrError(f"GitHub {method} {path}: {type(error).__name__}") from None
        if missing_ok and response.status_code == httpx.codes.NOT_FOUND:
            return None
        if response.is_error:
            raise ResultsPrError(f"GitHub {method} {path}: HTTP {response.status_code}")
        try:
            return response.json()
        except ValueError:
            raise ResultsPrError(f"GitHub {method} {path}: the reply is not JSON") from None


class Remote(Protocol):
    """Fetches the base branch and pushes results branches."""

    def fetch(self, repo: Path, branch: str) -> str:
        """Fetch ``branch`` and return its commit."""
        ...

    def push(self, repo: Path, commit: str, branch: str) -> None:
        """Create ``branch`` on the remote at ``commit``; never move an existing one."""
        ...


class HttpsRemote:
    """git over HTTPS to GitHub, with the token in an HTTP header passed through the environment."""

    def __init__(self, repo: GitHubRepo, token: TokenSource, *, timeout_s: float = 300.0) -> None:
        self._repo = repo
        self._token = token
        self._timeout_s = timeout_s

    def fetch(self, repo: Path, branch: str) -> str:
        ref = f"refs/goaly/upstream/{branch}"
        self._git(
            repo,
            [
                "fetch",
                "--no-tags",
                "--no-write-fetch-head",
                self._repo.git_url,
                f"+refs/heads/{branch}:{ref}",
            ],
        )
        return run_git(["rev-parse", "--verify", f"{ref}^{{commit}}"], cwd=repo).strip()

    def push(self, repo: Path, commit: str, branch: str) -> None:
        if not branch.startswith("rsi/"):
            raise ResultsPrError(f"refusing to push {branch}: goaly.rsi pushes only rsi/*")
        self._git(
            repo, ["push", "--no-verify", self._repo.git_url, f"{commit}:refs/heads/{branch}"]
        )

    def _git(self, repo: Path, args: list[str]) -> None:
        basic = base64.b64encode(f"x-access-token:{self._token()}".encode()).decode()
        _git_with_header(
            args, cwd=repo, header=f"Authorization: Basic {basic}", timeout_s=self._timeout_s
        )


def _git_with_header(args: list[str], *, cwd: Path, header: str, timeout_s: float) -> None:
    """Run git as goaly.rsi.git.run_git does, plus one HTTP header for github.com.

    run_git takes no credentials, so this is the only git that can push. The header reaches git
    through ``GIT_CONFIG_*`` variables, never an argument, and no output of git is kept but its
    error text, which never holds the header.
    """
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("GIT_") or name in _KEPT_GIT
    }
    environment.update(
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL="/dev/null",
        GIT_TERMINAL_PROMPT="0",
        GIT_ASKPASS="",
        SSH_ASKPASS="",
        LC_ALL="C",
        GIT_CONFIG_COUNT="1",
        GIT_CONFIG_KEY_0="http.https://github.com/.extraheader",
        GIT_CONFIG_VALUE_0=header,
    )
    settings = [item for setting in GIT_SETTINGS for item in ("-c", setting)]
    try:
        done = subprocess.run(  # noqa: S603  (git, with arguments goaly.rsi builds)
            ["git", *settings, *args],  # noqa: S607  (git from PATH)
            cwd=cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout_s,
            start_new_session=True,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise GitError(args, None, f"timed out after {timeout_s:g} s") from None
    if done.returncode != 0:
        reason = done.stderr.decode("utf-8", "replace").replace(header, "[redacted]")
        raise GitError(args, done.returncode, reason)


class ResultsPublisherImpl:
    """Opens a lane's results PRs on GitHub, keeping their numbers in the ledger.

    ``repo`` is the lane's repository, which holds the accepted candidates' commits.
    """

    def __init__(
        self,
        ledger: Ledger,
        repo: Path,
        *,
        github: GitHubApi,
        remote: Remote,
        notifier: Notifier,
        base: str = "main",
        accepted_per_pr: int = 3,
        interval: timedelta = timedelta(hours=3),
        identity: tuple[str, str] = _IDENTITY,
    ) -> None:
        self._ledger = ledger
        self._repo = repo
        self._github = github
        self._remote = remote
        self._notifier = notifier
        self._base = base
        self._accepted_per_pr = accepted_per_pr
        self._interval = interval
        self._identity = identity

    def due(self, lane: Lane.ValueType, now: datetime) -> bool:
        try:
            history = _History.read(self._ledger, lane)
            if history.final is not None:
                return False
            if len(history.new_accepted()) >= self._accepted_per_pr:
                return True
            since = history.since or history.first_start
            return (
                since is not None and now - since >= self._interval and bool(history.new_decided())
            )
        except Exception:
            _logger.exception("could not tell whether a results PR is due")
            return False

    def publish(self, lane: Lane.ValueType, *, final: bool, now: datetime) -> PublishedPr | None:
        try:
            return self._publish(lane, final=final, now=now)
        except Exception as error:  # noqa: BLE001  (a results PR never stops the lane)
            known = isinstance(error, ResultsPrError | GitError)
            _logger.error(
                "the results PR failed",
                extra={"reason": str(error) if known else type(error).__name__},
            )
            return None

    def _publish(self, lane: Lane.ValueType, *, final: bool, now: datetime) -> PublishedPr:
        info = lane_info(lane)
        history = _History.read(self._ledger, lane)
        if history.final is not None:
            return _published(history.final)
        branch = f"rsi/{info.name}/results-{history.next_number}"
        accepted = history.accepted()
        new_ids = [record.experiment_id for record in history.new_accepted()]
        title = (
            f"RSI results: {info.title}, results-{history.next_number}{' (final)' if final else ''}"
        )
        found = self._github.find_pr(branch)
        head = self._github.branch_head(branch)
        if head is None:
            if found is not None:
                raise ResultsPrError(f"a PR exists for {branch}, but the branch is gone")
            head = self._build(info, accepted, new_ids, history)
            self._remote.push(self._repo, head, branch)
        else:  # An earlier attempt pushed it: record what the branch really carries.
            new_ids = _carried(self._github.commit_message(head))
        if found is None:
            body = pr_body(info, history, branch=branch, final=final, carried=new_ids)
            found = self._github.open_pr(branch=branch, base=self._base, title=title, body=body)
        number, url = found
        event = LaneEvent(
            lane=lane,
            kind=LANE_EVENT_KIND_RESULTS_PR,
            detail=f"{branch}{' final' if final else ''}",
            pr_number=number,
            pr_url=url,
            head_commit=head,
            experiment_ids=new_ids,
        )
        event.occurred_at.CopyFrom(_timestamp(now))
        self._ledger.record_lane_event(event)
        _logger.info("results PR opened", extra={"pr_number": number, "branch": branch})
        self._notifier.notify(
            lane,
            NotificationKind.RESULTS_PR,
            f"{info.title}: results PR #{number} is open with {len(new_ids)} new accepted "
            f"change(s){', the final one' if final else ''}. {url}",
        )
        return PublishedPr(number, url, branch, head)

    def _build(
        self,
        info: LaneInfo,
        accepted: Sequence[ExperimentRecord],
        new_ids: Sequence[str],
        history: "_History",
    ) -> str:
        """Build the results commits on the base branch, in a temporary index; return the head."""
        parent = self._remote.fetch(self._repo, self._base)
        with tempfile.TemporaryDirectory(prefix="goaly-rsi-results-") as scratch:
            env = {"GIT_INDEX_FILE": str(Path(scratch) / "index")}
            git = _IndexGit(self._repo, env, self._identity)
            git.run(["read-tree", parent])
            tree = git.run(["write-tree"])
            for record in accepted:
                for pack in info.pack_dirs:
                    git.remove(str(pack))
                    if git.run(["ls-tree", "-d", record.candidate_commit, "--", str(pack)]):
                        git.run(
                            ["read-tree", f"--prefix={pack}/", f"{record.candidate_commit}:{pack}"]
                        )
                new_tree = git.run(["write-tree"])
                if new_tree == tree:  # The base already holds this change.
                    continue
                tree = new_tree
                message = (
                    f"rsi({info.name}): {record.experiment_id}, accepted in round "
                    f"{record.round}\n\n"
                    f"Candidate commit: {record.candidate_commit}\n"
                )
                parent = git.run(["commit-tree", tree, "-p", parent, "-F", "-"], text=message)
            directory = RESULTS_DIR / info.name
            git.remove(str(directory))
            for name, text in history.files().items():
                blob = git.run(["hash-object", "-w", "--stdin"], text=text)
                git.run(
                    ["update-index", "--add", "--cacheinfo", f"100644,{blob},{directory / name}"]
                )
            tree = git.run(["write-tree"])
            message = (
                f"rsi({info.name}): the ledger and lessons for this results PR\n\n"
                f"{_TRAILER} {' '.join(new_ids) or '-'}\n"
            )
            return git.run(["commit-tree", tree, "-p", parent, "-F", "-"], text=message)


class _IndexGit:
    """git on a temporary index (``env``), committing as ``identity``: a name and an email."""

    def __init__(self, repo: Path, env: Mapping[str, str], identity: tuple[str, str]) -> None:
        self._repo = repo
        self._env = env
        name, email = identity
        self._identity = ["-c", f"user.name={name}", "-c", f"user.email={email}"]

    def run(self, args: list[str], *, text: str | None = None) -> str:
        args = [*self._identity, *args]
        return run_git(args, cwd=self._repo, env=self._env, input_text=text).strip()

    def remove(self, directory: str) -> None:
        """Remove ``directory`` from the index, without touching any working tree."""
        names = run_git(["ls-files", "-z", "--", directory], cwd=self._repo, env=self._env)
        if names:
            run_git(
                ["update-index", "--force-remove", "-z", "--stdin"],
                cwd=self._repo,
                env=self._env,
                input_text=names,
            )


@dataclass
class _History:
    """A lane's results PRs and what the ledger holds since the last one."""

    lane: Lane.ValueType
    experiments: dict[str, ExperimentRecord]
    events: list[LaneEvent]
    lessons: list[Lesson]
    lessons_markdown: str | None

    @classmethod
    def read(cls, ledger: Ledger, lane: Lane.ValueType) -> "_History":
        with ledger.session() as files:
            records = files.read(EXPERIMENTS, ExperimentRecord)
            events = [event for event in files.read(LANE_EVENTS, LaneEvent) if event.lane == lane]
            lessons = [lesson for lesson in files.read(LESSONS, Lesson) if lesson.lane == lane]
            markdown = ledger.directory / LESSONS_MARKDOWN
            text = markdown.read_text(encoding="utf-8") if markdown.is_file() else None
        latest: dict[str, ExperimentRecord] = {}
        for record in records:
            if record.lane == lane:
                latest[record.experiment_id] = record
        return cls(lane, latest, events, lessons, text)

    @property
    def prs(self) -> list[LaneEvent]:
        return [event for event in self.events if event.kind == LANE_EVENT_KIND_RESULTS_PR]

    @property
    def final(self) -> LaneEvent | None:
        return next((pr for pr in self.prs if pr.detail.endswith(" final")), None)

    @property
    def next_number(self) -> int:
        numbers = [
            int(match.group(1)) for pr in self.prs if (match := _RESULTS_BRANCH.match(pr.detail))
        ]
        return max(numbers, default=0) + 1

    @property
    def since(self) -> datetime | None:
        prs = self.prs
        return prs[-1].occurred_at.ToDatetime(tzinfo=UTC) if prs else None

    @property
    def first_start(self) -> datetime | None:
        starts = [event for event in self.events if event.kind == LANE_EVENT_KIND_STARTED]
        return starts[0].occurred_at.ToDatetime(tzinfo=UTC) if starts else None

    def accepted(self) -> list[ExperimentRecord]:
        found = [
            record for record in self.experiments.values() if record.decision == DECISION_ACCEPTED
        ]
        return sorted(found, key=lambda record: record.round)

    def new_accepted(self) -> list[ExperimentRecord]:
        carried = {name for pr in self.prs for name in pr.experiment_ids}
        return [record for record in self.accepted() if record.experiment_id not in carried]

    def new_decided(self) -> list[ExperimentRecord]:
        since = self.since
        decided = [
            record for record in self.experiments.values() if record.decision != DECISION_PENDING
        ]
        if since is None:
            return decided
        return [record for record in decided if record.finished_at.ToDatetime(tzinfo=UTC) > since]

    def files(self) -> dict[str, str]:
        """The ledger files the branch carries, holding this lane's entries only."""
        files = {
            EXPERIMENTS: _lines(self.experiments.values()),
            LANE_EVENTS: _lines(self.events),
            LESSONS: _lines(self.lessons),
        }
        if self.lessons_markdown is not None:
            files[LESSONS_MARKDOWN] = self.lessons_markdown
        return files


def pr_body(
    info: LaneInfo, history: _History, *, branch: str, final: bool, carried: Sequence[str]
) -> str:
    """The PR's description, generated from the ledger."""
    lines = [
        f"Results of the {info.title} RSI lane{', its final PR' if final else ''}. "
        "goaly.rsi opened it and never merges it: the project lead reviews it.",
        "",
        f"- Branch: `{branch}`; the ledger and lessons are under `{RESULTS_DIR / info.name}/`.",
        f"- New accepted changes: {', '.join(f'`{name}`' for name in carried) or 'none'}.",
        f"- Accepted in all: {len(history.accepted())}; lessons: {len(history.lessons)}.",
    ]
    stops = [event for event in history.events if event.kind == LANE_EVENT_KIND_STOPPED]
    if final and stops:
        lines.append(f"- Stopped: {stops[-1].detail}.")
    lines += ["", "## Experiments decided since the last results PR", ""]
    decided = history.new_decided()
    if decided:
        lines += ["| Experiment | Round | Decision | Δ val | Reason |", "|---|---|---|---|---|"]
        for record in decided:
            delta = f"{record.validation.delta:+.3f}" if record.HasField("validation") else "-"
            reason = record.decision_reason.replace("|", "/").replace("\n", " ")[:300]
            decision = Decision.Name(record.decision).removeprefix("DECISION_").lower()
            lines.append(
                f"| `{record.experiment_id}` | {record.round} | {decision} | {delta} | {reason} |"
            )
    else:
        lines.append("None.")
    since = history.since
    notes = [
        event
        for event in history.events
        if event.kind in (LANE_EVENT_KIND_NOTIFIED, LANE_EVENT_KIND_NOTIFY_FAILED)
        and (since is None or event.occurred_at.ToDatetime(tzinfo=UTC) > since)
    ]
    lines += ["", "## Notifications since the last results PR", ""]
    for event in notes:
        state = (
            "delivered"
            if event.kind == LANE_EVENT_KIND_NOTIFIED
            else f"not delivered ({event.detail})"
        )
        moment = event.occurred_at.ToDatetime(tzinfo=UTC).strftime("%Y-%m-%d %H:%M UTC")
        lines.append(f"- {moment}, `{event.notification_id}`: {event.notification} ({state})")
    if not notes:
        lines.append("None.")
    body = "\n".join(lines) + "\n"
    return body if len(body) <= _MAX_BODY else body[:_MAX_BODY] + "\n\n(cut; see the ledger)\n"


def _carried(message: str) -> list[str]:
    for line in reversed(message.splitlines()):
        if line.startswith(_TRAILER):
            return [name for name in line.removeprefix(_TRAILER).split() if name != "-"]
    raise ResultsPrError("the results branch's head has no experiments trailer")


def _published(event: LaneEvent) -> PublishedPr:
    return PublishedPr(event.pr_number, event.pr_url, event.detail.split()[0], event.head_commit)


def _lines(messages: Iterable[Message]) -> str:
    return "".join(f"{to_json_line(message)}\n" for message in messages)


def _timestamp(moment: datetime) -> Timestamp:
    stamp = Timestamp()
    stamp.FromDatetime(moment)
    return stamp
