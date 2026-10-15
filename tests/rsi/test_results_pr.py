"""Tests for goaly.rsi.results_pr: building, pushing and opening results PRs, idempotently."""

import json
import os
import stat
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from google.protobuf.timestamp_pb2 import Timestamp

from goaly.contracts.gen.experiment_pb2 import (
    DECISION_ACCEPTED,
    DECISION_PENDING,
    DECISION_REJECTED,
    LANE_EFFISQL,
    LANE_EVENT_KIND_NOTIFIED,
    LANE_EVENT_KIND_NOTIFY_FAILED,
    LANE_EVENT_KIND_RESULTS_PR,
    LANE_EVENT_KIND_STARTED,
    LANE_EVENT_KIND_STOPPED,
    REJECTION_REASON_NO_SCREEN_GAIN,
    ExperimentRecord,
    Hypothesis,
    Lane,
    LaneEvent,
    Lesson,
    PhaseResult,
)
from goaly.rsi import results_pr
from goaly.rsi.ledger import Ledger
from goaly.rsi.notify import NotificationKind
from goaly.rsi.results_pr import (
    GitHubApi,
    GitHubRepo,
    HttpsRemote,
    ResultsPrError,
    ResultsPublisherImpl,
    runtime_github_token,
)
from tests.rsi.gitrepo import git, make_repo, write

T0 = datetime(2026, 10, 15, 16, 0, tzinfo=UTC)
TOKEN = "ghu_secret_token_123"  # noqa: S105  (a fake token)
REPO = GitHubRepo("Catflirter", "goaly")
PROMPT = "src/goaly/harness/packs/effisql/prompt.md"


def stamp(moment: datetime) -> Timestamp:
    value = Timestamp()
    value.FromDatetime(moment)
    return value


class Notes:
    def __init__(self) -> None:
        self.sent: list[tuple[Lane.ValueType, NotificationKind, str]] = []

    def notify(self, lane: Lane.ValueType, kind: NotificationKind, text: str) -> None:
        self.sent.append((lane, kind, text))

    def close(self, timeout_s: float = 30.0) -> None:
        pass


class LocalRemote:
    """The remote is a bare repository: fetch reads the local main, push pushes to the bare."""

    def __init__(self, bare: Path) -> None:
        self.bare = bare
        self.pushes: list[tuple[str, str]] = []

    def fetch(self, repo: Path, branch: str) -> str:
        return git(repo, "rev-parse", branch).strip()

    def push(self, repo: Path, commit: str, branch: str) -> None:
        self.pushes.append((commit, branch))
        git(repo, "push", "--quiet", str(self.bare), f"{commit}:refs/heads/{branch}")


@dataclass
class FakeGitHub:
    """GitHub's pulls, branches and commits endpoints, over the bare repository."""

    bare: Path
    pulls: list[dict[str, Any]] = field(default_factory=list)
    fail_next_post: int = 0
    requests: list[httpx.Request] = field(default_factory=list)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        assert request.headers["Authorization"] == f"Bearer {TOKEN}"
        path = request.url.path.removeprefix("/repos/Catflirter/goaly")
        if path == "/pulls" and request.method == "GET":
            head = request.url.params["head"].removeprefix("Catflirter:")
            return httpx.Response(200, json=[pr for pr in self.pulls if pr["head"] == head])
        if path == "/pulls" and request.method == "POST":
            if self.fail_next_post:
                self.fail_next_post -= 1
                return httpx.Response(502)
            body = json.loads(request.content)
            number = len(self.pulls) + 1
            pr = {**body, "number": number, "html_url": f"https://github.com/pr/{number}"}
            self.pulls.append(pr)
            return httpx.Response(201, json=pr)
        if path.startswith("/branches/"):
            ref = f"refs/heads/{path.removeprefix('/branches/')}"
            done = subprocess.run(  # noqa: S603  (git with arguments built by the tests)
                ["git", "rev-parse", "--verify", "--quiet", ref],  # noqa: S607
                cwd=self.bare,
                capture_output=True,
                text=True,
                check=False,
            )
            if done.returncode:
                return httpx.Response(404)
            return httpx.Response(200, json={"commit": {"sha": done.stdout.strip()}})
        if path.startswith("/commits/"):
            message = git(self.bare, "log", "-1", "--format=%B", path.removeprefix("/commits/"))
            return httpx.Response(200, json={"commit": {"message": message}})
        return httpx.Response(418)


@dataclass
class World:
    repo: Path
    bare: Path
    ledger: Ledger
    github: FakeGitHub
    remote: LocalRemote
    notes: Notes
    publisher: ResultsPublisherImpl
    rounds: int = 0

    def accept(self, name: str, prompt: str, *, at: datetime = T0) -> str:
        """Commit a candidate that sets the prompt, and record it as accepted."""
        self.rounds += 1
        git(self.repo, "checkout", "--quiet", "-B", f"rsi/effisql/{name}", "best/effisql")
        write(self.repo, {PROMPT: prompt})
        git(self.repo, "commit", "--quiet", "--all", "--allow-empty", f"--message={name}")
        commit = git(self.repo, "rev-parse", "HEAD").strip()
        git(self.repo, "tag", "--force", "best/effisql", commit)
        git(self.repo, "checkout", "--quiet", "main")
        record = ExperimentRecord(
            experiment_id=name,
            lane=LANE_EFFISQL,
            round=self.rounds,
            candidate_commit=commit,
            hypothesis=Hypothesis(hypothesis_id=f"h-{name}"),
            validation=PhaseResult(delta=0.07),
            decision=DECISION_ACCEPTED,
            decision_reason="val +7.0 R-VES | p=0.03",
            finished_at=stamp(at),
        )
        self.ledger.record_experiment(record)
        return commit

    def reject(self, name: str, *, at: datetime = T0) -> None:
        self.rounds += 1
        self.ledger.record_experiment(
            ExperimentRecord(
                experiment_id=name,
                lane=LANE_EFFISQL,
                round=self.rounds,
                hypothesis=Hypothesis(hypothesis_id=f"h-{name}"),
                decision=DECISION_REJECTED,
                rejection_reasons=[REJECTION_REASON_NO_SCREEN_GAIN],
                decision_reason="screen +0.0",
                finished_at=stamp(at),
            )
        )

    def event(self, kind: int, at: datetime = T0, **fields: Any) -> None:
        self.ledger.record_lane_event(
            LaneEvent(lane=LANE_EFFISQL, kind=kind, occurred_at=stamp(at), **fields)  # type: ignore[arg-type]  # A kind's number.
        )

    def prs(self) -> list[LaneEvent]:
        return [e for e in self.ledger.lane_events() if e.kind == LANE_EVENT_KIND_RESULTS_PR]

    def show(self, revision: str, path: str) -> str:
        return git(self.bare, "show", f"{revision}:{path}")


@pytest.fixture
def world(tmp_path: Path) -> World:
    repo = make_repo(tmp_path / "repo")
    bare = tmp_path / "remote.git"
    git(tmp_path, "init", "--quiet", "--bare", str(bare))
    ledger = Ledger(tmp_path / "ledger")
    github = FakeGitHub(bare)
    remote = LocalRemote(bare)
    notes = Notes()
    api = GitHubApi(REPO, lambda: TOKEN, transport=httpx.MockTransport(github.handle))
    publisher = ResultsPublisherImpl(ledger, repo, github=api, remote=remote, notifier=notes)
    return World(repo, bare, ledger, github, remote, notes, publisher)


def test_publish_builds_pushes_and_opens_a_pr_from_the_ledger(world: World) -> None:
    world.event(LANE_EVENT_KIND_STARTED)
    first = world.accept("e1", "Use indexes.\n")
    world.reject("e2")
    with world.ledger.session(write=True) as files:
        files.append(
            "lessons.jsonl", Lesson(lesson_id="l1", lane=LANE_EFFISQL, text="Indexes help.")
        )
    published = world.publisher.publish(LANE_EFFISQL, final=False, now=T0 + timedelta(hours=1))
    assert published is not None
    assert published.number == 1
    assert published.branch == "rsi/effisql/results-1"
    assert world.remote.pushes == [(published.head_commit, "rsi/effisql/results-1")]
    head = published.head_commit
    assert world.show(head, PROMPT) == "Use indexes.\n"
    assert world.show(head, "README.md") == "# goaly\n"  # From main.
    experiments = world.show(head, "rsi/results/effisql/experiments.jsonl").splitlines()
    assert [json.loads(line)["experiment_id"] for line in experiments] == ["e1", "e2"]
    assert "Indexes help." in world.show(head, "rsi/results/effisql/lessons.jsonl")
    log = git(world.bare, "log", "--format=%s", head).splitlines()
    assert log[:2] == [
        "rsi(effisql): the ledger and lessons for this results PR",
        "rsi(effisql): e1, accepted in round 1",
    ]
    assert first in git(world.bare, "log", "-1", "--format=%B", f"{head}~1")
    (pr,) = world.github.pulls
    assert pr["base"] == "main"
    assert pr["title"] == "RSI results: Effi-SQL, results-1"
    assert "`e1`" in pr["body"]
    assert "| `e2` | 2 | rejected | - | screen +0.0 |" in pr["body"]
    (event,) = world.prs()
    assert (event.pr_number, event.detail, list(event.experiment_ids)) == (
        1,
        "rsi/effisql/results-1",
        ["e1"],
    )
    assert event.head_commit == head
    assert world.notes.sent[0][1] is NotificationKind.RESULTS_PR
    assert "#1" in world.notes.sent[0][2]


def test_a_second_pr_numbers_on_and_carries_only_new_acceptances(world: World) -> None:
    world.accept("e1", "one\n")
    assert world.publisher.publish(LANE_EFFISQL, final=False, now=T0) is not None
    world.accept("e2", "two\n", at=T0 + timedelta(minutes=5))
    second = world.publisher.publish(LANE_EFFISQL, final=True, now=T0 + timedelta(hours=1))
    assert second is not None
    assert second.branch == "rsi/effisql/results-2"
    assert world.show(second.head_commit, PROMPT) == "two\n"
    assert [list(event.experiment_ids) for event in world.prs()] == [["e1"], ["e2"]]
    assert world.github.pulls[1]["title"] == "RSI results: Effi-SQL, results-2 (final)"


def test_a_final_pr_is_opened_once(world: World) -> None:
    world.accept("e1", "one\n")
    world.event(LANE_EVENT_KIND_STOPPED, detail="deadline: the deadline passed")
    first = world.publisher.publish(LANE_EFFISQL, final=True, now=T0)
    again = world.publisher.publish(LANE_EFFISQL, final=True, now=T0 + timedelta(minutes=1))
    assert first == again
    assert len(world.github.pulls) == 1
    assert "Stopped: deadline: the deadline passed." in world.github.pulls[0]["body"]
    assert not world.publisher.due(LANE_EFFISQL, T0 + timedelta(days=1))


def test_a_failed_open_is_finished_later_without_a_second_push(world: World) -> None:
    world.accept("e1", "one\n")
    world.github.fail_next_post = 1
    assert world.publisher.publish(LANE_EFFISQL, final=False, now=T0) is None
    assert world.prs() == []
    world.accept("e2", "two\n")  # Accepted after the push: not on that branch.
    published = world.publisher.publish(LANE_EFFISQL, final=False, now=T0)
    assert published is not None
    assert len(world.remote.pushes) == 1
    assert list(world.prs()[0].experiment_ids) == ["e1"]  # What the branch really carries.
    assert world.publisher.due(LANE_EFFISQL, T0) is False  # e2 alone is not 3.


def test_a_pr_opened_before_a_crash_is_recorded_not_opened_again(world: World) -> None:
    world.accept("e1", "one\n")
    original = world.ledger.record_lane_event

    def crash(event: LaneEvent) -> None:
        raise OSError("disk full")

    world.ledger.record_lane_event = crash  # type: ignore[method-assign]  # Simulate the crash.
    assert world.publisher.publish(LANE_EFFISQL, final=False, now=T0) is None
    world.ledger.record_lane_event = original  # type: ignore[method-assign]
    published = world.publisher.publish(LANE_EFFISQL, final=False, now=T0)
    assert published is not None
    assert published.number == 1
    assert len(world.github.pulls) == 1
    assert len(world.remote.pushes) == 1


def test_a_branch_whose_pr_exists_but_is_gone_fails_safely(world: World) -> None:
    world.github.pulls.append(
        {"head": "rsi/effisql/results-1", "number": 9, "html_url": "https://github.com/pr/9"}
    )
    assert world.publisher.publish(LANE_EFFISQL, final=False, now=T0) is None
    assert world.prs() == []


def test_a_change_already_on_main_adds_no_commit(world: World) -> None:
    world.accept("e1", "Rewrite the query to run faster.\n")  # The same text as v0.
    published = world.publisher.publish(LANE_EFFISQL, final=False, now=T0)
    assert published is not None
    subjects = git(world.bare, "log", "--format=%s", published.head_commit).splitlines()
    assert subjects == ["rsi(effisql): the ledger and lessons for this results PR", "v0"]


def test_due_after_three_acceptances_or_three_hours_with_news(world: World) -> None:
    world.event(LANE_EVENT_KIND_STARTED)
    assert not world.publisher.due(LANE_EFFISQL, T0 + timedelta(hours=4))  # Nothing decided.
    world.reject("r1", at=T0 + timedelta(minutes=1))
    assert not world.publisher.due(LANE_EFFISQL, T0 + timedelta(hours=2))
    assert world.publisher.due(LANE_EFFISQL, T0 + timedelta(hours=3))
    for name in ("a1", "a2", "a3"):
        world.accept(name, f"{name}\n")
    assert world.publisher.due(LANE_EFFISQL, T0)


def test_due_never_raises(world: World) -> None:
    (world.ledger.directory).mkdir(parents=True)
    (world.ledger.directory / "lane_events.jsonl").write_text("{bad\n{bad\n", encoding="utf-8")
    assert world.publisher.due(LANE_EFFISQL, T0) is False


def test_the_body_lists_notifications_since_the_last_pr(world: World) -> None:
    world.event(
        LANE_EVENT_KIND_NOTIFY_FAILED,
        notification="lane stopped: x",
        notification_id="stopped-1",
        detail="attempt 1: down",
    )
    world.event(
        LANE_EVENT_KIND_NOTIFIED, notification="lane stopped: x", notification_id="stopped-1"
    )
    world.ledger.record_experiment(
        ExperimentRecord(
            experiment_id="p1",
            lane=LANE_EFFISQL,
            round=1,
            hypothesis=Hypothesis(hypothesis_id="h"),
            decision=DECISION_PENDING,
        )
    )
    assert world.publisher.publish(LANE_EFFISQL, final=False, now=T0) is not None
    body = world.github.pulls[0]["body"]
    assert "`stopped-1`: lane stopped: x (not delivered (attempt 1: down))" in body
    assert "`stopped-1`: lane stopped: x (delivered)" in body
    assert "## Experiments decided since the last results PR\n\nNone." in body


# The GitHub client and the HTTPS remote.


def test_github_errors_name_the_status_but_never_the_token() -> None:
    api = GitHubApi(
        REPO,
        lambda: TOKEN,
        transport=httpx.MockTransport(lambda _: httpx.Response(401, text=TOKEN)),
    )
    with pytest.raises(ResultsPrError, match="HTTP 401") as caught:
        api.find_pr("rsi/effisql/results-1")
    assert TOKEN not in str(caught.value)

    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(str(request.headers))

    api = GitHubApi(REPO, lambda: TOKEN, transport=httpx.MockTransport(fail))
    with pytest.raises(ResultsPrError, match=r"ConnectError$"):
        api.branch_head("x")
    api = GitHubApi(
        REPO, lambda: TOKEN, transport=httpx.MockTransport(lambda _: httpx.Response(200, text="<"))
    )
    with pytest.raises(ResultsPrError, match="not JSON"):
        api.commit_message("abc")
    api.close()


def test_https_remote_passes_the_token_only_in_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[list[str], dict[str, str] | None]] = []

    def fake_git(
        args: list[str], *, cwd: Path, env: dict[str, str] | None = None, **_: object
    ) -> str:
        calls.append((list(args), env))
        return "abc\n"

    monkeypatch.setattr(results_pr, "run_git", fake_git)
    remote = HttpsRemote(REPO, lambda: TOKEN)
    assert remote.fetch(tmp_path, "main") == "abc"
    remote.push(tmp_path, "abc", "rsi/effisql/results-1")
    with pytest.raises(ResultsPrError, match="only rsi/"):
        remote.push(tmp_path, "abc", "main")
    pushes = [args for args, _ in calls if args[0] == "push"]
    assert pushes == [
        [
            "push",
            "--no-verify",
            "https://github.com/Catflirter/goaly.git",
            "abc:refs/heads/rsi/effisql/results-1",
        ]
    ]
    for args, env in calls:
        assert all(TOKEN not in arg for arg in args)
        if args[0] in ("fetch", "push"):
            assert env is not None
            assert env["GIT_CONFIG_KEY_0"] == "http.https://github.com/.extraheader"
            assert TOKEN not in env["GIT_CONFIG_VALUE_0"]  # Base64, in a header.
            assert env["GIT_CONFIG_VALUE_0"].startswith("Authorization: Basic ")


def test_runtime_token_prefers_the_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOALY_API_KEY", "k")
    monkeypatch.setenv("GOALY_GITHUB_TOKEN", TOKEN)
    assert runtime_github_token() == TOKEN


def fake_coder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, script: str) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    coder = bin_dir / "coder"
    coder.write_text(f"#!/bin/sh\n{script}\n", encoding="utf-8")
    coder.chmod(coder.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("GOALY_API_KEY", "k")


def test_runtime_token_falls_back_to_coder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_coder(
        tmp_path, monkeypatch, f'[ "$*" = "external-auth access-token github" ] && echo {TOKEN}'
    )
    assert runtime_github_token() == TOKEN


def test_runtime_token_failure_quotes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_coder(tmp_path, monkeypatch, f"echo {TOKEN} >&2; exit 2")
    with pytest.raises(ResultsPrError, match="coder exited with 2") as caught:
        runtime_github_token()
    assert TOKEN not in str(caught.value)
    monkeypatch.setenv("PATH", str(tmp_path / "nowhere"))
    with pytest.raises(ResultsPrError, match="FileNotFoundError"):
        runtime_github_token()
