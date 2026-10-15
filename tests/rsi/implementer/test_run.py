"""Tests for goaly.rsi.implementer.run: an implementer run, its exit and its one commit.

The agent is mini-swe-agent's own DeterministicToolcallModel, and the sandbox a fake environment
that runs the scripted commands with bash in the worktree.
"""

import asyncio
import concurrent.futures
import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from goaly.contracts.gen.experiment_pb2 import (
    CHANGE_KIND_SKILL,
    LANE_EFFISQL,
    Hypothesis,
    ImplementerExit,
)
from goaly.contracts.gen.llm_call_pb2 import CallPurpose
from goaly.harness.core.llm_errors import ContextOverflow, InvalidResponse, ServerError, TimedOut
from goaly.rsi.commit import commit_candidate
from goaly.rsi.implementer import msa
from goaly.rsi.implementer.environment import HostCommand, SandboxStoppedError
from goaly.rsi.implementer.model import (
    EXIT_COST_LIMIT,
    EXIT_INTERRUPTED,
    EXIT_STEP_LIMIT,
    EXIT_TIME_LIMIT,
)
from goaly.rsi.implementer.run import (
    EnvironmentFactory,
    Implementation,
    Implementer,
    exit_for_error,
    exit_for_status,
)
from goaly.rsi.implementer.smoke import SmokeRunner
from goaly.rsi.sandbox import SandboxConfig, SandboxError
from goaly.rsi.workspace import Workspace, WorkspaceError, Worktree
from tests.harness.core.llm_fakes import Gateway, Recorder, completion, connect
from tests.rsi.fakedocker import FakeDocker
from tests.rsi.gitrepo import git, make_repo
from tests.rsi.implementer.fakes import (
    FakeEnvironment,
    FakeRunner,
    NoLlm,
    ScriptedModel,
    bash_call,
    shell,
    submit,
)

E = ImplementerExit
EXPERIMENT = "exp-1"
HYPOTHESIS = Hypothesis(
    hypothesis_id="hyp-3",
    failure_pattern="correlated subqueries are kept",
    target_instance_ids=["12", "40"],
    change_kind=CHANGE_KIND_SKILL,
    change="Add a skill on decorrelating subqueries.",
    expected_effect="More rewrites of correlated subqueries.",
    risk="Longer prompts.",
)
TRAIN_IDS = ("12", "40", "77")
SKILL = "src/goaly/harness/packs/effisql/skills/decorrelate.md"
WRITE_SKILL = f"mkdir -p $(dirname {SKILL}) && printf 'Join instead.\\n' > {SKILL}"


def glm_reply(command: str) -> httpx.Response:
    return completion("", tool_calls=[bash_call(command)], finish_reason="tool_calls")


def exit_message(status: str) -> dict[str, Any]:
    return {"role": "exit", "content": status, "extra": {"exit_status": status, "submission": ""}}


class Lane:
    """A throwaway repository, its workspace, one experiment's worktree, and an Implementer."""

    def __init__(self, tmp_path: Path) -> None:
        self.workspace = Workspace(make_repo(tmp_path / "repo"), root=tmp_path / "worktrees")
        self.worktree = self.workspace.create(LANE_EFFISQL, EXPERIMENT)
        self.base = self.workspace.head(self.worktree)
        self.runs = tmp_path / "runs"
        self.environment = FakeEnvironment(shell(self.worktree.path))
        self.pool = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="implementer")
        self.stops: list[threading.Event] = []

    def implementer(
        self,
        *commands: str | BaseException,
        make_environment: EnvironmentFactory | None = None,
        smoke: SmokeRunner | None = None,
    ) -> Implementer:
        def make_model(experiment_id: str, stop: threading.Event) -> msa.Model:
            assert experiment_id == EXPERIMENT
            self.stops.append(stop)
            return ScriptedModel(*commands)

        return Implementer(
            NoLlm(),
            self.workspace,
            image="unused",
            runs_dir=self.runs,
            pool=self.pool,
            make_environment=make_environment or self.fake_sandbox,
            make_model=make_model,
            smoke=smoke,
            train_ids=TRAIN_IDS,
        )

    def fake_sandbox(self, worktree: Worktree, smoke: HostCommand | None) -> FakeEnvironment:
        self.environment.smoke = smoke
        return self.environment

    def run(self, *commands: str | BaseException) -> Implementation:
        return self.implementer(*commands).run(HYPOTHESIS, self.worktree, threading.Event())

    def commits(self) -> list[str]:
        return git(self.worktree.path, "rev-list", f"{self.base}..HEAD").split()

    def trajectory(self) -> dict[str, Any]:
        path = self.runs / "rsi" / "effisql" / EXPERIMENT / "implementer.traj.json"
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return data


@pytest.fixture
def lane(tmp_path: Path) -> Iterator[Lane]:
    lane = Lane(tmp_path)
    yield lane
    lane.pool.shutdown(wait=True)


def test_a_finished_run_ends_as_one_commit_with_the_trailers(lane: Lane) -> None:
    done = lane.run(WRITE_SKILL, submit("Add a decorrelation skill"))

    assert done.run.exit == E.IMPLEMENTER_EXIT_SUBMITTED
    assert done.run.summary == "Add a decorrelation skill"
    assert lane.commits() == [done.commit]
    message = git(lane.worktree.path, "log", "-1", "--format=%B")
    assert message.startswith("rsi(effisql): Add a decorrelation skill\n")
    assert "Experiment-Id: exp-1\nHypothesis-Id: hyp-3\n" in message
    changed = git(lane.worktree.path, "show", "--name-only", "--format=", "HEAD").split()
    assert changed == [SKILL]
    assert done.run.step_count == 2
    assert done.run.wall_time_s > 0
    assert done.run.trajectory_path == "rsi/effisql/exp-1/implementer.traj.json"
    assert lane.trajectory()["trajectory_format"] == "mini-swe-agent-1.1"
    assert lane.environment.cleanups == 1


def test_the_prompt_gives_the_hypothesis_paths_rules_and_limits(lane: Lane) -> None:
    lane.run(submit())

    system, task = (message["content"] for message in lane.trajectory()["messages"][:2])
    assert "at most 60 steps, $3.0 of model calls and 30 minutes" in system
    assert "stopped after 120 s" in system
    assert str(lane.worktree.path) in system
    assert "Add a skill on decorrelating subqueries." in task
    assert "Train tasks that fail this way: 12, 40" in task
    assert "Kind of change: CHANGE_KIND_SKILL" in task
    assert f"{lane.worktree.path}/src/goaly/harness/packs/effisql/" in task
    assert f"{lane.worktree.path}/tests/harness/packs/effisql/" in task
    assert "json, math, re" in task  # The allowlist's stdlib modules, in plain words.
    assert "`goaly.harness.core.tools`" in task
    assert "Literal" in task
    assert "do not use the network" in system
    assert "git does not work in the sandbox" in task
    assert "never `import goaly.harness.core.<module>`" in task
    assert 'never `data["key"]`' in task
    assert "no `from_attributes`" in task
    assert "A class body may define no dunder" in task
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in task


def test_a_finish_without_a_change_makes_no_commit(lane: Lane) -> None:
    done = lane.run(submit("nothing to do"))

    assert done.run.exit == E.IMPLEMENTER_EXIT_NO_CHANGES
    assert done.commit is None
    assert lane.commits() == []
    assert done.run.summary == "nothing to do"


@pytest.mark.parametrize(
    ("ending", "expected"),
    [
        pytest.param(
            msa.LimitsExceeded(exit_message(EXIT_STEP_LIMIT)),
            E.IMPLEMENTER_EXIT_STEP_LIMIT,
            id="steps",
        ),
        pytest.param(
            msa.LimitsExceeded(exit_message(EXIT_COST_LIMIT)),
            E.IMPLEMENTER_EXIT_COST_LIMIT,
            id="cost",
        ),
        pytest.param(
            msa.TimeExceeded(exit_message(EXIT_TIME_LIMIT)),
            E.IMPLEMENTER_EXIT_TIME_LIMIT,
            id="time",
        ),
        pytest.param(
            msa.InterruptAgentFlow(exit_message(EXIT_INTERRUPTED)),
            E.IMPLEMENTER_EXIT_INTERRUPTED,
            id="stopped",
        ),
        pytest.param(
            ContextOverflow("too long"), E.IMPLEMENTER_EXIT_CRASHED, id="context-overflow"
        ),
        pytest.param(
            InvalidResponse("not a completion"), E.IMPLEMENTER_EXIT_CRASHED, id="invalid-response"
        ),
        pytest.param(ValueError("a bug"), E.IMPLEMENTER_EXIT_CRASHED, id="crash"),
        pytest.param(
            ServerError("502 after 5 retries"), E.IMPLEMENTER_EXIT_INFRASTRUCTURE, id="gateway"
        ),
        pytest.param(
            TimedOut("no response"), E.IMPLEMENTER_EXIT_INFRASTRUCTURE, id="gateway-timeout"
        ),
        pytest.param(
            concurrent.futures.CancelledError(), E.IMPLEMENTER_EXIT_INTERRUPTED, id="cancelled"
        ),
    ],
)
def test_a_run_that_does_not_finish_makes_no_commit(
    lane: Lane, ending: BaseException, expected: ImplementerExit.ValueType
) -> None:
    done = lane.run(WRITE_SKILL, ending)  # The pack changed, but the run never finished.

    assert done.run.exit == expected
    assert done.commit is None
    assert lane.commits() == []
    assert done.run.step_count == 1
    assert done.run.summary == ""
    assert lane.trajectory()["messages"][-1]["role"] == "exit"
    assert lane.environment.cleanups == 1


def test_repeated_format_errors_make_no_commit(lane: Lane) -> None:
    error = msa.FormatError({"role": "user", "content": "no call", "extra": {}})
    done = lane.run(WRITE_SKILL, error, error, error)

    assert done.run.exit == E.IMPLEMENTER_EXIT_FORMAT_ERRORS
    assert done.commit is None
    assert lane.commits() == []


def test_a_docker_failure_mid_run_aborts_the_experiment(lane: Lane) -> None:
    def broken(command: str) -> dict[str, Any]:
        raise SandboxError("docker exec failed (1): the container stopped")

    lane.environment = FakeEnvironment(broken)

    done = lane.run(WRITE_SKILL, submit())

    assert done.run.exit == E.IMPLEMENTER_EXIT_INFRASTRUCTURE
    assert done.commit is None
    assert lane.environment.cleanups == 1


def test_a_sandbox_that_never_starts_aborts_the_experiment(lane: Lane) -> None:
    def no_docker(worktree: Worktree, smoke: HostCommand | None) -> FakeEnvironment:
        raise SandboxError("cannot run docker")

    implementer = lane.implementer(submit(), make_environment=no_docker)
    done = implementer.run(HYPOTHESIS, lane.worktree, threading.Event())

    assert done.run.exit == E.IMPLEMENTER_EXIT_INFRASTRUCTURE
    assert not done.run.HasField("step_count")
    assert done.run.trajectory_path == ""


def test_a_run_told_to_stop_commits_nothing_even_after_it_finished(lane: Lane) -> None:
    stop = threading.Event()

    def stop_then_run(command: str) -> dict[str, Any]:
        stop.set()
        return shell(lane.worktree.path)(command)

    lane.environment = FakeEnvironment(stop_then_run)

    done = lane.implementer(WRITE_SKILL, submit()).run(HYPOTHESIS, lane.worktree, stop)

    assert done.run.exit == E.IMPLEMENTER_EXIT_INTERRUPTED
    assert done.commit is None
    assert lane.commits() == []


def test_a_write_outside_the_pack_is_never_committed(lane: Lane) -> None:
    done = lane.run("printf 'x' >> README.md", WRITE_SKILL, submit())

    assert done.run.exit == E.IMPLEMENTER_EXIT_CRASHED
    assert done.commit is None
    assert lane.commits() == []


def test_a_worktree_on_another_branch_aborts_the_experiment(lane: Lane) -> None:
    git(lane.worktree.path, "checkout", "--quiet", "-b", "elsewhere")

    done = lane.run(WRITE_SKILL, submit())

    assert done.run.exit == E.IMPLEMENTER_EXIT_INFRASTRUCTURE
    assert done.commit is None


@pytest.mark.anyio
async def test_implement_runs_on_the_implementer_pool(lane: Lane) -> None:
    threads: list[str] = []

    def record_thread(command: str) -> dict[str, Any]:
        threads.append(threading.current_thread().name)
        return shell(lane.worktree.path)(command)

    lane.environment = FakeEnvironment(record_thread)

    done = await lane.implementer(WRITE_SKILL, submit()).implement(HYPOTHESIS, lane.worktree)

    assert done.run.exit == E.IMPLEMENTER_EXIT_SUBMITTED
    assert threads
    assert all(name.startswith("implementer") for name in threads)


@pytest.mark.anyio
async def test_cancelling_implement_stops_the_run_before_its_commit(lane: Lane) -> None:
    started, release = threading.Event(), threading.Event()

    def block(command: str) -> dict[str, Any]:
        started.set()
        release.wait(timeout=30)
        return shell(lane.worktree.path)(command)

    lane.environment = FakeEnvironment(block)
    implementer = lane.implementer(WRITE_SKILL, submit())
    task = asyncio.create_task(implementer.implement(HYPOTHESIS, lane.worktree))
    await asyncio.to_thread(started.wait, 30)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert lane.stops[0].is_set()
    release.set()
    await asyncio.to_thread(lane.pool.shutdown, wait=True)

    assert lane.commits() == []
    assert lane.environment.cleanups == 1


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("Submitted", E.IMPLEMENTER_EXIT_SUBMITTED),
        ("RepeatedFormatError", E.IMPLEMENTER_EXIT_FORMAT_ERRORS),
        ("LimitsExceeded", E.IMPLEMENTER_EXIT_CRASHED),  # The agent's own limits are off.
        ("", E.IMPLEMENTER_EXIT_CRASHED),
    ],
)
def test_exit_for_status(status: str, expected: ImplementerExit.ValueType) -> None:
    assert exit_for_status(status) == expected


def test_exit_for_error_aborts_on_goalys_own_failures() -> None:
    assert exit_for_error(WorkspaceError("not registered")) == E.IMPLEMENTER_EXIT_INFRASTRUCTURE
    assert exit_for_error(SandboxError("docker hung")) == E.IMPLEMENTER_EXIT_INFRASTRUCTURE


@pytest.mark.anyio
async def test_by_default_glm_runs_in_the_sandbox(tmp_path: Path, lane: Lane) -> None:
    docker = FakeDocker(tmp_path / "docker")
    docker.script({"exec": {"stdout": f"{msa.SUBMIT_MARKER}\nNothing to change.\n"}})
    gateway, recorder = Gateway(glm_reply("printf ...")), Recorder()
    async with connect(gateway, recorder=recorder) as llm:
        implementer = Implementer(
            llm,
            lane.workspace,
            image="goaly-rsi-sandbox:test",
            runs_dir=lane.runs,
            pool=lane.pool,
            sandbox=SandboxConfig(docker=str(docker.path)),
        )

        done = await implementer.implement(HYPOTHESIS, lane.worktree)

    assert done.run.exit == E.IMPLEMENTER_EXIT_NO_CHANGES
    assert done.run.summary == "Nothing to change."
    assert docker.commands() == ["ps", "run", "exec", "rm", "ps", "ps"]
    (record,) = recorder.records
    assert (record.purpose, record.run_id) == (CallPurpose.CALL_PURPOSE_RSI_IMPL, EXPERIMENT)
    assert lane.trajectory()["info"]["config"]["model"]["run_id"] == EXPERIMENT


def test_the_runs_smoke_tests_are_recorded_and_offered_in_the_prompt(lane: Lane) -> None:
    runner = FakeRunner()
    implementer = lane.implementer(
        WRITE_SKILL,
        "goaly-smoke 12",
        "goaly-smoke 31",  # Not a train task.
        "goaly-smoke 40",
        "goaly-smoke 77",  # The third run.
        submit("Add a skill"),
        smoke=runner,
    )

    done = implementer.run(HYPOTHESIS, lane.worktree, threading.Event())

    assert done.run.exit == E.IMPLEMENTER_EXIT_SUBMITTED
    assert [test.instance_id for test in done.run.smoke_tests] == ["12", "31", "40", "77"]
    assert [test.HasField("score") for test in done.run.smoke_tests] == [True, False, True, False]
    assert [instance for _, instance in runner.ran] == ["12", "40"]
    assert lane.environment.events == ["paused", "resumed"] * 2  # Once per smoke run.
    task = lane.trajectory()["messages"][1]["content"]
    assert "run `goaly-smoke <instance_id>`" in task
    assert "at most 2 runs per hypothesis" in task


def test_without_a_runner_goaly_smoke_is_not_offered(lane: Lane) -> None:
    done = lane.run(submit())

    assert len(done.run.smoke_tests) == 0
    assert "goaly-smoke" not in lane.trajectory()["messages"][1]["content"]


@pytest.mark.parametrize(
    "lock_out",
    [
        pytest.param(f"chmod 000 {SKILL}", id="unreadable-file"),
        pytest.param(f"chmod 000 $(dirname {SKILL})", id="unsearchable-directory"),
    ],
)
def test_a_pack_the_host_cannot_read_rejects_the_candidate(lane: Lane, lock_out: str) -> None:
    done = lane.run(WRITE_SKILL, lock_out, submit("Add a skill"))

    assert done.run.exit == E.IMPLEMENTER_EXIT_CRASHED  # Rejected: neither aborted nor empty.
    assert done.commit is None
    assert lane.commits() == []
    # Give the test's cleanup a readable tree.
    (lane.worktree.path / SKILL).parent.chmod(0o755)
    (lane.worktree.path / SKILL).chmod(0o644)


def test_the_commit_comes_only_after_the_sandbox_is_gone(
    lane: Lane, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    environment_exit = FakeEnvironment.__exit__

    def record_cleanup(self: FakeEnvironment, *exc_info: object) -> None:
        events.append("sandbox removed")
        environment_exit(self, *exc_info)

    def record_commit(*args: Any, **kwargs: Any) -> str | None:
        events.append("commit")
        return commit_candidate(*args, **kwargs)

    monkeypatch.setattr(FakeEnvironment, "__exit__", record_cleanup)
    monkeypatch.setattr("goaly.rsi.implementer.run.commit_candidate", record_commit)

    done = lane.run(WRITE_SKILL, submit("Add a skill"))

    assert done.commit is not None
    assert events == ["sandbox removed", "commit"]


def test_a_sandbox_the_agent_stopped_rejects_the_candidate(lane: Lane) -> None:
    def stopped(command: str) -> dict[str, Any]:
        raise SandboxStoppedError("the sandbox container stopped: its init exited (0)")

    lane.environment = FakeEnvironment(stopped)

    done = lane.run("pkill sleep", submit())

    assert done.run.exit == E.IMPLEMENTER_EXIT_CRASHED


def test_a_stopped_run_is_interrupted_whatever_it_saw(lane: Lane) -> None:
    stop = threading.Event()

    def removed(command: str) -> dict[str, Any]:
        stop.set()  # As implement() does when cancelled, removing the sandbox under the run.
        raise SandboxError("the sandbox container is not running")

    lane.environment = FakeEnvironment(removed)

    done = lane.implementer(WRITE_SKILL, submit()).run(HYPOTHESIS, lane.worktree, stop)

    assert done.run.exit == E.IMPLEMENTER_EXIT_INTERRUPTED


def test_the_summary_is_cleaned_and_bounded(lane: Lane) -> None:
    noisy = "Added a skill\\t\\x1b[31mred\\x1b[0m\\n" + "y" * 5000
    done = lane.run(WRITE_SKILL, f"printf '%b\\n' {msa.SUBMIT_MARKER} '{noisy}'")

    assert done.run.summary.startswith("Added a skill [31mred[0m\n")
    assert len(done.run.summary) == 4000


@pytest.mark.anyio
async def test_cancelling_removes_the_sandbox_under_the_command_in_flight(
    tmp_path: Path, lane: Lane
) -> None:
    docker = FakeDocker(tmp_path / "docker")
    docker.script({"exec": {"sleep": 2}})
    implementer = Implementer(
        NoLlm(),
        lane.workspace,
        image="goaly-rsi-sandbox:test",
        runs_dir=lane.runs,
        pool=lane.pool,
        sandbox=SandboxConfig(docker=str(docker.path)),
        make_model=lambda experiment_id, stop: ScriptedModel(WRITE_SKILL, submit()),
    )
    task = asyncio.create_task(implementer.implement(HYPOTHESIS, lane.worktree))
    while "exec" not in docker.commands():
        await asyncio.sleep(0.05)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.to_thread(lane.pool.shutdown, wait=True)

    # Right after the exec began: the removal of the experiment's containers, by label.
    assert docker.commands()[3] == "ps"
    assert docker.calls()[3]["args"][-1] == "--filter=label=goaly.rsi.experiment=exp-1"
    assert lane.commits() == []


def test_a_failed_removal_on_cancel_is_logged(lane: Lane, caplog: pytest.LogCaptureFixture) -> None:
    implementer = Implementer(
        NoLlm(),
        lane.workspace,
        image="unused",
        runs_dir=lane.runs,
        pool=lane.pool,
        sandbox=SandboxConfig(docker="/nonexistent/docker"),
    )

    implementer._abandon(lane.worktree)

    assert "could not remove a stopped run's sandbox" in caplog.text
