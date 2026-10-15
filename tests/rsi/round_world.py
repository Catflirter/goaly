"""A toy world for the round and lane tests: a git repository, a scripted GLM, a fake implementer
and a deterministic toy evaluator.

A hypothesis's ID names its candidate's behavior, by its first word:

- ``good``: solves the 5 failing train tasks and every val task, in every seed;
- ``meh``: solves the failing train tasks, and val tasks at credit 0.75;
- ``noisy``: like ``good`` in the validation seed, but solves nothing in the confirmation seed;
- ``bad``: changes nothing that the evaluator sees;
- ``flag``: like ``good``, but its first val prediction is flagged hard-coded;
- ``crash``: the implementer stops at its step limit and commits nothing;
- ``infra``: the implementer ends on a gateway failure;
- ``guard``: its candidate fails a guard;
- ``broken``: its evaluation raises, as when the solve container fails;
- ``late``: like ``good``, but its confirmation raises;
- ``short``: its validation comes back without one task's result;
- ``graderr``: its validation has a grader error;
- ``flaky``: like ``good``, but the infrastructure fails 2 of its 8 validation tasks (25%);
- ``wobbly``: like ``good``, but the infrastructure fails 1 of them (12.5%), under the 20% cap;
- ``hostfail``: like ``good``, but the host counts 2 infrastructure failures in its validation;
- ``extra``: its validation returns one result more than was asked for.

The best (v0) fails train tasks 1 to 5, solves 6 to 10, and scores 0.5 on every val task.
"""

import asyncio
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
from pydantic import SecretStr

from goaly.contracts.gen.experiment_pb2 import (
    CHANGE_KIND_PROMPT,
    GUARD_ALLOWED_PATHS,
    GUARD_UNIT_TESTS,
    HYPOTHESIS_SOURCE_ANALYST,
    IMPLEMENTER_EXIT_INFRASTRUCTURE,
    IMPLEMENTER_EXIT_STEP_LIMIT,
    IMPLEMENTER_EXIT_SUBMITTED,
    LANE_EFFISQL,
    GuardResult,
    Hypothesis,
    ImplementerRun,
)
from goaly.contracts.gen.llm_call_pb2 import LlmCallRecord
from goaly.contracts.gen.result_pb2 import TASK_ERROR_TYPE_LLM, TaskResult
from goaly.contracts.gen.task_pb2 import (
    GRADE_STATUS_GRADED,
    GRADE_STATUS_GRADER_ERROR,
    EffiSqlGrade,
    Grade,
)
from goaly.contracts.gen.trajectory_pb2 import Trajectory
from goaly.harness.core.llm import LlmClient
from goaly.harness.core.llm_models import IMPROVEMENT_MODEL
from goaly.harness.core.settings import Settings
from goaly.rsi.analyst import AnalystInput
from goaly.rsi.commit import commit_candidate
from goaly.rsi.decision import CostStats
from goaly.rsi.hypothesis_queue import HypothesisQueue
from goaly.rsi.ledger import Ledger
from goaly.rsi.lessons import LessonStore
from goaly.rsi.round import (
    Evaluation,
    EvaluationRequest,
    RoundConfig,
    RoundDeps,
    RoundRunner,
)
from goaly.rsi.workspace import Workspace, Worktree
from tests.rsi.gitrepo import git, make_repo

TRAIN = tuple(str(n) for n in range(1, 11))
FAILING = TRAIN[:5]
VAL = tuple(str(n) for n in range(101, 109))
_SPEEDUP = {0.25: 0.6, 0.5: 1.0, 0.75: 1.5, 1.0: 3.0}
_GOODLIKE = frozenset({"flaky", "wobbly", "hostfail", "extra"})


def hypothesis(hypothesis_id: str) -> Hypothesis:
    return Hypothesis(
        hypothesis_id=hypothesis_id,
        failure_pattern="timid rewrites: no speedup",
        target_instance_ids=FAILING,
        change_kind=CHANGE_KIND_PROMPT,
        change=f"Tell the agent to try {hypothesis_id}.",
        expected_effect="Faster queries.",
        risk="None.",
        source=HYPOTHESIS_SOURCE_ANALYST,
    )


def behavior(hypothesis_id: str) -> str:
    return hypothesis_id.split("-")[0]


def credit(kind: str, task: str, seed: int) -> float:
    """The checked credit a version of ``kind`` gets on ``task`` in ``seed``."""
    improves = kind in {"good", "meh", "noisy", "flag", "late", "short", "graderr"} | _GOODLIKE
    if task in TRAIN:
        return 1.0 if task not in FAILING or improves else 0.0
    if kind in {"good", "flag", "late", "short", "graderr"} | _GOODLIKE:
        return 1.0
    if kind == "meh":
        return 0.75
    if kind == "noisy":
        return 1.0 if seed == 0 else 0.0
    return 0.5


def task_result(run_id: str, task: str, seed: int, value: float, *, flagged: bool) -> TaskResult:
    grade = EffiSqlGrade(credit=value, matched=value > 0 or flagged)
    if grade.matched:
        grade.speedup = _SPEEDUP.get(value, 3.0)
    if flagged:
        grade.credit = 0.0
        grade.alt_matched = False
    return TaskResult(
        run_id=run_id,
        instance_id=task,
        seed=seed,
        grade=Grade(status=GRADE_STATUS_GRADED, effisql=grade),
        wall_time_s=999.0,  # The container's number, which the decisions never use (V27).
    )


@dataclass
class ToyEvaluator:
    """Evaluates commits by the behavior of the hypothesis that made them; v0 and the best
    behave as ``base`` until a candidate is accepted, whose commit then keeps its behavior.
    """

    kinds: dict[str, str] = field(default_factory=dict)  # Commit -> behavior.
    requests: list[EvaluationRequest] = field(default_factory=list)
    concurrency: int = 16
    in_flight: int = 0
    most_in_flight: int = 0
    gate: asyncio.Event | None = None  # When set, validations wait for it.
    train_known: bool = True  # Whether v0's train results exist.
    kind_of: Callable[[str], str] | None = None  # The behavior of a commit not in ``kinds``.

    def __post_init__(self) -> None:
        self._limit = asyncio.Semaphore(self.concurrency)

    async def evaluate(self, request: EvaluationRequest) -> Evaluation:
        self.requests.append(request)
        kind = self._kind(request.commit)
        if kind == "broken" or (kind == "late" and request.phase == "confirmation"):
            raise RuntimeError("the solve container exited with 137")
        if self.gate is not None and request.phase == "validation":
            await self.gate.wait()
        results = []
        for task in request.instance_ids:
            for seed in request.seeds:
                async with self._limit:
                    self.in_flight += 1
                    self.most_in_flight = max(self.most_in_flight, self.in_flight)
                    await asyncio.sleep(0)
                    self.in_flight -= 1
                flagged = kind == "flag" and task == VAL[0]
                value = credit(kind, task, seed)
                results.append(task_result(request.run_id, task, seed, value, flagged=flagged))
        if request.phase == "validation" and kind == "short":
            results.pop()
        if request.phase == "validation" and kind == "graderr":
            results[0].grade.status = GRADE_STATUS_GRADER_ERROR
        if request.phase == "validation" and kind in {"flaky", "wobbly"}:
            for result in results[: 2 if kind == "flaky" else 1]:
                result.error_type = TASK_ERROR_TYPE_LLM
        if request.phase == "validation" and kind == "extra":
            results.append(task_result(request.run_id, TRAIN[0], 0, 1.0, flagged=False))
        infra = 2 if request.phase == "validation" and kind == "hostfail" else None
        count = len(results)
        stats = CostStats(1000.0, 0.01, 10.0, count=count)
        return Evaluation(results, stats, infra_failures=infra)

    def _kind(self, commit: str) -> str:
        if commit in self.kinds:
            return self.kinds[commit]
        return "base" if self.kind_of is None else self.kind_of(commit)

    def known_train(self, commit: str) -> tuple[Sequence[TaskResult], Sequence[Trajectory]]:
        kind = self._kind(commit)
        if kind != "base" or not self.train_known:
            return [], []
        results = [
            task_result("v0-train", task, 0, credit("base", task, 0), flagged=False)
            for task in TRAIN
        ]
        unscored = task_result("v0-train", TRAIN[0], 1, 1.0, flagged=False)
        unscored.grade.status = GRADE_STATUS_GRADER_ERROR  # Left out: it says nothing.
        results.append(unscored)
        return results, [Trajectory(run_id="v0-train", instance_id=task) for task in TRAIN]


@dataclass(frozen=True)
class Built:
    """What the fake implementer left: T20's Implementation."""

    run: ImplementerRun
    commit: str | None


@dataclass
class FakeImplementer:
    """Writes a file into the pack and commits it, or ends as the hypothesis's behavior says."""

    workspace: Workspace
    evaluator: ToyEvaluator
    calls: list[str] = field(default_factory=list)
    gate: asyncio.Event | None = None  # When set, every run waits for it.

    async def implement(self, hypothesis: Hypothesis, worktree: Worktree) -> Built:
        self.calls.append(hypothesis.hypothesis_id)
        if self.gate is not None:
            await self.gate.wait()
        kind = behavior(hypothesis.hypothesis_id)
        if kind == "crash":
            return Built(ImplementerRun(exit=IMPLEMENTER_EXIT_STEP_LIMIT, step_count=60), None)
        if kind == "infra":
            return Built(ImplementerRun(exit=IMPLEMENTER_EXIT_INFRASTRUCTURE), None)
        source, _ = worktree.pack_dirs
        (source / f"{hypothesis.hypothesis_id}.md").write_text(f"{kind}\n", encoding="utf-8")
        commit = await asyncio.to_thread(
            commit_candidate,
            self.workspace,
            worktree,
            hypothesis_id=hypothesis.hypothesis_id,
            summary=f"Try {hypothesis.hypothesis_id}",
        )
        assert commit is not None
        self.evaluator.kinds[commit] = kind
        run = ImplementerRun(exit=IMPLEMENTER_EXIT_SUBMITTED, step_count=3, summary="done")
        return Built(run, commit)


class FakeGuards:
    """Passes every candidate except one whose evaluator behavior is ``guard``."""

    def __init__(self, evaluator: ToyEvaluator) -> None:
        self.evaluator = evaluator
        self.calls: list[tuple[str, str, str]] = []

    def __call__(self, experiment_id: str, parent: str, candidate: str) -> list[GuardResult]:
        self.calls.append((experiment_id, parent, candidate))
        paths = GuardResult(guard=GUARD_ALLOWED_PATHS, passed=True)
        if self.evaluator.kinds.get(candidate) == "guard":
            return [paths, GuardResult(guard=GUARD_UNIT_TESTS, passed=False, detail="1 failed")]
        return [paths]


@dataclass
class ScriptedGlm:
    """GLM behind core's real client: the analyst answers ``analyst_ids`` in turn, or invalid
    JSON while ``invalid`` is set or none is left, and every lesson call gets a lesson.
    """

    analyst_ids: list[list[str]] = field(default_factory=list)
    invalid: bool = False
    fail: bool = False
    bodies: list[dict[str, Any]] = field(default_factory=list)
    records: list[LlmCallRecord] = field(default_factory=list)

    def record(self, record: LlmCallRecord) -> None:
        self.records.append(record)

    async def handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        if self.fail:
            return httpx.Response(400, json={"error": {"message": "refused"}})
        prompt = body["messages"][-1]["content"]
        if "Write the lesson." in prompt:
            content = json.dumps({"lesson": "The prompt change helped on train but not val."})
        elif self.invalid or not self.analyst_ids:
            content = "I cannot answer in JSON."
        else:
            content = json.dumps(_analyst_answer(self.analyst_ids.pop(0)))
        reply = {
            "id": f"chatcmpl-{len(self.bodies)}",
            "object": "chat.completion",
            "model": IMPROVEMENT_MODEL,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content, "reasoning_content": "."},
                }
            ],
            "usage": {
                "prompt_tokens": 1001,
                "completion_tokens": 300,
                "completion_tokens_details": {"reasoning_tokens": 201},
            },
        }
        return httpx.Response(200, json=reply, headers={"x-litellm-response-cost": "0.03"})

    def client(self) -> LlmClient:
        settings = Settings(api_key=SecretStr("sk-test-never-sent"), base_url="https://gw.test/v1")
        return LlmClient(settings, self, transport=httpx.MockTransport(self.handle))

    def purposes(self) -> list[str]:
        return [body["messages"][0]["content"][:30] for body in self.bodies]


def _analyst_answer(ids: list[str]) -> dict[str, Any]:
    return {
        "failure_patterns": [
            {"name": "timid", "task_ids": list(FAILING), "description": "No speedup."}
        ],
        "hypotheses": [
            {
                "id": hypothesis_id,
                "target_pattern": "timid",
                "change_kind": "prompt",
                "change": f"In prompt.md, try {hypothesis_id}.",
                "expected_effect": "Faster.",
                "risk": "Slower easy tasks.",
                "source": "analysis",
            }
            for hypothesis_id in ids
        ],
    }


def analyst_input(
    results: Sequence[TaskResult], trajectories: Sequence[Trajectory], queued: Sequence[Hypothesis]
) -> AnalystInput:
    failed = frozenset(r.instance_id for r in results if r.grade.effisql.credit < 0.75)
    return AnalystInput(
        system="You analyze failures.",
        user=f"{len(failed)} train tasks failed; {len(queued)} hypotheses are queued.",
        failed_task_ids=failed,
        shown_task_ids=tuple(sorted(failed)),
        dropped_task_ids=(),
        taken_ids=frozenset(h.hypothesis_id for h in queued),
    )


class Clock:
    """A clock that advances one second per reading."""

    def __init__(self, start: datetime | None = None) -> None:
        self.now = start or datetime(2026, 10, 15, 16, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now


@dataclass
class World:
    """Everything a round needs, in ``root``."""

    root: Path
    repo: Path
    workspace: Workspace
    ledger: Ledger
    queue: HypothesisQueue
    lessons: LessonStore
    evaluator: ToyEvaluator
    implementer: FakeImplementer
    guards: FakeGuards
    glm: ScriptedGlm
    v0: str
    clock: Clock

    def deps(self, llm: LlmClient) -> RoundDeps:
        return RoundDeps(
            lane=LANE_EFFISQL,
            ledger=self.ledger,
            lessons=self.lessons,
            queue=self.queue,
            workspace=self.workspace,
            llm=llm,
            implementer=self.implementer,
            guards=self.guards,
            evaluator=self.evaluator,
            analyst_input=analyst_input,
            train_ids=TRAIN,
            val_ids=VAL,
            v0_commit=self.v0,
            clock=self.clock,
        )

    def runner(self, llm: LlmClient, config: RoundConfig | None = None) -> RoundRunner:
        return RoundRunner(self.deps(llm), config or RoundConfig())

    def queue_up(self, *ids: str) -> None:
        self.queue.push([hypothesis(hypothesis_id) for hypothesis_id in ids])

    def best(self) -> str | None:
        return self.workspace.best_commit(LANE_EFFISQL)


def make_world(root: Path) -> World:
    repo = make_repo(root / "repo")
    workspace = Workspace(repo, root=root / "worktrees")
    ledger = Ledger(root / "ledger")
    evaluator = ToyEvaluator()
    clock = Clock()
    return World(
        root=root,
        repo=repo,
        workspace=workspace,
        ledger=ledger,
        queue=HypothesisQueue(root / "ledger", LANE_EFFISQL),
        lessons=LessonStore(ledger, clock=clock),
        evaluator=evaluator,
        implementer=FakeImplementer(workspace, evaluator),
        guards=FakeGuards(evaluator),
        glm=ScriptedGlm(),
        v0=git(repo, "rev-parse", "HEAD").strip(),
        clock=clock,
    )
