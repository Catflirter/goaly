"""Tests for goaly.rsi.round: one RSI round in the toy world of tests.rsi.round_world."""

import asyncio
import json
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from goaly.bench.effisql.grading import Mode
from goaly.bench.effisql.grading_pool import Priority
from goaly.contracts.gen.experiment_pb2 import (
    DECISION_ABORTED,
    DECISION_ACCEPTED,
    DECISION_PENDING,
    DECISION_REJECTED,
    LANE_EFFISQL,
    REJECTION_REASON_GUARD_FAILED,
    REJECTION_REASON_HARD_CODED,
    REJECTION_REASON_IMPLEMENTATION_FAILED,
    REJECTION_REASON_NO_SCREEN_GAIN,
    REJECTION_REASON_NOT_CONFIRMED,
    REJECTION_REASON_OUTRANKED,
    ExperimentRecord,
)
from goaly.contracts.gen.llm_call_pb2 import CallPurpose
from goaly.contracts.gen.task_pb2 import SPLIT_TRAIN, SPLIT_VAL
from goaly.harness.core.llm import LlmClient
from goaly.rsi import round as round_module
from goaly.rsi.decision import DecisionConfig
from goaly.rsi.locks import LockTimeoutError
from goaly.rsi.resume import abort_experiments, reconcile_best
from goaly.rsi.round import EvalPhase, RoundConfig, lesson_id, tasks_digest
from tests.rsi.gitrepo import git
from tests.rsi.round_world import FAILING, TRAIN, VAL, World, make_world

pytestmark = pytest.mark.anyio


@pytest.fixture
def world(tmp_path: Path) -> World:
    return make_world(tmp_path)


@pytest.fixture
async def llm(world: World) -> AsyncIterator[LlmClient]:
    async with world.glm.client() as client:
        yield client


def latest(world: World) -> dict[str, ExperimentRecord]:
    return world.ledger.latest_experiments(LANE_EFFISQL)


def by_hypothesis(world: World) -> dict[str, ExperimentRecord]:
    return {record.hypothesis.hypothesis_id: record for record in latest(world).values()}


async def test_the_best_candidate_is_accepted_and_the_others_rejected(
    world: World, llm: LlmClient
) -> None:
    world.queue_up("good-1", "meh-1", "bad-1")
    outcome = await world.runner(llm).run(1)
    records = by_hypothesis(world)
    good, meh, bad = records["good-1"], records["meh-1"], records["bad-1"]
    assert outcome.accepted == good.experiment_id
    assert (outcome.decided, outcome.aborted) == (3, 0)
    assert good.decision == DECISION_ACCEPTED
    assert world.best() == good.candidate_commit
    assert world.ledger.best_version(LANE_EFFISQL) == good.candidate_commit
    assert good.confirmation.seed_count == 2
    assert good.confirmation.delta == pytest.approx(0.5)
    assert good.screening.task_count == 10  # 5 targeted failures and the 5 solved tasks.
    assert meh.decision == DECISION_REJECTED
    assert list(meh.rejection_reasons) == [REJECTION_REASON_OUTRANKED]
    assert not meh.HasField("confirmation")
    assert list(bad.rejection_reasons) == [REJECTION_REASON_NO_SCREEN_GAIN]
    assert not bad.HasField("validation")
    # The outranked hypothesis is queued to be rebuilt on the new best.
    assert [h.hypothesis_id for h in world.queue.pending()] == ["meh-1"]
    for record in records.values():
        assert record.parent_version == world.v0
        assert record.round == 1
        assert record.thresholds == DecisionConfig().thresholds()
        assert record.HasField("finished_at")
        assert record.diff_stats.files_changed == 1
        assert world.lessons.get(lesson_id(record.experiment_id)) is not None
        assert record.implementer.summary == "done"
    # Each decided experiment got its lesson, through one rsi-lesson call each.
    lessons = [r for r in world.glm.records if r.purpose == CallPurpose.CALL_PURPOSE_RSI_LESSON]
    assert sorted(r.run_id for r in lessons) == sorted(r.experiment_id for r in records.values())
    assert all(body["max_tokens"] == 24000 for body in world.glm.bodies)
    # The worktrees are gone; the branches keep the commits.
    assert world.workspace.worktrees(LANE_EFFISQL) == []


async def test_every_evaluation_names_what_the_host_expects(world: World, llm: LlmClient) -> None:
    world.queue_up("good-1")
    await world.runner(llm).run(1)
    good = by_hypothesis(world)["good-1"]
    requests = world.evaluator.requests
    assert {r.mode for r in requests} == {Mode.DEV}
    assert {r.priority for r in requests} == {Priority.RSI}
    assert EvalPhase.SMOKE.priority == Priority.SMOKE
    phases = {(r.phase, r.commit, r.split, r.seeds) for r in requests}
    assert (EvalPhase.SCREEN, good.candidate_commit, SPLIT_TRAIN, (0,)) in phases
    assert (EvalPhase.VALIDATION, good.candidate_commit, SPLIT_VAL, (0,)) in phases
    assert (EvalPhase.CONFIRMATION, good.candidate_commit, SPLIT_VAL, (1,)) in phases
    assert (EvalPhase.REFERENCE, world.v0, SPLIT_VAL, (0, 1)) in phases
    for request in requests:
        assert len(request.tree) == 40
        if request.split == SPLIT_VAL:
            assert request.instance_ids == VAL
    # The container's wall times (999 s) are never used; the evaluator's statistics are.
    assert good.validation.time_ratio_vs_best == pytest.approx(1.0)


async def test_a_gain_that_does_not_hold_fails_confirmation(world: World, llm: LlmClient) -> None:
    world.queue_up("noisy-1")
    outcome = await world.runner(llm).run(1)
    noisy = by_hypothesis(world)["noisy-1"]
    assert outcome.accepted is None
    assert noisy.validation.delta == pytest.approx(0.5)
    assert list(noisy.rejection_reasons) == [REJECTION_REASON_NOT_CONFIRMED]
    assert world.best() == world.v0
    assert "confirmation:" in noisy.decision_reason
    assert world.queue.pending() == []


async def test_a_flagged_prediction_fails_the_candidate(world: World, llm: LlmClient) -> None:
    world.queue_up("flag-1")
    await world.runner(llm).run(1)
    flagged = by_hypothesis(world)["flag-1"]
    assert list(flagged.rejection_reasons) == [REJECTION_REASON_HARD_CODED]
    assert list(flagged.validation.flagged_instance_ids) == [VAL[0]]
    assert VAL[0] not in flagged.decision_reason


async def test_failed_implementations_and_guards_reject_and_infrastructure_aborts(
    world: World, llm: LlmClient
) -> None:
    world.queue_up("crash-1", "infra-1", "guard-1")
    outcome = await world.runner(llm).run(1)
    records = by_hypothesis(world)
    assert list(records["crash-1"].rejection_reasons) == [REJECTION_REASON_IMPLEMENTATION_FAILED]
    assert records["crash-1"].implementer.step_count == 60
    assert not records["crash-1"].candidate_commit
    assert list(records["guard-1"].rejection_reasons) == [REJECTION_REASON_GUARD_FAILED]
    assert [g.passed for g in records["guard-1"].guards] == [True, False]
    infra = records["infra-1"]
    assert infra.decision == DECISION_ABORTED
    assert not infra.rejection_reasons
    assert "IMPLEMENTER_EXIT_INFRASTRUCTURE" in infra.decision_reason
    assert world.lessons.get(lesson_id(infra.experiment_id)) is None  # Aborts teach nothing.
    assert [h.hypothesis_id for h in world.queue.pending()] == ["infra-1"]
    assert (outcome.decided, outcome.aborted) == (2, 1)


async def test_a_failed_evaluation_aborts_the_experiment(world: World, llm: LlmClient) -> None:
    world.queue_up("broken-1", "good-1")
    world.evaluator.kinds["unused"] = "broken"
    original = world.implementer.implement

    async def implement_broken(hypothesis, worktree):  # type: ignore[no-untyped-def]  # A wrapper.
        built = await original(hypothesis, worktree)
        if hypothesis.hypothesis_id.startswith("broken") and built.commit:
            world.evaluator.kinds[built.commit] = "broken"
        return built

    world.implementer.implement = implement_broken  # type: ignore[method-assign]  # Patched.
    outcome = await world.runner(llm).run(1)
    records = by_hypothesis(world)
    assert records["broken-1"].decision == DECISION_ABORTED
    assert "infrastructure failure: RuntimeError" in records["broken-1"].decision_reason
    assert records["good-1"].decision == DECISION_ACCEPTED
    assert outcome.accepted == records["good-1"].experiment_id
    assert [h.hypothesis_id for h in world.queue.pending()] == ["broken-1"]


async def test_the_analyst_fills_the_queue_and_its_cost_is_split(
    world: World, llm: LlmClient
) -> None:
    world.glm.analyst_ids = [["good-a", "bad-a", "meh-a"]]
    config = RoundConfig(k=2)
    await world.runner(llm, config).run(1)
    records = latest(world)
    assert sorted(r.hypothesis.hypothesis_id for r in records.values()) == ["bad-a", "good-a"]
    assert [h.hypothesis_id for h in world.queue.pending()] == ["meh-a"]
    analyst = [
        next(
            u.usage for u in r.usage_by_purpose if u.purpose == CallPurpose.CALL_PURPOSE_RSI_ANALYST
        )
        for r in records.values()
    ]
    assert sorted(u.input_tokens for u in analyst) == [500, 501]
    assert sum(u.cost_usd for u in analyst) == pytest.approx(0.03)
    assert all(r.hypothesis.target_instance_ids == list(FAILING) for r in records.values())


async def test_an_analysis_that_stays_invalid_skips_the_round(world: World, llm: LlmClient) -> None:
    world.glm.invalid = True
    outcome = await world.runner(llm).run(1)
    assert outcome.skipped.startswith("3 invalid replies")
    assert not outcome.infrastructure
    assert latest(world) == {}


async def test_a_failed_analyst_call_aborts_the_round(world: World, llm: LlmClient) -> None:
    world.glm.fail = True
    outcome = await world.runner(llm).run(1)
    assert outcome.infrastructure
    assert "analyst" in outcome.skipped
    assert latest(world) == {}


async def test_a_queue_short_of_k_still_runs_when_the_analysis_is_skipped(
    world: World, llm: LlmClient
) -> None:
    world.glm.invalid = True
    world.queue_up("bad-1")
    outcome = await world.runner(llm).run(1)
    assert outcome.decided == 1


async def test_the_screen_targets_failures_then_draws_solved_tasks(
    world: World, llm: LlmClient
) -> None:
    world.queue_up("good-1")
    await world.runner(llm, RoundConfig(screen_failures=2, screen_solved=3)).run(1)
    screen = next(r for r in world.evaluator.requests if r.phase == EvalPhase.SCREEN)
    assert screen.instance_ids[:2] == FAILING[:2]
    assert set(screen.instance_ids[2:]) <= set(TRAIN[5:])
    assert len(screen.instance_ids) == 5


async def test_an_interrupted_round_resumes_without_duplicating_an_experiment(
    world: World, llm: LlmClient
) -> None:
    world.queue_up("good-1", "meh-1", "bad-1")
    world.evaluator.gate = asyncio.Event()  # Validations wait, so the round stops mid-way.
    runner = world.runner(llm)
    task = asyncio.create_task(runner.run(1))
    for _ in range(200):
        await asyncio.sleep(0.01)
        screened = [r for r in latest(world).values() if r.HasField("screening")]
        if len(screened) == 3:
            break
    task.cancel()  # As a crash would stop it.
    with pytest.raises(asyncio.CancelledError):
        await task
    before = latest(world)
    assert {r.decision for r in before.values()} == {DECISION_PENDING, DECISION_REJECTED}
    built = list(world.implementer.calls)

    # The restarted lane: nothing uncommitted to abort here, then the round resumes.
    world.evaluator.gate = None
    assert abort_experiments(
        world.ledger, world.queue, LANE_EFFISQL, "restart", now=world.clock(),
        uncommitted_only=True,
    ) == []  # fmt: skip
    pending = [r for r in latest(world).values() if r.round == 1]
    outcome = await world.runner(llm).run(1, resume=pending)
    after = latest(world)
    assert after.keys() == before.keys()
    assert world.implementer.calls == built  # No candidate was built again.
    assert outcome.accepted == by_hypothesis(world)["good-1"].experiment_id
    assert world.best() == by_hypothesis(world)["good-1"].candidate_commit


async def test_an_experiment_cut_before_its_commit_is_aborted_and_queued_again(
    world: World, llm: LlmClient
) -> None:
    world.queue_up("good-1")
    world.implementer.gate = asyncio.Event()
    task = asyncio.create_task(world.runner(llm).run(1))
    for _ in range(200):
        await asyncio.sleep(0.01)
        if world.implementer.calls:
            break
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    aborted = abort_experiments(
        world.ledger, world.queue, LANE_EFFISQL, "restart", now=world.clock(),
        uncommitted_only=True,
    )  # fmt: skip
    assert [r.hypothesis.hypothesis_id for r in aborted] == ["good-1"]
    record = by_hypothesis(world)["good-1"]
    assert record.decision == DECISION_ABORTED
    assert record.implementer.exit == 8  # IMPLEMENTER_EXIT_INTERRUPTED, never measured.
    assert not record.implementer.HasField("step_count")
    assert [h.hypothesis_id for h in world.queue.pending()] == ["good-1"]


def test_the_confirmation_needs_its_own_seed() -> None:
    with pytest.raises(ValueError, match="confirmation"):
        RoundConfig(validation_seed=1, confirmation_seed=1)
    assert RoundConfig().val_seeds == (0, 1)


@pytest.mark.parametrize(
    ("kind", "says"),
    [
        ("short", "1 missing"),
        ("extra", "1 unexpected"),
        ("graderr", "is not usable"),
        ("late", "RuntimeError"),
        ("flaky", "the validation: the infrastructure failed 2 of 8 tasks"),
        ("hostfail", "the validation: the infrastructure failed 2 of 8 tasks"),
    ],
)
async def test_an_unusable_evaluation_aborts_the_experiment(
    world: World, llm: LlmClient, kind: str, says: str
) -> None:
    world.queue_up(f"{kind}-1")
    outcome = await world.runner(llm).run(1)
    record = by_hypothesis(world)[f"{kind}-1"]
    assert record.decision == DECISION_ABORTED
    assert says in record.decision_reason
    assert outcome.aborted == 1
    assert world.best() == world.v0


async def test_without_train_results_the_experiment_aborts(world: World, llm: LlmClient) -> None:
    world.evaluator.train_known = False
    world.queue_up("good-1")
    await world.runner(llm).run(1)
    record = by_hypothesis(world)["good-1"]
    assert record.decision == DECISION_ABORTED
    assert "no train results" in record.decision_reason


async def test_a_lesson_call_that_fails_leaves_the_experiment_decided(
    world: World, llm: LlmClient
) -> None:
    world.queue_up("bad-1")
    world.glm.fail = True
    outcome = await world.runner(llm, RoundConfig(k=1)).run(1)
    record = by_hypothesis(world)["bad-1"]
    assert record.decision == DECISION_REJECTED
    assert world.lessons.get(lesson_id(record.experiment_id)) is None
    assert outcome.decided == 1


class Crash(BaseException):
    """The lane process dies: nothing in the round catches it."""


async def test_a_confirmed_candidate_found_pending_is_accepted(
    world: World, llm: LlmClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def crash(*_: object) -> None:
        raise Crash

    world.queue_up("good-1")
    monkeypatch.setattr(round_module, "_accept", crash)
    with pytest.raises(Crash):
        await world.runner(llm).run(1)
    monkeypatch.undo()
    good = by_hypothesis(world)["good-1"]
    assert good.decision == DECISION_PENDING
    assert good.HasField("confirmation")
    asked = len(world.evaluator.requests)
    outcome = await world.runner(llm).run(1, resume=[good])
    assert outcome.accepted == good.experiment_id
    assert world.best() == good.candidate_commit
    assert len(world.evaluator.requests) == asked  # Nothing was evaluated again.


async def test_a_lane_without_its_best_tag_cannot_run_a_round(world: World, llm: LlmClient) -> None:
    git(world.repo, "tag", "--delete", "best/effisql")
    with pytest.raises(RuntimeError, match="best/effisql"):
        await world.runner(llm).run(1)


async def test_infrastructure_failures_under_the_cap_are_scored(
    world: World, llm: LlmClient
) -> None:
    world.queue_up("wobbly-1")
    await world.runner(llm).run(1)
    record = by_hypothesis(world)["wobbly-1"]
    assert record.decision != DECISION_ABORTED
    assert record.validation.task_count == 7  # The failed task is left out of the comparison.
    assert "1 tasks left out: the infrastructure failed them" in record.decision_reason


async def test_a_cancelled_accept_finishes_before_the_cancellation_returns(
    world: World, llm: LlmClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = threading.Event()
    set_best = world.workspace.set_best

    def slow_set_best(*args: Any, **kwargs: Any) -> None:
        started.set()
        time.sleep(0.5)
        set_best(*args, **kwargs)

    monkeypatch.setattr(world.workspace, "set_best", slow_set_best)
    world.queue_up("good-1")
    task = asyncio.create_task(world.runner(llm).run(1))
    while not started.is_set():
        await asyncio.sleep(0.01)
    task.cancel()  # As the deadline does, while the tag moves.
    with pytest.raises(asyncio.CancelledError):
        await task
    # By the time the cancellation returns, the tag and the ledger agree, so the stop's
    # reconcile and abort find nothing to undo.
    good = by_hypothesis(world)["good-1"]
    assert good.decision == DECISION_ACCEPTED
    assert world.best() == good.candidate_commit
    assert abort_experiments(
        world.ledger, world.queue, LANE_EFFISQL, "deadline", now=world.clock(),
        uncommitted_only=False,
    ) == []  # fmt: skip


async def test_a_hypothesis_goes_back_at_most_twice(world: World, llm: LlmClient) -> None:
    runner = world.runner(llm, RoundConfig(k=1))
    world.glm.invalid = True
    world.queue_up("infra-1")
    for round_no in (1, 2, 3):
        await runner.run(round_no)
    records = [r for r in latest(world).values() if r.hypothesis.hypothesis_id == "infra-1"]
    assert [r.decision for r in records] == [DECISION_ABORTED] * 3
    assert world.queue.pending() == []  # Dropped after its second requeue.


async def test_reference_evaluations_are_shared_within_a_round(
    world: World, llm: LlmClient
) -> None:
    world.queue_up("good-1", "meh-1", "noisy-1")
    await world.runner(llm).run(1)
    keys = [
        (r.commit, r.split, r.instance_ids, r.seeds)
        for r in world.evaluator.requests
        if r.phase == EvalPhase.REFERENCE
    ]
    assert len(keys) == len(set(keys))
    validations = [r for r in world.evaluator.requests if r.phase == EvalPhase.VALIDATION]
    assert len(validations) == len({r.commit for r in validations}) == 3


async def test_every_request_names_its_pack_and_expected_tasks(
    world: World, llm: LlmClient
) -> None:
    world.queue_up("good-1")
    await world.runner(llm).run(1)
    for request in world.evaluator.requests:
        pack = git(world.repo, "rev-parse", f"{request.commit}:src/goaly/harness/packs/effisql")
        assert request.pack_tree == pack.strip()
        assert request.tasks_digest == tasks_digest(
            request.split, request.instance_ids, request.seeds
        )
    assert tasks_digest(SPLIT_VAL, ("2", "1"), (1, 0)) == tasks_digest(
        SPLIT_VAL, ("1", "2"), (0, 1)
    )
    assert tasks_digest(SPLIT_VAL, ("1",), (0,)) != tasks_digest(SPLIT_TRAIN, ("1",), (0,))


async def test_no_lesson_prompt_ever_holds_a_val_task_id(world: World, llm: LlmClient) -> None:
    # Lessons reach the analyst, so they must stay val-clean (spec §7): run every kind of
    # decision, then search every lesson call for any val ID.
    world.queue_up("good-1", "flag-1", "noisy-1")
    await world.runner(llm).run(1)
    world.queue_up("meh-2", "bad-2", "short-2")
    await world.runner(llm).run(2)
    lessons = [b for b in world.glm.bodies if "Write the lesson." in b["messages"][-1]["content"]]
    assert len(lessons) >= 5
    for body in lessons:
        text = json.dumps(body)
        for val_id in VAL:
            assert val_id not in text


async def test_infrastructure_failures_in_the_reference_never_help_a_candidate(
    world: World, llm: LlmClient
) -> None:
    # The best loses one val task to the infrastructure. Scored, it would hand a candidate that
    # changes nothing on val a free win; left out of both sides, it changes nothing.
    world.evaluator.reference_infra = 1
    world.queue_up("trainonly-1")
    await world.runner(llm).run(1)
    record = by_hypothesis(world)["trainonly-1"]
    assert record.validation.delta == 0
    assert record.validation.task_count == 7
    assert record.decision == DECISION_REJECTED
    assert world.best() == world.v0


async def test_a_failed_acceptance_write_is_retried_and_never_aborts(
    world: World, llm: LlmClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    record_experiment = world.ledger.record_experiment
    failures = {"left": 1}

    def flaky(record: ExperimentRecord) -> None:
        if record.decision == DECISION_ACCEPTED and failures["left"]:
            failures["left"] -= 1
            raise LockTimeoutError("the ledger lock is busy")
        record_experiment(record)

    monkeypatch.setattr(world.ledger, "record_experiment", flaky)
    monkeypatch.setattr(round_module, "_ACCEPT_RETRIES_S", (0.0,))
    world.queue_up("good-1")
    outcome = await world.runner(llm).run(1)
    good = by_hypothesis(world)["good-1"]
    assert outcome.accepted == good.experiment_id
    assert good.decision == DECISION_ACCEPTED


async def test_an_acceptance_that_cannot_be_written_is_repaired_from_the_tag(
    world: World, llm: LlmClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    record_experiment = world.ledger.record_experiment

    def broken(record: ExperimentRecord) -> None:
        if record.decision == DECISION_ACCEPTED:
            raise LockTimeoutError("the ledger lock is busy")
        record_experiment(record)

    monkeypatch.setattr(world.ledger, "record_experiment", broken)
    monkeypatch.setattr(round_module, "_ACCEPT_RETRIES_S", (0.0, 0.0))
    world.queue_up("good-1")
    with pytest.raises(LockTimeoutError):  # The lane fails; it never aborts the best.
        await world.runner(llm).run(1)
    monkeypatch.undo()
    good = by_hypothesis(world)["good-1"]
    assert good.decision == DECISION_PENDING
    assert world.best() == good.candidate_commit
    assert world.queue.pending() == []
    best = reconcile_best(
        world.ledger, world.workspace, LANE_EFFISQL, v0_commit=world.v0, now=world.clock()
    )
    assert best == good.candidate_commit
    assert by_hypothesis(world)["good-1"].decision == DECISION_ACCEPTED


def test_abort_reasons_keep_only_a_bounded_first_line() -> None:
    error = RuntimeError("goaly run failed\n" + "x" * 5000)
    expected = "infrastructure failure: RuntimeError: goaly run failed"
    assert round_module._infrastructure(error) == expected
    assert len(round_module._infrastructure(RuntimeError("y" * 5000))) < 400
