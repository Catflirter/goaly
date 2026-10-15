"""Unit tests of the setup run: its checks, exclusions, sampling, summary and phases."""

import asyncio
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from goaly.bench.effisql.data import EffiSqlInstance
from goaly.bench.effisql.grader import GraderError, QueryTimeoutError, ResourceError
from goaly.bench.effisql.grading import BaseCache
from goaly.bench.effisql.grading_pool import GradingInstance, GradingPool
from goaly.bench.effisql.grading_service import SlotRunner, ThreadRunner
from goaly.bench.effisql.setup_run import (
    NEGATIVE_SAMPLE,
    DataCheck,
    GradeSummary,
    RecordStore,
    SetupRecord,
    TimeStats,
    data_check,
    dump_exclusions,
    exclusion_reason,
    load_manual_exclusions,
    negative_sample,
    negative_task,
    negative_variants,
    run_phase,
    summarize,
    to_markdown,
)
from goaly.bench.effisql.splits import Exclusion
from goaly.contracts.gen.task_pb2 import GRADE_STATUS_GRADED, EffiSqlGrade, Grade
from tests.bench.effisql.grading_fakes import (
    ALT_ROWS,
    BASE_ROWS,
    BASE_SQL,
    FakeSession,
    digest_of,
    entry,
    plan,
    sessions,
)

GRADED = "GRADE_STATUS_GRADED"
GRADER_ERROR = "GRADE_STATUS_GRADER_ERROR"


def gold(**changes: Any) -> GradeSummary:
    fields: dict[str, Any] = {
        "status": GRADED,
        "matched": True,
        "alt_matched": True,
        "literal_result": False,
        "speedup": 3.0,
        "credit": 1.0,
        "timing_runs": 3,
        "eval_wall_s": 12.0,
    }
    fields.update(changes)
    return GradeSummary(**fields)


def record(task: str = "7", **changes: Any) -> SetupRecord:
    fields: dict[str, Any] = {
        "instance_id": task,
        "base": entry(instance_id=task),
        "base_wall_s": 30.0,
        "data": DataCheck(outcome="pass", plan_rows=3),
        "gold": gold(),
    }
    fields.update(changes)
    return SetupRecord(**fields)


def task(plan_json: Any) -> EffiSqlInstance:
    return EffiSqlInstance.model_validate(
        {"id": 7, "db": "tpch", "base_sql": BASE_SQL, "base_explain_analyze": plan_json}
    )


class TestDataCheck:
    def test_the_root_actual_rows_must_equal_the_count(self) -> None:
        recorded = {"Node Type": "Aggregate", "Actual Rows": 3, "Plan Rows": 3}
        assert data_check(task(recorded), 3) == DataCheck(outcome="pass", plan_rows=3)
        assert data_check(task(recorded), 4) == DataCheck(outcome="fail", plan_rows=3)

    def test_reads_a_wrapped_plan_too(self) -> None:
        assert data_check(task({"Plan": {"Actual Rows": 3}}), 3).outcome == "pass"

    def test_a_plan_without_actual_rows_is_skipped(self) -> None:
        assert data_check(task({"Node Type": "Seq Scan"}), 3).outcome == "no actual rows"
        assert data_check(task(None), 3).outcome == "no actual rows"

    def test_a_plan_on_another_scale_is_an_anomaly_not_a_failure(self) -> None:
        sf10 = {
            "Node Type": "Seq Scan",
            "Relation Name": "lineitem",
            "Actual Rows": 59_986_052,
            "Plan Rows": 59_986_052,
        }
        check = data_check(task(sf10), 17_996_609)
        assert check == DataCheck(outcome="foreign scale", plan_rows=59_986_052)

    def test_task_40s_filter_subplan_reveals_sf10(self) -> None:
        """Task 40's recorded plan, read through T09's foreign_scale_plan."""
        subplan = {
            "Node Type": "Aggregate",
            "Parent Relationship": "SubPlan",
            "Actual Rows": 1,
            "Actual Loops": 59_986_052,
        }
        task_40 = {
            "Node Type": "Seq Scan",
            "Relation Name": "lineitem",
            "Actual Rows": 15_240_492,
            "Actual Loops": 1,
            "Plan Rows": 299_959,
            "Plans": [subplan],
        }
        assert data_check(task(task_40), 4_572_053).outcome == "foreign scale"


class TestExclusionReason:
    def test_a_good_task_is_kept(self) -> None:
        assert exclusion_reason(record()) is None

    def test_a_base_timeout(self) -> None:
        failed = record(base=None, base_failure="cancelled after 300 s", base_timed_out=True)
        assert exclusion_reason(failed) == (
            "base query exceeds the 300 s statement timeout on this machine"
        )

    def test_a_base_error_never_quotes_the_message(self) -> None:
        failed = record(base=None, base_failure="syntax error at or near SECRET")
        assert exclusion_reason(failed) == "base query fails on this machine"

    def test_a_base_over_300_s(self) -> None:
        assert "300 s" in str(exclusion_reason(record(base=entry(base_ms=300_001.0))))

    @pytest.mark.parametrize(
        ("summary", "expected"),
        [
            (gold(matched=False, speedup=None), "does not match base_sql on this machine"),
            (
                gold(matched=False, status="GRADE_STATUS_PREDICTION_FAILED", feedback="x"),
                "does not match base_sql: GRADE_STATUS_PREDICTION_FAILED",
            ),
            (gold(speedup=None, status="GRADE_STATUS_TIMEOUT"), "could not be timed"),
            (gold(speedup=0.0, status="GRADE_STATUS_TIMEOUT"), "could not be timed"),
            (gold(speedup=1.19), "only 1.19x faster than base_sql (under 1.2x)"),
        ],
    )
    def test_a_failing_gold(self, summary: GradeSummary, expected: str) -> None:
        assert expected in str(exclusion_reason(record(gold=summary)))

    def test_a_gold_at_exactly_1_2_is_kept(self) -> None:
        assert exclusion_reason(record(gold=gold(speedup=1.2))) is None

    def test_a_flagged_gold_is_left_to_review(self) -> None:
        assert exclusion_reason(record(gold=gold(alt_matched=False, credit=0.0))) is None

    def test_a_grader_failure_never_excludes(self) -> None:
        assert exclusion_reason(record(gold=gold(status=GRADER_ERROR, matched=False))) is None


class TestNegativeVariants:
    def test_every_variant(self) -> None:
        variants = negative_variants("SELECT a, b FROM t;", ["text", "numeric"], 5)
        source = "(\nSELECT a, b FROM t\n) AS base (c1, c2)"
        assert variants == {
            "control": "SELECT c1, c2 FROM (\nSELECT a, b FROM t\n) AS base (c1, c2)",
            "drop": "SELECT c1, c2 FROM (\nSELECT a, b FROM t\n) AS base (c1, c2) LIMIT 4",
            "duplicate": "SELECT c1, c2 FROM (\nSELECT a, b FROM t\n) AS base (c1, c2) UNION ALL "
            "(SELECT c1, c2 FROM (\nSELECT a, b FROM t\n) AS base (c1, c2) LIMIT 1)",
            "reorder": "SELECT c2, c1 FROM (\nSELECT a, b FROM t\n) AS base (c1, c2)",
            "perturb": "SELECT c1, CASE WHEN rn = 1 THEN COALESCE(c2 + 1, 0) ELSE c2 END FROM "
            "(SELECT *, row_number() OVER () AS rn FROM (\nSELECT a, b FROM t\n) AS base "
            "(c1, c2)) AS numbered",
        }
        assert all(source in query for query in variants.values())

    @pytest.mark.parametrize(
        ("types", "change"),
        [
            (["date"], "COALESCE(c1 + 1, DATE '1970-01-01')"),
            (["varchar"], "COALESCE(c1 || '~', '~')"),
            (["bpchar", "int8"], "COALESCE(c2 + 1, 0)"),
        ],
    )
    def test_each_kind_of_perturbation(self, types: list[str], change: str) -> None:
        assert change in negative_variants("SELECT x", types, 1)["perturb"]

    def test_padded_text_is_never_perturbed(self) -> None:
        assert "perturb" not in negative_variants("SELECT a FROM t", ["bpchar"], 1)

    def test_an_empty_result_has_only_the_control_and_reorder(self) -> None:
        assert set(negative_variants("SELECT 1, 2", ["int4", "int4"], 0)) == {"control", "reorder"}

    def test_an_unknown_type_is_not_perturbed(self) -> None:
        assert "perturb" not in negative_variants("SELECT x", ["jsonb"], 3)


class TestNegativeSample:
    def test_a_fixed_sample_of_kept_tasks_with_small_nonempty_results(self) -> None:
        records = {str(i): record(str(i)) for i in range(1, 31)}
        records["5"] = record("5", base=entry(instance_id="5", row_count=10_000))
        records["6"] = record("6", gold=gold(matched=False))
        records["8"] = record("8", base=entry(instance_id="8", row_count=0))
        sample = negative_sample(records)
        assert len(sample) == NEGATIVE_SAMPLE
        assert not {"5", "6", "8"} & set(sample)
        assert sample == sorted(sample, key=int)
        assert negative_sample(records) == sample

    def test_fewer_candidates_than_the_sample(self) -> None:
        assert negative_sample({"1": record("1")}) == ["1"]


def clean_records() -> dict[str, SetupRecord]:
    records = {
        str(i): record(str(i), negatives={"control": True, "drop": False}) for i in range(1, 11)
    }
    for task_id in ("1", "2"):
        records[task_id] = records[task_id].model_copy(
            update={
                "official_gold": gold(eval_wall_s=40.0),
                "official_identity": gold(speedup=1.03),
            }
        )
    return records


VAL = ["1", "2", "99"]


def _remove(records: dict[str, SetupRecord], task_id: str) -> None:
    del records[task_id]


def _update(records: dict[str, SetupRecord], task_id: str, **fields: Any) -> None:
    records[task_id] = records[task_id].model_copy(update=fields)


class TestSummarize:
    def test_a_clean_setup_passes(self) -> None:
        records = clean_records()
        summary = summarize(list(records), records, val=VAL)
        assert summary.passed, to_markdown(summary)
        assert summary.exclusions == []
        assert summary.agreement_tasks == ["1", "2"]
        assert summary.agreement_missing == []
        assert summary.same_tier_share == 1.0
        assert summary.negative_sample == summary.negative_tasks
        assert summary.times["official grading (gold)"] is not None
        assert "All self-checks passed: True" in to_markdown(summary)

    @pytest.mark.parametrize(
        ("change", "field"),
        [
            pytest.param(lambda r: _remove(r, "3"), "missing", id="a task without a record"),
            pytest.param(
                lambda r: _update(r, "3", gold=gold(status=GRADER_ERROR, matched=False)),
                "grader_errors",
                id="a grader failure to rerun",
            ),
            pytest.param(
                lambda r: _update(r, "3", gold=gold(literal_result=True, credit=0.0)),
                "review",
                id="a flagged gold",
            ),
            pytest.param(
                lambda r: _update(r, "3", data=DataCheck(outcome="fail")),
                "data",
                id="a data check failure",
            ),
            pytest.param(
                lambda r: _update(r, "3", base=entry(instance_id="3", identity_speedup=1.2)),
                "identity_failures",
                id="an identity τ outside the tolerance",
            ),
            pytest.param(
                lambda r: _update(r, "3", negatives={"control": False}),
                "negative_failures",
                id="a negative check failure",
            ),
            pytest.param(
                lambda r: _update(r, "3", negatives=None),
                "negative_sample",
                id="a sampled task without negatives",
            ),
            pytest.param(
                lambda r: _update(r, "2", official_gold=None),
                "agreement_missing",
                id="a val task without its official grades",
            ),
            pytest.param(
                lambda r: _update(r, "2", official_gold=gold(matched=False)),
                "match_disagreements",
                id="the modes disagree on a match",
            ),
            pytest.param(
                lambda r: _update(r, "2", official_identity=gold(speedup=0.8)),
                "identity_disagreements",
                id="the official identity τ is off",
            ),
        ],
    )
    def test_each_failure_alone_fails_the_setup(
        self, change: Callable[[dict[str, SetupRecord]], None], field: str
    ) -> None:
        records = clean_records()
        ids = list(records)
        change(records)
        summary = summarize(ids, records, val=VAL)
        assert not summary.passed
        assert getattr(summary, field), field

    def test_no_official_phase_at_all_fails(self) -> None:
        records = {
            task_id: r.model_copy(update={"official_gold": None, "official_identity": None})
            for task_id, r in clean_records().items()
        }
        summary = summarize(list(records), records, val=VAL)
        assert summary.agreement_tasks == []
        assert summary.agreement_missing == ["1", "2"]
        assert not summary.passed

    def test_the_modes_identity_taus_must_agree_with_each_other(self) -> None:
        records = clean_records()
        _update(
            records,
            "1",
            base=entry(instance_id="1", identity_speedup=0.92),
            official_identity=gold(speedup=1.08),
        )
        summary = summarize(list(records), records, val=VAL)
        assert summary.identity_disagreements == {"1": [0.92, 1.08]}

    def test_the_tolerance_edge_counts_as_inside(self) -> None:
        records = clean_records()
        _update(records, "3", base=entry(instance_id="3", identity_speedup=1.1))
        summary = summarize(list(records), records, val=VAL)
        assert summary.identity_failures == {}

    def test_the_stop_rule_allows_up_to_10_percent(self) -> None:
        records = {str(i): record(str(i)) for i in range(1, 11)}
        records["1"] = record("1", gold=gold(speedup=1.0))
        assert not summarize(list(records), records, val=[]).stop
        records["2"] = record("2", gold=gold(speedup=1.0))
        assert summarize(list(records), records, val=[]).stop

    def test_reviewed_exclusions_are_merged_and_counted(self) -> None:
        records = clean_records()
        _update(records, "3", gold=gold(literal_result=True, credit=0.0))
        manual = [Exclusion(instance_id="3", reason="gold fails the plan check; review: gold")]
        summary = summarize(list(records), records, val=VAL, manual=manual)
        assert summary.exclusions == manual
        assert summary.review == []
        assert "3" not in summary.negative_sample

    def test_a_base_with_the_same_result_on_both_databases_is_listed(self) -> None:
        records = clean_records()
        same = digest_of(BASE_ROWS)
        base = entry(
            instance_id="4",
            alt_row_count=same.row_count,
            alt_row_hash_sum=f"{same.row_hash_sum:064x}",
        )
        _update(records, "4", base=base)
        assert summarize(list(records), records, val=VAL).scale_invariant_bases == ["4"]

    def test_decides_the_data_check_again_from_the_tasks(self) -> None:
        stale = record("7", data=DataCheck(outcome="fail", plan_rows=59_986_052))
        sf10 = {"Node Type": "Seq Scan", "Relation Name": "lineitem", "Actual Rows": 59_986_052}
        summary = summarize(["7"], {"7": stale}, val=[], tasks={"7": task(sf10)})
        assert summary.data["fail"] == []
        assert summary.data["foreign scale"] == ["7"]


def test_time_stats_p90_is_nearest_rank() -> None:
    stats = TimeStats.of([float(i) for i in range(1, 11)])
    assert stats is not None
    assert (stats.p90_s, stats.max_s, stats.median_s) == (9.0, 10.0, 5.5)
    assert TimeStats.of([]) is None


def test_dump_exclusions_is_the_committed_format() -> None:
    text = dump_exclusions([Exclusion(instance_id="12", reason="why")])
    assert json.loads(text) == [{"instance_id": "12", "reason": "why"}]
    assert text.endswith("\n")


def test_manual_exclusions_load_from_their_file(tmp_path: Path) -> None:
    path = tmp_path / "manual.json"
    assert load_manual_exclusions(path) == ()
    path.write_text('[{"instance_id": "3", "reason": "review: gold"}]', encoding="utf-8")
    assert load_manual_exclusions(path) == (Exclusion(instance_id="3", reason="review: gold"),)


def test_the_committed_manual_exclusions_file_is_valid() -> None:
    assert [e.instance_id for e in load_manual_exclusions()] == ["93"]


def test_grade_summary_of_a_grade() -> None:
    graded = Grade(
        status=GRADE_STATUS_GRADED,
        feedback="the prediction failed: near GOLD",
        effisql=EffiSqlGrade(matched=True, speedup=2.0, credit=1.0),
    )
    summary = GradeSummary.of(graded)
    assert summary == GradeSummary(
        status=GRADED, matched=True, speedup=2.0, credit=1.0, feedback=graded.feedback
    )
    assert not summary.flagged
    assert GradeSummary.of(graded, keep_feedback=False).feedback == ""


def test_the_record_store_round_trips_privately_and_skips_bad_files(tmp_path: Path) -> None:
    store = RecordStore(tmp_path)
    store.save(record("7"))
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    assert store.load() == {"7": record("7")}
    assert (tmp_path / "7.json").stat().st_mode & 0o777 == 0o600


class FakeSlot:
    instance = GradingInstance("fake", 1)

    def __init__(self, main: FakeSession, alt: FakeSession) -> None:
        self._sessions = sessions(main, alt)
        self.open_sessions = self._sessions
        self.resets = 0

    def run(self, _token: int, work: Callable[["FakeSlot"], Any]) -> Any:
        return work(self)

    def abort(self, _token: int) -> None:
        pass

    def sessions(self) -> Any:
        return self._sessions

    def reset(self) -> None:
        self.resets += 1


GOLD_SQL = "SELECT l_returnflag, count(*) FROM lineitem GROUP BY 1"


def instance(task_id: str, **fields: Any) -> EffiSqlInstance:
    return EffiSqlInstance.model_validate(
        {
            "id": task_id,
            "db": "tpch",
            "base_sql": BASE_SQL,
            "optimized_sql": GOLD_SQL,
            "base_explain_analyze": {"Node Type": "Aggregate", "Actual Rows": 3},
            **fields,
        }
    )


def fake_main(*, base_times: Sequence[float | Exception] | None = None) -> FakeSession:
    return FakeSession(
        plans={BASE_SQL: plan("lineitem"), GOLD_SQL: plan("lineitem")},
        results={BASE_SQL: BASE_ROWS, GOLD_SQL: BASE_ROWS},
        times={BASE_SQL: base_times or [1000.0] * 36, GOLD_SQL: [400.0] * 12},
    )


def fake_alt() -> FakeSession:
    return FakeSession(results={BASE_SQL: ALT_ROWS, GOLD_SQL: ALT_ROWS})


def pool_of(*slots: FakeSlot) -> GradingPool[SlotRunner]:
    return GradingPool([ThreadRunner(slot) for slot in slots])  # type: ignore[arg-type]  # Fakes.


def measure(slot: FakeSlot, store: RecordStore, cache: BaseCache, *tasks: EffiSqlInstance) -> int:
    return asyncio.run(run_phase("measure", pool_of(slot), list(tasks), store, cache))


class TestRunPhase:
    def test_measure_writes_each_record_and_resumes(self, tmp_path: Path) -> None:
        store, cache = RecordStore(tmp_path / "records"), BaseCache(tmp_path / "dev")
        slot = FakeSlot(fake_main(), fake_alt())
        assert measure(slot, store, cache, instance("7")) == 1
        measured = store.load()["7"]
        assert measured.base is not None
        assert measured.base.identity_speedup == 1.0
        assert measured.identity_speedups == (1.0,)
        assert measured.data == DataCheck(outcome="pass", plan_rows=3)
        assert measured.gold is not None
        assert measured.gold.speedup == 2.5
        assert measured.gold.feedback == "", "a gold's feedback is never kept"
        assert cache.get(instance("7")) is not None
        assert measure(slot, store, cache, instance("7")) == 0

    def test_a_base_timeout_is_recorded(self, tmp_path: Path) -> None:
        store = RecordStore(tmp_path)
        main = fake_main(base_times=[QueryTimeoutError("cancelled after 300 s")])
        measure(FakeSlot(main, fake_alt()), store, BaseCache(tmp_path / "c"), instance("7"))
        failed = store.load()["7"]
        assert failed.base_timed_out
        assert failed.gold is None

    def test_identity_off_by_more_than_10_percent_is_timed_again_and_averaged(
        self, tmp_path: Path
    ) -> None:
        main = fake_main(base_times=[1000.0, 1300.0] * 3 + [1000.0] * 6)
        store = RecordStore(tmp_path)
        measure(FakeSlot(main, fake_alt()), store, BaseCache(tmp_path / "c"), instance("7"))
        measured = store.load()["7"]
        assert measured.base is not None
        assert len(measured.identity_speedups) == 2
        assert measured.identity_speedups[1] == 1.0
        mean = sum(measured.identity_speedups) / 2
        assert measured.base.identity_speedup == pytest.approx(mean)
        assert main.kinds(BASE_SQL).count("time") == 12
        assert main.kinds(BASE_SQL).count("digest") == 1, "the retry only re-times"

    def test_a_grader_failure_is_never_recorded_and_resets_the_slot(self, tmp_path: Path) -> None:
        store, cache = RecordStore(tmp_path), BaseCache(tmp_path / "c")
        main = fake_main()
        main.results[GOLD_SQL] = GraderError("the connection failed")
        slot = FakeSlot(main, fake_alt())
        assert measure(slot, store, cache, instance("7")) == 0
        assert store.load() == {}
        assert slot.resets >= 1
        main.results[GOLD_SQL] = BASE_ROWS  # The rerun retries the task.
        assert measure(slot, store, cache, instance("7")) == 1

    def test_a_raising_base_measurement_resets_the_slot(self, tmp_path: Path) -> None:
        main = fake_main()
        main.plans[BASE_SQL] = GraderError("lost")
        slot = FakeSlot(main, fake_alt())
        store = RecordStore(tmp_path)
        measure(slot, store, BaseCache(tmp_path / "c"), instance("7"))
        assert slot.resets == 1
        assert store.load() == {}

    def test_an_old_grader_error_record_is_measured_again(self, tmp_path: Path) -> None:
        store = RecordStore(tmp_path / "records")
        store.save(record("7", gold=gold(status=GRADER_ERROR, matched=False)))
        slot = FakeSlot(fake_main(), fake_alt())
        assert measure(slot, store, BaseCache(tmp_path / "c"), instance("7")) == 1
        redone = store.load()["7"].gold
        assert redone is not None
        assert redone.status == GRADED

    def test_official_grades_the_kept_val_golds_and_identity(self, tmp_path: Path) -> None:
        store, cache = RecordStore(tmp_path / "records"), BaseCache(tmp_path / "dev")
        tasks = [instance("7"), instance("8"), instance("9")]
        measure(FakeSlot(fake_main(), fake_alt()), store, cache, *tasks)
        store.save(store.load()["9"].model_copy(update={"gold": gold(speedup=1.0)}))  # Excluded.
        main = FakeSession(
            plans={BASE_SQL: plan("lineitem"), GOLD_SQL: plan("lineitem")},
            results={BASE_SQL: BASE_ROWS, GOLD_SQL: BASE_ROWS},
            times={BASE_SQL: [1000.0] * 9, GOLD_SQL: [500.0] * 3},
        )
        official_pool = pool_of(FakeSlot(main, fake_alt()))
        ran = asyncio.run(run_phase("official", official_pool, tasks, store, cache, val=["7", "9"]))
        assert ran == 1
        records = store.load()
        assert records["7"].official_gold is not None
        assert records["7"].official_gold.speedup == 2.0
        assert records["7"].official_identity is not None
        assert records["7"].official_identity.speedup == 1.0
        assert records["8"].official_gold is None
        assert records["9"].official_gold is None, "an excluded task is not graded officially"

    def test_negatives_wait_for_every_record(self, tmp_path: Path) -> None:
        store = RecordStore(tmp_path / "records")
        store.save(record("7"))
        pool = pool_of(FakeSlot(fake_main(), fake_alt()))
        tasks = [instance("7"), instance("8")]
        assert asyncio.run(run_phase("negatives", pool, tasks, store, BaseCache(tmp_path))) == 0

    def test_negatives_run_on_the_sample(self, tmp_path: Path) -> None:
        store = RecordStore(tmp_path / "records")
        store.save(record("7"))
        main = fake_main()
        types = ["text", "int8"]
        main.column_types = lambda _sql: types  # type: ignore[attr-defined]  # The fake's own.
        variants = negative_variants(BASE_SQL, types, 3)
        main.results[variants["control"]] = BASE_ROWS
        main.results[variants["drop"]] = BASE_ROWS[:-1]
        main.results[variants["duplicate"]] = [*BASE_ROWS, BASE_ROWS[0]]
        main.results[variants["reorder"]] = [(b, a) for a, b in BASE_ROWS]
        main.results[variants["perturb"]] = [("A", 11), *BASE_ROWS[1:]]
        pool = pool_of(FakeSlot(main, fake_alt()))
        asyncio.run(run_phase("negatives", pool, [instance("7")], store, BaseCache(tmp_path)))
        assert store.load()["7"].negatives == {
            "control": True,
            "drop": False,
            "duplicate": False,
            "reorder": False,
            "perturb": False,
        }


def test_negative_task_resets_the_slot_on_failure() -> None:
    main = fake_main()

    def broken(_sql: str) -> list[str]:
        raise GraderError("lost")

    main.column_types = broken  # type: ignore[attr-defined]  # The fake's own.
    slot = FakeSlot(main, fake_alt())
    with pytest.raises(GraderError):
        negative_task(slot, instance("7"), entry())  # type: ignore[arg-type]  # A fake slot.
    assert slot.resets == 1


def test_a_failing_task_is_logged_and_left_for_a_rerun(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = RecordStore(tmp_path)
    main = fake_main()
    main.results[BASE_SQL] = GraderError("the connection failed")
    assert measure(FakeSlot(main, fake_alt()), store, BaseCache(tmp_path / "c"), instance("7")) == 0
    assert store.load() == {}
    assert "setup task failed" in caplog.text


class TestRulingsOfTheTail:
    def test_a_resource_failure_is_retried_once_then_stands(self, tmp_path: Path) -> None:
        store, cache = RecordStore(tmp_path / "records"), BaseCache(tmp_path / "dev")
        main = fake_main()
        main.results[BASE_SQL] = ResourceError("the server ran out of resources: shm")
        slot = FakeSlot(main, fake_alt())
        assert measure(slot, store, cache, instance("7")) == 1
        first = store.load()["7"]
        assert (first.resource_failures, first.base_timed_out) == (1, False)
        assert exclusion_reason(first) == "base query fails on this machine"
        assert measure(slot, store, cache, instance("7")) == 1, "retried once"
        assert store.load()["7"].resource_failures == 2
        assert measure(slot, store, cache, instance("7")) == 0, "then the exclusion stands"

    def test_an_old_record_that_ran_out_of_shared_memory_is_retried(self, tmp_path: Path) -> None:
        store, cache = RecordStore(tmp_path / "records"), BaseCache(tmp_path / "dev")
        failure = 'could not resize shared memory segment "/PostgreSQL.7" to 16777216 bytes'
        store.save(record("7", base=None, gold=None, data=None, base_failure=failure))
        assert measure(FakeSlot(fake_main(), fake_alt()), store, cache, instance("7")) == 1
        assert store.load()["7"].base is not None

    def test_a_base_timeout_from_the_old_clock_is_measured_again(self, tmp_path: Path) -> None:
        store, cache = RecordStore(tmp_path / "records"), BaseCache(tmp_path / "dev")
        old = record("7", base=None, gold=None, data=None, base_failure="late", base_timed_out=True)
        store.save(old)
        slot = FakeSlot(fake_main(base_times=[QueryTimeoutError("statement timeout")]), fake_alt())
        assert measure(slot, store, cache, instance("7")) == 1
        again = store.load()["7"]
        assert again.base_timed_out
        assert again.timeout_rechecked
        assert measure(slot, store, cache, instance("7")) == 0

    def test_marginal_golds_and_off_identities_are_graded_again_officially(
        self, tmp_path: Path
    ) -> None:
        store, cache = RecordStore(tmp_path / "records"), BaseCache(tmp_path / "dev")
        store.save(record("7", gold=gold(speedup=1.1)))
        store.save(record("8", base=entry(instance_id="8", identity_speedup=1.3)))
        store.save(record("9"))
        main = FakeSession(
            plans={BASE_SQL: plan("lineitem"), GOLD_SQL: plan("lineitem")},
            results={BASE_SQL: BASE_ROWS, GOLD_SQL: BASE_ROWS},
            times={BASE_SQL: [1000.0] * 12, GOLD_SQL: [400.0] * 3},
        )
        pool = pool_of(FakeSlot(main, fake_alt()))
        tasks = [instance("7"), instance("8"), instance("9")]
        assert asyncio.run(run_phase("recheck", pool, tasks, store, cache)) == 2
        records = store.load()
        assert records["7"].gold_recheck is not None
        assert records["7"].gold_recheck.speedup == 2.5
        assert records["7"].identity_recheck is None
        assert exclusion_reason(records["7"]) is None, "the official re-measure decides"
        assert records["8"].identity_recheck is not None
        assert records["8"].identity_recheck.speedup == 1.0
        assert records["8"].gold_recheck is None
        assert records["9"].gold_recheck is None
        assert asyncio.run(run_phase("recheck", pool, tasks, store, cache)) == 0
        summary = summarize(["7", "8", "9"], records, val=[])
        assert summary.gold_rechecks == {"7": [1.1, 2.5]}
        assert summary.identity_rechecks == {"8": [1.3, 1.0]}
        assert "8" not in summary.identity_failures, "the idle re-measure is within tolerance"

    def test_exclusions_are_counted_by_reason_and_split(self) -> None:
        records = {str(i): record(str(i)) for i in range(1, 21)}
        records["1"] = record("1", gold=gold(speedup=1.1))
        records["2"] = record("2", base=None, gold=None, base_failure="x", base_timed_out=True)
        splits = {"1": "train", "2": "heldout"}
        summary = summarize(list(records), records, val=[], splits=splits)
        assert summary.exclusions_by_reason == {
            "base query exceeds the 300 s statement timeout on this machine": 1,
            "gold under 1.2x": 1,
        }
        assert summary.exclusions_by_split == {"heldout": 1, "train": 1}
        assert "by split" in to_markdown(summary)


def test_the_negative_sample_leaves_out_reviewed_exclusions_as_the_summary_does(
    tmp_path: Path,
) -> None:
    store = RecordStore(tmp_path / "records")
    for i in range(1, 13):
        store.save(record(str(i)))
    records = store.load()
    reviewed = ["3"]
    manual = [Exclusion(instance_id="3", reason="review")]
    expected = summarize(list(records), records, val=[], manual=manual).negative_sample
    ran: list[str] = []

    class Runner:
        instance = GradingInstance("fake", 1)

        async def call(self, _work: Any, task: EffiSqlInstance, *_: Any) -> dict[str, bool]:
            ran.append(task.instance_id)
            return {"control": True}

        def close(self) -> None:
            pass

    pool: GradingPool[SlotRunner] = GradingPool([Runner()])  # type: ignore[list-item]  # A fake.
    tasks = [instance(str(i)) for i in range(1, 13)]
    asyncio.run(run_phase("negatives", pool, tasks, store, BaseCache(tmp_path), manual=reviewed))
    assert sorted(ran, key=int) == expected
    assert "3" not in ran
