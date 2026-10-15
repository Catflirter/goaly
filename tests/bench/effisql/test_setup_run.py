"""Unit tests of the setup run: its checks, exclusions, sampling, summary and phases."""

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from goaly.bench.effisql.data import EffiSqlInstance
from goaly.bench.effisql.grader import GraderError, QueryTimeoutError
from goaly.bench.effisql.grading import BaseCache
from goaly.bench.effisql.grading_pool import GradingInstance, GradingPool
from goaly.bench.effisql.grading_service import SlotRunner, ThreadRunner
from goaly.bench.effisql.setup_run import (
    NEGATIVE_SAMPLE,
    DataCheck,
    GradeSummary,
    RecordStore,
    SetupRecord,
    data_check,
    dump_exclusions,
    exclusion_reason,
    negative_sample,
    negative_variants,
    official_credit,
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
    entry,
    plan,
    sessions,
)

GRADED = "GRADE_STATUS_GRADED"


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

    def test_a_filter_that_drops_most_of_an_sf10_table_is_an_anomaly(self) -> None:
        """Task 40's plan: a lineitem scan returning 15.2M of SF10's 60M rows."""
        task_40 = {
            "Node Type": "Seq Scan",
            "Relation Name": "lineitem",
            "Actual Rows": 15_240_492,
            "Rows Removed by Filter": 44_745_560,
            "Actual Loops": 1,
            "Plan Rows": 299_959,
        }
        check = data_check(task(task_40), 4_572_053)
        assert check == DataCheck(outcome="foreign scale", plan_rows=15_240_492)

    def test_a_filter_subplan_that_ran_once_per_sf10_row_is_an_anomaly(self) -> None:
        """Task 40's recorded plan, which leaves out Rows Removed by Filter."""
        subplan = {
            "Node Type": "Aggregate",
            "Parent Relationship": "SubPlan",
            "Actual Loops": 59_986_052,
        }
        task_40 = {
            "Node Type": "Seq Scan",
            "Relation Name": "lineitem",
            "Actual Rows": 15_240_492,
            "Actual Loops": 1,
            "Plans": [subplan],
        }
        assert data_check(task(task_40), 4_572_053).outcome == "foreign scale"
        initplan = {**subplan, "Parent Relationship": "InitPlan"}
        assert data_check(task({**task_40, "Plans": [initplan]}), 4_572_053).outcome == "fail"

    def test_a_filtered_sf3_scan_still_counts(self) -> None:
        sf3 = {
            "Node Type": "Seq Scan",
            "Relation Name": "lineitem",
            "Actual Rows": 4_572_053,
            "Rows Removed by Filter": 13_424_556,
        }
        assert data_check(task(sf3), 4_572_053).outcome == "pass"
        nested = {"Node Type": "Gather", "Actual Rows": 3, "Plans": [{**sf3, "Actual Rows": 3}]}
        assert data_check(task(nested), 3).outcome == "pass"

    def test_a_parallel_scan_counts_every_process(self) -> None:
        parallel = {
            "Node Type": "Gather",
            "Actual Rows": 10,
            "Plans": [
                {
                    "Node Type": "Seq Scan",
                    "Relation Name": "orders",
                    "Parallel Aware": True,
                    "Actual Loops": 3,
                    "Actual Rows": 3,
                    "Rows Removed by Filter": 5_000_000,
                }
            ],
        }
        assert data_check(task(parallel), 10).outcome == "foreign scale"


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
        reason = exclusion_reason(failed)
        assert reason == "base query fails on this machine"

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
            (gold(speedup=1.19), "only 1.19x faster than base_sql (under 1.2x)"),
        ],
    )
    def test_a_failing_gold(self, summary: GradeSummary, expected: str) -> None:
        assert expected in str(exclusion_reason(record(gold=summary)))

    def test_a_gold_at_exactly_1_2_is_kept(self) -> None:
        assert exclusion_reason(record(gold=gold(speedup=1.2))) is None

    def test_a_flagged_gold_is_left_to_review(self) -> None:
        assert exclusion_reason(record(gold=gold(alt_matched=False, credit=0.0))) is None


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
            "perturb": "SELECT c1, CASE WHEN rn = 1 THEN c2 + 1 ELSE c2 END FROM (SELECT *, "
            "row_number() OVER () AS rn FROM (\nSELECT a, b FROM t\n) AS base (c1, c2)) "
            "AS numbered",
        }
        assert all(source in query for query in variants.values())

    def test_text_is_perturbed_when_no_number_or_date_is_there(self) -> None:
        variants = negative_variants("SELECT a FROM t", ["bpchar"], 1)
        assert "c1 || '~'" in variants["perturb"]
        assert "reorder" not in variants

    def test_an_empty_result_has_only_the_control_and_reorder(self) -> None:
        assert set(negative_variants("SELECT 1, 2", ["int4", "int4"], 0)) == {"control", "reorder"}

    def test_an_unknown_type_is_not_perturbed(self) -> None:
        assert "perturb" not in negative_variants("SELECT x", ["jsonb"], 3)


class TestNegativeSample:
    def test_a_fixed_sample_of_kept_tasks_with_small_results(self) -> None:
        records = {str(i): record(str(i)) for i in range(1, 31)}
        records["5"] = record("5", base=entry(instance_id="5", row_count=10_000))
        records["6"] = record("6", gold=gold(matched=False))
        sample = negative_sample(records)
        assert len(sample) == NEGATIVE_SAMPLE
        assert "5" not in sample
        assert "6" not in sample
        assert sample == sorted(sample, key=int)
        assert negative_sample(records) == sample

    def test_fewer_candidates_than_the_sample(self) -> None:
        assert negative_sample({"1": record("1")}) == ["1"]


class TestSummarize:
    def test_a_clean_setup_passes(self) -> None:
        records = {
            str(i): record(str(i), negatives={"control": True, "drop": False}) for i in range(1, 11)
        }
        for task in ("1", "2"):
            records[task] = records[task].model_copy(
                update={
                    "official_gold": gold(eval_wall_s=40.0),
                    "official_identity": gold(speedup=1.03),
                }
            )
        summary = summarize(list(records), records, val=["1", "2", "99"])
        assert summary.passed
        assert summary.exclusions == []
        assert not summary.stop
        assert summary.agreement_tasks == ["1", "2"]
        assert summary.same_tier_share == 1.0
        assert summary.times["official grading (gold)"] is not None
        assert "All self-checks passed: True" in to_markdown(summary)

    def test_reports_every_failure(self) -> None:
        records = {
            "1": record("1", base=None, base_failure="late", base_timed_out=True, gold=None),
            "2": record("2", data=DataCheck(outcome="fail", plan_rows=9)),
            "3": record("3", base=entry(instance_id="3", identity_speedup=1.2)),
            "4": record("4", gold=gold(literal_result=True, credit=0.0)),
            "5": record("5", negatives={"control": False, "drop": True}),
            "6": record(
                "6",
                official_gold=gold(matched=False, speedup=None, credit=0.0),
                official_identity=gold(speedup=0.8),
            ),
            "7": record(
                "7",
                official_gold=gold(speedup=1.5, credit=0.75),
                official_identity=gold(speedup=1.0),
            ),
        }
        summary = summarize([*records, "8"], records, val=["6", "7"])
        assert summary.missing == ["8"]
        assert summary.exclusions == [
            Exclusion(
                instance_id="1",
                reason="base query exceeds the 300 s statement timeout on this machine",
            )
        ]
        assert summary.stop  # 1 of 8 is over 10%.
        assert summary.data["fail"] == ["2"]
        assert summary.identity_failures == {"3": 1.2}
        assert summary.review == ["4"]
        assert summary.gold_failures == ["4"]
        assert summary.negative_failures == {"5": ["control", "drop"]}
        assert summary.match_disagreements == ["6"]
        assert summary.same_tier_share == 0.0
        assert summary.identity_disagreements == {"6": [1.02, 0.8]}
        assert not summary.passed
        assert "TRIGGERED" in to_markdown(summary)

    def test_the_stop_rule_allows_up_to_10_percent(self) -> None:
        records = {str(i): record(str(i)) for i in range(1, 11)}
        records["1"] = record("1", gold=gold(speedup=1.0))
        assert not summarize(list(records), records, val=[]).stop


def test_dump_exclusions_is_the_committed_format() -> None:
    text = dump_exclusions([Exclusion(instance_id="12", reason="why")])
    assert json.loads(text) == [{"instance_id": "12", "reason": "why"}]
    assert text.endswith("\n")


def test_official_credit_ignores_the_flags() -> None:
    assert official_credit(gold(alt_matched=False, credit=0.0, speedup=2.5)) == 1.0
    assert official_credit(gold(matched=False)) == 0.0


def test_grade_summary_of_a_grade() -> None:
    summary = GradeSummary.of(
        Grade(
            status=GRADE_STATUS_GRADED,
            feedback="ok",
            effisql=EffiSqlGrade(matched=True, speedup=2.0, credit=1.0),
        )
    )
    assert summary == GradeSummary(
        status=GRADED, matched=True, speedup=2.0, credit=1.0, feedback="ok"
    )
    assert not summary.flagged


def test_the_record_store_round_trips_and_skips_bad_files(tmp_path: Path) -> None:
    store = RecordStore(tmp_path)
    store.save(record("7"))
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    assert store.load() == {"7": record("7")}


class FakeSlot:
    instance = GradingInstance("fake", 1)

    def __init__(self, main: FakeSession, alt: FakeSession) -> None:
        self._sessions = sessions(main, alt)
        self.open_sessions = self._sessions

    def sessions(self) -> Any:
        return self._sessions

    def reset(self) -> None:
        pass


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
        times={BASE_SQL: base_times or [1000.0] * 24, GOLD_SQL: [400.0] * 6},
    )


def fake_alt() -> FakeSession:
    return FakeSession(results={BASE_SQL: ALT_ROWS, GOLD_SQL: ALT_ROWS})


def pool_of(*slots: FakeSlot) -> GradingPool[SlotRunner]:
    return GradingPool([ThreadRunner(slot) for slot in slots])  # type: ignore[arg-type]  # Fakes.


class TestRunPhase:
    def test_measure_writes_each_record_and_resumes(self, tmp_path: Path) -> None:
        store, cache = RecordStore(tmp_path / "records"), BaseCache(tmp_path / "dev")
        pool = pool_of(FakeSlot(fake_main(), fake_alt()))
        tasks = [instance("7")]
        assert asyncio.run(run_phase("measure", pool, tasks, store, cache)) == 1
        measured = store.load()["7"]
        assert measured.base is not None
        assert measured.base.identity_speedup == 1.0
        assert measured.data == DataCheck(outcome="pass", plan_rows=3)
        assert measured.gold is not None
        assert measured.gold.speedup == 2.5
        assert cache.get(tasks[0]) is not None
        assert asyncio.run(run_phase("measure", pool, tasks, store, cache)) == 0

    def test_a_base_timeout_is_recorded(self, tmp_path: Path) -> None:
        store = RecordStore(tmp_path)
        main = fake_main(base_times=[QueryTimeoutError("cancelled after 300 s")])
        pool = pool_of(FakeSlot(main, fake_alt()))
        asyncio.run(run_phase("measure", pool, [instance("7")], store, BaseCache(tmp_path / "c")))
        failed = store.load()["7"]
        assert failed.base_timed_out
        assert failed.gold is None

    def test_identity_off_by_more_than_10_percent_is_measured_again(self, tmp_path: Path) -> None:
        main = fake_main(base_times=[1000.0, 1300.0] * 3 + [1000.0] * 6)
        store = RecordStore(tmp_path)
        pool = pool_of(FakeSlot(main, fake_alt()))
        asyncio.run(run_phase("measure", pool, [instance("7")], store, BaseCache(tmp_path / "c")))
        measured = store.load()["7"].base
        assert measured is not None
        assert measured.identity_speedup == 1.0
        assert main.kinds(BASE_SQL).count("time") == 12

    def test_official_grades_the_val_golds_and_identity(self, tmp_path: Path) -> None:
        store, cache = RecordStore(tmp_path / "records"), BaseCache(tmp_path / "dev")
        pool = pool_of(FakeSlot(fake_main(), fake_alt()))
        tasks = [instance("7"), instance("8")]
        asyncio.run(run_phase("measure", pool, tasks, store, cache))
        main = FakeSession(
            plans={BASE_SQL: plan("lineitem"), GOLD_SQL: plan("lineitem")},
            results={BASE_SQL: BASE_ROWS, GOLD_SQL: BASE_ROWS},
            times={BASE_SQL: [1000.0] * 9, GOLD_SQL: [500.0] * 3},
        )
        official_pool = pool_of(FakeSlot(main, fake_alt()))
        ran = asyncio.run(run_phase("official", official_pool, tasks, store, cache, val=["7"]))
        assert ran == 1
        records = store.load()
        assert records["7"].official_gold is not None
        assert records["7"].official_gold.speedup == 2.0
        assert records["7"].official_identity is not None
        assert records["7"].official_identity.speedup == 1.0
        assert records["8"].official_gold is None

    def test_negatives_run_on_the_sample(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store, cache = RecordStore(tmp_path / "records"), BaseCache(tmp_path / "dev")
        store.save(record("7"))
        calls: list[str] = []

        def fake_negative(_slot: Any, inst: EffiSqlInstance, _entry: Any) -> dict[str, bool]:
            calls.append(inst.instance_id)
            return {"control": True, "drop": False}

        from goaly.bench.effisql import setup_run

        monkeypatch.setattr(setup_run, "negative_task", fake_negative)
        pool = pool_of(FakeSlot(fake_main(), fake_alt()))
        asyncio.run(run_phase("negatives", pool, [instance("7")], store, cache))
        assert calls == ["7"]
        assert store.load()["7"].negatives == {"control": True, "drop": False}


def test_a_failing_task_is_logged_and_left_for_a_rerun(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = RecordStore(tmp_path)
    main = fake_main()
    main.results[BASE_SQL] = GraderError("the connection failed")
    pool = pool_of(FakeSlot(main, fake_alt()))
    tasks = [instance("7")]
    assert asyncio.run(run_phase("measure", pool, tasks, store, BaseCache(tmp_path / "c"))) == 0
    assert store.load() == {}
    assert "setup task failed" in caplog.text


def test_summarize_decides_the_data_check_again_from_the_tasks() -> None:
    stale = record("7", data=DataCheck(outcome="fail", plan_rows=15_240_492))
    plan = {
        "Node Type": "Seq Scan",
        "Relation Name": "lineitem",
        "Actual Rows": 15_240_492,
        "Rows Removed by Filter": 44_745_560,
    }
    summary = summarize(["7"], {"7": stale}, val=[], tasks={"7": task(plan)})
    assert summary.data["fail"] == []
    assert summary.data["foreign scale"] == ["7"]
