"""Unit tests of grading in dev and official mode, the base cache and identity normalization."""

import random
from pathlib import Path
from typing import Any

import pytest

from goaly.bench.effisql.grader import LONG_RUN_MS, GraderError, QueryError, QueryTimeoutError
from goaly.bench.effisql.grading import (
    BaseCache,
    Mode,
    evaluate,
    measure_base,
    normalize_sql,
    prediction_sql,
    quick_grade,
)
from goaly.contracts.gen.task_pb2 import (
    GRADE_STATUS_GRADED,
    GRADE_STATUS_GRADER_ERROR,
    GRADE_STATUS_PREDICTION_FAILED,
    GRADE_STATUS_TIMEOUT,
    EffiSqlPrediction,
    Grade,
    Prediction,
    SwePrediction,
)
from tests.bench.effisql.grading_fakes import (
    ALT_ROWS,
    BASE_ROWS,
    BASE_SQL,
    INST,
    FakeSession,
    Row,
    entry,
    plan,
    sessions,
)

PRED = "SELECT l_returnflag, count(*) FROM lineitem GROUP BY 1"


class TestNormalizeSql:
    @pytest.mark.parametrize(
        "variant",
        [
            "select L_RETURNFLAG, COUNT(*) from LINEITEM group by L_RETURNFLAG;",
            "SELECT  l_returnflag ,\n\tcount( * )\nFROM lineitem -- the table\n"
            "GROUP BY l_returnflag",
            "SELECT /* a /* nested */ comment */ l_returnflag, count(*) FROM lineitem "
            "GROUP BY l_returnflag ;; ",
        ],
    )
    def test_equivalent_spellings_normalize_the_same(self, variant: str) -> None:
        assert normalize_sql(variant) == normalize_sql(BASE_SQL)

    def test_the_normal_form(self) -> None:
        assert (
            normalize_sql(BASE_SQL)
            == "select l_returnflag,count(*)from lineitem group by l_returnflag"
        )

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            ("SELECT 'A'", "SELECT 'a'"),
            ('SELECT "Col" FROM t', "SELECT col FROM t"),
            ("SELECT 'a  b'", "SELECT 'a b'"),
            ("SELECT $$A$$", "SELECT $$a$$"),
            ("SELECT $x$ A $x$", "SELECT $x$ a $x$"),
            ("SELECT E'\\'A'", "SELECT E'\\'a'"),
            ("SELECT 'it''s A'", "SELECT 'it''s a'"),
            ("SELECT a FROM t", "SELECT a FROM u"),
            ("SELECT a b", "SELECT ab"),
        ],
    )
    def test_quoted_text_and_tokens_stay_distinct(self, left: str, right: str) -> None:
        assert normalize_sql(left) != normalize_sql(right)

    def test_comment_markers_inside_quotes_are_text(self) -> None:
        assert normalize_sql("SELECT '-- x', \"/* y\"") == "select '-- x',\"/* y\""
        assert normalize_sql("SELECT $q$ -- not a comment $q$") == "select $q$ -- not a comment $q$"

    def test_unterminated_quotes_and_comments_run_to_the_end(self) -> None:
        assert normalize_sql("SELECT 'abc") == "select 'abc"
        assert normalize_sql("SELECT E'a\\") == "select E'a\\"
        assert normalize_sql("SELECT $t$ open") == "select $t$ open"
        assert normalize_sql("SELECT 1 /* open") == "select 1"

    def test_a_dollar_inside_an_identifier_is_not_a_quote(self) -> None:
        assert normalize_sql("SELECT a$b$ FROM T") == "select a$b$ from t"


class TestPredictionSql:
    def test_the_effisql_answer(self) -> None:
        assert prediction_sql(INST, Prediction(effisql=EffiSqlPrediction(sql=PRED))) == PRED

    def test_no_branch_is_base_sql_unchanged(self) -> None:
        assert prediction_sql(INST, Prediction(instance_id="7")) == BASE_SQL

    def test_another_benchmarks_answer_is_none(self) -> None:
        assert prediction_sql(INST, Prediction(swe=SwePrediction(patch=""))) is None


def main_session(
    *,
    pred_rows: list[Row] | Exception = BASE_ROWS,
    pred_plan: dict[str, Any] | Exception | None = None,
    pred_times: list[float | Exception] | None = None,
    base_times: list[float | Exception] | None = None,
    base_rows: list[Row] | Exception = BASE_ROWS,
) -> FakeSession:
    return FakeSession(
        plans={
            PRED: plan("lineitem") if pred_plan is None else pred_plan,
            BASE_SQL: plan("lineitem"),
        },
        results={PRED: pred_rows, BASE_SQL: base_rows},
        times={PRED: pred_times or [250.0] * 3, BASE_SQL: base_times or [1000.0] * 3},
    )


def alt_session(pred_rows: list[Row] | Exception = ALT_ROWS) -> FakeSession:
    return FakeSession(results={PRED: pred_rows, BASE_SQL: ALT_ROWS})


@pytest.fixture
def cache(tmp_path: Path) -> BaseCache:
    filled = BaseCache(tmp_path / "dev")
    filled.put(entry())
    return filled


def grade_dev(main: FakeSession, alt: FakeSession, cache: BaseCache, sql: str = PRED) -> Grade:
    return evaluate(sessions(main, alt), INST, sql, Mode.DEV, cache=cache, rng=random.Random(0))  # noqa: S311  # Reproducible order.


def grade_official(main: FakeSession, alt: FakeSession, sql: str = PRED) -> Grade:
    return evaluate(sessions(main, alt), INST, sql, Mode.OFFICIAL, cache=None, rng=random.Random(0))  # noqa: S311  # Reproducible order.


class TestDevMode:
    def test_a_matched_prediction_is_checked_and_timed_alone(self, cache: BaseCache) -> None:
        main, alt = main_session(), alt_session()
        grade = grade_dev(main, alt, cache)
        details = grade.effisql
        assert grade.status == GRADE_STATUS_GRADED
        assert grade.correct
        assert details.matched
        assert details.alt_matched
        assert details.HasField("literal_result")
        assert not details.literal_result
        assert (details.base_time_s, details.pred_time_s, details.timing_runs) == (1.0, 0.25, 3)
        assert details.speedup == 4.0
        assert details.credit == grade.score == 1.0
        assert details.base_cached
        assert not details.identity_cached
        assert details.HasField("eval_wall_s")
        assert main.kinds(BASE_SQL) == [], "the base side comes from the cache"
        assert main.kinds(PRED) == ["plan", "digest", "time", "time", "time"]
        assert ("digest", PRED, 3) in main.calls, "fetching stops past the base's row count"
        assert alt.calls == [("digest", PRED, 3)]

    def test_identity_gets_the_setups_tau_without_running(self, cache: BaseCache) -> None:
        main, alt = main_session(), alt_session()
        grade = grade_dev(
            main,
            alt,
            cache,
            sql="select l_returnflag, count(*)\nfrom lineitem group by l_returnflag;",
        )
        details = grade.effisql
        assert grade.status == GRADE_STATUS_GRADED
        assert grade.correct
        assert details.matched
        assert details.identity_cached
        assert details.base_cached
        assert details.speedup == 1.02
        assert details.credit == 0.5
        assert details.base_time_s == 1.0
        assert not details.HasField("pred_time_s")
        assert details.timing_runs == 0
        assert not details.HasField("eval_wall_s")
        assert not details.HasField("alt_matched")
        assert main.calls == []
        assert alt.calls == []

    def test_identity_without_a_measured_tau_is_timed(self, tmp_path: Path) -> None:
        cache = BaseCache(tmp_path)
        cache.put(entry(identity_speedup=None))
        main = main_session()
        main.plans[BASE_SQL] = plan("lineitem")
        main.times[BASE_SQL] = [1000.0] * 3
        grade = grade_dev(main, alt_session(), cache, sql=BASE_SQL)
        assert not grade.effisql.identity_cached
        assert grade.effisql.timing_runs == 3

    def test_a_missing_entry_is_measured_and_cached(self, tmp_path: Path) -> None:
        cache = BaseCache(tmp_path)
        main = main_session(base_times=[900.0, 1100.0, 1000.0])
        first = grade_dev(main, alt_session(), cache)
        assert first.effisql.matched
        assert not first.effisql.base_cached
        stored = cache.get(INST)
        assert stored is not None
        assert (stored.base_ms, stored.timing_runs, stored.relations) == (1000.0, 3, ("lineitem",))
        assert stored.identity_speedup is None
        assert stored.digest.row_count == 3
        second = grade_dev(main_session(), alt_session(), cache)
        assert second.effisql.base_cached

    def test_a_long_cached_base_stops_the_timing_after_two_runs(self, tmp_path: Path) -> None:
        cache = BaseCache(tmp_path)
        cache.put(entry(base_ms=LONG_RUN_MS))
        grade = grade_dev(main_session(), alt_session(), cache)
        assert grade.effisql.timing_runs == 2


class TestOfficialMode:
    def test_everything_is_measured_interleaved(self) -> None:
        main, alt = main_session(), alt_session()
        grade = grade_official(main, alt)
        details = grade.effisql
        assert grade.status == GRADE_STATUS_GRADED
        assert details.matched
        assert details.alt_matched
        assert details.speedup == 4.0
        assert details.timing_runs == 3
        assert not details.base_cached
        assert main.kinds(PRED) == ["plan", "digest", "time", "time", "time"]
        assert main.kinds(BASE_SQL) == ["plan", "digest", "time", "time", "time"]
        assert ("digest", PRED, None) in main.calls
        assert ("digest", BASE_SQL, 3) in main.calls, "the base is fetched up to the pred's count"
        assert alt.calls == [("digest", BASE_SQL, None), ("digest", PRED, 3)]

    def test_identity_is_measured_like_any_prediction(self) -> None:
        main = main_session()
        main.times[BASE_SQL] = [1000.0] * 6
        grade = grade_official(main, alt_session(), sql=BASE_SQL)
        assert not grade.effisql.identity_cached
        assert grade.effisql.timing_runs == 3
        assert grade.effisql.speedup == 1.0

    def test_a_base_timeout_scores_like_the_official_evaluator(self) -> None:
        main = main_session(base_times=[QueryTimeoutError("late")])
        grade = grade_official(main, alt_session())
        assert grade.status == GRADE_STATUS_TIMEOUT
        assert grade.score == 0
        assert "base query failed" in grade.feedback

    def test_a_base_error_is_the_graders(self) -> None:
        main = main_session(base_rows=QueryError("relation does not exist"))
        grade = grade_official(main, alt_session())
        assert grade.status == GRADE_STATUS_GRADER_ERROR


class TestOutcomes:
    """Each way an evaluation can end, the same in both modes."""

    @pytest.fixture(params=[Mode.DEV, Mode.OFFICIAL])
    def grade(self, request: pytest.FixtureRequest, cache: BaseCache) -> Any:
        mode: Mode = request.param

        def run(main: FakeSession, alt: FakeSession | None = None, sql: str = PRED) -> Grade:
            used = cache if mode is Mode.DEV else None
            return evaluate(
                sessions(main, alt or alt_session()),
                INST,
                sql,
                mode,
                cache=used,
                rng=random.Random(0),  # noqa: S311  # Reproducible order.
            )

        return run

    @pytest.mark.parametrize("sql", ["", "  \n"])
    def test_an_empty_prediction_fails(self, grade: Any, sql: str) -> None:
        result = grade(main_session(), sql=sql)
        assert result.status == GRADE_STATUS_PREDICTION_FAILED
        assert result.HasField("effisql")

    def test_a_prediction_that_does_not_plan_fails(self, grade: Any) -> None:
        result = grade(main_session(pred_plan=QueryError('syntax error at or near "SELEC"')))
        assert result.status == GRADE_STATUS_PREDICTION_FAILED
        assert "syntax error" in result.feedback
        assert not result.effisql.matched

    def test_planning_that_times_out(self, grade: Any) -> None:
        result = grade(main_session(pred_plan=QueryTimeoutError("cancelled")))
        assert result.status == GRADE_STATUS_TIMEOUT

    def test_a_data_modifying_plan_is_refused_before_it_runs(self, grade: Any) -> None:
        modifying = {"Plan": {"Node Type": "ModifyTable", "Relation Name": "lineitem"}}
        main = main_session(pred_plan=modifying)
        result = grade(main)
        assert result.status == GRADE_STATUS_PREDICTION_FAILED
        assert not result.effisql.matched
        assert "refused" in result.feedback
        assert main.kinds(PRED) == ["plan"]

    def test_an_unmatched_prediction_runs_no_check_and_no_timing(self, grade: Any) -> None:
        main = main_session(pred_rows=[("A", 10), ("N", 20)], pred_plan=plan())
        alt = alt_session()
        result = grade(main, alt)
        details = result.effisql
        assert result.status == GRADE_STATUS_GRADED
        assert not result.correct
        assert not details.matched
        assert not details.HasField("alt_matched")
        assert not details.HasField("literal_result")
        assert not details.HasField("speedup")
        assert (details.pred_row_count, details.base_row_count) == (2, 3)
        assert details.credit == 0
        assert "time" not in main.kinds(PRED)
        assert ("digest", PRED, 3) not in alt.calls

    def test_too_many_rows_do_not_match(self, grade: Any) -> None:
        result = grade(main_session(pred_rows=[*BASE_ROWS, ("X", 1)]))
        assert not result.effisql.matched

    @pytest.mark.parametrize(
        ("error", "status"),
        [
            (QueryTimeoutError("cancelled after 300 s"), GRADE_STATUS_TIMEOUT),
            (QueryError("division by zero"), GRADE_STATUS_PREDICTION_FAILED),
        ],
    )
    def test_a_failed_comparison_run_is_unmatched(
        self, grade: Any, error: Exception, status: int
    ) -> None:
        result = grade(main_session(pred_rows=error))
        assert result.status == status
        assert not result.effisql.matched
        assert result.score == 0

    def test_a_literal_result_is_flagged_timed_and_gets_no_credit(self, grade: Any) -> None:
        main = main_session(pred_plan=plan())
        result = grade(main)
        details = result.effisql
        assert result.status == GRADE_STATUS_GRADED
        assert not result.correct
        assert details.matched
        assert details.literal_result
        assert details.speedup == 4.0
        assert details.credit == 0
        assert "literal_result" in result.feedback

    def test_a_result_that_differs_on_the_alternate_database_is_hard_coded(
        self, grade: Any
    ) -> None:
        result = grade(main_session(), alt_session(pred_rows=BASE_ROWS))
        details = result.effisql
        assert details.matched
        assert details.HasField("alt_matched")
        assert not details.alt_matched
        assert details.HasField("speedup")
        assert details.credit == 0
        assert not result.correct
        assert "hard_coded" in result.feedback

    @pytest.mark.parametrize(
        ("error", "status"),
        [
            (QueryTimeoutError("late"), GRADE_STATUS_TIMEOUT),
            (QueryError("out of memory"), GRADE_STATUS_PREDICTION_FAILED),
        ],
    )
    def test_a_failed_timing_stays_matched_with_tau_zero(
        self, grade: Any, error: Exception, status: int
    ) -> None:
        result = grade(main_session(pred_times=[250.0, error]))
        details = result.effisql
        assert result.status == status
        assert details.matched
        assert details.HasField("speedup")
        assert details.speedup == 0.0
        assert details.credit == 0
        assert not result.correct
        assert "τ = 0" in result.feedback

    @pytest.mark.parametrize(
        ("pred_ms", "credit"),
        [
            (500.0, 1.0),  # τ = 2
            (501.0, 0.75),
            (1000 / 1.2, 0.75),  # τ = 1.2
            (834.0, 0.5),
            (1000.0, 0.5),  # τ = 1
            (1001.0, 0.25),
            (2000.0, 0.25),  # τ = 0.5
            (2001.0, 0.0),
            (0.0, 1.0),  # A zero time counts as 0.001 ms: τ capped at 1000.
        ],
    )
    def test_credit_tiers(self, grade: Any, pred_ms: float, credit: float) -> None:
        result = grade(main_session(pred_times=[pred_ms] * 3))
        assert result.effisql.credit == credit
        assert result.score == credit

    def test_a_grader_error_is_reported_not_raised(self, grade: Any) -> None:
        result = grade(main_session(pred_rows=GraderError("the connection failed")))
        assert result.status == GRADE_STATUS_GRADER_ERROR
        assert not result.correct
        assert "the connection failed" in result.feedback

    def test_not_an_effisql_answer(self, grade: Any) -> None:
        result = grade(main_session(), sql=None)
        assert result.status == GRADE_STATUS_PREDICTION_FAILED


class TestBaseCache:
    def test_round_trip(self, tmp_path: Path) -> None:
        cache = BaseCache(tmp_path / "dev")
        cache.put(entry())
        assert cache.get(INST) == entry()
        assert (tmp_path / "dev" / "7.json").stat().st_mode & 0o777 == 0o600

    def test_the_grading_mode_is_part_of_the_key(self, tmp_path: Path) -> None:
        official = BaseCache(tmp_path, mode=Mode.OFFICIAL)
        BaseCache(tmp_path).put(entry())
        assert official.get(INST) is None
        with pytest.raises(ValueError, match="does not belong"):
            official.put(entry())

    def test_an_entry_for_another_base_sql_is_ignored(self, tmp_path: Path) -> None:
        cache = BaseCache(tmp_path)
        cache.put(entry(base_sql_sha256="0" * 64))
        assert cache.get(INST) is None

    def test_a_missing_or_unreadable_entry_is_none(self, tmp_path: Path) -> None:
        cache = BaseCache(tmp_path)
        assert cache.get(INST) is None
        cache.path(INST.instance_id).write_text("{not json", encoding="utf-8")
        assert cache.get(INST) is None


class TestMeasureBase:
    def test_measures_the_plan_both_digests_and_the_median(self) -> None:
        main = main_session(base_times=[900.0, 1200.0, 1000.0])
        measured = measure_base(sessions(main, alt_session()), INST)
        assert measured == entry(
            base_ms=1000.0,
            identity_speedup=None,
            measured_at=measured.measured_at,
        )

    def test_with_an_rng_times_the_base_against_itself_for_the_identity_tau(self) -> None:
        main = main_session(base_times=[1000.0, 1100.0, 900.0, 1000.0, 1000.0, 1000.0])
        rng = random.Random(0)  # noqa: S311  # A reproducible order, not a secret.
        measured = measure_base(sessions(main, alt_session()), INST, identity_rng=rng)
        assert measured.timing_runs == 3
        assert main.kinds(BASE_SQL).count("time") == 6
        assert measured.identity_speedup == pytest.approx(measured.base_ms / 1000.0)

    @pytest.mark.parametrize("error", [QueryError("boom"), QueryTimeoutError("late")])
    def test_a_failed_base_timing_raises_its_error(self, error: Exception) -> None:
        main = main_session(base_times=[error])
        with pytest.raises(type(error)):
            measure_base(sessions(main, alt_session()), INST)


class TestReviewFixes:
    def test_a_data_change_is_refused_before_a_cache_miss_is_measured(self, tmp_path: Path) -> None:
        modifying = {"Plan": {"Node Type": "ModifyTable", "Relation Name": "lineitem"}}
        main = main_session(pred_plan=modifying)
        grade = grade_dev(main, alt_session(), BaseCache(tmp_path))
        assert grade.status == GRADE_STATUS_PREDICTION_FAILED
        assert main.kinds(BASE_SQL) == [], "nothing was measured"

    def test_a_quick_grade_is_not_measured(self, cache: BaseCache) -> None:
        main = main_session()
        grade = grade_dev(main, alt_session(), cache, sql="")
        assert not grade.effisql.HasField("eval_wall_s")
        assert quick_grade(INST, BASE_SQL, Mode.DEV, cache) is not None
        assert quick_grade(INST, BASE_SQL, Mode.OFFICIAL, cache) is None
        assert quick_grade(INST, PRED, Mode.DEV, cache) is None
        assert quick_grade(INST, None, Mode.OFFICIAL, None) is not None

    def test_the_robustness_reason_reaches_the_feedback(self, cache: BaseCache) -> None:
        alt = alt_session(pred_rows=QueryError("division by zero"))
        grade = grade_dev(main_session(), alt, cache)
        assert "hard_coded: the prediction failed on the alternate database" in grade.feedback
        assert not grade.correct

    def test_the_plan_checks_reasons_reach_the_feedback(self, cache: BaseCache) -> None:
        grade = grade_dev(main_session(pred_plan=plan()), alt_session(), cache)
        assert "literal_result: the plan scans none of the base query's tables" in grade.feedback

    def test_an_environment_probe_is_flagged(self, cache: BaseCache) -> None:
        probe = (
            "SELECT l_returnflag, count(*) FROM lineitem "
            "WHERE current_database() = 'tpch' GROUP BY 1"
        )
        main = main_session()
        main.plans[probe] = plan("lineitem")
        main.results[probe] = BASE_ROWS
        main.times[probe] = [250.0] * 3
        alt = alt_session()
        alt.results[probe] = ALT_ROWS
        grade = grade_dev(main, alt, cache, sql=probe)
        assert grade.effisql.literal_result
        assert "probes the environment: current_database" in grade.feedback

    def test_a_truncated_base_count_says_more_than(self) -> None:
        main = main_session(pred_rows=[("A", 10)], base_rows=BASE_ROWS)
        grade = grade_official(main, alt_session())
        assert "returned 1 rows, the base more than 2 rows" in grade.feedback

    def test_an_on_the_fly_measurement_keeps_the_setups_identity_tau(self, tmp_path: Path) -> None:
        cache = BaseCache(tmp_path)
        cache.put_measured(INST, entry(identity_speedup=None))
        assert cache.get(INST) == entry(identity_speedup=None)
        cache.put(entry())
        cache.put_measured(INST, entry(identity_speedup=None, base_ms=5.0))
        kept = cache.get(INST)
        assert kept is not None
        assert kept.identity_speedup == 1.02

    def test_the_plan_check_does_not_apply_to_a_base_without_tables(self) -> None:
        main = main_session(pred_plan=plan())
        main.plans[BASE_SQL] = plan()
        grade = grade_official(main, alt_session())
        assert grade.status == GRADE_STATUS_GRADED
        assert grade.effisql.matched
        assert not grade.effisql.literal_result
