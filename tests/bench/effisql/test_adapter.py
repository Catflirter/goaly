"""Tests for goaly.bench.effisql.adapter: the official prediction record (spec §3.3).

Each answer is checked against tests/bench/effisql/official_extractor.py, a reimplementation of
the official runner's extraction: patch mode must rebuild the query exactly, and end2end must pick
it. Every task here is synthetic.
"""

import json
import os
from pathlib import Path

import pytest

from goaly.bench.effisql.adapter import (
    PREDICTION_FIELD,
    OfficialAnswer,
    QueryFormatError,
    encode_prediction,
    official_answer,
    official_query,
    to_official,
)
from goaly.bench.effisql.data import GOLD_FIELDS, EffiSqlInstance, load_instances
from goaly.contracts.gen.task_pb2 import EffiSqlPrediction, Prediction, SwePrediction
from tests.bench.effisql.official_extractor import (
    FINAL_SQL_MARKER,
    extract_end2end,
    extract_patch,
    looks_like_diff,
)
from tests.bench.effisql.synthetic import BASE_SQL, task

FIXTURE = Path(__file__).parent / "fixtures" / "official_record.jsonl"


def instance(base_sql: str = BASE_SQL) -> EffiSqlInstance:
    return EffiSqlInstance.model_validate(task(base_sql=base_sql))


def answer(sql: str) -> Prediction:
    return Prediction(instance_id="7", effisql=EffiSqlPrediction(sql=sql))


def assert_both_modes_extract(base_sql: str, prediction: str, query: str) -> None:
    """Both official modes must read exactly ``query``, ended with ``;``."""
    assert extract_patch(base_sql, prediction) == f"{query};"
    assert extract_end2end(base_sql, prediction) == f"{query};"


def blocks(prediction: str) -> tuple[str, str]:
    """Return the sql block's and the diff block's bodies, checking nothing else is there."""
    sql_part, diff_part = prediction.split("\n```\n\n```diff\n")
    assert sql_part.startswith("```sql\n")
    assert diff_part.endswith("\n```")
    return sql_part.removeprefix("```sql\n"), diff_part.removesuffix("\n```")


class TestFixture:
    def test_the_record_matches_the_official_format_fixture(self) -> None:
        expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
        record = {name: value for name, value in expected.items() if name != PREDICTION_FIELD}
        inst = EffiSqlInstance.model_validate({**record, "optimized_sql": "SELECT 'gold'"})
        submitted = (
            "-- final query\n"
            "SELECT l_orderkey,\n"
            "       sum(l_extendedprice * (1 - l_discount)) AS revenue -- same rows\n"
            "FROM lineitem\n"
            "GROUP BY l_orderkey;\n"
        )

        official = to_official(inst, answer(submitted))

        assert official == expected
        assert list(official) == list(expected)  # The input record's order, then prediction.
        assert_both_modes_extract(
            inst.base_sql, expected[PREDICTION_FIELD], official_query(submitted)
        )


# Each case: (base_sql, submitted sql, its official form).
PARITY_CASES = {
    "multi-line query": (
        BASE_SQL,
        "SELECT count(*)\nFROM orders o\nWHERE EXISTS (\n  SELECT 1 FROM orders i\n"
        "  WHERE i.o_orderkey = o.o_orderkey\n)",
        "SELECT count(*)\nFROM orders o\nWHERE EXISTS (\n  SELECT 1 FROM orders i\n"
        "  WHERE i.o_orderkey = o.o_orderkey\n)",
    ),
    "base lines that begin with --": (
        "-- the slow query\n--- with a header-like comment\nSELECT *\nFROM nation",
        "SELECT * FROM nation",
        "SELECT * FROM nation",
    ),
    "a base equal to the prediction": (BASE_SQL, BASE_SQL, BASE_SQL),
    "a -- final query comment in the source": (
        BASE_SQL,
        "-- final query\nSELECT count(*) -- optimized query\nFROM orders /* rewritten sql */",
        "SELECT count(*)\nFROM orders",
    ),
    "lines that begin with + or -": (
        BASE_SQL,
        "SELECT sum(l_extendedprice\n- l_discount\n+ l_tax)\nFROM lineitem",
        "SELECT sum(l_extendedprice - l_discount + l_tax)\nFROM lineitem",
    ),
    "lines end2end's cleanup would drop": (
        BASE_SQL,
        "SELECT a.x\nFROM t a\n  analysis_join\n### x\nDiagnosis",
        "SELECT a.x\nFROM t a analysis_join ### x Diagnosis",
    ),
    "a CTE written AS MATERIALIZED": (
        "WITH d AS MATERIALIZED (SELECT * FROM orders)\nSELECT count(*) FROM d",
        "WITH d AS MATERIALIZED (\n  SELECT * FROM orders\n)\nSELECT count(*) FROM d",
        "WITH d AS MATERIALIZED (\n  SELECT * FROM orders\n)\nSELECT count(*) FROM d",
    ),
    "a SELECT with FROM( and no spaced FROM": (
        BASE_SQL,
        "SELECT x FROM(SELECT 1 AS x) s",
        "SELECT x FROM(SELECT 1 AS x) s",
    ),
    "CRLF line ends, blank lines and a final semicolon": (
        "SELECT 1\r\nFROM t\r\n",
        "SELECT 1\r\n\r\nFROM t ;  \r\n",
        "SELECT 1\nFROM t",
    ),
    "comment markers inside literals and quoted identifiers": (
        BASE_SQL,
        "SELECT '-- kept', \"a--b\", $q$ /* kept */ $q$, E'it\\'s -- kept'\nFROM t -- dropped",
        "SELECT '-- kept', \"a--b\", $q$ /* kept */ $q$, E'it\\'s -- kept'\nFROM t",
    ),
    "a nested block comment": (
        BASE_SQL,
        "SELECT /* a /* nested */ comment */ 1 AS one\nFROM t",
        "SELECT   1 AS one\nFROM t",
    ),
    "a multi-line literal keeps its text": (
        BASE_SQL,
        "SELECT 'first  \n\n  second'\nFROM t",
        "SELECT 'first  \n\n  second'\nFROM t",
    ),
    "a line that begins with the @@ operator": (
        BASE_SQL,
        "SELECT count(*) FROM docs\nWHERE to_tsvector(body)\n@@ to_tsquery('x')",
        "SELECT count(*) FROM docs\nWHERE to_tsvector(body) @@ to_tsquery('x')",
    ),
    "a SELECT glued to a parenthesis": (BASE_SQL, "SELECT(1) FROM t", "SELECT (1) FROM t"),
    "a SELECT glued to a star": (BASE_SQL, "SELECT*FROM t", "SELECT *FROM t"),
    "a WITH glued to a quoted name": (
        BASE_SQL,
        'WITH"d" AS (SELECT 1 AS x FROM t) SELECT x FROM "d"',
        'WITH "d" AS (SELECT 1 AS x FROM t) SELECT x FROM "d"',
    ),
    "dollar signs inside identifiers": (
        BASE_SQL,
        "SELECT a$b$c, $1 FROM t$x -- $b$ is no quote here",
        "SELECT a$b$c, $1 FROM t$x",
    ),
    "a base line holding a marker or diff header text": (
        "SELECT 'final answer' AS a,\n'+++' AS b,\n'<think>' AS c\nFROM t -- @@ ```\nWHERE 1 = 1",
        "SELECT 1 AS a\nFROM t",
        "SELECT 1 AS a\nFROM t",
    ),
}


class TestParity:
    @pytest.mark.parametrize(("base_sql", "sql", "query"), PARITY_CASES.values(), ids=PARITY_CASES)
    def test_both_modes_extract_the_official_form(
        self, base_sql: str, sql: str, query: str
    ) -> None:
        result = official_answer(instance(base_sql), answer(sql))

        assert result == OfficialAnswer(query, encode_prediction(base_sql, query))
        assert_both_modes_extract(base_sql, result.prediction, query)

    @pytest.mark.parametrize(("base_sql", "sql", "query"), PARITY_CASES.values(), ids=PARITY_CASES)
    def test_the_blocks_keep_the_format_rules(self, base_sql: str, sql: str, query: str) -> None:
        sql_block, diff_block = blocks(official_answer(instance(base_sql), answer(sql)).prediction)
        header, *lines = diff_block.split("\n")
        removed = [line for line in lines if line.startswith("-")]
        added = [line for line in lines if line.startswith("+")]

        assert sql_block == f"{query};"
        assert header == f"@@ -1,{len(removed)} +1,{len(added)} @@"
        assert len(removed) == len(base_sql.replace("\r\n", "\n").splitlines())
        assert removed + added == lines
        assert "\n".join(line[1:] for line in added) == sql_block
        assert not any(line.startswith(("---", "+++")) for line in lines)
        assert not looks_like_diff(sql_block)
        assert sum(line.startswith(("-", "+")) for line in sql_block.split("\n")) <= 1
        assert not FINAL_SQL_MARKER.search(sql_block + diff_block)

    def test_both_modes_survive_comment_lines_around_every_line(self) -> None:
        sql = "\n-- final query\n".join(["SELECT a", "  , b", "FROM t", "WHERE a > 0"])
        query = official_query(sql)

        assert query == "SELECT a\n  , b\nFROM t\nWHERE a > 0"
        assert_both_modes_extract(BASE_SQL, encode_prediction(BASE_SQL, query), query)

    def test_a_raw_comment_would_have_added_a_competing_end2end_candidate(self) -> None:
        # Why comments go: the same query with its comment kept, in the sql block alone, makes
        # end2end read the text after the marker instead of the block.
        raw = "```sql\n-- final query\nSELECT a FROM t;\n```\n\nfinal answer\nSELECT b FROM t;"

        assert extract_end2end(BASE_SQL, raw) == "SELECT b FROM t;"


class TestFallbackAndEmpty:
    def test_a_prediction_without_an_answer_submits_base_sql(self) -> None:
        inst = instance()

        result = official_answer(inst, Prediction(instance_id="7"))

        assert result.query == BASE_SQL
        assert result.problem is None
        assert_both_modes_extract(BASE_SQL, result.prediction, BASE_SQL)

    def test_the_budget_fallback_is_an_identity_diff_plus_base_sql(self) -> None:
        inst = instance("-- slow\nSELECT count(*)\nFROM orders")
        fallback = Prediction(
            instance_id="7", fallback=True, effisql=EffiSqlPrediction(sql=inst.base_sql)
        )

        result = official_answer(inst, fallback)

        assert result.query == "SELECT count(*)\nFROM orders"
        assert blocks(result.prediction)[1] == (
            "@@ -1,3 +1,2 @@\n-\n-SELECT count(*)\n-FROM orders\n+SELECT count(*)\n+FROM orders;"
        )
        assert_both_modes_extract(inst.base_sql, result.prediction, result.query)

    @pytest.mark.parametrize("sql", ["", "   \n", "-- only a comment\n", ";", "/* x */ ;"])
    def test_an_empty_prediction_fails_in_both_modes(self, sql: str) -> None:
        result = official_answer(instance(), answer(sql))

        assert result == OfficialAnswer(None, "", "the query is empty")
        assert extract_patch(BASE_SQL, result.prediction) is None
        assert extract_end2end(BASE_SQL, result.prediction) is None

    def test_a_swe_answer_has_no_official_effisql_form(self) -> None:
        swe = Prediction(instance_id="7", swe=SwePrediction(patch="diff"))

        result = official_answer(instance(), swe)

        assert result == OfficialAnswer(None, "", "not an Effi-SQL answer: it has a swe answer")


class TestNoOfficialForm:
    @pytest.mark.parametrize(
        ("sql", "problem"),
        [
            ("SELECT 1 FROM t; SELECT 2 FROM t", "more than one statement"),
            ("SELECT ';' FROM t; DELETE FROM t", "more than one statement"),
            ("EXPLAIN SELECT 1 FROM t", "does not begin with SELECT or WITH"),
            ("(SELECT 1 FROM t) UNION (SELECT 2 FROM t)", "does not begin with SELECT or WITH"),
            ("SELECT 'open FROM t", "unterminated literal"),
            ('SELECT "open FROM t', "unterminated literal"),
            ("SELECT $a$ open FROM t", "unterminated dollar-quoted"),
            ("SELECT 1 /* open FROM t", "unterminated block comment"),
            ("SELECT 'final answer' FROM t", "answer marker"),
            ("SELECT '```' FROM t", "```"),
            ("SELECT '---' FROM t", "reads as a diff"),
            ("SELECT '<think>' FROM t", "reasoning tag"),
            ("SELECT 1\x0cFROM t", "line break other than a line feed"),
            ("SELECT 1\rFROM t", "line break other than a line feed"),
            ("SELECT 'a\n- b' FROM t", "inside a literal"),
        ],
    )
    def test_the_answer_says_why_and_its_prediction_is_empty(self, sql: str, problem: str) -> None:
        result = official_answer(instance(), answer(sql))

        assert result.query is None
        assert result.prediction == ""
        assert result.problem is not None
        assert problem in result.problem
        with pytest.raises(QueryFormatError, match=r"query|answer|line"):
            official_query(sql)

    def test_a_base_sql_with_no_official_form_falls_back_to_an_empty_prediction(self) -> None:
        inst = instance("SELECT 'final query' FROM t")

        assert official_answer(inst, Prediction(instance_id="7")).prediction == ""


class TestRecord:
    def test_the_record_is_the_task_record_plus_prediction_without_gold(self) -> None:
        inst = instance()

        record = to_official(inst, answer("SELECT count(*) FROM orders"))

        assert {name: value for name, value in record.items() if name != PREDICTION_FIELD} == (
            dict(inst.record)
        )
        assert not set(GOLD_FIELDS) & set(record)
        assert record[PREDICTION_FIELD] == encode_prediction(
            BASE_SQL, "SELECT count(*) FROM orders"
        )
        assert "synthetic gold answer" not in json.dumps(record)

    def test_an_identifier_starting_with_e_is_not_an_escape_string(self) -> None:
        sql = "SELECT name'x' FROM t -- dropped"  # A typed literal after a name: no E''.

        assert official_query(sql) == "SELECT name'x' FROM t"

    def test_a_doubled_quote_stays_inside_its_literal(self) -> None:
        assert official_query("SELECT 'it''s -- kept' FROM t") == "SELECT 'it''s -- kept' FROM t"

    def test_leading_space_and_comments_go(self) -> None:
        assert official_query("  /* why */  \n\t SELECT 1 FROM t") == "SELECT 1 FROM t"

    def test_a_dollar_parameter_is_not_a_dollar_quote(self) -> None:
        assert official_query("SELECT $1 FROM t -- x") == "SELECT $1 FROM t"


@pytest.mark.integration
def test_every_dataset_query_comes_back_exactly_in_both_modes() -> None:
    """The ticket's acceptance sweep: all 300 base and 300 gold queries of the real dataset."""
    local_copy = os.environ.get("GOALY_EFFISQL_DATA")
    if not local_copy:
        pytest.skip("set GOALY_EFFISQL_DATA to a local copy of effi-sql-pg.jsonl")
    instances = load_instances(Path(local_copy))
    materialized = 0
    checked = 0
    for inst in instances:
        materialized += "as materialized" in " ".join(inst.base_sql.lower().split())
        for sql in (inst.base_sql, inst.optimized_sql):
            if sql is None:
                continue
            query = official_query(sql)
            assert_both_modes_extract(inst.base_sql, encode_prediction(inst.base_sql, query), query)
            checked += 1

    assert len(instances) == 300
    assert checked == 600
    assert materialized == 12
