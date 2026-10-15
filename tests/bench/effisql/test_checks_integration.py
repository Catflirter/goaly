"""The grader's checks on real plans and the real alternate database.

Run with ``GOALY_EFFISQL_ITEST_REAL=1 pytest --integration``; without the variable they skip.
"""

from collections.abc import Iterator

import pytest

from goaly.bench.effisql.checks import (
    alt_matches,
    inspect_plan,
    read_only_problems,
    scanned_relations,
)
from goaly.bench.effisql.grader import PgSession, QueryError
from goaly.bench.effisql.pg_settings import ALT_DB
from tests.bench.effisql.real_grading import real_session

pytestmark = pytest.mark.integration

BASE = """
SELECT n_name, sum(l_extendedprice) AS revenue
FROM lineitem JOIN supplier ON l_suppkey = s_suppkey JOIN nation ON s_nationkey = n_nationkey
GROUP BY n_name
"""


@pytest.fixture
def main() -> Iterator[PgSession]:
    opened = real_session()
    yield opened
    opened.close()


@pytest.fixture
def alt() -> Iterator[PgSession]:
    opened = real_session(db=ALT_DB)
    yield opened
    opened.close()


def base_relations(session: PgSession) -> frozenset[str]:
    return scanned_relations(session.plan(BASE))


@pytest.mark.parametrize(
    "prediction",
    [
        "SELECT 'ALGERIA' AS n_name, 12345.67 AS revenue",
        "SELECT * FROM (VALUES ('ALGERIA', 1.5), ('BRAZIL', 2.5)) v(n_name, revenue)",
        "SELECT n_name, CASE n_name WHEN 'ALGERIA' THEN 1.5 ELSE 2.5 END FROM nation",
        "SELECT n_name, v.r FROM nation JOIN (VALUES ('ALGERIA', 1.5)) v(n, r) ON v.n = n_name",
        "SELECT r_name, 1.5 FROM region",
        "SELECT * FROM (VALUES ('ALGERIA', 1.5), ('BRAZIL', 2.5)) v(n_name, revenue) "
        "WHERE current_database() = 'tpch' UNION ALL SELECT n_name, sum(l_extendedprice) "
        "FROM lineitem JOIN supplier ON l_suppkey = s_suppkey JOIN nation "
        "ON s_nationkey = n_nationkey GROUP BY n_name HAVING current_database() <> 'tpch'",
        "SELECT * FROM (VALUES ('ALGERIA', 1.5), ('BRAZIL', 2.5)) v(n_name, revenue) "
        "UNION ALL SELECT n_name, 0 FROM nation WHERE false",
        "WITH c AS MATERIALIZED (VALUES ('ALGERIA', 1.5)) SELECT * FROM c "
        "WHERE EXISTS (SELECT 1 FROM lineitem)",
    ],
)
def test_constant_answers_are_flagged(main: PgSession, prediction: str) -> None:
    verdict = inspect_plan(main.plan(prediction), base_relations(main), sql=prediction)
    assert verdict.literal_result


@pytest.mark.parametrize(
    "prediction",
    [
        BASE,
        """SELECT n_name, sum(x) FROM lineitem JOIN supplier ON l_suppkey = s_suppkey
           JOIN nation ON s_nationkey = n_nationkey,
           LATERAL (VALUES (l_extendedprice)) t(x) GROUP BY n_name""",
        "SELECT 'all', (SELECT sum(l_extendedprice) FROM lineitem)",
        "SELECT 'all', max(l_extendedprice) FROM lineitem LIMIT 1",
        "WITH c AS MATERIALIZED (SELECT max(l_extendedprice) m FROM lineitem) "
        "SELECT 'all', m FROM c ORDER BY 1",
    ],
)
def test_queries_that_read_the_data_pass(main: PgSession, prediction: str) -> None:
    verdict = inspect_plan(main.plan(prediction), base_relations(main), sql=prediction)
    assert not verdict.literal_result
    assert not verdict.modifies_data


def test_the_grader_role_cannot_even_plan_a_data_change(main: PgSession) -> None:
    """PostgreSQL refuses the plan for lack of privilege, before the ModifyTable check could."""
    with pytest.raises(QueryError, match="permission denied"):
        main.plan("WITH d AS (DELETE FROM region RETURNING *) SELECT * FROM d")


def test_the_grading_role_is_read_only(main: PgSession, alt: PgSession) -> None:
    assert read_only_problems(main.connection) == []
    assert read_only_problems(alt.connection) == []


def test_the_robustness_check(main: PgSession, alt: PgSession) -> None:
    base_alt = alt.digest(BASE)
    assert alt_matches(alt, BASE, base_alt).matched
    rewrite = """SELECT n_name, sum(l_extendedprice) FROM nation JOIN supplier
                 ON s_nationkey = n_nationkey JOIN lineitem ON l_suppkey = s_suppkey
                 GROUP BY n_name"""
    assert alt_matches(alt, rewrite, base_alt).matched
    # The main database's answer, copied in as constants, fails on the alternate database.
    rows = main.connection.execute(BASE).fetchall()
    values = ", ".join(f"('{name}', {revenue}::numeric)" for name, revenue in rows)
    constant = f"SELECT * FROM (VALUES {values}) v(n_name, revenue)"  # noqa: S608  # Test data.
    assert main.digest(constant).matches(main.digest(BASE))
    outcome = alt_matches(alt, constant, base_alt)
    assert not outcome.matched
    assert outcome.reason == "the prediction's result differs from the base's there"
