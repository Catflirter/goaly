"""Sessions on the real grading instances, for integration tests that only read.

They skip unless ``GOALY_EFFISQL_ITEST_REAL=1``, as T09's real-environment tests do.
"""

import os

import pytest

from goaly.bench.effisql.grader import STATEMENT_TIMEOUT_S, PgSession
from goaly.bench.effisql.pg_build import connector
from goaly.bench.effisql.pg_settings import GRADER_ROLE, MAIN_DB, load_credentials

GRADE_1_PORT = 55434
GRADE_2_PORT = 55435


def real_session(
    port: int = GRADE_1_PORT,
    db: str = MAIN_DB,
    *,
    timeout_s: float = STATEMENT_TIMEOUT_S,
    hard_cap_s: float | None = None,
) -> PgSession:
    """Open a grading session as the grader's role, or skip the test without the real instances."""
    if os.environ.get("GOALY_EFFISQL_ITEST_REAL") != "1":
        pytest.skip("needs the real instances: set GOALY_EFFISQL_ITEST_REAL=1")
    password = load_credentials().grader_password
    connect = connector("127.0.0.1", port, password, user=GRADER_ROLE)
    session = PgSession.open(connect, db, timeout_s=timeout_s)
    if hard_cap_s is not None:
        session._hard_cap_s = hard_cap_s
    return session
