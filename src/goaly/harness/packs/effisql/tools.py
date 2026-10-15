"""Effi-SQL's v0 tools (spec §5.4): execute_sql, to run and time queries; verify_equivalence,
to check a rewrite's rows against the original query's; and submit.

execute_sql runs one query in the task's read-only solving database and shows the model its first
rows, or the error, with how long it ran and, apart from that, how long it waited for a statement
slot. Its statement timeout is enforced by the environment, which cancels the statement from the
client. The policy binds it with the task's wall budget as core's executor backstop (600 s in v0),
which also bounds the queue wait, so a timed-out statement is an ordinary result.

verify_equivalence runs both queries once in one statement, the bidirectional EXCEPT ALL diff of
their results, and answers EQUIVALENT, the number of differing rows, or the database's error. It
reuses execute_sql's plumbing: the same statement timeout and the wall budget as its backstop.
"""

from pydantic import BaseModel, ConfigDict, Field

from goaly.contracts.gen.environment_pb2 import SqlResult
from goaly.contracts.gen.task_pb2 import EffiSqlPrediction, Prediction
from goaly.contracts.gen.trajectory_pb2 import ToolResult
from goaly.contracts.interfaces import SqlEnvironment

# PostgreSQL's message for a statement past its timeout, which the environment reports.
STATEMENT_TIMEOUT = "canceling statement due to statement timeout"


class ExecuteSqlConfig(BaseModel):
    """execute_sql's settings, from the ``pack.execute_sql`` section of config.yaml."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # The statement timeout, enforced by client-side cancellation in the environment. The
    # solving instances' capacity rests on it (spec §6.2), so a pack may lower it but never
    # raise it: the trusted environment runs each statement for at most its own timeout
    # (SqlEnvOptions.timeout_s, which the wiring sets to 120 s), whatever is asked here.
    statement_timeout_s: float = Field(default=120.0, gt=0, le=120)
    # How many rows the model sees, at most; the environment keeps no more than its own limit.
    rows_shown: int = Field(default=100, ge=0)
    # How many characters of each value the model sees: enough for a plan's EXPLAIN text.
    value_chars: int = Field(default=2000, ge=1)


class ExecuteSqlArgs(BaseModel):
    """Run one SQL query in the task's database, read-only, and see its result and timing."""

    sql: str = Field(min_length=1, pattern=r"\S", description="The query to run.")


class SubmitArgs(BaseModel):
    """Hand in the final query; this ends the task."""

    # Whitespace alone is no query: such a submit is refused, so the fallback still applies.
    sql: str = Field(
        pattern=r"\S",
        description="The rewritten query: one statement that begins with SELECT or WITH.",
    )


class ExecuteSql:
    """Runs a query with the statement timeout and reports rows, errors and timing."""

    name = "execute_sql"
    description = (
        "Run one SQL query in the task's PostgreSQL database, which is read-only. Returns the "
        "row count and the first rows, with long values cut, or the error; how long the query "
        "ran; and how long it first waited for a free statement slot."
    )
    args_model = ExecuteSqlArgs

    def __init__(self, config: ExecuteSqlConfig) -> None:
        self.config = config

    async def run(self, args: ExecuteSqlArgs, env: SqlEnvironment) -> ToolResult:
        result = await env.execute(args.sql, timeout_s=self.config.statement_timeout_s)
        return ToolResult(content=render(result, self.config))


class VerifyEquivalenceArgs(BaseModel):
    """Check a rewrite against the original query before you submit it.

    Call this once with your final rewrite before submitting. If the result is not EQUIVALENT,
    or the call errors or times out, submit the base query unchanged instead.
    """

    base_sql: str = Field(min_length=1, pattern=r"\S", description="The original query, unchanged.")
    candidate_sql: str = Field(
        min_length=1, pattern=r"\S", description="The rewritten query you plan to submit."
    )


class VerifyEquivalence:
    """Counts the rows on which a rewrite and the original query disagree, both directions."""

    name = "verify_equivalence"
    description = (
        "Check your final rewrite against the original query in one call: it counts the rows "
        "one result has that the other does not, in both directions (EXCEPT ALL), and answers "
        "EQUIVALENT when there are none, NOT EQUIVALENT with the number of differing rows, or "
        "the database's error, such as a column-count mismatch. It runs the queries, so it can "
        "take as long as the slower of the two."
    )
    args_model = VerifyEquivalenceArgs

    def __init__(self, config: ExecuteSqlConfig) -> None:
        self.config = config

    async def run(self, args: VerifyEquivalenceArgs, env: SqlEnvironment) -> ToolResult:
        verdict = await verify_equivalence(
            args.base_sql, args.candidate_sql, env, self.config.statement_timeout_s
        )
        return ToolResult(content=verdict)


async def verify_equivalence(
    base_sql: str, candidate_sql: str, env: SqlEnvironment, timeout_s: float
) -> str:
    """Diff the two queries' results in one statement and report whether they match.

    Call this once with your final rewrite before submitting. If the result is not EQUIVALENT,
    or the call errors or times out, submit the base query unchanged instead.
    """
    base = base_sql.strip().rstrip(";").rstrip()
    candidate = candidate_sql.strip().rstrip(";").rstrip()
    statement = (
        "SELECT count(*) AS diffs FROM ("  # noqa: S608  # The model's own queries, read-only.
        f"(SELECT * FROM ({candidate}) c EXCEPT ALL SELECT * FROM ({base}) b)"
        f" UNION ALL (SELECT * FROM ({base}) b EXCEPT ALL SELECT * FROM ({candidate}) c)"
        ") d"
    )
    result = await env.execute(statement, timeout_s=timeout_s)
    if result.error:
        return result.error
    values = iter(result.rows[0].items()) if result.rows else iter(())
    value = next(values, None)
    if not isinstance(value, float | str):
        return "the database returned no count, so equivalence is unknown"
    diffs = int(value)
    return "EQUIVALENT" if diffs == 0 else f"NOT EQUIVALENT: {diffs} differing rows"


class Submit:
    """Submits the final query, which ends the episode."""

    name = "submit"
    description = (
        "Submit your final query, which must return the same result as the original query on "
        "any data. This ends the task."
    )
    args_model = SubmitArgs

    async def run(self, args: SubmitArgs, env: SqlEnvironment) -> ToolResult:
        prediction = Prediction(effisql=EffiSqlPrediction(sql=args.sql))
        return ToolResult(content="Submitted.", submission=prediction)


def render(result: SqlResult, config: ExecuteSqlConfig) -> str:
    """The text the model sees for ``result``: its rows or its error, then its timing."""
    lines = []
    if result.error:
        lines.append(f"The query failed: {result.error}")
        if result.error == STATEMENT_TIMEOUT:
            limit = config.statement_timeout_s
            lines.append(f"It was stopped at the {limit:g} s statement timeout.")
    else:
        lines.extend(render_rows(result, config))
    lines.append(
        f"Execution time: {result.elapsed_s:.3f} s"
        f" (it waited {result.queue_wait_s:.3f} s for a statement slot before that)."
    )
    return "\n".join(lines)


def render_rows(result: SqlResult, config: ExecuteSqlConfig) -> list[str]:
    """The result's column names, row count and first rows, as lines."""
    rows = [
        " | ".join(cell(value, config.value_chars) for value in row.items())
        for row in result.rows[: config.rows_shown]
    ]
    count = f"{result.row_count} row" + ("" if result.row_count == 1 else "s")
    if not result.columns:  # Not a query: the count is of the rows the statement affected.
        return ["Columns: (none)", f"Rows: {count}."]
    if len(rows) < result.row_count:
        count += f", the first {len(rows)} shown"
    return [f"Columns: {', '.join(result.columns)}", f"Rows: {count}.", *rows]


def cell(value: object, limit: int) -> str:
    """One value as text: NULL, a number without a needless .0, or the string, cut to ``limit``.

    Rows hold JSON-like values: None for NULL, a bool, a float, or a string, which numeric and
    large integer values are, so they lose no precision.
    """
    if value is None:
        text = "NULL"
    elif isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, float):
        text = str(int(value)) if value.is_integer() else repr(value)
    else:
        text = str(value)
    return text if len(text) <= limit else text[:limit] + "…"
