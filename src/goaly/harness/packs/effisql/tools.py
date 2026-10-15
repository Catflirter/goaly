"""Effi-SQL's two v0 tools (spec §5.4): execute_sql, to run and time queries, and submit.

execute_sql runs one query in the task's read-only solving database and shows the model its first
rows, or the error, with how long it ran and, apart from that, how long it waited for a statement
slot. Its statement timeout is enforced by the environment, which cancels the statement from the
client. The policy binds it with the task's wall budget as core's executor backstop (600 s in v0),
which also bounds the queue wait, so a timed-out statement is an ordinary result.

Two guards keep one call from spending that whole budget. The runaway guard cancels a call that
has run longer than its share of the time the task has left, so a slow probe can no longer run
until the wall clock ends the task mid-statement; the environment stops the statement and opens a
new session for the next call, as a cancelled execute always does. The batch guard runs only the
first statement of a string that holds several, which the server would refuse anyway.
"""

import asyncio
import math
import re
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field

from goaly.contracts.gen.environment_pb2 import SqlResult
from goaly.contracts.gen.task_pb2 import EffiSqlPrediction, Prediction
from goaly.contracts.gen.trajectory_pb2 import ToolResult
from goaly.contracts.interfaces import SqlEnvironment

# PostgreSQL's message for a statement past its timeout, which the environment reports.
STATEMENT_TIMEOUT = "canceling statement due to statement timeout"

# What the model is told when its call held more than one statement.
BATCH_NOTE = "one SQL statement per call - the rest was not run"
# The runaway guard's cap when the tool cannot know the task's remaining time, as when it runs
# with no task to answer for: a unit test, say.
GUARD_FALLBACK_S = 30.0
# A dollar quote's opening tag: ``$$``, or ``$tag$``, which only the same tag closes.
DOLLAR_TAG = re.compile(r"\$[A-Za-z_][A-Za-z_0-9]*\$|\$\$")


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
    # The runaway guard: a call is cancelled once it has run longer than this fraction of the
    # time the task has left, and never sooner than guard_min_s. Half, and 10 s, leave the task
    # time to try another shape or submit, where a statement that runs to the wall budget would
    # end the task mid-exploration, with nothing submitted.
    guard_fraction: float = Field(default=0.5, gt=0, le=1)
    guard_min_s: float = Field(default=10.0, gt=0)


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
        "ran; and how long it first waited for a free statement slot. A query that runs past "
        "its share of the task's remaining time is cancelled, and a call with more than one "
        "statement runs only the first."
    )
    args_model = ExecuteSqlArgs

    def __init__(
        self, config: ExecuteSqlConfig, remaining_s: Callable[[], float] | None = None
    ) -> None:
        self.config = config
        # Says how many seconds of the task's wall budget are left, as the agent loop's clock
        # counts them. None when the tool has no task to answer for, and the fixed cap stands in.
        self.remaining_s = remaining_s

    async def run(self, args: ExecuteSqlArgs, env: SqlEnvironment) -> ToolResult:
        sql, batched = first_statement(args.sql)
        notes = [BATCH_NOTE] if batched else []
        guard = self.guard_limit_s()
        if guard is None:  # The statement timeout is the tighter bound already.
            result = await env.execute(sql, timeout_s=self.config.statement_timeout_s)
            return ToolResult(content=joined([render(result, self.config), *notes]))
        try:
            result = await asyncio.wait_for(
                env.execute(sql, timeout_s=self.config.statement_timeout_s), timeout=guard
            )
        except TimeoutError:
            return ToolResult(content=joined([cancelled(guard), *notes]))
        return ToolResult(content=joined([render(result, self.config), *notes]))

    def guard_limit_s(self) -> float | None:
        """How long one call may take before the tool cancels it, or None for no guard.

        The guard is there to stop a call before it spends the task's whole wall budget, so it
        has nothing to do when the statement timeout is already the tighter bound: the
        environment then cancels the statement itself, and the model sees the ordinary timeout
        result. A budget that is spent leaves the minimum, so the last call still gets its turn.
        """
        if self.remaining_s is None:
            share = GUARD_FALLBACK_S
        else:
            left = self.remaining_s()
            share = (
                GUARD_FALLBACK_S
                if not math.isfinite(left)
                else max(left * self.config.guard_fraction, self.config.guard_min_s)
            )
        return None if share >= self.config.statement_timeout_s else share


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


def cancelled(limit: float) -> str:
    """What the model sees when the tool cancelled its call at ``limit`` seconds."""
    return (
        f"cancelled after {limit:g} s - too slow; do not re-run this statement, it cannot be "
        "your submission; try a cheaper probe or submit your best candidate"
    )


def joined(parts: list[str]) -> str:
    """The parts as one text, a blank line between each two, empty ones left out."""
    return "\n\n".join(part for part in parts if part)


def first_statement(sql: str) -> tuple[str, bool]:
    """``sql``'s first statement, and whether the string held more than one.

    Only a semicolon that PostgreSQL would treat as ending a statement splits: one inside a
    quoted literal, a quoted identifier, a dollar-quoted string or a comment does not, and a
    trailing one, or one followed by nothing but whitespace and comments, ends the only
    statement there is, which then runs as the model wrote it.
    """
    statements = split_statements(sql)
    if len(statements) > 1:
        return statements[0], True
    return sql, False


def split_statements(sql: str) -> list[str]:
    """Split ``sql`` into its statements, each stripped, empty ones left out."""
    statements: list[str] = []
    start = 0  # Where the statement being read began.
    body = False  # Whether it has a character outside whitespace and comments.
    i = 0
    while i < len(sql):
        char = sql[i]
        if char in ("'", '"'):
            i = quoted_end(sql, i, char)
            body = True
        elif char == "$":
            end = dollar_quote_end(sql, i)
            if end is None:
                i += 1  # A dollar that opens no dollar quote, as in an operator.
            else:
                i = end
                body = True
        elif sql.startswith("--", i):
            i = line_comment_end(sql, i)
        elif sql.startswith("/*", i):
            i = block_comment_end(sql, i)
        elif char == ";":
            if body:
                statements.append(sql[start:i].strip())
            i += 1
            start = i
            body = False
        else:
            body = body or not char.isspace()
            i += 1
    if body:
        statements.append(sql[start:].strip())
    return statements


def quoted_end(sql: str, start: int, quote: str) -> int:
    """The index just past the quoted literal or identifier that begins at ``start``.

    A doubled quote is a quote inside, so the run ends only at a lone one; an unterminated run
    ends the text, and the caller reads the rest as one statement.
    """
    i = start + 1
    while i < len(sql):
        if sql[i] == quote:
            if sql[i + 1 : i + 2] == quote:
                i += 2
                continue
            return i + 1
        i += 1
    return len(sql)


def line_comment_end(sql: str, start: int) -> int:
    """The index of the newline that ends the ``--`` comment at ``start``, or of the text's end."""
    end = sql.find("\n", start)
    return len(sql) if end < 0 else end


def block_comment_end(sql: str, start: int) -> int:
    """The index just past the block comment at ``start``, nested as PostgreSQL nests them."""
    depth = 1
    i = start + 2
    while i < len(sql) and depth > 0:
        if sql.startswith("/*", i):
            depth += 1
            i += 2
        elif sql.startswith("*/", i):
            depth -= 1
            i += 2
        else:
            i += 1
    return i


def dollar_quote_end(sql: str, start: int) -> int | None:
    """The index just past the dollar-quoted string that begins at ``start``, or None.

    The string ends at the first tag like its opening one, and at the text's end if there is
    none; an unterminated dollar quote still ends the statement, as it does on the server.
    """
    found = DOLLAR_TAG.match(sql, start)
    if found is None:
        return None
    tag = found.group(0)
    end = sql.find(tag, found.end())
    return len(sql) if end < 0 else end + len(tag)


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
