"""The v0 Effi-SQL policy (spec §5.4): one agent loop with execute_sql and submit.

The harness factory reads prompt.md and config.yaml, gives core the sampling, the budgets and the
context window, and calls :func:`build_policy` with the config's ``pack`` section and the prompt.
When a budget runs out before submit, the fallback submits ``base_sql`` unchanged, which the
grader credits with its identity τ (spec §5.4).

On top of prompt.md the pack holds one rule, TIME_GATE_RULE, which the task prompt carries: a
rewrite is handed in only once one EXPLAIN ANALYZE has shown it at least 1.2x faster than the base
plan's time, so no rewrite slower than identity is ever submitted.
"""

from collections.abc import Mapping
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from goaly.contracts.gen.task_pb2 import (
    EffiSqlPrediction,
    EffiSqlTaskInput,
    Prediction,
    TaskInput,
)
from goaly.contracts.interfaces import JsonValue, SqlEnvironment
from goaly.harness.core.agent import AgentContext, AgentHooks, run_agent
from goaly.harness.core.budgets import Budgets
from goaly.harness.core.tools import ToolRegistry, bind

from .tools import ExecuteSql, ExecuteSqlConfig, Submit


class PackConfig(BaseModel):
    """The pack's own settings: the ``pack`` section of config.yaml."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    execute_sql: ExecuteSqlConfig = ExecuteSqlConfig()
    # How long a submit call may take; it does no I/O.
    submit_timeout_s: float = Field(default=10.0, gt=0)


class EffiSqlPolicy:
    """Solves an Effi-SQL task with core's agent loop, falling back to ``base_sql``."""

    def __init__(self, config: PackConfig, system_prompt: str) -> None:
        self.config = config
        self.system_prompt = system_prompt

    async def solve(self, task: TaskInput, env: SqlEnvironment, ctx: AgentContext) -> Prediction:
        """Run the agent on ``task`` in ``env`` until it submits or a budget runs out."""
        if task.WhichOneof("benchmark") != "effisql":
            raise ValueError("the Effi-SQL policy solves Effi-SQL tasks only")
        sql_task = task.effisql
        hooks = AgentHooks(
            system_prompt=self.system_prompt,
            task_prompt=task_prompt(sql_task, self.config, ctx.budgets),
            tools=registry(self.config, ctx.budgets),
            fallback=lambda: fallback(sql_task),
        )
        return await run_agent(env, ctx, hooks)


def registry(config: PackConfig, budgets: Budgets) -> ToolRegistry[SqlEnvironment]:
    """The tools: execute_sql with the task's wall budget as its backstop, and submit."""
    execute_sql = ExecuteSql(config.execute_sql)
    return ToolRegistry(
        [
            bind(execute_sql, timeout_s=budgets.wall_clock_s),
            bind(Submit(), timeout_s=config.submit_timeout_s),
        ]
    )


def fallback(task: EffiSqlTaskInput) -> Prediction:
    """The answer when a budget runs out before submit: ``base_sql`` unchanged."""
    return Prediction(effisql=EffiSqlPrediction(sql=task.base_sql))


# The one rule this pack adds on top of prompt.md: hand in a rewrite only once it has been timed.
# Under a 1.2x speedup the credit is the base query's anyway, and a rewrite slower than the base
# earns less than submitting the base itself, so an unmeasured rewrite is never worth submitting.
# The rule is prose for the executor, so it reaches it through the task prompt.
TIME_GATE_RULE: Final = (
    "Time-gated submission: before submitting a rewrite, run EXPLAIN ANALYZE on it exactly once "
    "and read its total execution time. Compare it with the base plan's actual execution time "
    "given in the task. Submit the rewrite only if base_time / candidate_time >= 1.2; otherwise "
    "submit the base query unchanged. If the remaining budget cannot afford that one EXPLAIN "
    "ANALYZE, submit the base query unchanged. Never submit a rewrite you have not timed."
)


def task_prompt(task: EffiSqlTaskInput, config: PackConfig, budgets: Budgets) -> str:
    """The first user message: the query, its database and plan, the limits, and the rule.

    The rule comes last, and the opening messages never fold, so the model still reads it at the
    step where it submits.
    """
    minutes = budgets.wall_clock_s / 60
    timeout = config.execute_sql.statement_timeout_s
    return "\n\n".join(
        [
            f"Rewrite this query on the database {task.db} to run faster."
            f" Its difficulty is rated {task.difficulty or 'unknown'}.",
            f"The query:\n```sql\n{task.base_sql}\n```",
            f"The database schema:\n{task.schema}",
            f"The query's plan:\n```json\n{task.base_plan}\n```",
            f"Limits: execute_sql stops a query after {timeout:g} s. You have {budgets.steps}"
            f" steps and {minutes:g} minutes; if they run out before you submit, the query above"
            " is submitted unchanged.",
            TIME_GATE_RULE,
        ]
    )


def build_policy(config: Mapping[str, JsonValue], system_prompt: str) -> EffiSqlPolicy:
    """The policy, from config.yaml's ``pack`` section and prompt.md; ValidationError if invalid."""
    return EffiSqlPolicy(PackConfig.model_validate(config), system_prompt)
