"""Tests for goaly.harness.core.agent: the agent loop, its budgets and errors, and its trajectory.

FakeLLM (agent_fakes.py) scripts the executor model, with real Qwen replies among its items, and
the tools run in tests/fakes.py's FakeSqlEnvironment, so no test needs a model, a database or
the network.
"""

import asyncio
import hashlib
import inspect
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, SecretStr

from goaly.contracts.errors import EnvironmentFailure
from goaly.contracts.gen.environment_pb2 import SqlResult
from goaly.contracts.gen.llm_call_pb2 import CallPurpose, LlmCallRecord, LlmCallStatus, Usage
from goaly.contracts.gen.task_pb2 import EffiSqlPrediction, Prediction
from goaly.contracts.gen.trajectory_pb2 import (
    ChatMessage,
    ChatRole,
    ContextFold,
    Step,
    StopReason,
    ToolCall,
    ToolCallSource,
    ToolResult,
    Trajectory,
)
from goaly.contracts.interfaces import Environment, SqlEnvironment
from goaly.contracts.jsonl import append_jsonl, read_jsonl
from goaly.harness.core.agent import AgentContext, AgentHooks, run_agent
from goaly.harness.core.budgets import CALLS_PER_REPLY, Budgets, BudgetUse
from goaly.harness.core.context_window import ContextWindow
from goaly.harness.core.llm import BoundLlm, CallTags, LlmClient, LlmReply
from goaly.harness.core.llm_errors import ContextOverflow, ServerError
from goaly.harness.core.llm_models import EXECUTOR_MODEL, ModelParams
from goaly.harness.core.settings import Settings
from goaly.harness.core.tools import ToolRegistry, bind, environment_failure_in
from tests.contracts.test_records import prompt
from tests.fakes import FakeSqlEnvironment, SubmitTool
from tests.harness.core.agent_fakes import (
    DEFAULT_USAGE,
    HANG,
    QWEN_BASE_SQL,
    FakeLLM,
    call_record,
    qwen_reply,
    reply,
    tool_call,
)

pytestmark = pytest.mark.anyio

SYSTEM, USER = ChatRole.CHAT_ROLE_SYSTEM, ChatRole.CHAT_ROLE_USER
ASSISTANT, TOOL = ChatRole.CHAT_ROLE_ASSISTANT, ChatRole.CHAT_ROLE_TOOL
CONTENT_XML = ToolCallSource.TOOL_CALL_SOURCE_CONTENT_XML
REASONING_XML = ToolCallSource.TOOL_CALL_SOURCE_REASONING_XML

SUBMITTED = StopReason.STOP_REASON_SUBMITTED
STEPS = StopReason.STOP_REASON_STEP_BUDGET_EXHAUSTED
TIME = StopReason.STOP_REASON_TIME_BUDGET_EXHAUSTED
TOKENS = StopReason.STOP_REASON_TOKEN_BUDGET_EXHAUSTED
COST = StopReason.STOP_REASON_COST_BUDGET_EXHAUSTED
ERRORS = StopReason.STOP_REASON_CONSECUTIVE_ERRORS
CRASHED = StopReason.STOP_REASON_CRASHED
CANCELLED = StopReason.STOP_REASON_CANCELLED

EXPLAIN = f"EXPLAIN {QWEN_BASE_SQL}"
FAST = "SELECT count(*) FROM orders JOIN customer ON c_custkey = o_custkey WHERE c_nationkey = 7"
SLOW = "SELECT pg_sleep(60)"
PAUSE = "SELECT pg_sleep(0.6)"
DEAD = "SELECT 'the connection dies'"
SYSTEM_PROMPT = "You rewrite slow PostgreSQL queries into equivalent, faster ones."
TASK_PROMPT = f"Rewrite this query:\n{QWEN_BASE_SQL}"
FALLBACK = Prediction(effisql=EffiSqlPrediction(sql=QWEN_BASE_SQL))
SUBMISSION = Prediction(effisql=EffiSqlPrediction(sql=FAST))
PARAMS = ModelParams(max_tokens=4096)


class ExecuteSqlArgs(BaseModel):
    """execute_sql's arguments."""

    sql: str
    limit: int = 20


class ExecuteSql:
    """Effi-SQL's execute_sql, as v0 has it: it runs one statement, and a failed one is a result."""

    name = "execute_sql"
    description = "Run one SQL statement and return the rows or the error."
    args_model = ExecuteSqlArgs

    async def run(self, args: ExecuteSqlArgs, env: SqlEnvironment) -> ToolResult:
        result = await env.execute(args.sql, timeout_s=120.0)
        return ToolResult(content=result.error or f"rows: {result.row_count}")


class NoArgs(BaseModel):
    """No arguments."""


class FragileTool:
    """A tool with a bug: it raises, which the executor turns into an error result."""

    name = "fragile"
    description = "Fail."
    args_model = NoArgs

    async def run(self, args: NoArgs, env: Environment) -> ToolResult:
        raise RuntimeError("the tool broke")


class PairArgs(BaseModel):
    """PairTool's arguments."""

    first: str
    second: str


class PairTool:
    """Runs two statements at once, in a TaskGroup."""

    name = "pair"
    description = "Run two SQL statements at once."
    args_model = PairArgs

    async def run(self, args: PairArgs, env: SqlEnvironment) -> ToolResult:
        async with asyncio.TaskGroup() as group:
            for sql in (args.first, args.second):
                group.create_task(env.execute(sql, timeout_s=120.0))
        return ToolResult(content="Both ran.")


def big(n: int) -> str:
    """A 4,000-character output, a different one for each ``n``."""
    return f"{n}".ljust(4000, "x")


class PageArgs(BaseModel):
    """PageTool's arguments."""

    n: int


class PageTool:
    """A short call with a long output, as reading a big plan is: page ``n`` is big(n)."""

    name = "page"
    description = "Read page n."
    args_model = PageArgs

    async def run(self, args: PageArgs, env: Environment) -> ToolResult:
        return ToolResult(content=big(args.n))


def environment() -> FakeSqlEnvironment:
    return FakeSqlEnvironment(
        {
            EXPLAIN: SqlResult(row_count=12),
            FAST: SqlResult(row_count=1),
            SLOW: SqlResult(row_count=1, elapsed_s=60.0),
            PAUSE: SqlResult(row_count=1, elapsed_s=0.6),
            DEAD: EnvironmentFailure("the database connection died"),
        }
    )


def registry() -> ToolRegistry[SqlEnvironment]:
    return ToolRegistry[SqlEnvironment](
        [
            bind(ExecuteSql()),
            bind(SubmitTool()),
            bind(FragileTool()),
            bind(PairTool()),
            bind(PageTool()),
        ]
    )


TOOL_NAMES = "execute_sql, submit, fragile, pair, page"
SCHEMAS = registry().json_schemas()


def give_fallback() -> Prediction:
    return FALLBACK


def hooks(
    *,
    on_tool_result: Callable[[ToolCall, ToolResult], str | None] | None = None,
    on_step: Callable[[Step, BudgetUse], str | None] | None = None,
    fallback: Callable[[], Prediction] = give_fallback,
) -> AgentHooks[SqlEnvironment]:
    return AgentHooks(
        system_prompt=SYSTEM_PROMPT,
        task_prompt=TASK_PROMPT,
        tools=registry(),
        fallback=fallback,
        on_tool_result=on_tool_result,
        on_step=on_step,
    )


def context(llm: BoundLlm, *, window: ContextWindow | None = None, **budgets: Any) -> AgentContext:
    limits = Budgets.model_validate({"steps": 30, "wall_clock_s": 600.0} | budgets)
    return AgentContext(llm, budgets=limits, params=PARAMS, window=window)


def execute_sql(sql: str) -> ToolCall:
    return tool_call("execute_sql", {"sql": sql})


def submit(sql: str = FAST) -> ToolCall:
    return tool_call("submit", {"sql": sql})


def system(text: str) -> ChatMessage:
    return ChatMessage(role=SYSTEM, content=text)


def user(text: str) -> ChatMessage:
    return ChatMessage(role=USER, content=text)


def tool_message(result: ToolResult) -> ChatMessage:
    return ChatMessage(role=TOOL, content=result.content, tool_call_id=result.tool_call_id)


def reference_prompts(trajectory: Trajectory) -> list[list[ChatMessage]]:
    """Each step's prompt, as the contract's reference rebuild (test_records.prompt) makes it."""
    return [prompt(trajectory, index) for index in range(len(trajectory.steps))]


def function_block(name: str, sql: str) -> str:
    """A Qwen3-Coder XML call to ``name``, bare."""
    return f"<function={name}>\n<parameter=sql>\n{sql}\n</parameter>\n</function>"


def tool_call_block(name: str, sql: str) -> str:
    """A Qwen3-Coder XML call to ``name``, wrapped in its <tool_call> tags."""
    return f"<tool_call>\n{function_block(name, sql)}\n</tool_call>"


def sent(llm: FakeLLM) -> list[list[ChatMessage]]:
    return [list(request.messages) for request in llm.requests]


def injected(trajectory: Trajectory) -> list[list[str]]:
    return [[message.content for message in step.injected] for step in trajectory.steps]


# A normal run, and what the model and the trajectory get.


async def test_a_submission_ends_the_loop_with_its_prediction() -> None:
    llm = FakeLLM([reply(execute_sql(FAST)), reply(submit(FAST))])
    ctx = context(llm)

    prediction = await run_agent(environment(), ctx, hooks())

    assert prediction == SUBMISSION
    assert not prediction.fallback
    trajectory = ctx.trajectory()
    assert trajectory.stop_reason == SUBMITTED
    assert trajectory.error_message == ""
    assert [chat.label for chat in trajectory.chats] == ["main"]
    assert list(trajectory.chats[0].initial_messages) == [system(SYSTEM_PROMPT), user(TASK_PROMPT)]
    first, second = trajectory.steps
    assert [result.content for result in first.tool_results] == ["rows: 1"]
    assert second.tool_results[0].submission == SUBMISSION
    assert trajectory.usage == Usage(input_tokens=2000, output_tokens=400, cost_usd=0.002)
    assert sent(llm)[1] == [
        system(SYSTEM_PROMPT),
        user(TASK_PROMPT),
        first.assistant,
        tool_message(first.tool_results[0]),
    ]


async def test_each_call_offers_the_tools_and_sends_the_packs_parameters() -> None:
    llm = FakeLLM([reply(submit())])

    await run_agent(environment(), context(llm), hooks())

    [request] = llm.requests
    assert list(request.tools) == SCHEMAS
    assert request.params == PARAMS


async def test_a_step_records_its_call_with_its_timing_and_usage() -> None:
    usage = Usage(input_tokens=5000, output_tokens=700, reasoning_tokens=600, cost_usd=0.004)
    scripted = reply(submit(), usage=usage, latency_s=3.5)
    ctx = context(FakeLLM([scripted]))

    await run_agent(environment(), ctx, hooks())

    [step] = ctx.trajectory().steps
    assert (step.index, step.chat) == (0, 0)
    assert step.call_id == scripted.record.call_id
    assert step.started_at == scripted.record.started_at
    assert step.latency_s == 3.5  # The record's: the client's waits included.
    assert step.usage == usage
    assert step.assistant == scripted.message


async def test_every_prompt_can_be_rebuilt_from_the_trajectory() -> None:
    """The trajectory schema: each step holds only what it added, and the prompts follow."""
    script = [
        qwen_reply("standard_call"),
        reply(content="The query is already fast."),
        qwen_reply("xml_in_content"),
        qwen_reply("call_in_reasoning"),  # Its call repeats the content's: the first failure.
        qwen_reply("truncated_standard_call"),
        reply(tool_call("explain_query", {"sql": QWEN_BASE_SQL})),  # The third: the reminder.
        reply(content="Done."),  # A nudge.
        reply(submit()),
    ]
    llm = FakeLLM(script)
    ctx = context(llm)

    await run_agent(
        environment(),
        ctx,
        hooks(
            on_tool_result=lambda call, result: f"Checked {call.name}.",
            on_step=lambda step, used: f"{30 - used.steps} steps left.",
        ),
    )

    trajectory = ctx.trajectory()
    assert trajectory.stop_reason == SUBMITTED
    assert len(trajectory.steps) == len(script)
    assert trajectory.steps[5].injected[0].content.startswith("You have made 3 errors in a row.")
    assert reference_prompts(trajectory) == sent(llm)


BARE_CALL = function_block("execute_sql", "SELECT 2")
WRAPPED_CALL = tool_call_block("execute_sql", "SELECT 1")


def markup_script() -> list[LlmReply]:
    """Replies with both kinds of call markup, in content and reasoning, with calls from each
    source; then six plain replies, which push the first two out of the 8-turn replay window."""
    return [
        # Calls from the content's <tool_call> block; the reasoning drafts it too.
        reply(
            content=f"Run it.\n{WRAPPED_CALL}\nNot this: {BARE_CALL}",
            reasoning=f"Draft: {WRAPPED_CALL}\n{BARE_CALL}",
        ),
        # A call from a bare block, in a content with no <tool_call>; a bare one in reasoning.
        reply(content=f"Then: {function_block('execute_sql', 'SELECT 3')}", reasoning=BARE_CALL),
        # A call from the reasoning's <tool_call> block.
        reply(reasoning=f"Thinking.\n{tool_call_block('execute_sql', 'SELECT 4')}"),
        # A standard call, with markup in both fields all the same.
        reply(
            execute_sql("SELECT 5"), content=f"Also {WRAPPED_CALL}", reasoning=f"Also {BARE_CALL}"
        ),
        *(reply(execute_sql(f"SELECT {n}"), reasoning=f"Plain {n}.") for n in range(6, 12)),
        reply(submit()),
    ]


def replies_sent(llm: FakeLLM, call: int) -> list[tuple[str, str]]:
    """The content and reasoning of each reply that call number ``call`` was sent."""
    return [(m.content, m.reasoning_content) for m in sent(llm)[call] if m.role == ASSISTANT]


async def test_replies_go_out_without_call_markup_and_older_ones_without_reasoning() -> None:
    """Spec §5.1: an executor chat replays no call markup, and reasoning only in its window."""
    llm = FakeLLM(markup_script())
    ctx = context(llm)

    await run_agent(environment(), ctx, hooks())

    trajectory = ctx.trajectory()
    assert [len(step.assistant.tool_calls) for step in trajectory.steps] == [1] * 11
    assert [chat.keep_turns for chat in trajectory.chats] == [8]
    stripped = [
        # Every <tool_call> block goes, then every bare <function=...> block, in either field,
        # whatever field the calls came from: a standard call's reply loses its markup too.
        ("Run it.\n\nNot this: ", "Draft: \n"),
        ("Then: ", ""),
        ("", "Thinking.\n"),
        ("Also ", "Also "),
    ]
    assert replies_sent(llm, 4) == stripped  # All four within the window.
    assert (
        replies_sent(llm, 10)
        == [
            ("Run it.\n\nNot this: ", ""),  # Out of the window: no reasoning at all.
            ("Then: ", ""),
            *stripped[2:],
            *(("", f"Plain {n}.") for n in range(6, 12)),
        ]
    )
    assert trajectory.steps[3].assistant.content == f"Also {WRAPPED_CALL}"  # Kept as it came.


@pytest.mark.parametrize("keep_turns", [0, 2, 8])
async def test_the_loop_sends_the_prompts_the_contracts_reference_rebuilds(keep_turns: int) -> None:
    llm = FakeLLM(markup_script())
    ctx = context(llm, window=ContextWindow(keep_turns=keep_turns))

    await run_agent(environment(), ctx, hooks())

    trajectory = ctx.trajectory()
    assert [chat.keep_turns for chat in trajectory.chats] == [keep_turns]
    assert reference_prompts(trajectory) == sent(llm)


async def test_a_window_of_0_replays_the_chat_unchanged() -> None:
    llm = FakeLLM(markup_script())
    ctx = context(llm, window=ContextWindow(keep_turns=0))

    await run_agent(environment(), ctx, hooks())

    as_written = [
        (s.assistant.content, s.assistant.reasoning_content) for s in ctx.trajectory().steps
    ]
    assert replies_sent(llm, 10) == as_written[:10]  # Markup and reasoning, all as they came.


async def test_a_trajectory_round_trips_through_jsonl(tmp_path: Path) -> None:
    ctx = context(FakeLLM([qwen_reply("xml_in_content"), reply(submit())]))
    await run_agent(environment(), ctx, hooks())
    path = tmp_path / "trajectories.jsonl"

    append_jsonl(path, ctx.trajectory())

    [read] = read_jsonl(path, Trajectory)
    assert read == ctx.trajectory()


# Qwen's replies: XML calls, repaired arguments and made-up IDs.


async def test_a_call_in_the_content_runs_and_is_replayed_without_its_markup() -> None:
    xml = qwen_reply("xml_in_content")
    llm = FakeLLM([xml, reply(submit())])
    ctx = context(llm)

    await run_agent(environment(), ctx, hooks())

    first = ctx.trajectory().steps[0]
    [call] = first.assistant.tool_calls
    assert (call.name, call.source) == ("execute_sql", CONTENT_XML)
    assert first.assistant.content == xml.message.content  # The trajectory keeps it as written.
    assert [result.content for result in first.tool_results] == ["rows: 12"]
    replay = sent(llm)[1][2]
    assert "<tool_call>" not in replay.content
    assert replay.content.startswith("Let me first examine the query plan")
    assert list(replay.tool_calls) == [call]


async def test_a_call_drafted_in_the_reasoning_is_taken_out_of_the_replay_too() -> None:
    def xml(name: str, sql: str) -> str:
        return (
            f"<tool_call>\n<function={name}>\n<parameter=sql>\n{sql}\n</parameter>\n"
            "</function>\n</tool_call>"
        )

    drafted = reply(
        content=f"Running it.\n{xml('execute_sql', FAST)}",
        reasoning=f"Submit it now?\n{xml('submit', FAST)}\nNo: run it first.",
    )
    llm = FakeLLM([drafted, reply(submit())])
    ctx = context(llm)

    await run_agent(environment(), ctx, hooks())

    first = ctx.trajectory().steps[0]
    assert [(call.name, call.source) for call in first.assistant.tool_calls] == [
        ("execute_sql", CONTENT_XML)  # The content's call wins; the drafted submit never ran.
    ]
    assert first.assistant.reasoning_content == drafted.message.reasoning_content
    replay = sent(llm)[1][2]
    assert replay.content == "Running it.\n"
    assert replay.reasoning_content == "Submit it now?\n\nNo: run it first."


async def test_a_call_in_the_reasoning_runs_and_is_replayed_without_its_markup() -> None:
    xml = qwen_reply("call_in_reasoning")
    llm = FakeLLM([xml, reply(submit())])
    ctx = context(llm)

    await run_agent(environment(), ctx, hooks())

    first = ctx.trajectory().steps[0]
    assert first.assistant.tool_calls[0].source == REASONING_XML
    assert first.assistant.reasoning_content == xml.message.reasoning_content
    assert [result.content for result in first.tool_results] == ["rows: 12"]
    replay = sent(llm)[1][2]
    assert "<tool_call>" not in replay.reasoning_content
    assert replay.reasoning_content.rstrip().endswith("Let's do that.")


async def test_a_call_without_an_id_gets_one_that_replays_identically() -> None:
    arguments = json.dumps({"sql": FAST})
    llm = FakeLLM([reply(tool_call("execute_sql", arguments, call_id="")), reply(submit())])
    ctx = context(llm)

    await run_agent(environment(), ctx, hooks())

    made_up = "call_" + hashlib.sha256(f"0\nexecute_sql\n{arguments}".encode()).hexdigest()[:24]
    first = ctx.trajectory().steps[0]
    assert first.assistant.tool_calls[0].id == made_up
    assert first.tool_results[0].tool_call_id == made_up
    assert sent(llm)[1][2].tool_calls[0].id == made_up


async def test_malformed_arguments_run_once_repaired() -> None:
    llm = FakeLLM([reply(tool_call("execute_sql", '{"sql": "' + FAST + '"')), reply(submit())])
    ctx = context(llm)

    await run_agent(environment(), ctx, hooks())

    first = ctx.trajectory().steps[0]
    assert json.loads(first.assistant.tool_calls[0].repaired_arguments_json) == {"sql": FAST}
    assert [result.content for result in first.tool_results] == ["rows: 1"]


# Budgets: each one ends the loop with the fallback.


async def test_the_step_budget_ends_in_the_fallback() -> None:
    llm = FakeLLM([reply(execute_sql(FAST)), reply(execute_sql(FAST))])
    ctx = context(llm, steps=2)

    prediction = await run_agent(
        environment(), ctx, hooks(on_step=lambda step, used: "Keep going.")
    )

    assert prediction.fallback
    assert prediction.effisql == FALLBACK.effisql
    assert not FALLBACK.fallback  # The loop marks a copy.
    assert len(llm.requests) == 2
    trajectory = ctx.trajectory()
    assert trajectory.stop_reason == STEPS
    assert injected(trajectory) == [["Keep going."], []]  # No call followed the last step.


async def test_the_wall_clock_cancels_a_call_and_ends_in_the_fallback() -> None:
    llm = FakeLLM([HANG])
    ctx = context(llm, wall_clock_s=0.05)

    async with asyncio.timeout(10):  # A guard: the budget must end the call long before.
        prediction = await run_agent(environment(), ctx, hooks())

    assert prediction.fallback
    trajectory = ctx.trajectory()
    assert trajectory.stop_reason == TIME
    assert not trajectory.steps  # The cancelled call got no reply.


async def test_the_wall_clock_stops_a_running_tool() -> None:
    env = environment()
    ctx = context(FakeLLM([reply(execute_sql(SLOW), execute_sql(FAST))]), wall_clock_s=0.2)

    async with asyncio.timeout(10):
        prediction = await run_agent(env, ctx, hooks())

    assert prediction.fallback
    assert env.stopped == [SLOW]
    assert [sql for sql, _ in env.executed] == [SLOW]
    [step] = ctx.trajectory().steps
    assert [(result.content, result.is_error) for result in step.tool_results] == [
        ("Error: no result: the task's time ran out.", True)
    ] * 2  # The call that was stopped, and the one after it.
    assert ctx.trajectory().stop_reason == TIME


async def test_the_token_budget_ends_in_the_fallback() -> None:
    llm = FakeLLM([reply(execute_sql(FAST)) for _ in range(3)])  # 1,200 tokens each.
    ctx = context(llm, tokens=3000)

    prediction = await run_agent(environment(), ctx, hooks())

    assert prediction.fallback
    assert len(llm.requests) == 3  # 2,400 tokens leave room for a third call; 3,600 do not.
    assert ctx.trajectory().stop_reason == TOKENS


async def test_the_cost_budget_ends_in_the_fallback() -> None:
    llm = FakeLLM([reply(execute_sql(FAST)) for _ in range(2)])  # $0.001 each.
    ctx = context(llm, cost_usd=0.002)

    prediction = await run_agent(environment(), ctx, hooks())

    assert prediction.fallback
    assert len(llm.requests) == 2
    assert ctx.trajectory().stop_reason == COST


async def test_a_context_overflow_ends_in_the_fallback() -> None:
    overflow = ContextOverflow("qwen3.6-35b-a3b: HTTP 400: maximum context length is 131072")
    ctx = context(FakeLLM([reply(execute_sql(FAST)), overflow]))

    prediction = await run_agent(environment(), ctx, hooks())

    assert prediction.fallback
    trajectory = ctx.trajectory()
    assert trajectory.stop_reason == TOKENS
    assert trajectory.error_message == f"ContextOverflow: {overflow}"


async def test_a_failed_calls_usage_is_counted() -> None:
    record = call_record(Usage(input_tokens=0, output_tokens=0, cost_usd=0.0005))
    overflow = ContextOverflow("too long", record=record)
    ctx = context(FakeLLM([reply(execute_sql(FAST)), overflow]))

    await run_agent(environment(), ctx, hooks())

    assert ctx.trajectory().usage.cost_usd == pytest.approx(DEFAULT_USAGE.cost_usd + 0.0005)


# Errors: none raises, each counts, and the streak reminds, then stops.


async def test_errors_in_a_row_bring_one_reminder_then_the_fallback() -> None:
    llm = FakeLLM([reply(tool_call("explain_query")) for _ in range(5)])
    ctx = context(llm)

    prediction = await run_agent(environment(), ctx, hooks())

    assert prediction.fallback
    assert len(llm.requests) == 5
    trajectory = ctx.trajectory()
    assert trajectory.stop_reason == ERRORS
    reminder = (
        "You have made 3 errors in a row. Call the tools exactly as these schemas define "
        f"them:\n{json.dumps(SCHEMAS, ensure_ascii=False, indent=2)}"
    )
    assert injected(trajectory) == [[], [], [reminder], [], []]
    assert sent(llm)[3][-1] == user(reminder)  # Two replies came after it, before the stop.


async def test_every_kind_of_error_counts_toward_the_streak() -> None:
    llm = FakeLLM(
        [
            reply(tool_call("explain_query")),  # A tool the registry does not hold.
            reply(tool_call("execute_sql", {"query": FAST})),  # Invalid arguments.
            reply(tool_call("execute_sql", '{"sql": ')),  # JSON that cannot be repaired.
            reply(tool_call("fragile")),  # A tool that raised.
            reply(content="The query is already as fast as it gets."),  # No call at all.
        ]
    )
    ctx = context(llm)

    prediction = await run_agent(environment(), ctx, hooks())

    assert prediction.fallback
    assert len(llm.requests) == 5
    assert ctx.trajectory().stop_reason == ERRORS


async def test_a_reply_whose_parallel_calls_all_failed_is_one_error() -> None:
    parallel = reply(*(tool_call("explain_query") for _ in range(5)))
    error = reply(tool_call("explain_query"))
    llm = FakeLLM([parallel, error, error, reply(submit())])
    ctx = context(llm)

    prediction = await run_agent(environment(), ctx, hooks())

    assert prediction == SUBMISSION  # Five failed calls in one reply did not stop the chat:
    trajectory = ctx.trajectory()
    assert [result.is_error for result in trajectory.steps[0].tool_results] == [True] * 5
    # they made one failed reply, so the streak reached 3, and the reminder, at the third.
    assert [bool(texts) for texts in injected(trajectory)] == [False, False, True, False]


async def test_a_reply_with_one_call_that_worked_ends_the_streak() -> None:
    error = reply(tool_call("explain_query"))
    mixed = reply(execute_sql(FAST), tool_call("explain_query"))
    llm = FakeLLM([error, error, mixed, error, error, reply(submit())])
    ctx = context(llm)

    prediction = await run_agent(environment(), ctx, hooks())

    assert prediction == SUBMISSION
    assert not any(injected(ctx.trajectory()))  # The streak never reached 3.


async def test_cut_off_replies_count_once_each_however_many_calls_they_had() -> None:
    cut = reply(execute_sql(FAST), execute_sql(EXPLAIN), finish_reason="length")
    llm = FakeLLM([cut] * 5)
    ctx = context(llm)

    prediction = await run_agent(environment(), ctx, hooks())

    assert prediction.fallback
    assert len(llm.requests) == 5  # Ten failed calls, but five failed replies.
    assert ctx.trajectory().stop_reason == ERRORS


async def test_a_call_that_works_ends_the_streak() -> None:
    errors = [reply(tool_call("explain_query")) for _ in range(4)]
    llm = FakeLLM([*errors, reply(execute_sql(FAST)), *errors, reply(submit())])
    ctx = context(llm)

    prediction = await run_agent(environment(), ctx, hooks())

    assert prediction == SUBMISSION
    reminders = [texts for texts in injected(ctx.trajectory()) if texts]
    assert len(reminders) == 2  # One for each streak of 3 or more.


async def test_a_reply_that_calls_no_tool_is_answered_with_a_nudge() -> None:
    llm = FakeLLM([reply(content="It is fast now."), reply(submit())])
    ctx = context(llm)

    prediction = await run_agent(environment(), ctx, hooks())

    assert prediction == SUBMISSION
    nudge = (
        f"Error: your reply called no tool. Each reply must call one of the tools: {TOOL_NAMES}."
    )
    assert injected(ctx.trajectory()) == [[nudge], []]
    assert sent(llm)[1][-1] == user(nudge)


async def test_a_reply_cut_off_at_the_output_limit_runs_none_of_its_calls() -> None:
    env = environment()
    cut = reply(submit(QWEN_BASE_SQL), execute_sql(FAST), finish_reason="length")
    llm = FakeLLM([cut, reply(submit())])
    ctx = context(llm)

    prediction = await run_agent(env, ctx, hooks())

    assert prediction == SUBMISSION  # The cut-off submit never ran.
    assert env.executed == []
    first = ctx.trajectory().steps[0]
    assert [result.tool_call_id for result in first.tool_results] == [
        call.id for call in cut.message.tool_calls
    ]
    for result in first.tool_results:
        assert result.is_error
        assert result.content.startswith("Error: your reply was cut off at the output limit")
        assert "more concisely" in result.content


async def test_calls_past_the_cap_of_a_reply_are_answered_but_not_run() -> None:
    env = environment()
    queries = [f"SELECT {n}" for n in range(CALLS_PER_REPLY)]
    calls = [*map(execute_sql, queries), submit(QWEN_BASE_SQL), execute_sql(FAST)]
    llm = FakeLLM([reply(*calls), reply(submit())])
    ctx = context(llm)

    prediction = await run_agent(env, ctx, hooks())

    assert prediction == SUBMISSION  # The capped submit never ran.
    assert [sql for sql, _ in env.executed] == queries
    first = ctx.trajectory().steps[0]
    assert [result.tool_call_id for result in first.tool_results] == [call.id for call in calls]
    assert [result.is_error for result in first.tool_results] == [False] * 16 + [True] * 2
    for result in first.tool_results[16:]:
        assert result.content == (
            "Error: a reply may make at most 16 tool calls, so this call was not run. "
            "Make fewer calls in each reply."
        )
    assert not first.injected  # Calls of the reply worked, so it is no error.


async def test_a_submission_drafted_in_the_reasoning_does_not_submit() -> None:
    drafted = reply(
        reasoning=(
            "The join is faster, so I will submit it.\n<tool_call>\n<function=submit>\n"
            f"<parameter=sql>\n{QWEN_BASE_SQL}\n</parameter>\n</function>\n</tool_call>"
        ),
        finish_reason="stop",
    )
    llm = FakeLLM([drafted, reply(submit())])
    ctx = context(llm)

    prediction = await run_agent(environment(), ctx, hooks())

    assert prediction == SUBMISSION  # The submit in the reply, not the drafted one.
    [result] = ctx.trajectory().steps[0].tool_results
    assert not result.HasField("submission")
    assert result.is_error
    assert result.content == (
        "Error: call submit in your reply, not in your reasoning. A call in your reasoning "
        "cannot submit, so nothing was submitted."
    )


SUBMIT_XML = f"<function=submit>\n<parameter=sql>\n{QWEN_BASE_SQL}\n</parameter>\n</function>"


@pytest.mark.parametrize(
    ("field", "text"),
    [
        ("content", f"Done.\n<tool_call>\n{SUBMIT_XML}"),  # Never closed.
        ("reasoning", f"Submit it.\n<tool_call>\n{SUBMIT_XML}"),  # Never closed.
        ("reasoning", f"Submit it.\n{SUBMIT_XML}"),  # No <tool_call> around it.
    ],
    ids=["unclosed-in-content", "unclosed-in-reasoning", "bare-in-reasoning"],
)
async def test_a_submission_that_is_no_call_never_submits(field: str, text: str) -> None:
    unfinished = (
        reply(content=text, finish_reason="stop")
        if field == "content"
        else reply(reasoning=text, finish_reason="stop")
    )
    llm = FakeLLM([unfinished, reply(submit())])
    ctx = context(llm)

    prediction = await run_agent(environment(), ctx, hooks())

    assert prediction == SUBMISSION  # The submit of the next reply.
    first = ctx.trajectory().steps[0]
    assert not first.assistant.tool_calls  # The parser read no call there.
    assert not first.tool_results
    assert first.injected[0].content.startswith("Error: your reply called no tool.")


async def test_calls_after_a_submission_are_answered_but_not_run() -> None:
    env = environment()
    ctx = context(FakeLLM([reply(submit(), execute_sql(FAST))]))

    prediction = await run_agent(env, ctx, hooks())

    assert prediction == SUBMISSION
    assert env.executed == []
    later = ctx.trajectory().steps[0].tool_results[1]
    assert (later.content, later.is_error) == (
        "Error: not run: an earlier call in this reply submitted, which ended the episode.",
        True,
    )


async def test_a_reply_cut_off_before_any_call_is_nudged_to_think_less() -> None:
    thinking = reply(reasoning="Let me weigh every plan...", finish_reason="length")
    llm = FakeLLM([thinking, reply(submit())])
    ctx = context(llm)

    prediction = await run_agent(environment(), ctx, hooks())

    assert prediction == SUBMISSION
    nudge = (
        "Error: your reply was cut off at the output limit before it called a tool. "
        f"Think briefly, then call one of the tools: {TOOL_NAMES}."
    )
    assert injected(ctx.trajectory()) == [[nudge], []]
    assert sent(llm)[1][-1] == user(nudge)


# The pack's hooks.


async def test_tool_result_guidance_is_appended_to_what_the_model_sees() -> None:
    seen: list[tuple[str, str]] = []

    def guide(call: ToolCall, result: ToolResult) -> str | None:
        seen.append((call.name, result.content))
        return "Hint: compare the plans." if call.name == "execute_sql" else None

    llm = FakeLLM([reply(execute_sql(FAST)), reply(submit())])
    ctx = context(llm)

    await run_agent(environment(), ctx, hooks(on_tool_result=guide))

    assert seen == [("execute_sql", "rows: 1"), ("submit", "Submitted.")]
    first, second = ctx.trajectory().steps
    assert first.tool_results[0].content == "rows: 1\n\nHint: compare the plans."
    assert second.tool_results[0].content == "Submitted."
    assert sent(llm)[1][-1].content == "rows: 1\n\nHint: compare the plans."


async def test_step_guidance_is_sent_between_steps() -> None:
    seen: list[tuple[int, BudgetUse]] = []

    def guide(step: Step, used: BudgetUse) -> str | None:
        seen.append((step.index, used))
        return f"{30 - used.steps} steps left." if used.steps < 2 else ""

    llm = FakeLLM([reply(execute_sql(FAST)), reply(execute_sql(FAST)), reply(submit())])
    ctx = context(llm)

    await run_agent(environment(), ctx, hooks(on_step=guide))

    assert [index for index, _ in seen] == [0, 1]  # Not after the last step: none follows it.
    first_use = seen[0][1]
    assert (first_use.steps, first_use.tokens, first_use.cost_usd) == (1, 1200, 0.001)
    assert first_use.elapsed_s >= 0
    assert injected(ctx.trajectory()) == [["29 steps left."], [], []]  # "" is no guidance.
    assert sent(llm)[1][-1] == user("29 steps left.")


async def test_the_hooks_get_copies() -> None:
    def spoil_result(call: ToolCall, result: ToolResult) -> None:
        call.name = "spoiled"
        result.content = "spoiled"

    def spoil_step(step: Step, used: BudgetUse) -> None:
        step.assistant.content = "spoiled"
        del step.tool_results[:]

    ctx = context(FakeLLM([reply(execute_sql(FAST)), reply(submit())]))

    await run_agent(environment(), ctx, hooks(on_tool_result=spoil_result, on_step=spoil_step))

    first = ctx.trajectory().steps[0]
    assert first.assistant.tool_calls[0].name == "execute_sql"
    assert first.assistant.content == ""
    assert [result.content for result in first.tool_results] == ["rows: 1"]


# What propagates: environment failures, LLM failures, the pack's bugs, and cancellation.


async def test_an_environment_failure_propagates() -> None:
    ctx = context(FakeLLM([reply(execute_sql(DEAD), execute_sql(FAST))]))

    with pytest.raises(EnvironmentFailure, match="connection died"):
        await run_agent(environment(), ctx, hooks())

    trajectory = ctx.trajectory()
    assert trajectory.stop_reason == CRASHED
    assert trajectory.error_message == "EnvironmentFailure: the database connection died"
    [step] = trajectory.steps  # The step that hit it is kept, with its calls answered.
    assert [result.content for result in step.tool_results] == [
        "Error: no result: the task stopped on EnvironmentFailure."
    ] * 2


async def test_an_environment_failure_in_a_tools_task_group_propagates() -> None:
    call = tool_call("pair", {"first": FAST, "second": DEAD})
    ctx = context(FakeLLM([reply(call)]))

    with pytest.raises(EnvironmentFailure, match="connection died"):
        await run_agent(environment(), ctx, hooks())


async def test_a_group_that_holds_an_environment_failure_is_never_converted() -> None:
    def broken_hook(call: ToolCall, result: ToolResult) -> str:
        raise ExceptionGroup("the hook's tasks", [EnvironmentFailure("the database died")])

    ctx = context(FakeLLM([reply(execute_sql(FAST))]))

    with pytest.raises(ExceptionGroup) as raised:
        await run_agent(environment(), ctx, hooks(on_tool_result=broken_hook))

    assert environment_failure_in(raised.value) is not None
    trajectory = ctx.trajectory()
    assert trajectory.stop_reason == CRASHED
    assert trajectory.error_message == (
        "ExceptionGroup: the hook's tasks (1 sub-exception) [EnvironmentFailure: the database died]"
    )


async def test_an_llm_failure_propagates_with_its_usage_counted() -> None:
    failure = ServerError(
        "qwen3.6-35b-a3b: HTTP 503 (6 attempts)",
        record=call_record(Usage(cost_usd=0.0)),
    )
    ctx = context(FakeLLM([reply(execute_sql(FAST)), failure]))

    with pytest.raises(ServerError):
        await run_agent(environment(), ctx, hooks())

    trajectory = ctx.trajectory()
    assert trajectory.stop_reason == CRASHED
    assert trajectory.error_message == "ServerError: qwen3.6-35b-a3b: HTTP 503 (6 attempts)"
    assert trajectory.usage.input_tokens == DEFAULT_USAGE.input_tokens
    assert trajectory.usage.cost_usd == DEFAULT_USAGE.cost_usd  # Plus the failed call's 0.


async def test_a_bug_in_a_hook_propagates() -> None:
    def broken_hook(step: Step, used: BudgetUse) -> str:
        raise KeyError("missing")

    ctx = context(FakeLLM([reply(execute_sql(FAST))]))

    with pytest.raises(KeyError):
        await run_agent(environment(), ctx, hooks(on_step=broken_hook))

    assert ctx.trajectory().stop_reason == CRASHED
    assert ctx.trajectory().error_message == "KeyError: 'missing'"


async def test_a_hooks_own_timeout_is_not_the_wall_clock() -> None:
    def impatient_hook(step: Step, used: BudgetUse) -> str:
        raise TimeoutError("the hook gave up")

    ctx = context(FakeLLM([reply(execute_sql(FAST))]))

    with pytest.raises(TimeoutError, match="the hook gave up"):
        await run_agent(environment(), ctx, hooks(on_step=impatient_hook))

    assert ctx.trajectory().stop_reason == CRASHED


async def test_a_fallback_that_raises_propagates() -> None:
    def broken_fallback() -> Prediction:
        raise ValueError

    ctx = context(FakeLLM([reply(execute_sql(FAST))]), steps=1)

    with pytest.raises(ValueError):  # noqa: PT011 - the bare error the fallback raised.
        await run_agent(environment(), ctx, hooks(fallback=broken_fallback))

    assert ctx.trajectory().stop_reason == CRASHED
    assert ctx.trajectory().error_message == "ValueError"


async def test_cancellation_propagates() -> None:
    llm = FakeLLM([reply(execute_sql(FAST)), HANG])
    ctx = context(llm)
    task = asyncio.create_task(run_agent(environment(), ctx, hooks()))
    while len(llm.requests) < 2:  # Until the loop waits in the hanging call.
        await asyncio.sleep(0)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    trajectory = ctx.trajectory()
    assert trajectory.stop_reason == CANCELLED
    assert trajectory.error_message == ""


async def test_cancellation_in_a_tool_answers_the_steps_calls() -> None:
    env = environment()
    ctx = context(FakeLLM([reply(execute_sql(SLOW))]))
    task = asyncio.create_task(run_agent(env, ctx, hooks()))
    while not env.executed:  # Until the statement runs.
        await asyncio.sleep(0)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert env.stopped == [SLOW]
    [result] = ctx.trajectory().steps[0].tool_results
    assert result.content == "Error: no result: the task was cancelled."


# The context: one per task, shared by the task's loops, with no way to choose a purpose.


async def test_a_second_loop_adds_a_chat_and_spends_the_same_budgets() -> None:
    llm = FakeLLM([reply(execute_sql(FAST)), reply(submit()), reply(execute_sql(FAST))])
    ctx = context(llm, steps=3)

    first = await run_agent(environment(), ctx, hooks())
    second = await run_agent(environment(), ctx, hooks(), label="verify")

    assert first == SUBMISSION
    assert second.fallback  # The third step was the last of the task's 3.
    trajectory = ctx.trajectory()
    assert [chat.label for chat in trajectory.chats] == ["main", "verify"]
    assert [step.chat for step in trajectory.steps] == [0, 0, 1]
    assert [step.index for step in trajectory.steps] == [0, 1, 2]
    assert trajectory.stop_reason == STEPS  # The last loop's.
    assert sent(llm)[2] == [system(SYSTEM_PROMPT), user(TASK_PROMPT)]  # A chat of its own.


async def test_a_label_another_chat_has_is_refused_with_a_stop_reason() -> None:
    ctx = context(FakeLLM([reply(submit())]))
    await run_agent(environment(), ctx, hooks())

    with pytest.raises(ValueError, match="already has a chat named 'main'"):
        await run_agent(environment(), ctx, hooks())

    trajectory = ctx.trajectory()
    assert [chat.label for chat in trajectory.chats] == ["main"]
    assert trajectory.stop_reason == CRASHED
    assert trajectory.error_message.startswith("ValueError: the task already has a chat named")


async def test_a_blank_label_is_refused_with_a_stop_reason() -> None:
    ctx = context(FakeLLM([]))

    with pytest.raises(ValueError, match="label"):
        await run_agent(environment(), ctx, hooks(), label="")

    assert ctx.trajectory().stop_reason == CRASHED


async def test_the_wall_clock_runs_from_the_first_loop() -> None:
    llm = FakeLLM([HANG])
    ctx = context(llm, wall_clock_s=0.05)
    await run_agent(environment(), ctx, hooks())

    second = await run_agent(environment(), ctx, hooks(), label="again")

    assert second.fallback
    assert len(llm.requests) == 1  # The second loop found the time spent and made no call.
    assert ctx.trajectory().stop_reason == TIME


async def test_a_later_loop_gets_only_the_time_left() -> None:
    llm = FakeLLM([reply(execute_sql(PAUSE)), reply(submit()), HANG])
    ctx = context(llm, wall_clock_s=1.0)
    await run_agent(environment(), ctx, hooks())  # Its query takes 0.6 s.

    started = time.monotonic()
    second = await run_agent(environment(), ctx, hooks(), label="again")

    assert second.fallback
    assert time.monotonic() - started < 0.7  # The 0.4 s left, not a new 1.0 s.
    assert ctx.trajectory().stop_reason == TIME


def test_the_model_handle_takes_no_purpose_or_model() -> None:
    parameters = inspect.signature(BoundLlm.complete).parameters
    assert list(parameters) == ["self", "messages", "tools", "params"]


def test_the_context_exposes_no_model_handle() -> None:
    ctx = context(FakeLLM([]))

    public = {name for name in dir(ctx) if not name.startswith("_")}

    assert public == {"budgets", "trajectory"}
    assert ctx.budgets == Budgets(steps=30, wall_clock_s=600.0)


def test_the_context_starts_with_an_empty_trajectory() -> None:
    assert context(FakeLLM([])).trajectory() == Trajectory()


# The real client, bound to purpose exec, against a scripted gateway.

EXEC_TAGS = CallTags(CallPurpose.CALL_PURPOSE_EXEC, run_id="run-1", instance_id="q7", seed=2)


class Recorder:
    """A CallRecorder that keeps the record of each call."""

    def __init__(self) -> None:
        self.records: list[LlmCallRecord] = []

    def record(self, record: LlmCallRecord) -> None:
        self.records.append(record)


def settings() -> Settings:
    return Settings(
        api_key=SecretStr("sk-test-0123456789abcdef"), base_url="https://gateway.test/v1"
    )


def completion(call_id: str, name: str, arguments: dict[str, str]) -> httpx.Response:
    """A chat completion that makes one tool call, as the gateway sends one."""
    call = {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }
    message = {"role": "assistant", "content": "", "tool_calls": [call]}
    body = {
        "choices": [{"finish_reason": "tool_calls", "index": 0, "message": message}],
        "usage": {"prompt_tokens": 900, "completion_tokens": 100},
    }
    return httpx.Response(200, json=body, headers={"x-litellm-response-cost": "0.0004"})


async def test_the_loop_runs_on_the_real_exec_bound_client() -> None:
    bodies: list[Any] = []
    replies = iter(
        [
            completion("c1", "execute_sql", {"sql": FAST}),
            completion("c2", "submit", {"sql": FAST}),
        ]
    )

    def gateway(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return next(replies)

    recorder = Recorder()
    async with LlmClient(settings(), recorder, transport=httpx.MockTransport(gateway)) as client:
        llm: BoundLlm = client.bind(model=EXECUTOR_MODEL, tags=EXEC_TAGS)  # mypy checks the fit.
        ctx = context(llm)
        prediction = await run_agent(environment(), ctx, hooks())

    assert prediction == SUBMISSION
    records = recorder.records
    assert [record.status for record in records] == [LlmCallStatus.LLM_CALL_STATUS_OK] * 2
    assert {(r.purpose, r.model, r.run_id, r.instance_id, r.seed) for r in records} == {
        (CallPurpose.CALL_PURPOSE_EXEC, EXECUTOR_MODEL, "run-1", "q7", 2)
    }
    steps = ctx.trajectory().steps
    assert [step.call_id for step in steps] == [record.call_id for record in records]
    assert [step.latency_s for step in steps] == [record.latency_s for record in records]
    second = bodies[1]
    assert second["max_tokens"] == PARAMS.max_tokens  # The pack's parameters, over the profile's.
    assert [tool["function"]["name"] for tool in second["tools"]] == TOOL_NAMES.split(", ")
    assert [message["role"] for message in second["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert second["messages"][2]["tool_calls"][0]["id"] == "c1"
    assert second["messages"][3] == {"role": "tool", "content": "rows: 1", "tool_call_id": "c1"}


async def test_the_wall_clock_cancels_a_real_call_and_its_record_says_so() -> None:
    async def gateway(request: httpx.Request) -> httpx.Response:
        await asyncio.Event().wait()  # Never answers.
        raise AssertionError("the gateway answered")

    recorder = Recorder()
    async with LlmClient(settings(), recorder, transport=httpx.MockTransport(gateway)) as client:
        ctx = context(client.bind(model=EXECUTOR_MODEL, tags=EXEC_TAGS), wall_clock_s=0.1)
        async with asyncio.timeout(10):
            prediction = await run_agent(environment(), ctx, hooks())

    assert prediction.fallback
    assert ctx.trajectory().stop_reason == TIME
    [record] = recorder.records
    assert record.status == LlmCallStatus.LLM_CALL_STATUS_CANCELLED


# Context folding (spec §5.1): older tool outputs fold, and older reasoning stays behind.

FOLD_MARKER = "\n[... folded: the first 500 of 4000 characters ...]"


def folded(n: int) -> str:
    return big(n)[:500] + FOLD_MARKER


def page(n: int) -> ToolCall:
    return tool_call("page", {"n": n})


def counted(input_tokens: int) -> Usage:
    """The usage of a call whose prompt the gateway counted as ``input_tokens``."""
    return Usage(input_tokens=input_tokens, output_tokens=10)


async def test_a_long_chat_folds_its_oldest_tool_outputs() -> None:
    calls = [page(n) for n in range(4)]
    script = [
        reply(call, usage=counted(n))
        for call, n in zip(calls, (500, 1500, 2500, 3500), strict=True)
    ]
    llm = FakeLLM([*script, reply(submit())])
    ctx = context(llm, window=ContextWindow(fold_above_tokens=3000, keep_turns=2))

    prediction = await run_agent(environment(), ctx, hooks())

    assert prediction == SUBMISSION
    trajectory = ctx.trajectory()
    assert [list(step.folds) for step in trajectory.steps] == [
        [],
        [],
        [],
        [ContextFold(step=0, tool_call_id=calls[0].id, replacement=folded(0))],
        [ContextFold(step=1, tool_call_id=calls[1].id, replacement=folded(1))],
    ]
    estimates = [step.estimated_prompt_tokens for step in trajectory.steps]
    assert 0 < estimates[0] < estimates[1] < estimates[2] < 3000
    assert estimates[3] <= 3000  # One fold brought it back under the threshold.
    assert estimates[4] > 3000  # The two outputs it may fold were not enough; it went anyway.
    assert [step.tool_results[0].content for step in trajectory.steps[:4]] == [
        big(n) for n in range(4)
    ]
    assert [m.content for m in sent(llm)[4] if m.role == TOOL] == [
        folded(0),
        folded(1),
        big(2),
        big(3),
    ]
    assert reference_prompts(trajectory) == sent(llm)


async def test_the_last_eight_turns_never_fold() -> None:
    calls = [page(n) for n in range(10)]
    script = [reply(call, usage=counted(10_000 * (n + 1))) for n, call in enumerate(calls)]
    llm = FakeLLM([*script, reply(submit())])
    ctx = context(llm, window=ContextWindow(fold_above_tokens=1000))  # The default 8-turn tail.

    await run_agent(environment(), ctx, hooks())

    last = [m.content for m in sent(llm)[10] if m.role == TOOL]
    assert last == [folded(0), folded(1), *(big(n) for n in range(2, 10))]
    assert reference_prompts(ctx.trajectory()) == sent(llm)


async def test_each_prompt_keeps_the_reasoning_of_the_chats_last_turns_before_it() -> None:
    script = [reply(execute_sql(f"SELECT {n}"), reasoning=f"Thought {n}.") for n in range(4)]
    llm = FakeLLM([*script, reply(submit())])
    ctx = context(llm, window=ContextWindow(keep_turns=2))

    await run_agent(environment(), ctx, hooks())

    kept = [[m.reasoning_content for m in prompt if m.role == ASSISTANT] for prompt in sent(llm)]
    assert kept == [
        [],
        ["Thought 0."],
        ["Thought 0.", "Thought 1."],
        ["", "Thought 1.", "Thought 2."],
        ["", "", "Thought 2.", "Thought 3."],
    ]
    assert reference_prompts(ctx.trajectory()) == sent(llm)


async def test_only_the_last_turns_go_out_with_their_reasoning() -> None:
    script = [reply(execute_sql(FAST), reasoning=f"Thought {n}.") for n in range(3)]
    llm = FakeLLM([*script, reply(submit())])
    ctx = context(llm, window=ContextWindow(keep_turns=2))

    await run_agent(environment(), ctx, hooks())

    replies = [m for m in sent(llm)[3] if m.role == ASSISTANT]
    assert [m.reasoning_content for m in replies] == ["", "Thought 1.", "Thought 2."]
    trajectory = ctx.trajectory()
    assert trajectory.steps[0].assistant.reasoning_content == "Thought 0."  # Kept here.
    assert reference_prompts(trajectory) == sent(llm)


async def test_a_prompt_too_long_for_the_model_folds_and_goes_once_more() -> None:
    overflow = ContextOverflow("qwen3.6-35b-a3b: HTTP 400: maximum context length is 131072")
    calls = [page(n) for n in range(3)]
    llm = FakeLLM([*(reply(call) for call in calls), overflow, reply(submit())])
    ctx = context(llm, window=ContextWindow(keep_turns=1))  # The estimate never passes 96K.

    prediction = await run_agent(environment(), ctx, hooks())

    assert prediction == SUBMISSION
    trajectory = ctx.trajectory()
    assert trajectory.stop_reason == SUBMITTED
    assert len(trajectory.steps) == 4  # The call that overflowed has no step.
    assert [(fold.step, fold.tool_call_id) for fold in trajectory.steps[3].folds] == [
        (0, calls[0].id),
        (1, calls[1].id),
    ]
    overflowed, retried = sent(llm)[3], sent(llm)[4]
    assert [m.content for m in overflowed if m.role == TOOL] == [big(0), big(1), big(2)]
    assert [m.content for m in retried if m.role == TOOL] == [folded(0), folded(1), big(2)]
    assert reference_prompts(trajectory) == [*sent(llm)[:3], retried]


async def test_a_prompt_too_long_after_its_fold_ends_in_the_fallback() -> None:
    first = ContextOverflow("too long")
    second = ContextOverflow("still too long")
    llm = FakeLLM([reply(page(0)), reply(page(1)), first, second])
    ctx = context(llm, window=ContextWindow(keep_turns=1))

    prediction = await run_agent(environment(), ctx, hooks())

    assert prediction.fallback
    assert len(llm.requests) == 4
    trajectory = ctx.trajectory()
    assert trajectory.stop_reason == TOKENS
    assert trajectory.error_message == "ContextOverflow: still too long"


async def test_a_prompt_too_long_with_nothing_to_fold_is_not_sent_again() -> None:
    llm = FakeLLM([reply(page(0)), ContextOverflow("too long")])
    ctx = context(llm, window=ContextWindow(keep_turns=1))  # The one turn is the tail.

    prediction = await run_agent(environment(), ctx, hooks())

    assert prediction.fallback
    assert len(llm.requests) == 2
    assert ctx.trajectory().error_message == "ContextOverflow: too long"


# Repeated calls (spec §5.1): the same call again counts as an error, and says so.

REPEATED = (
    "Note: this call repeats your previous one, the same tool with the same arguments, so its "
    "result is unchanged. Repeating a call does not help: change the arguments or try another "
    "approach."
)


async def test_a_call_that_repeats_the_one_before_runs_with_a_note_and_counts_as_an_error() -> None:
    env = environment()
    llm = FakeLLM([*(reply(execute_sql(FAST)) for _ in range(4)), reply(submit())])
    ctx = context(llm)

    prediction = await run_agent(env, ctx, hooks())

    assert prediction == SUBMISSION
    assert [sql for sql, _ in env.executed] == [FAST] * 4  # Each repeat still ran.
    steps = ctx.trajectory().steps
    assert steps[0].tool_results[0].content == "rows: 1"
    for step in steps[1:4]:
        assert step.tool_results[0].content == f"rows: 1\n\n{REPEATED}"
        assert not step.tool_results[0].is_error
    # The three repeats are three failed replies in a row: the reminder came after them.
    assert injected(ctx.trajectory())[:3] == [[], [], []]
    assert steps[3].injected[0].content.startswith("You have made 3 errors in a row.")


async def test_arguments_that_differ_only_in_layout_repeat() -> None:
    one = tool_call("execute_sql", f'{{"sql": "{FAST}",  "limit": 20}}')
    other = tool_call("execute_sql", f'{{"limit":20,"sql":"{FAST}"}}')
    ctx = context(FakeLLM([reply(one), reply(other), reply(submit())]))

    await run_agent(environment(), ctx, hooks())

    assert ctx.trajectory().steps[1].tool_results[0].content.endswith(REPEATED)


async def test_a_repeat_in_the_same_reply_counts() -> None:
    ctx = context(FakeLLM([reply(execute_sql(FAST), execute_sql(FAST)), reply(submit())]))

    await run_agent(environment(), ctx, hooks())

    results = ctx.trajectory().steps[0].tool_results
    assert [result.content.endswith(REPEATED) for result in results] == [False, True]


async def test_only_the_immediately_preceding_call_that_ran_counts() -> None:
    cut = reply(execute_sql(EXPLAIN), finish_reason="length")  # It does not run.
    script = [
        reply(execute_sql(FAST)),
        reply(execute_sql(EXPLAIN)),
        reply(execute_sql(FAST)),  # Not right after the other FAST.
        cut,
        reply(execute_sql(EXPLAIN)),  # The cut-off call came between, but never ran.
        reply(submit()),
    ]
    ctx = context(FakeLLM(script))

    await run_agent(environment(), ctx, hooks())

    contents = [step.tool_results[0].content for step in ctx.trajectory().steps]
    assert [content.endswith(REPEATED) for content in contents] == [False] * 6


async def test_the_pack_sees_the_note_and_its_guidance_follows_it() -> None:
    seen: list[str] = []

    def guide(call: ToolCall, result: ToolResult) -> str:
        seen.append(result.content)
        return "Hint: try a join."

    ctx = context(FakeLLM([reply(execute_sql(FAST)), reply(execute_sql(FAST)), reply(submit())]))

    await run_agent(environment(), ctx, hooks(on_tool_result=guide))

    assert seen[1] == f"rows: 1\n\n{REPEATED}"
    assert ctx.trajectory().steps[1].tool_results[0].content == (
        f"rows: 1\n\n{REPEATED}\n\nHint: try a join."
    )
