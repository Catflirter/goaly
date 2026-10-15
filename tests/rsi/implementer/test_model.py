"""Tests for goaly.rsi.implementer.model: mini-swe-agent's agent on GLM, behind core's LLM client.

The agent runs in a worker thread, as on the lane's implementer pool, and the client on the test's
event loop, in front of a scripted gateway.
"""

import asyncio
import json
import threading
from collections.abc import Callable, Iterator
from typing import Any

import httpx
import pytest

from goaly.contracts.gen.llm_call_pb2 import CallPurpose
from goaly.contracts.gen.trajectory_pb2 import ChatMessage, ChatRole, ToolCall
from goaly.harness.core.llm_errors import ClientError, ModelNotAllowed
from goaly.rsi.implementer import msa
from goaly.rsi.implementer.config import Limits, load_config
from goaly.rsi.implementer.model import (
    EXIT_COST_LIMIT,
    EXIT_INTERRUPTED,
    EXIT_STEP_LIMIT,
    EXIT_TIME_LIMIT,
    GlmModel,
    assistant_message,
    chat_message,
)
from tests.harness.core.llm_fakes import (
    GLM,
    KEY,
    Gateway,
    Recorder,
    completion,
    connect,
    failure,
)
from tests.rsi.implementer.fakes import (
    INSTANCE_TEMPLATE,
    SUBMIT,
    SYSTEM_TEMPLATE,
    FakeEnvironment,
    bash_call,
)

pytestmark = pytest.mark.anyio

EXPERIMENT = "exp-7"
LIMITS = load_config().limits
TEMPLATES = load_config().model


def reply(
    command: str, *, cost: str = "0.25", reasoning: str = "", call_id: str = "call_1"
) -> httpx.Response:
    """A GLM reply that makes one bash call."""
    return completion(
        "",
        reasoning=reasoning,
        tool_calls=[bash_call(command, call_id)],
        finish_reason="tool_calls",
        cost=cost,
    )


def limits(**changes: Any) -> Limits:
    return LIMITS.model_copy(update=changes)


class Run:
    """One agent run on GlmModel, and what it left behind."""

    def __init__(self, gateway: Gateway) -> None:
        self.gateway = gateway
        self.recorder = Recorder()
        self.environment = FakeEnvironment()
        self.result: dict[str, Any] = {}
        self._agent: msa.DefaultAgent | None = None
        self._model: GlmModel | None = None

    async def go(
        self,
        *,
        run_limits: Limits = LIMITS,
        stop: threading.Event | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        async with connect(self.gateway, recorder=self.recorder) as llm:
            options: dict[str, Any] = {} if clock is None else {"clock": clock}
            self._model = GlmModel(
                llm,
                experiment_id=EXPERIMENT,
                limits=run_limits,
                templates=TEMPLATES,
                stop=stop,
                **options,
            )
            self._agent = msa.make_agent(
                self._model,
                self.environment,
                system_template=SYSTEM_TEMPLATE,
                instance_template=INSTANCE_TEMPLATE,
                max_consecutive_format_errors=run_limits.max_consecutive_format_errors,
            )
            self.result = await asyncio.to_thread(self._agent.run, "make it faster")

    @property
    def agent(self) -> msa.DefaultAgent:
        assert self._agent is not None, "the run has not started"
        return self._agent

    @property
    def model(self) -> GlmModel:
        assert self._model is not None, "the run has not started"
        return self._model

    @property
    def messages(self) -> list[dict[str, Any]]:
        return self.agent.messages


async def test_reasoning_and_tool_calls_go_back_unchanged() -> None:
    reasoning = "Plan: look first.\n  Then edit — carefully. ✓"
    run = Run(Gateway(reply("ls", reasoning=reasoning), reply(f"{SUBMIT}\nadded a skill")))

    await run.go()

    assert run.result == {"exit_status": "Submitted", "submission": "added a skill"}
    first, second = run.gateway.bodies
    assert first["model"] == GLM
    assert first["tools"] == [msa.BASH_TOOL]
    assert [message["role"] for message in first["messages"]] == ["system", "user"]
    assert first["messages"][0]["content"] == "You implement hypotheses. You have 60 steps."
    assert first["messages"][1]["content"] == "Hypothesis: make it faster"
    system, task, assistant, tool = second["messages"]
    assert (system, task) == tuple(first["messages"])
    assert assistant == {
        "role": "assistant",
        "content": "",
        "reasoning_content": reasoning,  # Byte for byte.
        "tool_calls": [bash_call("ls")],
    }
    assert tool["role"] == "tool"
    assert tool["tool_call_id"] == "call_1"
    assert json.loads(tool["content"]) == {"returncode": 0, "output": ""}
    assert run.environment.commands == ["ls", f"{SUBMIT}\nadded a skill"]


async def test_each_call_is_recorded_with_its_purpose_and_the_gateways_cost() -> None:
    run = Run(Gateway(reply("ls", cost="0.25"), reply(SUBMIT, cost="0.5")))

    await run.go()

    records = run.recorder.records
    assert [record.purpose for record in records] == [CallPurpose.CALL_PURPOSE_RSI_IMPL] * 2
    assert {
        (record.run_id, record.instance_id, record.seed, record.model) for record in records
    } == {(EXPERIMENT, "", 0, GLM)}
    assert [record.usage.cost_usd for record in records] == [0.25, 0.5]
    steps = [message for message in run.messages if message["role"] == "assistant"]
    assert [step["extra"]["call_id"] for step in steps] == [record.call_id for record in records]
    assert [step["extra"]["cost"] for step in steps] == [0.25, 0.5]
    assert steps[0]["extra"]["actions"] == [{"command": "ls", "tool_call_id": "call_1"}]
    assert run.agent.cost == run.model.cost_usd == 0.75
    assert run.model.calls == 2


async def test_the_run_stops_at_the_cost_limit() -> None:
    run = Run(Gateway(*(reply("ls", cost="1.25") for _ in range(5))))

    await run.go()

    # $1.25, $2.50, then $3.75: the limit of $3 is reached after the third call.
    assert run.result == {"exit_status": EXIT_COST_LIMIT, "submission": ""}
    assert len(run.gateway.requests) == 3
    assert len(run.recorder.records) == 3


async def test_the_run_stops_as_soon_as_the_cost_reaches_the_limit_exactly() -> None:
    run = Run(Gateway(*(reply("ls", cost="1.5") for _ in range(3))))

    await run.go()

    assert run.result == {"exit_status": EXIT_COST_LIMIT, "submission": ""}
    assert len(run.gateway.requests) == 2  # $1.50 and $1.50 make $3.00, the limit.


async def test_the_run_stops_at_the_step_limit() -> None:
    run = Run(Gateway(*(reply("ls") for _ in range(5))))

    await run.go(run_limits=limits(step_limit=2))

    assert run.result == {"exit_status": EXIT_STEP_LIMIT, "submission": ""}
    assert len(run.gateway.requests) == 2


async def test_the_run_stops_at_the_wall_time_limit() -> None:
    # Read at the first call's start, at its check, and at the second call's check.
    times: Iterator[float] = iter([0.0, 0.0, 1800.0])
    run = Run(Gateway(*(reply("ls") for _ in range(3))))

    await run.go(clock=lambda: next(times))

    assert run.result == {"exit_status": EXIT_TIME_LIMIT, "submission": ""}
    assert len(run.gateway.requests) == 1


async def test_a_stop_ends_the_run_before_its_next_call() -> None:
    stop = threading.Event()
    run = Run(Gateway(reply("ls"), reply("ls")))

    def stop_after(command: str) -> dict[str, Any]:
        stop.set()
        return {}

    run.environment = FakeEnvironment(stop_after)

    await run.go(stop=stop)

    assert run.result == {"exit_status": EXIT_INTERRUPTED, "submission": ""}
    assert len(run.gateway.requests) == 1


async def test_a_reply_without_a_bash_call_is_shown_to_the_model_as_a_format_error() -> None:
    unparsable = completion("I will look around.", reasoning="hmm", cost="0.125")
    run = Run(Gateway(unparsable, reply(SUBMIT)))

    await run.go()

    assert run.result["exit_status"] == "Submitted"
    error = run.messages[2]
    assert error["role"] == "user"
    assert "No tool calls found" in error["content"]
    # The agent charged the failed reply, and the trajectory keeps it.
    assert error["extra"]["cost"] == 0.125
    assert error["extra"]["call_id"] == run.recorder.records[0].call_id
    assert error["extra"]["response"]["content"] == "I will look around."
    assert run.agent.cost == 0.125 + 0.25
    assert run.gateway.bodies[1]["messages"][-1]["content"] == error["content"]


async def test_three_format_errors_in_a_row_end_the_run() -> None:
    run = Run(Gateway(*(completion("no call", cost="0.1") for _ in range(5))))

    await run.go()

    assert run.result == {"exit_status": "RepeatedFormatError", "submission": ""}
    assert len(run.gateway.requests) == 3


async def test_a_failed_call_ends_the_run_with_its_error_and_its_cost_counts() -> None:
    run = Run(Gateway(failure(400, "bad parameter", cost="0.01")))

    with pytest.raises(ClientError):
        await run.go()

    assert run.model.cost_usd == 0.01
    assert run.model.calls == 0
    assert run.messages[-1]["extra"]["exit_status"] == "ClientError"


async def test_a_query_on_the_clients_own_loop_refuses_rather_than_deadlock() -> None:
    gateway = Gateway(reply("ls"))
    async with connect(gateway) as llm:
        model = GlmModel(llm, experiment_id=EXPERIMENT, limits=LIMITS, templates=TEMPLATES)

        with pytest.raises(RuntimeError, match="block the loop"):
            model.query([{"role": "user", "content": "hi"}])

    assert gateway.requests == []


async def test_the_trajectory_names_the_model_and_holds_no_key() -> None:
    run = Run(Gateway(reply(SUBMIT)))

    await run.go()

    data = run.agent.serialize()
    assert data["info"]["config"]["model"] == {
        "model": GLM,
        "purpose": "CALL_PURPOSE_RSI_IMPL",
        "run_id": EXPERIMENT,
        **LIMITS.model_dump(),
    }
    assert data["info"]["config"]["model_type"] == "goaly.rsi.implementer.model.GlmModel"
    assert KEY not in json.dumps(data)
    assert KEY not in json.dumps(run.gateway.bodies)


def test_chat_message_keeps_a_tool_answer_and_its_call() -> None:
    message = chat_message(
        {"role": "tool", "content": "ok", "tool_call_id": "call_9", "extra": {"returncode": 0}}
    )

    assert message == ChatMessage(role=ChatRole.CHAT_ROLE_TOOL, content="ok", tool_call_id="call_9")


@pytest.mark.parametrize(
    "message",
    [
        pytest.param({"role": "exit", "content": "Submitted"}, id="exit"),
        pytest.param({"content": "no role"}, id="no-role"),
        pytest.param({"role": "user", "content": [{"type": "text", "text": "x"}]}, id="parts"),
    ],
)
def test_chat_message_refuses_what_the_gateway_cannot_take(message: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=r"cannot send|must be text"):
        chat_message(message)


def test_assistant_message_gives_an_id_to_a_call_without_one() -> None:
    message = assistant_message(
        ChatMessage(
            role=ChatRole.CHAT_ROLE_ASSISTANT,
            content="x",
            tool_calls=[ToolCall(name="bash", arguments_json='{"command": "ls"}')],
        )
    )

    (call,) = message["tool_calls"]
    assert call["id"].startswith("call_")
    assert call["function"] == {"name": "bash", "arguments": '{"command": "ls"}'}
    assert "reasoning_content" not in message
    assert chat_message(message).tool_calls[0].id == call["id"]


class Unreachable:
    """A client whose every call fails before it is made, so no record exists."""

    def complete_sync(self, *args: Any, **kwargs: Any) -> Any:
        raise ModelNotAllowed("no call purpose may use this model")


def test_a_call_that_never_started_costs_nothing() -> None:
    model = GlmModel(Unreachable(), experiment_id=EXPERIMENT, limits=LIMITS, templates=TEMPLATES)

    with pytest.raises(ModelNotAllowed):
        model.query([{"role": "user", "content": "hi"}])

    assert (model.calls, model.cost_usd) == (0, 0.0)
