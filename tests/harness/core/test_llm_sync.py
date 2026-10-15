"""Tests for LlmClient.complete_sync: calls from worker threads, on the client's own event loop."""

import asyncio
import threading

import httpx
import pytest

from goaly.contracts.gen.llm_call_pb2 import CallPurpose, LlmCallStatus
from goaly.harness.core.llm import CallTags, LlmClient
from goaly.harness.core.llm_errors import ClientError, ModelNotAllowed
from goaly.harness.core.llm_models import ModelParams
from tests.harness.core.llm_fakes import (
    GLM,
    QWEN,
    Gateway,
    Recorder,
    completion,
    connect,
    failure,
    hang,
    settings,
    user,
)

pytestmark = pytest.mark.anyio

IMPL = CallTags(CallPurpose.CALL_PURPOSE_RSI_IMPL, run_id="rsi-1")


async def test_a_worker_thread_calls_through_the_clients_own_loop() -> None:
    seen: list[tuple[asyncio.AbstractEventLoop, int]] = []

    async def answer(request: httpx.Request) -> httpx.Response:
        seen.append((asyncio.get_running_loop(), threading.get_ident()))
        return completion()

    gateway, recorder = Gateway(answer), Recorder()
    async with connect(gateway, recorder=recorder) as llm:
        reply = await asyncio.to_thread(llm.complete_sync, [user("hi")], model=GLM, tags=IMPL)
        here = (asyncio.get_running_loop(), threading.get_ident())

    assert seen == [here]  # On the client's loop and in its thread, not the worker's.
    assert reply.message.content == "391"
    assert recorder.records == [reply.record]
    assert reply.record.purpose == CallPurpose.CALL_PURPOSE_RSI_IMPL


async def test_calls_from_threads_and_from_the_loop_share_one_limit() -> None:
    in_flight, peak = 0, 0

    async def answer(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return completion()

    gateway = Gateway(*(answer for _ in range(8)))
    async with LlmClient(
        settings(),
        Recorder(),
        shares={QWEN: 1, GLM: 2},
        transport=httpx.MockTransport(gateway.handle),
    ) as llm:
        from_threads = [
            asyncio.to_thread(llm.complete_sync, [user("thread")], model=GLM, tags=IMPL)
            for _ in range(4)
        ]
        from_the_loop = [llm.complete([user("loop")], model=GLM, tags=IMPL) for _ in range(4)]
        replies = await asyncio.gather(*from_threads, *from_the_loop)

    assert len(replies) == 8
    assert peak == 2  # One set of limits: not 2 per thread, nor 2 more for the threads.


async def test_an_error_reaches_the_calling_thread() -> None:
    gateway = Gateway(failure(400, "bad request"))
    async with connect(gateway) as llm:
        with pytest.raises(ClientError, match="bad request"):
            await asyncio.to_thread(llm.complete_sync, [user("hi")], model=GLM, tags=IMPL)


async def test_refuses_to_block_the_thread_its_loop_runs_in() -> None:
    gateway = Gateway()
    async with connect(gateway) as llm:
        with pytest.raises(RuntimeError, match="await complete"):
            # The timeout is a watchdog: were the guard gone, this would fail, not hang.
            llm.complete_sync([user("hi")], model=GLM, tags=IMPL, timeout_s=5)

    assert gateway.requests == []


async def test_needs_a_client_that_was_entered_or_used() -> None:
    gateway = Gateway()
    llm = connect(gateway)
    try:
        with pytest.raises(RuntimeError, match="async with"):
            await asyncio.to_thread(llm.complete_sync, [user("hi")], model=GLM, tags=IMPL)
    finally:
        await llm.aclose()

    assert gateway.requests == []


async def test_a_client_first_used_without_async_with_serves_that_loop() -> None:
    gateway = Gateway(completion(), completion())
    llm = connect(gateway)
    try:
        await llm.complete([user("hi")], model=GLM, tags=IMPL)
        reply = await asyncio.to_thread(llm.complete_sync, [user("hi")], model=GLM, tags=IMPL)
    finally:
        await llm.aclose()

    assert reply.message.content == "391"


def test_refuses_once_its_loop_has_stopped() -> None:
    async def use_once() -> LlmClient:
        async with connect(Gateway()) as llm:
            return llm

    llm = asyncio.run(use_once())

    with pytest.raises(RuntimeError, match="not running"):
        llm.complete_sync([user("hi")], model=GLM, tags=IMPL)


async def test_a_call_after_the_client_closed_is_refused_with_no_record() -> None:
    gateway, recorder = Gateway(), Recorder()
    async with connect(gateway, recorder=recorder) as llm:
        pass

    with pytest.raises(RuntimeError, match="closed"):
        await llm.complete([user("hi")], model=GLM, tags=IMPL)
    with pytest.raises(RuntimeError, match="closed"):  # A worker left running, say.
        await asyncio.to_thread(llm.complete_sync, [user("hi")], model=GLM, tags=IMPL)
    assert (gateway.requests, recorder.records) == ([], [])


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        pytest.param({"model": QWEN}, ModelNotAllowed, id="model-not-allowed"),
        pytest.param({"params": ModelParams(enable_thinking=False)}, ValueError, id="bad-params"),
    ],
)
async def test_a_refused_call_reaches_the_calling_thread_with_no_io_and_no_record(
    kwargs: dict[str, object], error: type[Exception]
) -> None:
    gateway, recorder = Gateway(), Recorder()
    call: dict[str, object] = {"model": GLM, "tags": IMPL, **kwargs}
    async with connect(gateway, recorder=recorder) as llm:
        with pytest.raises(error):
            await asyncio.to_thread(lambda: llm.complete_sync([user("hi")], **call))  # type: ignore[arg-type]  # The kwargs vary by case.

    assert (gateway.requests, recorder.records) == ([], [])


async def test_a_call_past_its_timeout_is_cancelled_and_recorded() -> None:
    gateway, recorder = Gateway(hang), Recorder()
    async with connect(gateway, recorder=recorder) as llm:
        with pytest.raises(TimeoutError, match=r"no reply within 0\.05 s"):
            await asyncio.to_thread(
                llm.complete_sync, [user("hi")], model=GLM, tags=IMPL, timeout_s=0.05
            )
        while not recorder.records:  # The loop takes in the cancellation after the thread.
            await asyncio.sleep(0.01)

    (record,) = recorder.records
    assert record.status == LlmCallStatus.LLM_CALL_STATUS_CANCELLED


async def test_a_call_the_loop_cancels_reaches_the_thread_as_a_cancellation() -> None:
    arrived = asyncio.Event()

    async def hold(request: httpx.Request) -> httpx.Response:
        arrived.set()
        return await hang(request)

    gateway, recorder = Gateway(hold), Recorder()
    outcome: list[BaseException] = []

    def worker(llm: LlmClient) -> None:
        try:
            llm.complete_sync([user("hi")], model=GLM, tags=IMPL)
        except BaseException as error:  # noqa: BLE001 - The test inspects whatever it was.
            outcome.append(error)

    async with connect(gateway, recorder=recorder) as llm:
        thread = threading.Thread(target=worker, args=(llm,))
        thread.start()
        await arrived.wait()
        for task in asyncio.all_tasks():  # The call's own task on the loop, as at shutdown.
            if task.get_coro().cr_code.co_name == "complete":  # type: ignore[union-attr]  # A coroutine.
                task.cancel()
        await asyncio.to_thread(thread.join)

    (error,) = outcome
    assert type(error) is asyncio.CancelledError  # Not swallowed by an `except Exception`.
    (record,) = recorder.records
    assert record.status == LlmCallStatus.LLM_CALL_STATUS_CANCELLED


async def test_a_loop_closing_as_the_call_is_sent_leaves_no_coroutine_behind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def closed(*args: object) -> object:
        raise RuntimeError("Event loop is closed")

    gateway = Gateway()
    async with connect(gateway) as llm:
        monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", closed)
        with pytest.raises(RuntimeError, match="Event loop is closed"):
            await asyncio.to_thread(llm.complete_sync, [user("hi")], model=GLM, tags=IMPL)

    assert gateway.requests == []  # And no warning of a coroutine never awaited.
