"""Tests for goaly.harness.core.llm_errors: one error per way a call can fail."""

import pytest

from goaly.contracts.gen.llm_call_pb2 import LlmCallRecord, LlmCallStatus
from goaly.harness.core.llm_errors import (
    ClientError,
    ConnectionFailed,
    ContextOverflow,
    InvalidResponse,
    LlmError,
    ModelNotAllowed,
    RateLimited,
    ServerError,
    TimedOut,
    error_type,
)

S = LlmCallStatus
FAILURES = [
    (ContextOverflow, S.LLM_CALL_STATUS_CONTEXT_OVERFLOW),
    (RateLimited, S.LLM_CALL_STATUS_RATE_LIMITED),
    (ServerError, S.LLM_CALL_STATUS_SERVER_ERROR),
    (TimedOut, S.LLM_CALL_STATUS_TIMEOUT),
    (ConnectionFailed, S.LLM_CALL_STATUS_CONNECTION_ERROR),
    (ClientError, S.LLM_CALL_STATUS_CLIENT_ERROR),
    (InvalidResponse, S.LLM_CALL_STATUS_INVALID_RESPONSE),
]
# OK is no failure, a cancelled call raises asyncio.CancelledError, and UNSPECIFIED is never set.
NOT_FAILURES = [S.LLM_CALL_STATUS_UNSPECIFIED, S.LLM_CALL_STATUS_OK, S.LLM_CALL_STATUS_CANCELLED]


@pytest.mark.parametrize(("error", "status"), FAILURES)
def test_each_failure_status_has_its_own_error(
    error: type[LlmError], status: LlmCallStatus.ValueType
) -> None:
    assert issubclass(error, LlmError)
    assert error.status == status
    assert error_type(status) is error


def test_every_status_a_call_can_fail_with_has_an_error() -> None:
    failures = {status for _, status in FAILURES}

    assert failures == set(LlmCallStatus.values()) - set(NOT_FAILURES)


@pytest.mark.parametrize("status", NOT_FAILURES)
def test_a_status_that_is_no_failure_has_no_error(status: LlmCallStatus.ValueType) -> None:
    with pytest.raises(ValueError, match="is not a failure"):
        error_type(status)


def test_a_disallowed_model_has_the_status_of_a_refused_request() -> None:
    assert issubclass(ModelNotAllowed, LlmError)
    assert ModelNotAllowed.status == S.LLM_CALL_STATUS_CLIENT_ERROR
    # A call that fails with CLIENT_ERROR got a 4xx from the gateway: that is a ClientError.
    assert error_type(S.LLM_CALL_STATUS_CLIENT_ERROR) is ClientError


def test_an_error_carries_its_call_record() -> None:
    record = LlmCallRecord(call_id="c-1")

    error = RateLimited("busy", record=record)

    assert str(error) == "busy"
    assert error.record is record
    assert ModelNotAllowed("refused").record is None
