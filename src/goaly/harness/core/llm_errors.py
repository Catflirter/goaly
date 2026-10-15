"""The errors the LLM client raises (spec §5.1): one type for each way a call can fail.

Each is an ``LlmError``, whose ``status`` is the LlmCallStatus the call's record ends with. A
cancelled call raises asyncio.CancelledError instead, and its record ends with
LLM_CALL_STATUS_CANCELLED. No message holds the API key: the client removes it from every text it
quotes from the gateway.
"""

from typing import ClassVar

from goaly.contracts.gen.llm_call_pb2 import LlmCallRecord, LlmCallStatus

__all__ = [
    "ClientError",
    "ConnectionFailed",
    "ContextOverflow",
    "InvalidResponse",
    "LlmError",
    "ModelNotAllowed",
    "RateLimited",
    "ServerError",
    "TimedOut",
    "error_type",
]


class LlmError(Exception):
    """A call to the model gateway failed.

    ``record`` is the call's record, the one the recorder got, or None if the call never started.
    """

    status: ClassVar[LlmCallStatus.ValueType]

    def __init__(self, message: str, *, record: LlmCallRecord | None = None) -> None:
        super().__init__(message)
        self.record = record


class ContextOverflow(LlmError):
    """The prompt, with room for max_tokens, does not fit in the model's context. Not retried."""

    status = LlmCallStatus.LLM_CALL_STATUS_CONTEXT_OVERFLOW


class RateLimited(LlmError):
    """The gateway answered 429 Too Many Requests, to the last retry too."""

    status = LlmCallStatus.LLM_CALL_STATUS_RATE_LIMITED


class ServerError(LlmError):
    """The gateway answered with a 5xx status, to the last retry too."""

    status = LlmCallStatus.LLM_CALL_STATUS_SERVER_ERROR


class TimedOut(LlmError):
    """No response came within the attempt timeout, for the last retry either."""

    status = LlmCallStatus.LLM_CALL_STATUS_TIMEOUT


class ConnectionFailed(LlmError):
    """The gateway could not be reached, for the last retry either."""

    status = LlmCallStatus.LLM_CALL_STATUS_CONNECTION_ERROR


class ClientError(LlmError):
    """The gateway refused the request with another 4xx, e.g. for a parameter. Not retried."""

    status = LlmCallStatus.LLM_CALL_STATUS_CLIENT_ERROR


class InvalidResponse(LlmError):
    """A response came but was no chat completion, e.g. a body that is not one. Not retried."""

    status = LlmCallStatus.LLM_CALL_STATUS_INVALID_RESPONSE


class ModelNotAllowed(LlmError):
    """The call's purpose may not use the model it named (the model allowlist, spec §5.1).

    The client raises it before any I/O: no call was made, and no record exists. Its status is
    nominal, since no record carries it: that of a request the gateway refuses.
    """

    status = LlmCallStatus.LLM_CALL_STATUS_CLIENT_ERROR


_BY_STATUS: dict[LlmCallStatus.ValueType, type[LlmError]] = {
    error.status: error
    for error in (
        ContextOverflow,
        RateLimited,
        ServerError,
        TimedOut,
        ConnectionFailed,
        ClientError,
        InvalidResponse,
    )
}


def error_type(status: LlmCallStatus.ValueType) -> type[LlmError]:
    """Return the error a call raises when it fails with ``status``."""
    try:
        return _BY_STATUS[status]
    except KeyError:
        raise ValueError(f"{LlmCallStatus.Name(status)} is not a failure") from None
