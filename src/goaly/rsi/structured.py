"""Structured GLM calls: a chat whose answer must be one JSON object of a pydantic model.

goaly.rsi asks GLM for structured answers twice a round: the analyst's hypotheses (spec §8.2
step 1) and the lesson writer's lessons (step 8). Both go through :func:`complete_structured`:

1. The JSON Schema of the model goes into the prompt, at the end of the last user message. The
   gateway's ``response_format`` is not used: it does not enforce a schema reliably.
2. The reply's content is parsed as the model's JSON, inside a ```json fence or bare.
3. A reply that does not validate is answered in the same chat: the assistant's reply goes back
   exactly as it came, reasoning included (spec §5.1), followed by a user message that quotes the
   validation errors. That is a retry, up to ``max_attempts`` calls in all.

The caller binds the client to its purpose and run (LlmClient.bind), so every attempt is
recorded under them. A gateway failure, which the client already retried, raises its LlmError
unchanged; output that is still invalid after the last attempt raises StructuredOutputError.
"""

import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from pydantic import BaseModel, ValidationError

from goaly.contracts.gen.llm_call_pb2 import Usage
from goaly.contracts.gen.trajectory_pb2 import ChatMessage, ChatRole
from goaly.harness.core.llm import BoundLlm, LlmReply
from goaly.harness.core.llm_models import ModelParams

__all__ = [
    "DEFAULT_MAX_ATTEMPTS",
    "StructuredOutputError",
    "StructuredReply",
    "add_usage",
    "complete_structured",
    "format_instructions",
]

DEFAULT_MAX_ATTEMPTS: Final = 3
# A fenced block: its language label, if any, and its body. Matched left to right, so a closing
# fence is never taken for an opening one.
_FENCE = re.compile(r"```([\w+-]*)[ \t]*\n(.*?)\n?```", re.DOTALL)
_ERRORS_SHOWN: Final = 20  # How many validation errors a retry quotes.
_INPUT_QUOTED: Final = 200  # How much of an invalid value an error quotes.
_CUT_OFF: Final = (
    "The answer was cut off at the token limit. Think more briefly, and write a shorter answer."
)

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class StructuredReply[M: BaseModel]:
    """A valid answer: the model instance, and the calls it took, in order, with their usage."""

    value: M
    call_ids: tuple[str, ...]
    usage: Usage


class StructuredOutputError(Exception):
    """Every attempt's reply was invalid. ``errors`` holds the last attempt's problems."""

    def __init__(self, message: str, *, call_ids: Sequence[str], usage: Usage, errors: str) -> None:
        super().__init__(message)
        self.call_ids = tuple(call_ids)
        self.usage = usage
        self.errors = errors


def add_usage(total: Usage, part: Usage) -> None:
    """Add ``part`` into ``total``, as Usage sums: an optional field is set if either has it."""
    total.input_tokens += part.input_tokens
    total.output_tokens += part.output_tokens
    if part.HasField("reasoning_tokens"):
        total.reasoning_tokens = total.reasoning_tokens + part.reasoning_tokens
    if part.HasField("cost_usd"):
        total.cost_usd = total.cost_usd + part.cost_usd


async def complete_structured[M: BaseModel](
    llm: BoundLlm,
    messages: Sequence[ChatMessage],
    schema: type[M],
    *,
    context: Mapping[str, Any] | None = None,
    params: ModelParams | None = None,
    attempt_timeout_s: float | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> StructuredReply[M]:
    """Ask ``llm`` for one JSON object of ``schema`` and return it validated.

    ``messages`` must end with a user message; the schema is appended to a copy of it.
    ``context`` is pydantic's validation context, for checks that depend on the call, such as
    which task IDs exist. ``params`` and ``attempt_timeout_s`` go with every call, as
    LlmClient.complete takes them: a long answer needs both a larger max_tokens and more than the
    default 300 s per attempt. Raises StructuredOutputError after ``max_attempts`` invalid replies,
    and the LlmError of a call that failed.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be at least 1, not {max_attempts}")
    if not messages or messages[-1].role != ChatRole.CHAT_ROLE_USER:
        raise ValueError("a structured call's messages must end with a user message")
    last = ChatMessage()
    last.CopyFrom(messages[-1])
    last.content = f"{last.content}\n\n{format_instructions(schema)}"
    chat = [*messages[:-1], last]
    call_ids: list[str] = []
    usage = Usage()
    problems = ""
    for attempt in range(1, max_attempts + 1):
        reply = await llm.complete(chat, params=params, attempt_timeout_s=attempt_timeout_s)
        call_ids.append(reply.record.call_id)
        add_usage(usage, reply.usage)
        value, problems = _validate(reply, schema, context)
        if value is not None:
            return StructuredReply(value, tuple(call_ids), usage)
        _logger.warning(
            "a structured reply was invalid",
            extra={
                "event": "structured_reply_invalid",
                "schema": schema.__name__,
                "attempt": attempt,
                "call_id": reply.record.call_id,
                "problems": problems,
            },
        )
        chat += [reply.message, ChatMessage(role=ChatRole.CHAT_ROLE_USER, content=_retry(problems))]
    raise StructuredOutputError(
        f"{max_attempts} replies in a row were no valid {schema.__name__}",
        call_ids=call_ids,
        usage=usage,
        errors=problems,
    )


def format_instructions(schema: type[BaseModel]) -> str:
    """Return what complete_structured appends to the last user message for ``schema``."""
    return (
        "Answer with exactly one JSON object and nothing else, optionally inside a ```json "
        "fence. It must validate against this JSON Schema:\n\n```json\n"
        f"{json.dumps(schema.model_json_schema(), indent=1, ensure_ascii=False)}\n```"
    )


def _retry(problems: str) -> str:
    return (
        "Your answer is not valid:\n\n"
        f"{problems}\n\n"
        "Answer again with the complete, corrected JSON object only."
    )


def _validate[M: BaseModel](
    reply: LlmReply, schema: type[M], context: Mapping[str, Any] | None
) -> tuple[M | None, str]:
    """Return the reply's value, or None and what is wrong with it, in words."""
    if reply.finish_reason == "length":
        return None, _CUT_OFF
    candidates = _json_candidates(reply.message.content)
    if not candidates:
        return None, "The answer held no JSON object: the content was empty."
    problems = ""
    for text in candidates:
        try:
            return schema.model_validate_json(text, context=dict(context or {})), ""
        except ValidationError as error:
            problems = problems or _describe(error)  # The likeliest candidate's problems.
    return None, problems


def _json_candidates(content: str) -> list[str]:
    """Return where the JSON may be in ``content``, likeliest first.

    That is: ```json blocks, then other fenced blocks, then the text from the first { to the
    last }, or else the whole content. Empty candidates are left out.
    """
    blocks = [(label.lower(), body.strip()) for label, body in _FENCE.findall(content)]
    candidates = [body for label, body in blocks if label == "json"]
    candidates += [body for label, body in blocks if label != "json"]
    start, end = content.find("{"), content.rfind("}")
    candidates.append(content[start : end + 1] if 0 <= start < end else content.strip())
    return [text for text in dict.fromkeys(candidates) if text]


def _describe(error: ValidationError) -> str:
    lines = []
    for detail in error.errors(include_url=False)[:_ERRORS_SHOWN]:
        where = ".".join(str(part) for part in detail["loc"]) or "(the whole object)"
        quoted = json.dumps(detail.get("input"), ensure_ascii=False, default=str)
        if len(quoted) > _INPUT_QUOTED:
            quoted = quoted[:_INPUT_QUOTED] + "…"
        lines.append(f"- {where}: {detail['msg']} (got {quoted})")
    hidden = error.error_count() - _ERRORS_SHOWN
    if hidden > 0:
        lines.append(f"- and {hidden} more")
    return "\n".join(lines)
