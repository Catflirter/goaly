"""The model allowlist and the model profiles (spec §5.1).

The allowlist gives each call purpose the one model it may use: the executor runs on Qwen alone,
and only RSI calls GLM. It lives in core, which RSI cannot edit (spec §8.5), so no change RSI
makes can lean on the improvement model at runtime.

A profile holds the request parameters every call to its model sends, and a call may override
them field by field.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from goaly.contracts.gen.llm_call_pb2 import CallPurpose

__all__ = [
    "DEFAULT_PROFILES",
    "EXECUTOR_MODEL",
    "GLM_PROFILE",
    "IMPROVEMENT_MODEL",
    "QWEN_PROFILE",
    "ModelParams",
    "ModelProfile",
    "allowed_model",
]

EXECUTOR_MODEL: Final = "qwen3.6-35b-a3b"
IMPROVEMENT_MODEL: Final = "glm-5.3"

_ALLOWED_MODELS: Final[Mapping[CallPurpose.ValueType, str]] = MappingProxyType(
    {
        CallPurpose.CALL_PURPOSE_EXEC: EXECUTOR_MODEL,
        CallPurpose.CALL_PURPOSE_RSI_ANALYST: IMPROVEMENT_MODEL,
        CallPurpose.CALL_PURPOSE_RSI_IMPL: IMPROVEMENT_MODEL,
        CallPurpose.CALL_PURPOSE_RSI_LESSON: IMPROVEMENT_MODEL,
    }
)
# GLM always thinks, and the gateway rejects its `thinking` parameter.
_SWITCHABLE_THINKING: Final = frozenset({EXECUTOR_MODEL})


def allowed_model(purpose: CallPurpose.ValueType) -> str | None:
    """Return the one model a call with ``purpose`` may use, or None if it may use none."""
    return _ALLOWED_MODELS.get(purpose)


class ModelParams(BaseModel):
    """Request parameters: a profile's defaults, or one call's overrides of them.

    A field left None is not sent. The client sends each sampling field under its own name, and
    the thinking switches inside ``chat_template_kwargs``, the only place Qwen takes them: the
    gateway rejects ``reasoning_effort``, and nothing here can send it or a top-level
    ``enable_thinking``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    temperature: float | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, gt=0, le=1)
    top_k: int | None = Field(default=None, ge=1)
    max_tokens: int | None = Field(default=None, ge=1)
    # Qwen only: whether the model thinks before it answers.
    enable_thinking: bool | None = None
    # Qwen only: whether the chat template keeps the reasoning of earlier turns in the prompt.
    preserve_thinking: bool | None = None


class ModelProfile(BaseModel):
    """A model, with the parameters every call to it sends unless the call overrides them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    params: ModelParams = ModelParams()

    @model_validator(mode="after")
    def _check_thinking_switches(self) -> Self:
        switched = (
            self.params.enable_thinking is not None or self.params.preserve_thinking is not None
        )
        if switched and self.model not in _SWITCHABLE_THINKING:
            raise ValueError(
                f"{self.model} always thinks: enable_thinking and preserve_thinking are for "
                f"{EXECUTOR_MODEL} only"
            )
        return self

    def with_overrides(self, overrides: ModelParams | None) -> Self:
        """Return this profile with the fields that ``overrides`` sets replaced."""
        if overrides is None:
            return self
        params = self.params.model_copy(update=overrides.model_dump(exclude_none=True))
        return type(self)(model=self.model, params=params)  # Validated: GLM cannot switch.


# The defaults (spec §5.4): Qwen's recommended sampling for thinking mode, and 16K tokens per call.
QWEN_PROFILE: Final = ModelProfile(
    model=EXECUTOR_MODEL,
    params=ModelParams(
        temperature=0.6, top_p=0.95, top_k=20, max_tokens=16384, enable_thinking=True
    ),
)
GLM_PROFILE: Final = ModelProfile(model=IMPROVEMENT_MODEL)
DEFAULT_PROFILES: Final = (QWEN_PROFILE, GLM_PROFILE)
