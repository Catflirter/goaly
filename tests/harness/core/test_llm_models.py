"""Tests for goaly.harness.core.llm_models: the model allowlist and the model profiles."""

import math
from typing import Any

import pytest
from pydantic import ValidationError

from goaly.contracts.gen.llm_call_pb2 import CallPurpose
from goaly.harness.core.llm_models import (
    DEFAULT_PROFILES,
    EXECUTOR_MODEL,
    GLM_PROFILE,
    IMPROVEMENT_MODEL,
    QWEN_PROFILE,
    ModelParams,
    ModelProfile,
    allowed_model,
)

P = CallPurpose

# The allowlist (spec §5.1).


@pytest.mark.parametrize(
    ("purpose", "model"),
    [
        (P.CALL_PURPOSE_EXEC, "qwen3.6-35b-a3b"),
        (P.CALL_PURPOSE_RSI_ANALYST, "glm-5.3"),
        (P.CALL_PURPOSE_RSI_IMPL, "glm-5.3"),
        (P.CALL_PURPOSE_RSI_LESSON, "glm-5.3"),
    ],
)
def test_each_purpose_may_use_exactly_one_model(purpose: CallPurpose.ValueType, model: str) -> None:
    assert allowed_model(purpose) == model


def test_only_the_executor_may_use_the_executor_model() -> None:
    assert EXECUTOR_MODEL == "qwen3.6-35b-a3b"
    assert IMPROVEMENT_MODEL == "glm-5.3"
    users = [purpose for purpose in P.values() if allowed_model(purpose) == EXECUTOR_MODEL]
    assert users == [P.CALL_PURPOSE_EXEC]


def test_an_unspecified_purpose_may_use_no_model() -> None:
    assert allowed_model(P.CALL_PURPOSE_UNSPECIFIED) is None


def test_every_purpose_is_on_the_allowlist() -> None:
    # A purpose added to llm_call.proto must be given its model here deliberately.
    for purpose in P.values():
        if purpose != P.CALL_PURPOSE_UNSPECIFIED:
            assert allowed_model(purpose) is not None, P.Name(purpose)


# Profiles (spec §5.1, §5.4).


def test_qwen_profile_sends_the_recommended_sampling_with_thinking_on() -> None:
    expected = ModelProfile(
        model="qwen3.6-35b-a3b",
        params=ModelParams(
            temperature=0.6, top_p=0.95, top_k=20, max_tokens=16384, enable_thinking=True
        ),
    )

    assert expected == QWEN_PROFILE
    assert QWEN_PROFILE.params.preserve_thinking is None


def test_glm_profile_sets_only_a_max_tokens_that_fits_one_attempt() -> None:
    # 12K tokens take about 270 s at 45 tokens/s, GLM's speed under load: within 300 s.
    expected = ModelProfile(model="glm-5.3", params=ModelParams(max_tokens=12000))

    assert expected == GLM_PROFILE


def test_the_default_profiles_cover_every_allowed_model() -> None:
    models = {profile.model for profile in DEFAULT_PROFILES}

    assert models == {allowed_model(purpose) for purpose in P.values()} - {None}


@pytest.mark.parametrize(
    "switch", [{"enable_thinking": False}, {"enable_thinking": True}, {"preserve_thinking": True}]
)
def test_glm_thinking_cannot_be_switched(switch: dict[str, bool]) -> None:
    with pytest.raises(ValidationError, match="always thinks"):
        ModelProfile(model="glm-5.3", params=ModelParams.model_validate(switch))


def test_overrides_replace_only_the_fields_they_set() -> None:
    profile = QWEN_PROFILE.with_overrides(ModelParams(max_tokens=512, enable_thinking=False))

    assert profile == ModelProfile(
        model="qwen3.6-35b-a3b",
        params=ModelParams(
            temperature=0.6, top_p=0.95, top_k=20, max_tokens=512, enable_thinking=False
        ),
    )
    assert QWEN_PROFILE.params.max_tokens == 16384  # The profile itself is unchanged.


def test_no_overrides_keep_the_profile() -> None:
    assert QWEN_PROFILE.with_overrides(None) is QWEN_PROFILE
    assert QWEN_PROFILE.with_overrides(ModelParams()) == QWEN_PROFILE


def test_an_override_cannot_switch_glm_thinking() -> None:
    with pytest.raises(ValidationError, match="always thinks"):
        GLM_PROFILE.with_overrides(ModelParams(enable_thinking=False))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", -0.1),
        ("temperature", math.nan),
        ("temperature", math.inf),
        ("top_p", 0.0),
        ("top_p", 1.01),
        ("top_k", 0),
        ("max_tokens", 0),
        ("presence_penalty", -2.1),
        ("presence_penalty", 2.1),
    ],
)
def test_params_reject_values_out_of_range(field: str, value: float) -> None:
    with pytest.raises(ValidationError, match=field):
        ModelParams.model_validate({field: value})


@pytest.mark.parametrize("name", ["reasoning_effort", "thinking", "stream"])
def test_params_reject_any_other_parameter(name: str) -> None:
    # The gateway rejects reasoning_effort and GLM's thinking; the client never streams.
    with pytest.raises(ValidationError, match=name):
        ModelParams.model_validate({name: "low"})


def test_profiles_and_params_are_frozen() -> None:
    target: Any = QWEN_PROFILE
    with pytest.raises(ValidationError, match="frozen"):
        target.model = "glm-5.3"
    target = QWEN_PROFILE.params
    with pytest.raises(ValidationError, match="frozen"):
        target.temperature = 1.0
