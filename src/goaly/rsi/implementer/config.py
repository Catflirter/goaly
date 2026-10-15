"""The implementer's configuration, ``implementer.yaml``: its limits and templates (spec §8.8)."""

from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt

from goaly.rsi.implementer.msa import load_yaml

__all__ = [
    "DEFAULT_CONFIG",
    "AgentTemplates",
    "ImplementerConfig",
    "Limits",
    "ModelTemplates",
    "load_config",
]

DEFAULT_CONFIG: Final = Path(__file__).with_name("implementer.yaml")


class Limits(BaseModel):
    """One implementer run's limits."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    # Model calls; the model adapter enforces it, and the next three.
    step_limit: PositiveInt
    # The GLM cost, as the gateway reports it for each call.
    cost_limit_usd: PositiveFloat
    # From the run's first model call.
    wall_time_limit_s: PositiveFloat
    # Replies in a row that the agent cannot parse; mini-swe-agent's agent enforces it.
    max_consecutive_format_errors: PositiveInt
    # Each command, inside the sandbox.
    command_timeout_s: PositiveFloat


class ModelTemplates(BaseModel):
    """The Jinja templates with which the model adapter shows the agent what happened."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # A command's output, as the tool message that answers its call; it gets ``output``.
    observation_template: str
    # A reply the agent cannot parse; it gets ``error``, ``has_tool_calls`` and ``finish_reason``.
    format_error_template: str


class AgentTemplates(BaseModel):
    """The prompt: the Jinja templates of the agent's first two messages."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    system_template: str
    # The hypothesis, the allowed paths, the pack's layout and rules, and how to finish.
    instance_template: str


class ImplementerConfig(BaseModel):
    """``implementer.yaml``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    limits: Limits
    model: ModelTemplates
    agent: AgentTemplates


def load_config(path: Path = DEFAULT_CONFIG) -> ImplementerConfig:
    """Load and validate the implementer's configuration.

    Raises FileNotFoundError if ``path`` does not exist, and pydantic.ValidationError if it is not
    a valid configuration.
    """
    return ImplementerConfig.model_validate(load_yaml(path))
