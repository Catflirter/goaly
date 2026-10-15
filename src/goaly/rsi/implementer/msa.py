"""The one module that imports mini-swe-agent (spec §8.8, §11); the rest of goaly.rsi uses it here.

Importing ``minisweagent`` has side effects, which this module contains:

- it prints a banner unless ``MSWEA_SILENT_STARTUP`` is set;
- it creates its global config directory, ``$MSWEA_GLOBAL_CONFIG_DIR`` or
  ``~/.config/mini-swe-agent``, and loads that directory's ``.env`` into ``os.environ``;
- it gives the ``minisweagent`` logger a Rich handler at DEBUG.

So the first import runs with ``MSWEA_SILENT_STARTUP=1`` and ``MSWEA_GLOBAL_CONFIG_DIR`` pointing at
a new, empty directory of goaly's own, which is removed afterwards: nothing reads it again outside
mini-swe-agent's CLI, which goaly never runs. ``os.environ`` is then put back exactly as it was, so
importing this module changes no variable, and the ``minisweagent`` logger loses its handler and
propagates to the root logger, where goaly's JSON handler is (goaly.harness.core.logs).

mini-swe-agent ships ``py.typed`` but leaves many functions unannotated, so a mypy override scoped
to this module allows calling them; this module re-exports them with types.
"""

import logging
import os
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final, cast

import yaml  # type: ignore[import-untyped]  # PyYAML ships no types; mini-swe-agent requires it.

from goaly.contracts.interfaces import JsonValue

__all__ = [
    "BASH_TOOL",
    "SUBMIT_MARKER",
    "VERSION",
    "DefaultAgent",
    "DockerEnvironment",
    "FormatError",
    "InterruptAgentFlow",
    "LimitsExceeded",
    "Submitted",
    "TimeExceeded",
    "format_toolcall_observations",
    "load_yaml",
    "make_agent",
    "parse_toolcall_actions",
]

# The first line of the output with which an agent's command finishes its run.
SUBMIT_MARKER: Final = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


@contextmanager
def _contained_import() -> Iterator[None]:
    """Run the first import of minisweagent silently, with an empty config directory of our own,
    and leave ``os.environ`` exactly as it was before.
    """
    before = dict(os.environ)
    try:
        with tempfile.TemporaryDirectory(prefix="goaly-mswea-config-") as config_dir:
            os.environ["MSWEA_SILENT_STARTUP"] = "1"
            os.environ["MSWEA_GLOBAL_CONFIG_DIR"] = config_dir
            yield
    finally:
        for name in set(os.environ) - set(before):
            del os.environ[name]
        for name, value in before.items():
            if os.environ.get(name) != value:
                os.environ[name] = value


with _contained_import():
    import minisweagent
    from minisweagent import Environment, Model
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments.docker import DockerEnvironment
    from minisweagent.exceptions import (
        FormatError,
        InterruptAgentFlow,
        LimitsExceeded,
        Submitted,
        TimeExceeded,
    )
    from minisweagent.models.utils.actions_toolcall import BASH_TOOL as _BASH_TOOL
    from minisweagent.models.utils.actions_toolcall import format_toolcall_observation_messages
    from minisweagent.models.utils.actions_toolcall import (
        parse_toolcall_actions as _parse_toolcall_actions,
    )

VERSION: Final = minisweagent.__version__
# The agent's one tool, bash, as an OpenAI tool entry for the client's ``tools``.
BASH_TOOL: Final = cast("Mapping[str, JsonValue]", _BASH_TOOL)  # Inferred as Collection[str].


def _route_logger() -> None:
    """Send the ``minisweagent`` logger's records to the root logger's handlers alone."""
    logger = logging.getLogger("minisweagent")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(logging.NOTSET)  # The root logger's level decides, as for goaly's loggers.
    logger.propagate = True


_route_logger()


class _Function:
    """A tool call's function, shaped as mini-swe-agent's parser reads it."""

    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _ToolCall:
    """A tool call, shaped as mini-swe-agent's parser reads it: ``id`` and ``function``."""

    def __init__(self, call_id: str, name: str, arguments: str) -> None:
        self.id = call_id
        self.function = _Function(name, arguments)


def parse_toolcall_actions(
    tool_calls: Sequence[Mapping[str, Any]],
    *,
    format_error_template: str,
    finish_reason: str,
) -> list[dict[str, Any]]:
    """Return the bash actions of OpenAI-format ``tool_calls``, as mini-swe-agent's agent runs
    them: ``{"command": ..., "tool_call_id": ...}`` each.

    Raises FormatError, whose one message tells the model what was wrong, if there is no call, a
    call is not to the bash tool, or its arguments hold no command. ``finish_reason`` lets the
    template tell a reply cut off at max_tokens from a malformed one.
    """
    calls = [
        _ToolCall(call["id"], call["function"]["name"], call["function"]["arguments"])
        for call in tool_calls
    ]
    actions: list[dict[str, Any]] = _parse_toolcall_actions(
        calls,
        format_error_template=format_error_template,
        template_kwargs={"finish_reason": finish_reason},
    )
    return actions


def format_toolcall_observations(
    actions: Sequence[Mapping[str, Any]],
    outputs: Sequence[Mapping[str, Any]],
    *,
    observation_template: str,
    template_vars: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Return one tool message for each of ``actions``, rendering its output with
    ``observation_template``; an action that did not run gets an output that says so.
    """
    messages: list[dict[str, Any]] = format_toolcall_observation_messages(
        actions=[dict(action) for action in actions],
        outputs=[dict(output) for output in outputs],
        observation_template=observation_template,
        template_vars=None if template_vars is None else dict(template_vars),
    )
    return messages


def make_agent(
    model: Model,
    environment: Environment,
    *,
    system_template: str,
    instance_template: str,
    max_consecutive_format_errors: int,
    output_path: Path | None = None,
) -> DefaultAgent:
    """Return mini-swe-agent's non-interactive agent on ``model`` and ``environment``.

    It stops after ``max_consecutive_format_errors`` unparsable replies in a row, and saves its
    trajectory to ``output_path`` after every step. Its own step, cost and wall-time limits are
    off: the model adapter enforces them, each with an exit status of its own.
    """
    return DefaultAgent(
        model,
        environment,
        system_template=system_template,
        instance_template=instance_template,
        step_limit=0,
        cost_limit=0,
        wall_time_limit_seconds=0,
        max_consecutive_format_errors=max_consecutive_format_errors,
        output_path=output_path,
    )


def load_yaml(path: Path) -> object:
    """Return the YAML document at ``path``, read with PyYAML's safe loader.

    It reads ``path`` and nothing else: mini-swe-agent's own config lookup would fall back to its
    built-in configs. Raises FileNotFoundError if ``path`` does not exist, and yaml.YAMLError if
    it is not YAML.
    """
    document: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    return document
