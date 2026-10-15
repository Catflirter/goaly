"""Doubles for the implementer tests: a fake sandbox environment and scripted bash calls.

pytest puts the repository root on sys.path, so tests import these as
``tests.rsi.implementer.fakes``.
"""

import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final, Self

from goaly.rsi.implementer import msa

# After msa, which makes the first import of minisweagent.
from minisweagent.models.test_models import (  # isort: skip
    DeterministicToolcallModel,
    make_toolcall_output,
)

SUBMIT = f"echo {msa.SUBMIT_MARKER}"


def submit(summary: str = "") -> str:
    """The command with which the agent finishes, with ``summary`` on the lines after the marker."""
    return f"printf '%s\\n' {msa.SUBMIT_MARKER} '{summary}'"


# The agent templates the unit tests run with; implementer.yaml holds the real ones.
SYSTEM_TEMPLATE: Final = "You implement hypotheses. You have {{ step_limit }} steps."
INSTANCE_TEMPLATE: Final = "Hypothesis: {{ task }}"

type Handler = Callable[[str], dict[str, Any]]


def bash_call(command: str, call_id: str = "call_1") -> dict[str, Any]:
    """A bash tool call, as the gateway sends one."""
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "bash", "arguments": json.dumps({"command": command})},
    }


class FakeEnvironment:
    """A mini-swe-agent environment that runs nothing: it records each command and answers it
    with ``handler``'s output, or with an empty success; without a handler, ``SUBMIT`` echoes
    what follows it. Output whose first line is the submit marker finishes the run, as in the
    sandbox, with the lines after it as the submission.

    As a context manager it stands in for the sandbox: leaving it counts as a cleanup.
    """

    def __init__(self, handler: Handler | None = None) -> None:
        self.config: dict[str, Any] = {}
        self.commands: list[str] = []
        self.cleanups = 0
        self._handler = handler

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.cleanups += 1

    def execute(self, action: dict[str, Any], cwd: str = "") -> dict[str, Any]:
        command = action["command"]
        self.commands.append(command)
        output: dict[str, Any] = {"output": "", "returncode": 0, "exception_info": ""}
        if self._handler is not None:
            output.update(self._handler(command))
        elif command.startswith(SUBMIT):
            output["output"] = command.removeprefix("echo ")
        # As the sandbox does: output whose first line is the marker finishes the run.
        lines = output["output"].lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == msa.SUBMIT_MARKER and output["returncode"] == 0:
            submission = "".join(lines[1:])
            raise msa.Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )
        return output

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        return dict(kwargs)

    def serialize(self) -> dict[str, Any]:
        return {"info": {"config": {"environment": {}, "environment_type": "FakeEnvironment"}}}


def shell(worktree: Path) -> Handler:
    """A handler that runs each command with bash in ``worktree``, as the sandbox would, but on
    the host and unconfined: for tests that need the agent's edits to land.
    """

    def run(command: str) -> dict[str, Any]:
        completed = subprocess.run(  # noqa: S603  (bash, with the test's own scripted commands)
            ["bash", "-c", command],  # noqa: S607
            cwd=worktree,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return {"output": completed.stdout + completed.stderr, "returncode": completed.returncode}

    return run


class ScriptedModel:
    """mini-swe-agent's DeterministicToolcallModel, making one bash call per reply, except that
    an exception in ``script`` is raised in its reply's place, as when a model call fails or a
    limit ends the run. The exceptions stay out of the model's config, which the trajectory holds.
    """

    def __init__(self, *script: str | BaseException) -> None:
        self._script = list(script)
        outputs = []
        for number, command in enumerate(script, start=1):
            if isinstance(command, str):
                call_id = f"call_{number}"
                actions = [{"command": command, "tool_call_id": call_id}]
                outputs.append(make_toolcall_output(None, [bash_call(command, call_id)], actions))
        self._model = DeterministicToolcallModel(outputs=outputs)  # type: ignore[no-untyped-call]  # Its __init__ is unannotated.
        self.config = self._model.config

    def query(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        step = self._script.pop(0)
        if isinstance(step, BaseException):
            raise step
        reply: dict[str, Any] = self._model.query(messages, **kwargs)
        return reply

    def format_message(self, **kwargs: Any) -> dict[str, Any]:
        message: dict[str, Any] = self._model.format_message(**kwargs)
        return message

    def format_observation_messages(
        self,
        message: dict[str, Any],
        outputs: list[dict[str, Any]],
        template_vars: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = self._model.format_observation_messages(
            message, outputs, template_vars
        )
        return messages

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        variables: dict[str, Any] = self._model.get_template_vars(**kwargs)
        return variables

    def serialize(self) -> dict[str, Any]:
        data: dict[str, Any] = self._model.serialize()
        return data


class NoLlm:
    """An LLM client for runs whose scripted model never calls one."""

    def complete_sync(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the scripted model makes no call")
