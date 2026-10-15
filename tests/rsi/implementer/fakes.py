"""Doubles for the implementer tests: a fake sandbox environment and scripted bash calls.

pytest puts the repository root on sys.path, so tests import these as
``tests.rsi.implementer.fakes``.
"""

import json
from collections.abc import Callable, Mapping
from typing import Any, Final

from goaly.rsi.implementer import msa

SUBMIT = f"echo {msa.SUBMIT_MARKER}"

# The agent templates the unit tests run with; implementer.yaml holds the real ones.
SYSTEM_TEMPLATE: Final = "You implement hypotheses. You have {{ step_limit }} steps."
INSTANCE_TEMPLATE: Final = "Hypothesis: {{ task }}"

type Handler = Callable[[str], Mapping[str, Any]]


def bash_call(command: str, call_id: str = "call_1") -> dict[str, Any]:
    """A bash tool call, as the gateway sends one."""
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "bash", "arguments": json.dumps({"command": command})},
    }


class FakeEnvironment:
    """A mini-swe-agent environment that runs nothing: it records each command and answers it
    with ``handler``'s output, or with an empty success. A command that echoes the submit marker
    finishes the run, as the sandbox's does, with the rest of its line as the submission.
    """

    def __init__(self, handler: Handler | None = None) -> None:
        self.config: dict[str, Any] = {}
        self.commands: list[str] = []
        self._handler = handler

    def execute(self, action: dict[str, Any], cwd: str = "") -> dict[str, Any]:
        command = action["command"]
        self.commands.append(command)
        if command.startswith(SUBMIT):
            submission = command.removeprefix(SUBMIT).strip()
            raise msa.Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )
        if self._handler is None:
            return {"output": "", "returncode": 0, "exception_info": ""}
        return {"output": "", "returncode": 0, "exception_info": "", **self._handler(command)}

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        return dict(kwargs)

    def serialize(self) -> dict[str, Any]:
        return {"info": {"config": {"environment": {}, "environment_type": "FakeEnvironment"}}}
