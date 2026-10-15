"""A fake docker CLI for the sandbox unit tests, which run without Docker.

It records each call, with its environment, as one JSON line in ``calls.jsonl``, and answers from
``script.json``: for each subcommand, what to print, how long to take and how to exit.
"""

import json
import stat
import sys
from pathlib import Path
from typing import Any

SCRIPT = """\
#!{python}
import json, os, sys, time
from pathlib import Path

here = Path({here!r})
args = sys.argv[1:]
command = " ".join(args[:2]) if args[:1] == ["image"] else args[0]
context = None
if command == "build":
    context = sorted(p.name for p in Path(args[-1]).iterdir())
with (here / "calls.jsonl").open("a") as calls:
    calls.write(json.dumps({{"args": args, "env": dict(os.environ), "context": context}}) + "\\n")
answer = json.loads((here / "script.json").read_text()).get(command, {{}})
time.sleep(answer.get("sleep", 0))
sys.stdout.write(answer.get("stdout", ""))
sys.stderr.write(answer.get("stderr", ""))
sys.exit(answer.get("exit", 0))
"""


class FakeDocker:
    """A docker executable in ``directory`` that answers as ``script`` says."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "docker"
        self.path.write_text(
            SCRIPT.format(python=sys.executable, here=str(directory)), encoding="utf-8"
        )
        self.path.chmod(self.path.stat().st_mode | stat.S_IXUSR)
        self.script({})

    def script(self, answers: dict[str, dict[str, Any]]) -> None:
        """Set how each subcommand answers: ``{"wait": {"stdout": "0\\n"}, ...}``."""
        (self.directory / "script.json").write_text(json.dumps(answers), encoding="utf-8")

    def calls(self) -> list[dict[str, Any]]:
        """Every call so far, in order: its arguments, environment and build context."""
        log = self.directory / "calls.jsonl"
        if not log.exists():
            return []
        return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]

    def commands(self) -> list[str]:
        """The subcommand of each call so far, as ``run``, ``wait`` or ``image ls``."""
        return [
            " ".join(call["args"][:2]) if call["args"][:1] == ["image"] else call["args"][0]
            for call in self.calls()
        ]
