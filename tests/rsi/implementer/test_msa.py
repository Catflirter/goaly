"""Tests for goaly.rsi.implementer.msa: importing mini-swe-agent with its side effects contained."""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from goaly.rsi.implementer import msa
from tests.rsi.implementer.fakes import bash_call

REPO_ROOT = Path(__file__).resolve().parents[3]

# Run in a fresh interpreter, where nothing has imported minisweagent yet.
PROBE = """\
import json, logging, os, sys
before = dict(os.environ)
import goaly.rsi.implementer.msa
logger = logging.getLogger("minisweagent")
print(json.dumps({
    "environ_unchanged": dict(os.environ) == before,
    "handlers": len(logger.handlers),
    "propagate": logger.propagate,
    "level": logger.level,
    "heavy": sorted(
        name for name in ("litellm", "openai", "datasets", "textual", "prompt_toolkit")
        if name in sys.modules
    ),
}))
"""


def test_importing_mini_swe_agent_changes_no_environment_variable(tmp_path: Path) -> None:
    # A .env where mini-swe-agent looks by default, and one where a caller's
    # MSWEA_GLOBAL_CONFIG_DIR points: neither may reach os.environ.
    xdg = tmp_path / "xdg"
    (xdg / "mini-swe-agent").mkdir(parents=True)
    (xdg / "mini-swe-agent" / ".env").write_text("PLANTED_DEFAULT=1\n", encoding="utf-8")
    theirs = tmp_path / "theirs"
    theirs.mkdir()
    (theirs / ".env").write_text("PLANTED_THEIRS=1\nMSWEA_SILENT_STARTUP=\n", encoding="utf-8")
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path / "home"),
        "XDG_CONFIG_HOME": str(xdg),
        "MSWEA_GLOBAL_CONFIG_DIR": str(theirs),
        "TMPDIR": str(scratch),
    }

    probe = subprocess.run(  # noqa: S603  (this interpreter, with a fixed script)
        [sys.executable, "-c", PROBE],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )

    assert probe.stderr == ""
    # One line of JSON, so no banner was printed.
    lines = probe.stdout.splitlines()
    assert len(lines) == 1, probe.stdout
    assert json.loads(lines[0]) == {
        "environ_unchanged": True,
        "handlers": 0,
        "propagate": True,
        "level": logging.NOTSET,
        "heavy": [],
    }
    # The config directory mini-swe-agent used was goaly's own, and it is gone.
    assert list(scratch.iterdir()) == []
    assert sorted(path.name for path in theirs.iterdir()) == [".env"]


def test_the_contained_import_puts_back_every_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MSWEA_SILENT_STARTUP", "")
    monkeypatch.delenv("MSWEA_GLOBAL_CONFIG_DIR", raising=False)
    before = dict(os.environ)

    with msa._contained_import():
        assert os.environ["MSWEA_SILENT_STARTUP"] == "1"
        config_dir = Path(os.environ["MSWEA_GLOBAL_CONFIG_DIR"])
        assert list(config_dir.iterdir()) == []
        os.environ["LOADED_FROM_DOTENV"] = "1"

    assert dict(os.environ) == before
    assert not config_dir.exists()


def test_the_minisweagent_logger_reaches_the_root_handlers(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        logging.getLogger("minisweagent.environment").info("started a container")

    assert [(record.name, record.message) for record in caplog.records] == [
        ("minisweagent.environment", "started a container")
    ]
    assert logging.getLogger("minisweagent").handlers == []


def test_parse_toolcall_actions_reads_bash_calls() -> None:
    actions = msa.parse_toolcall_actions(
        [bash_call("ls", "call_a"), bash_call("pwd", "call_b")],
        format_error_template="{{ error }}",
        finish_reason="tool_calls",
    )

    assert actions == [
        {"command": "ls", "tool_call_id": "call_a"},
        {"command": "pwd", "tool_call_id": "call_b"},
    ]


@pytest.mark.parametrize(
    ("tool_calls", "error"),
    [
        pytest.param([], "No tool calls found", id="no-call"),
        pytest.param(
            [{"id": "c", "function": {"name": "python", "arguments": '{"command": "ls"}'}}],
            "Unknown tool 'python'",
            id="other-tool",
        ),
        pytest.param(
            [{"id": "c", "function": {"name": "bash", "arguments": "{not json"}}],
            "Error parsing tool call arguments",
            id="bad-json",
        ),
    ],
)
def test_parse_toolcall_actions_raises_a_format_error_for_the_model(
    tool_calls: list[dict[str, object]], error: str
) -> None:
    with pytest.raises(msa.FormatError) as raised:
        msa.parse_toolcall_actions(
            tool_calls, format_error_template="{{ error }} ({{ finish_reason }})", finish_reason="x"
        )

    (message,) = raised.value.messages
    assert message["role"] == "user"
    assert error in message["content"]
    assert message["content"].endswith("(x)")


def test_format_toolcall_observations_answers_each_call() -> None:
    actions = [{"command": "ls", "tool_call_id": "a"}, {"command": "pwd", "tool_call_id": "b"}]

    messages = msa.format_toolcall_observations(
        actions,
        [{"output": "x.py\n", "returncode": 0, "exception_info": ""}],
        observation_template="{{ output.returncode }}:{{ output.output }}{{ tag }}",
        template_vars={"tag": "!"},
    )

    assert [(m["role"], m["tool_call_id"], m["content"]) for m in messages] == [
        ("tool", "a", "0:x.py\n!"),
        ("tool", "b", "-1:!"),  # Not run: the agent stopped before it.
    ]


def test_load_yaml_reads_a_document(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("limits:\n  step_limit: 60\n", encoding="utf-8")

    assert msa.load_yaml(path) == {"limits": {"step_limit": 60}}
