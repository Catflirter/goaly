"""Tests for the pack's skill guides: what they hold and how they reach the model.

The harness factory reads only prompt.md as the system prompt (bench's load_pack), and pack code
may read no file, so every guide in skills/ is embedded in prompt.md verbatim. These tests hold
that wiring: a guide the model never sees is a file, not a skill.
"""

from pathlib import Path

from goaly.harness.packs import effisql

PACK = Path(effisql.__file__).parent
DECORRELATE = "decorrelate-join.md"
# A guide is one focused template; the spec asks for few guides, each focused.
MAX_LINES = 12
MAX_GUIDES = 3


def guides() -> list[Path]:
    """Every skill guide the pack holds, sorted by name."""
    return sorted((PACK / "skills").glob("*.md"))


def guide_text(name: str) -> str:
    """A guide's text without its trailing newline, as prompt.md embeds it."""
    return (PACK / "skills" / name).read_text(encoding="utf-8").rstrip("\n")


class TestSkillGuides:
    def test_the_pack_holds_the_decorrelation_guide(self) -> None:
        assert 1 <= len(guides()) <= MAX_GUIDES
        assert DECORRELATE in {path.name for path in guides()}

    def test_every_guide_is_one_short_template(self) -> None:
        for path in guides():
            lines = guide_text(path.name).splitlines()

            assert len(lines) <= MAX_LINES
            # One template a guide, and a shape, never a task's own query.
            assert lines[0].startswith("# ")
            assert len(guide_text(path.name).split("rewrite as")) == 2

    def test_every_guide_reaches_the_model_through_the_system_prompt(self) -> None:
        system_prompt = (PACK / "prompt.md").read_text(encoding="utf-8")

        for path in guides():
            assert guide_text(path.name) in system_prompt

    def test_the_decorrelation_guide_states_the_rewrite(self) -> None:
        body = " ".join(guide_text(DECORRELATE).split())

        for phrase in (
            "correlated scalar subquery",
            "rewrite as SELECT t.*, s.expr FROM (outer) t LEFT JOIN",
            "ON (s.k, s.k2) = (t.k, t.k2)",
            "keep NULL like the subquery does",
            "GROUP BY the correlation columns in the derived table",
        ):
            assert phrase in body
