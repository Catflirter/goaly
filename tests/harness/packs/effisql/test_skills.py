"""Tests for the pack's skills: what they hold, and that they reach the executor.

Pack code may open no file (GUARD_PACK_CODE_ALLOWLIST refuses pathlib and open), and the harness
hands the policy prompt.md alone as its system prompt, so a skill reaches the executor only
inlined in prompt.md, as spec 5.2's first option has it: test_policy checks that prompt.md is
the system prompt, and these tests check that every skill is part of prompt.md and stays short,
so it costs little on every call of every task.
"""

from pathlib import Path

from goaly.harness.packs import effisql

PACK = Path(effisql.__file__).parent
SKILLS = PACK / "skills"

# Spec 5.2 keeps skills "few and focused"; a skill is one template of at most a dozen lines,
# so the prompt that carries it stays cheap (an earlier, longer skill cost more than it gained).
MAX_SKILLS = 3
MAX_LINES = 12


def skill_files() -> list[Path]:
    """The pack's skills, by name: every .md file directly under skills/."""
    return sorted(path for path in SKILLS.glob("*.md") if path.is_file())


class TestSkills:
    def test_the_pack_holds_few_short_skills(self) -> None:
        files = skill_files()

        assert files, "the pack ships at least one skill"
        assert len(files) <= MAX_SKILLS
        for path in files:
            assert len(path.read_text(encoding="utf-8").splitlines()) <= MAX_LINES, path.name

    def test_every_skill_is_inlined_in_the_system_prompt(self) -> None:
        prompt = " ".join((PACK / "prompt.md").read_text(encoding="utf-8").split())

        for path in skill_files():
            skill = " ".join(path.read_text(encoding="utf-8").split())
            assert skill in prompt, f"{path.name} reaches the executor only through prompt.md"


class TestTopKLateral:
    def test_the_skill_teaches_the_lateral_shape(self) -> None:
        text = (SKILLS / "topk-lateral.md").read_text(encoding="utf-8")

        assert "never merge the two aggregates into one GROUP BY over the largest join" in text
        assert "keep subquery (a) unchanged as a derived table t" in text
        assert "correlates on the group key" in text
        assert "k groups instead of all groups" in text

    def test_the_skill_is_one_template(self) -> None:
        text = (SKILLS / "topk-lateral.md").read_text(encoding="utf-8")

        assert text.count("CROSS JOIN LATERAL") == 1
        assert text.count("When a query") == 1

    def test_the_skill_carries_its_guards(self) -> None:
        text = (SKILLS / "topk-lateral.md").read_text(encoding="utf-8")

        assert "must GROUP BY exactly the correlation key" in text
        assert "must be unique, or rows are lost" in text
        assert "at least 1.2x faster than the base" in text
