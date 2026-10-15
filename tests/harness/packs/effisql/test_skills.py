"""Tests for the pack's skill guides: what they say, and that the executor reads them.

Pack code may not read files, so policy.py holds each guide's text as well as skills/*.md. These
tests keep the two the same and check the guides stay short, generic and carried in the prompt.
"""

import re
from pathlib import Path

import yaml

from goaly.bench.effisql.pg_build import TPCH_TABLES
from goaly.harness.packs import effisql
from goaly.harness.packs.effisql.policy import DECORRELATE_JOIN, SKILLS, build_policy, with_skills

PACK = Path(effisql.__file__).parent
GUIDES = PACK / "skills"


def prompt_md() -> str:
    """prompt.md, as the harness factory reads it."""
    return (PACK / "prompt.md").read_text(encoding="utf-8")


def guide_files() -> list[Path]:
    """The pack's skill guides, in name order."""
    return sorted(GUIDES.glob("*.md"), key=lambda path: path.name)


class TestGuides:
    def test_every_guide_is_a_file_and_every_file_a_guide(self) -> None:
        written = [path.read_text(encoding="utf-8") for path in guide_files()]

        assert set(written) == set(SKILLS)
        assert 1 <= len(SKILLS) <= 3  # Spec §5.2: few and focused.

    def test_the_system_prompt_carries_every_guide_after_prompt_md(self) -> None:
        sent = with_skills(prompt_md())

        assert sent.startswith(prompt_md().rstrip() + "\n\nSkills:\n\n")
        for guide in SKILLS:
            assert guide.rstrip() in sent
        assert sent.endswith("\n")

    def test_the_policy_sends_the_guides(self) -> None:
        loaded = yaml.safe_load((PACK / "config.yaml").read_text(encoding="utf-8"))
        pack = loaded.get("pack")
        assert isinstance(pack, dict)

        policy = build_policy(pack, prompt_md())

        assert policy.system_prompt == with_skills(prompt_md())


class TestDecorrelateJoin:
    def test_the_guide_is_one_short_template(self) -> None:
        lines = DECORRELATE_JOIN.splitlines()

        assert len(lines) <= 12
        assert DECORRELATE_JOIN.count("rewrite as") == 1

    def test_the_guide_keeps_the_rewrite_safe(self) -> None:
        # The two lines that keep a misapplied template from unmatched rows.
        assert "LEFT JOIN" in DECORRELATE_JOIN
        assert "unique in the subquery's table" in DECORRELATE_JOIN

    def test_the_template_names_no_table_of_the_benchmark(self) -> None:
        words = set(re.findall(r"[a-z]+", DECORRELATE_JOIN.lower()))

        assert words.isdisjoint(TPCH_TABLES)
