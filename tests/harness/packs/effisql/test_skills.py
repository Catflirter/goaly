"""Tests for the pack's skill guides: the file and the text with_skills() carries never drift.

Pack code may not read files (GUARD_PACK_CODE_ALLOWLIST), so policy.py holds each guide's text as
a constant and skills/<name>.md commits the same text; these tests compare the two, so changing
one without the other fails. They also hold the guides to what the RSI spec asks of a skill
(spec §5.2): short, mechanical, naming no task's SQL, and never wording a rule as a look at the
rows the model happens to see, which an earlier prompt checklist did and which nudged the model
into handing back literal outputs.
"""

from pathlib import Path

import pytest

from goaly.harness.packs import effisql
from goaly.harness.packs.effisql import policy
from goaly.harness.packs.effisql.policy import (
    EQUIVALENCE_INVARIANTS,
    SKILLS,
    build_policy,
    with_skills,
)

PACK = Path(effisql.__file__).parent
GUIDE = PACK / "skills" / "equivalence-invariants.md"
# A guide rides in the system prompt of every task, so it stays this short.
MAX_LINES = 12


def prompt() -> str:
    return (PACK / "prompt.md").read_text(encoding="utf-8")


def folded(text: str) -> str:
    return " ".join(text.split())


class TestTheGuideAndItsFile:
    def test_the_file_holds_the_carried_text(self) -> None:
        assert GUIDE.read_text(encoding="utf-8") == EQUIVALENCE_INVARIANTS

    def test_the_guide_is_the_one_carried(self) -> None:
        assert SKILLS == (EQUIVALENCE_INVARIANTS,)

    def test_the_guide_is_short(self) -> None:
        assert len(EQUIVALENCE_INVARIANTS.splitlines()) <= MAX_LINES


class TestWithSkills:
    def test_the_guides_follow_the_prompt(self) -> None:
        assert with_skills(prompt()) == (
            prompt().rstrip() + "\n\nSkill guides:\n\n" + EQUIVALENCE_INVARIANTS
        )

    def test_a_prompt_with_no_guide_is_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(policy, "SKILLS", ())

        assert with_skills(prompt()) == prompt()

    def test_the_policy_opens_every_chat_with_the_guides(self) -> None:
        assert build_policy({}, prompt()).system_prompt == with_skills(prompt())


class TestTheEquivalenceInvariants:
    def test_it_teaches_the_three_shapes(self) -> None:
        text = folded(EQUIVALENCE_INVARIANTS)

        for rule in ("STAR EXPANSION", "AGGREGATE TYPE", "TIES UNDER LIMIT"):
            assert rule in text

    def test_it_names_no_task_sql(self) -> None:
        text = folded(EQUIVALENCE_INVARIANTS)

        assert "FROM" not in text
        assert ";" not in text

    def test_it_points_at_no_rows_the_model_sees(self) -> None:
        text = folded(EQUIVALENCE_INVARIANTS).lower()

        assert "your rows" not in text
        assert "compare" not in text
