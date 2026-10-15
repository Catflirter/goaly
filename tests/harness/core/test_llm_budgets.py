"""Tests for goaly.harness.core.llm_budgets: the LLM concurrency budgets and shares (spec §6.2)."""

from pathlib import Path

import pytest

from goaly.harness.core.llm_budgets import BudgetsError, LlmBudgets, load_budgets

QWEN, GLM = "qwen3.6-35b-a3b", "glm-5.3"

BUDGETS = """
[budgets]
"qwen3.6-35b-a3b" = 48
"glm-5.3" = 16

[phases.rsi]
effisql = { "qwen3.6-35b-a3b" = 32, "glm-5.3" = 8 }
background = { "qwen3.6-35b-a3b" = 8 }

[phases.final]
effisql = { "qwen3.6-35b-a3b" = 40 }
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "llm_budgets.toml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def budgets(tmp_path: Path) -> LlmBudgets:
    return load_budgets(write(tmp_path, BUDGETS))


def test_a_process_gets_the_shares_its_phase_gives_it(budgets: LlmBudgets) -> None:
    assert budgets.share("rsi", "effisql") == {QWEN: 32, GLM: 8}
    assert budgets.share("rsi", "background") == {QWEN: 8}
    assert budgets.share("final", "effisql") == {QWEN: 40}


def test_a_share_is_the_callers_own_copy(budgets: LlmBudgets) -> None:
    budgets.share("rsi", "effisql")[QWEN] = 48

    assert budgets.share("rsi", "effisql") == {QWEN: 32, GLM: 8}


def test_there_is_no_fallback_to_a_models_whole_budget(budgets: LlmBudgets) -> None:
    # A process the phase does not name gets no share: an error, never the whole budget.
    with pytest.raises(BudgetsError, match="'final' gives no share to 'background'") as found:
        budgets.share("final", "background")
    assert "effisql" in str(found.value)
    # A model the phase gives the process no share of is absent: the process must not call it.
    assert GLM not in budgets.share("final", "effisql")


def test_an_unknown_phase_names_the_phases(budgets: LlmBudgets) -> None:
    with pytest.raises(BudgetsError, match="no phase 'baseline'; the phases are: final, rsi"):
        budgets.share("baseline", "effisql")


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        (
            "[budgets]\nm = 10\n[phases.p]\na = { m = 6 }\nb = { m = 5 }\n",
            "phase 'p' gives out 11 of 'm', whose budget is 10",
        ),
        ("[budgets]\nm = 10\n[phases.p]\na = { other = 1 }\n", "'other', which has no budget"),
        ("[budgets]\nm = 0\n[phases]\n", "greater than 0"),
        ("[budgets]\nm = 10\n[phases.p]\na = { m = 0 }\n", "greater than 0"),
        ("[budgets]\nm = 10\n[phases.p]\na = { m = 2.5 }\n", "valid integer"),
        ("[budgets]\nm = 10\n[phases]\n[extra]\nx = 1\n", "Extra inputs"),
        ("[phases]\n", "budgets"),
        ("not toml", "line 1"),
    ],
    ids=[
        "over budget",
        "no budget",
        "zero budget",
        "zero share",
        "fraction",
        "extra",
        "missing",
        "toml",
    ],
)
def test_a_budgets_file_that_does_not_add_up_is_refused(
    tmp_path: Path, text: str, problem: str
) -> None:
    path = write(tmp_path, text)

    with pytest.raises(BudgetsError) as refused:
        load_budgets(path)

    assert str(path) in str(refused.value)
    assert problem in str(refused.value)


def test_a_missing_budgets_file_is_named(tmp_path: Path) -> None:
    with pytest.raises(BudgetsError, match=r"missing\.toml"):
        load_budgets(tmp_path / "missing.toml")


def test_shares_may_use_the_whole_budget(tmp_path: Path) -> None:
    path = write(tmp_path, "[budgets]\nm = 10\n[phases.p]\na = { m = 6 }\nb = { m = 4 }\n")

    assert load_budgets(path).share("p", "b") == {"m": 4}


def test_a_process_may_have_no_share_of_any_model(tmp_path: Path) -> None:
    path = write(tmp_path, "[budgets]\nm = 10\n[phases.p]\nquiet = {}\n")

    assert load_budgets(path).share("p", "quiet") == {}
