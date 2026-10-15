"""Tests for goaly.bench.effisql.report, on toy run directories and ledgers."""

from pathlib import Path

import pytest

from goaly.bench.effisql.metrics import GradeTable, Scoring, compare
from goaly.bench.effisql.report import ROLES, ReportInputs, render, sections
from goaly.bench.effisql.report_inputs import RunRecord, load_run
from goaly.contracts.gen.experiment_pb2 import (
    DECISION_ABORTED,
    DECISION_REJECTED,
    REJECTION_REASON_REGRESSION,
    Lesson,
)
from goaly.contracts.gen.task_pb2 import SPLIT_HELDOUT, SPLIT_VAL
from tests.bench.effisql.rundirs import experiment, write_run

TEMPLATE = Path(__file__).resolve().parents[3] / "docs" / "results" / "final-report-template.md"
DIFFICULTY = {"1": "easy", "2": "hard", "3": "hard"}


def runs(tmp_path: Path, *roles: str) -> dict[str, RunRecord]:
    """v0 at τ 1 everywhere; the final faster on task 1, and flagged on task 3 in seed 2."""
    speedups = {
        "v0-heldout": {"1": [1.0], "2": [1.0], "3": [1.0]},
        "final-heldout": {"1": [2.0, 2.0, 1.2], "2": [1.0, 1.0, 1.0], "3": [1.0, 1.0, 1.0]},
        "v0-val": {"1": [1.0, 0.5], "2": [1.0, 1.0]},
        "final-val": {"1": [2.0, 2.0, 2.0], "2": [1.0, 1.0, 1.0]},
    }
    loaded = {}
    for role in roles or ROLES:
        heldout = "heldout" in role
        path = write_run(
            tmp_path / role,
            speedups[role],
            run_id=f"run-{role}",
            split=SPLIT_HELDOUT if heldout else SPLIT_VAL,
            mode="official" if heldout else "dev",
        )
        loaded[role] = load_run(path)
    return loaded


def test_the_headline_compares_v0_with_the_final_on_both_scorings(tmp_path: Path) -> None:
    inputs = ReportInputs(runs(tmp_path), DIFFICULTY)
    v0, final = inputs.table("v0-heldout"), inputs.table("final-heldout")
    assert v0 is not None
    assert final is not None

    headline = sections(inputs)["headline"]

    gain = compare(v0, final, scoring=Scoring.CHECKED)
    assert (
        f"| checked | 3 | 50.0 | {100 * gain.candidate_score:.1f} | +{100 * gain.delta:.1f} |"
        in headline
    )
    assert "| official | 3 |" in headline
    assert "| v0-heldout | official |" in headline
    assert "| final-heldout | checked | 3 | 3 |" in headline


def test_the_runs_section_traces_every_role_to_its_manifest(tmp_path: Path) -> None:
    loaded = runs(tmp_path, "final-heldout")

    text = sections(ReportInputs(loaded, DIFFICULTY))["runs"]

    assert f"`{tmp_path / 'final-heldout' / 'manifest.json'}`" in text
    assert "`run-final-heldout`" in text
    assert "| v0-heldout | pending |" in text


def test_a_role_without_a_run_renders_as_pending(tmp_path: Path) -> None:
    built = sections(ReportInputs(runs(tmp_path, "final-heldout"), DIFFICULTY))

    assert "Pending: needs v0-heldout and final-heldout." in built["headline"]
    assert "- heldout: Pending" in built["efficiency"]
    assert "| final-heldout | 2 | checked |" in built["variation"]


def test_compares_only_the_tasks_both_runs_graded(tmp_path: Path) -> None:
    loaded = runs(tmp_path, "v0-heldout")
    write_run(tmp_path / "f", {"1": [2.0], "2": [1.0]}, run_id="f")
    loaded["final-heldout"] = load_run(tmp_path / "f")

    headline = sections(ReportInputs(loaded, DIFFICULTY))["headline"]

    assert "Compared on the 2 tasks both runs graded." in headline


def test_breaks_the_held_out_results_down_by_difficulty(tmp_path: Path) -> None:
    text = sections(ReportInputs(runs(tmp_path), DIFFICULTY))["difficulty"]

    assert "| v0-heldout, easy | official | 1 | 1 |" in text
    assert "| final-heldout, hard | checked | 2 | 3 |" in text


def test_reports_cost_per_task_against_the_efficiency_cap(tmp_path: Path) -> None:
    loaded = runs(tmp_path, "v0-heldout")
    write_run(tmp_path / "f", {"1": [2.0], "2": [1.0], "3": [1.0]}, run_id="f", cost=1.0)
    loaded["final-heldout"] = load_run(tmp_path / "f")

    text = sections(ReportInputs(loaded, DIFFICULTY))["efficiency"]

    assert "| final-heldout | 3 | 4 | 0 | 400 | 40 | 4.00 | 1.3333 | 3.5 | 140.0 | 60.0 |" in text
    assert "- heldout: cost 4.00x v0's and wall time 1.00x v0's: **over the cap**." in text


def test_lists_flagged_predictions_with_their_checks(tmp_path: Path) -> None:
    loaded = runs(tmp_path, "final-heldout")
    final = loaded["final-heldout"]
    assert final.table is not None
    grades = {task: dict(seeds) for task, seeds in final.table.grades.items()}
    grades["3"][2] = grades["3"][2].__class__(matched=True, speedup=1000, alt_matched=False)
    loaded["final-heldout"] = RunRecord(final.path, final.manifest, GradeTable(grades), (), None)

    text = sections(ReportInputs(loaded, DIFFICULTY))["flags"]

    assert "- final-heldout: task 3 seed 2 (hard_coded)" in text


def test_counts_finished_rounds_and_plots_the_best_by_round(tmp_path: Path) -> None:
    experiments = [
        experiment("e1", 1, val_score=0.6),
        experiment("e2", 1, DECISION_REJECTED),
        experiment("e3", 2, DECISION_ABORTED),
    ]
    experiments[1].rejection_reasons.append(REJECTION_REASON_REGRESSION)
    inputs = ReportInputs(runs(tmp_path, "v0-val"), DIFFICULTY, experiments=experiments)

    built = sections(inputs)

    rsi = built["rsi"]
    assert "1 of 2 rounds finished" in rsi
    assert "| 1 | finished | 2 | 1 | 60.0 |" in rsi
    assert "| 2 | incomplete | 1 | 0 | 60.0 |" in rsi
    assert '    x-axis ["v0", "1", "2"]' in rsi
    assert "    line [43.8, 60.0, 60.0]" in rsi  # v0's val R-VES is (0.5 + 0.25 + 0.5 + 0.5) / 4.
    assert r"prompt: Ask for a plan \| then rewrite" in rsi
    assert "Accepted changes by source: 1 from idea_bank." in rsi
    assert "| effisql | exec | 3000 | 300 | 4.50 |" in built["efficiency"]
    assert "best/effisql; 1 rejected for a regression" in built["criteria"]
    assert "| Automation: several rounds | **at risk** |" in built["criteria"]


def test_says_plainly_when_no_round_ran(tmp_path: Path) -> None:
    built = sections(ReportInputs({}, DIFFICULTY))

    assert "No RSI experiment is in the ledger, so no round finished." in built["rsi"]
    assert "| Automation: several rounds | **not met** |" in built["criteria"]


def test_links_each_lesson_to_its_experiments(tmp_path: Path) -> None:
    lesson = Lesson(lesson_id="l1", text="Plans\nhelp.", experiment_ids=["e1"], helped_by=["e1"])
    inputs = ReportInputs({}, DIFFICULTY, experiments=[experiment("e1", 1)], lessons=[lesson])

    built = sections(inputs)

    assert "- Plans help. (helped 1, hurt 0; evidence: [`e1`](#exp-e1))" in built["lessons"]
    assert '<a id="exp-e1"></a>`e1`' in built["rsi"]


def test_renders_the_committed_template_with_every_section(tmp_path: Path) -> None:
    inputs = ReportInputs(runs(tmp_path), DIFFICULTY)

    report = render(inputs, TEMPLATE.read_text(encoding="utf-8"))

    for text in sections(inputs).values():
        assert text.rstrip("\n") in report
    assert "<!-- report:" not in report


def test_without_a_template_writes_the_sections_in_order() -> None:
    report = render(ReportInputs({}, DIFFICULTY))

    assert report.index("## Runs") < report.index("## Held-out") < report.index("## Acceptance")


@pytest.mark.parametrize(
    ("template", "message"),
    [
        ("<!-- report:nonsense -->", "unknown section 'nonsense'"),
        ("<!-- report:runs -->", "must place each"),
    ],
)
def test_refuses_a_template_that_misplaces_a_section(template: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        render(ReportInputs({}, DIFFICULTY), template)


def test_refuses_an_unknown_role() -> None:
    with pytest.raises(ValueError, match="unknown run roles"):
        ReportInputs({"v1-heldout": None}, DIFFICULTY)  # type: ignore[dict-item]  # The wrong role.


def test_says_plainly_when_the_rounds_ran_but_none_finished() -> None:
    inputs = ReportInputs({}, DIFFICULTY, experiments=[experiment("e1", 1, DECISION_ABORTED)])

    rsi = sections(inputs)["rsi"]

    assert "**No round finished**, so the final harness is v0." in rsi
    assert "(No val score to plot yet.)" in rsi


def test_the_cap_cannot_be_checked_without_a_call_log(tmp_path: Path) -> None:
    loaded = runs(tmp_path, "v0-heldout", "final-heldout")
    (tmp_path / "v0-heldout" / "llm_calls.jsonl").unlink()
    loaded["v0-heldout"] = load_run(tmp_path / "v0-heldout")

    text = sections(ReportInputs(loaded, DIFFICULTY))["efficiency"]

    assert "| v0-heldout | 3 | no call log |" in text
    assert "- heldout: cost n/a and wall time n/a: cannot be checked." in text


def test_lists_matched_predictions_for_the_constant_branch_audit(tmp_path: Path) -> None:
    gated = "SELECT CASE WHEN (SELECT count(*) FROM orders) > 4 THEN 1.5 ELSE (SELECT 2) END"
    path = write_run(
        tmp_path / "f",
        {"1": [2.0], "2": [1.0], "3": [None]},
        queries={("1", 0): gated, ("3", 0): gated},
    )
    loaded = {"final-heldout": load_run(path)}

    text = sections(ReportInputs(loaded, DIFFICULTY))["audit"]

    official = tmp_path / "f" / "official" / "seed-0.jsonl"
    assert (
        "| final-heldout | 1 | 0 | 2.00 | 1.0 | a CASE gated on a subquery; a SELECT of literals"
        f" only | `{official}` | TODO |" in text
    )
    assert "| final-heldout | 3 |" not in text  # Unmatched, so it scores nothing anyway.


def test_the_audit_says_when_nothing_needs_a_review(tmp_path: Path) -> None:
    loaded = runs(tmp_path, "final-heldout")
    (tmp_path / "final-heldout" / "official" / "seed-0.jsonl").unlink()
    (tmp_path / "final-heldout" / "official" / "seed-1.jsonl").unlink()
    (tmp_path / "final-heldout" / "official" / "seed-2.jsonl").unlink()
    loaded["final-heldout"] = load_run(tmp_path / "final-heldout")

    text = sections(ReportInputs(loaded, DIFFICULTY))["audit"]

    assert "No matched prediction needs a review." in text
    assert "No official records in final-heldout: listed by speedup alone." in text
