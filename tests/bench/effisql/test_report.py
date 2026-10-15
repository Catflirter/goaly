"""Tests for goaly.bench.effisql.report, on toy run directories and ledgers."""

from pathlib import Path

import pytest

from goaly.bench.effisql.metrics import GradeTable, Scoring, compare
from goaly.bench.effisql.report import ROLES, ReportInputs, render, sections
from goaly.bench.effisql.report_inputs import RunRecord, Verdict, load_run
from goaly.contracts.gen.experiment_pb2 import (
    DECISION_ABORTED,
    DECISION_PENDING,
    DECISION_REJECTED,
    REJECTION_REASON_REGRESSION,
    Lesson,
)
from goaly.contracts.gen.task_pb2 import SPLIT_HELDOUT, SPLIT_VAL, EffiSqlGrade
from tests.bench.effisql.rundirs import experiment, write_plans, write_run, write_stages

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


def test_refuses_runs_on_different_tasks_unless_partial(tmp_path: Path) -> None:
    loaded = runs(tmp_path, "v0-heldout")
    write_run(tmp_path / "f", {"1": [2.0], "2": [1.0]}, run_id="f")
    loaded["final-heldout"] = load_run(tmp_path / "f")

    with pytest.raises(ValueError, match="v0-heldout and final-heldout name different tasks"):
        ReportInputs(loaded, DIFFICULTY)
    headline = sections(ReportInputs(loaded, DIFFICULTY, allow_partial=True))["headline"]

    assert "Compared on the 2 tasks both runs graded." in headline


def test_breaks_the_held_out_results_down_by_difficulty(tmp_path: Path) -> None:
    text = sections(ReportInputs(runs(tmp_path), DIFFICULTY))["difficulty"]

    assert "| v0-heldout, easy | official | 1 | 1 |" in text
    assert "| final-heldout, hard | checked | 2 | 3 |" in text


def test_reports_cost_per_task_against_the_efficiency_cap(tmp_path: Path) -> None:
    loaded = runs(tmp_path, "v0-heldout")
    write_run(tmp_path / "f", {"1": [2.0], "2": [1.0], "3": [1.0]}, run_id="f", cost=1.0)
    loaded["final-heldout"] = load_run(tmp_path / "f")

    built = sections(ReportInputs(loaded, DIFFICULTY))

    text = built["efficiency"]
    row = "| final-heldout | 3 | 4 | 0 | 400 | 40 | 4.00 | 1.3333 | n/a | 30.0 | results | 60.0 |"
    assert row in text
    assert (
        "- heldout: cost 4.00x v0's and wall time 1.00x v0's (wall time from results and"
        " results): **over the cap**." in text
    )
    assert "| Efficiency: time, tokens and cost, 3x cap | **over the cap** |" in built["criteria"]


def test_takes_wall_time_from_the_host_clock_around_the_solve_segments(tmp_path: Path) -> None:
    loaded = runs(tmp_path, "v0-heldout")
    final = write_run(tmp_path / "final" / "heldout", {"1": [2.0], "2": [1.0], "3": [1.0]})
    write_stages(tmp_path / "final", final, 100.0, 50.0)  # A solve, resumed after a crash.
    loaded["final-heldout"] = load_run(final)

    built = sections(ReportInputs(loaded, DIFFICULTY))

    # 150 s of solving, times the concurrency of 2, over the 3 tasks.
    assert "| 0.3333 | 2.5 | 100.0 | solve phase | 60.0 |" in built["efficiency"]
    assert "wall time 3.33x v0's (wall time from solve phase and results)" in built["efficiency"]


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
        experiment("e4", 3, DECISION_PENDING),
    ]
    experiments[1].rejection_reasons.append(REJECTION_REASON_REGRESSION)
    inputs = ReportInputs(runs(tmp_path, "v0-val"), DIFFICULTY, experiments=experiments)

    built = sections(inputs)

    rsi = built["rsi"]
    assert "2 of 3 rounds finished" in rsi
    assert "| 1 | finished | 2 | 1 | 60.0 |" in rsi
    assert "| 2 | finished | 1 | 0 | 60.0 |" in rsi  # An aborted experiment is settled.
    assert "| 3 | incomplete | 1 | 0 | 60.0 |" in rsi
    assert '    x-axis ["v0", "1", "2", "3"]' in rsi
    # v0's val R-VES is (0.5 + 0.25 + 0.5 + 0.5) / 4.
    assert "    line [43.8, 60.0, 60.0, 60.0]" in rsi
    assert "No experiment was accepted" not in rsi
    assert r"prompt: Ask for a plan \| then rewrite" in rsi
    assert "Accepted changes by source: 1 from idea_bank." in rsi
    assert "| effisql | exec | 4000 | 400 | 6.00 |" in built["efficiency"]
    assert "best/effisql; 1 rejected for a regression" in built["criteria"]
    assert "| Automation: several rounds | met | 2 rounds finished |" in built["criteria"]


def test_a_round_with_an_aborted_straggler_still_finished(tmp_path: Path) -> None:
    experiments = [experiment("e1", 1), experiment("e2", 1, DECISION_ABORTED)]

    built = sections(ReportInputs({}, DIFFICULTY, experiments=experiments))

    assert "1 of 1 rounds finished" in built["rsi"]
    assert "final harness is v0" not in built["rsi"]
    assert "| Automation: several rounds | **at risk** | 1 rounds finished |" in built["criteria"]


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


def test_says_plainly_when_no_experiment_was_accepted() -> None:
    inputs = ReportInputs({}, DIFFICULTY, experiments=[experiment("e1", 1, DECISION_PENDING)])

    rsi = sections(inputs)["rsi"]

    assert "**No experiment was accepted**, so the final harness is v0." in rsi
    assert "0 of 1 rounds finished" in rsi
    assert "(No val score to plot yet.)" in rsi


def test_the_cap_cannot_be_checked_without_a_call_log(tmp_path: Path) -> None:
    loaded = runs(tmp_path, "v0-heldout", "final-heldout")
    (tmp_path / "v0-heldout" / "llm_calls.jsonl").unlink()
    loaded["v0-heldout"] = load_run(tmp_path / "v0-heldout")

    text = sections(ReportInputs(loaded, DIFFICULTY))["efficiency"]

    assert "| v0-heldout | 3 | no call log |" in text
    assert "- heldout: cost n/a and wall time 1.00x v0's" in text
    assert "cannot be checked." in text


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


def test_the_audit_names_every_seed_it_could_not_read(tmp_path: Path) -> None:
    loaded = runs(tmp_path, "final-heldout")
    (tmp_path / "final-heldout" / "official" / "seed-0.jsonl").unlink()
    (tmp_path / "final-heldout" / "official" / "seed-2.jsonl").unlink()
    loaded["final-heldout"] = load_run(tmp_path / "final-heldout")

    text = sections(ReportInputs(loaded, DIFFICULTY))["audit"]

    assert "No matched prediction is listed." in text
    assert (
        "- final-heldout: stored plans for 0 of 9 matched predictions; the literal and pattern"
        " rules for the rest. **No official records for seed 0, 2**: those are listed by τ"
        " alone." in text
    )


# A stored plan of the r5 CASE: a Result over two InitPlans, reading no table itself.
CASE_PLAN = [
    {
        "Plan": {
            "Node Type": "Result",
            "Plans": [
                {
                    "Node Type": "Aggregate",
                    "Parent Relationship": "InitPlan",
                    "Plans": [{"Node Type": "Seq Scan", "Relation Name": "orders"}],
                },
                {
                    "Node Type": "Aggregate",
                    "Parent Relationship": "InitPlan",
                    "Plans": [{"Node Type": "Seq Scan", "Relation Name": "lineitem"}],
                },
            ],
        }
    }
]


def test_reads_the_stored_plans_where_there_are_any(tmp_path: Path) -> None:
    path = write_run(tmp_path / "f", {"1": [2.0], "2": [1.0]}, run_id="f")
    write_plans(path, {("1", 0): CASE_PLAN, ("2", 0): "not json"})

    text = sections(ReportInputs({"final-heldout": load_run(path)}, DIFFICULTY))["audit"]

    assert (
        "| final-heldout | 1 | 0 | 2.00 | 1.0 | plan: a Result that reads no table, from 2" in text
    )
    assert "| final-heldout | 2 | 0 | 1.00 | 0.5 | plan: unreadable, so read the query |" in text
    assert "- final-heldout: stored plans for 2 of 2 matched predictions." in text


def test_a_verdict_of_constant_branch_takes_the_checked_credit(tmp_path: Path) -> None:
    loaded = runs(tmp_path, "final-heldout")
    verdicts = {
        ("final-heldout", "1", 0): Verdict(constant_branch=True, note="gated on count(orders)"),
        ("final-heldout", "2", 0): Verdict(constant_branch=False),
    }
    gated = "SELECT 12345.67"
    path = write_run(tmp_path / "f", {"1": [2.0, 2.0, 2.0], "2": [1.0] * 3, "3": [1.0] * 3},
                     run_id="f", queries={("1", 0): gated, ("2", 0): gated})  # fmt: skip
    loaded["final-heldout"] = load_run(path)
    plain = ReportInputs(loaded, DIFFICULTY)
    judged = ReportInputs(loaded, DIFFICULTY, verdicts=verdicts)

    audit, flags = sections(judged)["audit"], sections(judged)["flags"]

    assert "| constant branch: no credit (gated on count(orders)) |" in audit
    assert "| clean |" in audit
    assert "task 1 seed 0 (literal_result)" in flags
    before, after = plain.table("final-heldout"), judged.table("final-heldout")
    assert before is not None
    assert after is not None
    assert after.grades["1"][0].credit == 0.0
    assert before.grades["1"][0].credit == 1.0
    assert after.grades["1"][1].credit == 1.0  # Only the judged seed.


def test_refuses_a_verdict_on_a_run_the_report_lacks() -> None:
    with pytest.raises(ValueError, match="verdicts name runs the report lacks"):
        ReportInputs({}, DIFFICULTY, verdicts={("final-heldout", "1", 0): Verdict(True)})


def test_labels_an_unmatched_flag_by_the_check_that_set_it(tmp_path: Path) -> None:
    loaded = runs(tmp_path, "final-heldout")
    final = loaded["final-heldout"]
    assert final.table is not None
    grades = {task: dict(seeds) for task, seeds in final.table.grades.items()}
    grades["3"][2] = EffiSqlGrade(matched=False, alt_matched=False, literal_result=True)
    loaded["final-heldout"] = RunRecord(final.path, final.manifest, GradeTable(grades), (), None)

    text = sections(ReportInputs(loaded, DIFFICULTY))["flags"]

    assert "- final-heldout: task 3 seed 2 (literal_result)" in text
