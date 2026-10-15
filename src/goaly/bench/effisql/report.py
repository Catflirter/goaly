"""The final report's generated sections (spec §2, §6.3, §8.6, §9, §10).

Each section is Markdown built from run directories and the RSI ledger, never typed in by hand,
so every number traces to a run manifest or an experiment. The runs have fixed roles:
``v0-heldout`` (T12's single official run), ``final-heldout`` (3 official runs), and ``v0-val``
and ``final-val`` (dev mode). The "Runs" section maps each role to its run ID and manifest, and
every other table names the role it reads. A role without a run renders as pending, so the
report can be drafted before and while the final runs. The constant-branch audit lists the
matched predictions a person must read before publishing (goaly.bench.effisql.audit).

:func:`render` fills a narrative template: each line ``<!-- report:NAME -->`` becomes the
section NAME, and the template must place every section exactly once.
"""

import re
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from goaly.bench.effisql.audit import AUDIT_SPEEDUP, audit_reasons
from goaly.bench.effisql.metrics import (
    GradeTable,
    Scoring,
    by_difficulty,
    compare,
    delta_rves_ci,
    is_flagged,
    markdown_table,
    seed_metrics,
)
from goaly.bench.effisql.report_inputs import RunRecord
from goaly.contracts.gen.experiment_pb2 import (
    DECISION_ACCEPTED,
    DECISION_REJECTED,
    REJECTION_REASON_REGRESSION,
    ChangeKind,
    Decision,
    ExperimentRecord,
    HypothesisSource,
    Lane,
    Lesson,
    PhaseResult,
)
from goaly.contracts.gen.llm_call_pb2 import CallPurpose
from goaly.contracts.gen.task_pb2 import Split

__all__ = ["BOOTSTRAP_SEED", "EFFICIENCY_CAP", "ROLES", "ReportInputs", "render", "sections"]

BOOTSTRAP_SEED = 20261014  # The split's seed (spec §7), so the intervals are reproducible.
EFFICIENCY_CAP = 3.0  # The final harness's cost and wall time per task, at most 3x v0's (§8.3).
ROLES = ("v0-heldout", "final-heldout", "v0-val", "final-val")
_DECIDED = (DECISION_ACCEPTED, DECISION_REJECTED)
_AT_RISK = "**at risk**: Effi-SQL only, as SWE is on hold"
_MARKER = re.compile(r"<!-- report:([a-z]+) -->")


@dataclass(frozen=True)
class ReportInputs:
    """Everything the report reads: runs by role, task difficulties and the ledger."""

    runs: Mapping[str, RunRecord]
    difficulty: Mapping[str, str]
    experiments: Sequence[ExperimentRecord] = ()
    lessons: Sequence[Lesson] = ()
    bootstrap_seed: int = BOOTSTRAP_SEED

    def __post_init__(self) -> None:
        if unknown := self.runs.keys() - set(ROLES):
            raise ValueError(f"unknown run roles {sorted(unknown)}; the roles are {ROLES}")

    def table(self, role: str) -> GradeTable | None:
        run = self.runs.get(role)
        return None if run is None else run.table


def sections(inputs: ReportInputs) -> dict[str, str]:
    """Every generated section, by name, in report order."""
    return {name: build(inputs) for name, build in _SECTIONS.items()}


def render(inputs: ReportInputs, template: str | None = None) -> str:
    """The report: ``template`` with its markers filled, or every section in order."""
    built = sections(inputs)
    if template is None:
        return "\n".join(built.values())
    placed: list[str] = []

    def fill(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in built:
            raise ValueError(f"the template names an unknown section {name!r}")
        placed.append(name)
        return built[name].rstrip("\n")

    text = _MARKER.sub(fill, template)
    if sorted(placed) != sorted(built):
        raise ValueError(f"the template must place each of {list(built)} once, not {placed}")
    return text


# The sections.


def _runs(inputs: ReportInputs) -> str:
    lines = [
        "## Runs",
        "",
        "Every number below comes from these runs; each manifest records the code, data and"
        " models that produced it.",
        "",
        "| Role | Run ID | Manifest | Split | Grading | Seeds | Tasks | Commit | Harness hash"
        " | Graded |",
        "|---|---|---|---|---|--:|--:|---|---|---|",
    ]
    for role in ROLES:
        run = inputs.runs.get(role)
        if run is None:
            lines.append(f"| {role} | pending | | | | | | | | |")
            continue
        manifest = run.manifest
        split = Split.Name(manifest.split).removeprefix("SPLIT_").lower()
        commit = manifest.git_commit[:12] + (" (dirty)" if manifest.git_dirty else "")
        graded = "all" if not run.missing else f"partial: {len(run.missing)} not yet"
        cells: list[object] = [role, f"`{run.run_id}`", f"`{run.path / 'manifest.json'}`", split]
        cells += [run.grading_mode, len(manifest.seeds), len(manifest.instance_ids)]
        cells += [f"`{commit}`", f"`{manifest.harness_hash[:12]}`", graded]
        lines.append(_row(cells))
    return _text(lines)


def _headline(inputs: ReportInputs) -> str:
    lines = ["## Held-out: v0 against the final harness", ""]
    if (pair := _heldout_pair(inputs)) is None:
        return _text([*lines, _pending("v0-heldout and final-heldout")])
    v0, final = pair
    if any(len(table.tasks) > len(v0.tasks) for table in _tables(inputs, "heldout")):
        lines += [f"Compared on the {len(v0.tasks)} tasks both runs graded.", ""]
    lines += [
        "R-VES, Coverage@2x and GMS, official and checked (spec §4.1). The final harness shows"
        " the mean ± standard deviation over its seeds; v0's single run shows its bootstrap 95%"
        " CI over tasks.",
        "",
        markdown_table(
            {"v0-heldout": v0, "final-heldout": final}, bootstrap_seed=inputs.bootstrap_seed
        ),
        "The improvement, on each task's seed-mean credit. Its CI is a paired bootstrap over"
        " tasks; the sign test is one-sided.",
        "",
        "| Scoring | Tasks | v0 R-VES | Final R-VES | ΔR-VES | Paired 95% CI | Wins | Losses"
        " | Ties | Sign-test p |",
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for scoring in Scoring:
        result = compare(v0, final, scoring=scoring)
        ci = delta_rves_ci(v0, final, scoring=scoring, bootstrap_seed=inputs.bootstrap_seed)
        sign = result.sign
        cells = [scoring, result.task_count, _num(100 * result.best_score)]
        cells += [_num(100 * result.candidate_score), f"{100 * result.delta:+.1f}"]
        cells += [f"[{ci.low:+.1f}, {ci.high:+.1f}]", sign.wins, sign.losses, sign.ties]
        lines.append(_row([*cells, f"{sign.p_value:.3g}"]))
    return _text(lines)


def _variation(inputs: ReportInputs) -> str:
    lines = [
        "## Variation between runs",
        "",
        "Each seed is one independent run. v0 has one held-out run, so its run-to-run variation"
        " comes from its 2 val seeds (spec §10).",
        "",
    ]
    val = {role: inputs.table(role) for role in ("v0-val", "final-val")}
    present = {role: table for role, table in val.items() if table is not None}
    if present:
        lines.append(markdown_table(present, bootstrap_seed=inputs.bootstrap_seed))
    lines += ["| Run | Seed | Scoring | R-VES | Coverage@2x | GMS |", "|---|--:|---|--:|--:|--:|"]
    for role in ROLES:
        if (table := inputs.table(role)) is None:
            continue
        for scoring in Scoring:
            for seed, metrics in seed_metrics(table, scoring=scoring).items():
                cells = [role, seed, scoring, _num(metrics.rves), _num(metrics.coverage_2x)]
                lines.append(_row([*cells, _num(metrics.gms, 2)]))
    return _text(lines)


def _difficulty(inputs: ReportInputs) -> str:
    lines = ["## Held-out results by difficulty", ""]
    rows: dict[str, GradeTable] = {}
    for role in ("v0-heldout", "final-heldout"):
        if (table := inputs.table(role)) is not None:
            for label, part in by_difficulty(table, inputs.difficulty).items():
                rows[f"{role}, {label}"] = part
    if not rows:
        return _text([*lines, _pending("v0-heldout and final-heldout")])
    return _text([*lines, markdown_table(rows, bootstrap_seed=inputs.bootstrap_seed)])


def _efficiency(inputs: ReportInputs) -> str:
    lines = [
        "## Time, tokens and cost",
        "",
        "Tokens and cost come from the host-side proxy's call log. The solve span runs from the"
        " first LLM call to the last; wall time per task spreads it over the tasks and multiplies"
        " by the run's task concurrency (spec §8.3). The run span, from the manifest, includes"
        " grading.",
        "",
        "| Run | Tasks x seeds | LLM calls | Unpriced | Input tokens | Output tokens | Cost (USD)"
        " | Cost per task | Solve span (min) | Wall per task (s) | Run span (min) |",
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for role in ROLES:
        if (run := inputs.runs.get(role)) is None:
            continue
        usage = run.usage
        cells: list[object] = [role, run.task_seeds]
        if usage is None:
            cells += ["no call log"] + [""] * 5
        else:
            cells += [usage.calls, usage.unpriced_calls, usage.input_tokens, usage.output_tokens]
            cells += [f"{usage.cost_usd:.2f}", _num(run.cost_per_task_usd(), 4)]
        cells += [_num(_minutes(None if usage is None else usage.solve_span_s))]
        lines.append(_row([*cells, _num(run.wall_per_task_s()), _num(_minutes(run.run_span_s))]))
    lines += ["", f"Against the efficiency cap, {EFFICIENCY_CAP:g}x v0 per task (spec §8.3):", ""]
    for split in ("heldout", "val"):
        v0, final = inputs.runs.get(f"v0-{split}"), inputs.runs.get(f"final-{split}")
        if v0 is None or final is None:
            lines.append(f"- {split}: {_pending(f'v0-{split} and final-{split}')}")
            continue
        cost = _ratio(final.cost_per_task_usd(), v0.cost_per_task_usd())
        wall = _ratio(final.wall_per_task_s(), v0.wall_per_task_s())
        verdict = _cap_verdict(cost, wall)
        lines.append(f"- {split}: cost {_times(cost)} and wall time {_times(wall)}: {verdict}.")
    return _text([*lines, "", *_experiment_cost(inputs.experiments)])


def _experiment_cost(experiments: Sequence[ExperimentRecord]) -> list[str]:
    lines = [
        "RSI experiment cost, separate from the final harness's, summed from the ledger (spec §9):",
        "",
        "| Lane | Purpose | Input tokens | Output tokens | Cost (USD) |",
        "|---|---|--:|--:|--:|",
    ]
    totals: dict[tuple[str, str], list[float]] = {}
    for record in experiments:
        lane = _enum(Lane.Name(record.lane), "LANE_")
        for entry in record.usage_by_purpose:
            purpose = _enum(CallPurpose.Name(entry.purpose), "CALL_PURPOSE_")
            total = totals.setdefault((lane, purpose), [0, 0, 0.0])
            total[0] += entry.usage.input_tokens
            total[1] += entry.usage.output_tokens
            total[2] += entry.usage.cost_usd
    for (lane, purpose), (tokens_in, tokens_out, cost) in sorted(totals.items()):
        lines.append(_row([lane, purpose, int(tokens_in), int(tokens_out), f"{cost:.2f}"]))
    all_cost = sum(total[2] for total in totals.values())
    return [*lines, _row(["**all**", "", "", "", f"**{all_cost:.2f}**"])]


def _flags(inputs: ReportInputs) -> str:
    lines = [
        "## Hard-coding flags",
        "",
        "Matched predictions that a check against hard-coded results flagged (spec §4.1). The"
        " official numbers count them as the evaluator would; the checked numbers give them 0.",
        "",
    ]
    for role in ROLES:
        if (table := inputs.table(role)) is None:
            continue
        found = []
        for task, seeds in table.grades.items():
            for seed, grade in seeds.items():
                if is_flagged(grade):
                    alt = grade.HasField("alt_matched") and not grade.alt_matched
                    checks = ["hard_coded"] * alt + ["literal_result"] * grade.literal_result
                    found.append(f"task {task} seed {seed} ({', '.join(checks)})")
        lines.append(f"- {role}: {'; '.join(found) if found else 'none'}")
    return _text(lines)


def _audit(inputs: ReportInputs) -> str:
    lines = [
        "## Constant-branch audit",
        "",
        "The grader's checks cannot flag a single-row constant gated on the data, a residual risk"
        " the project lead accepted (spec §12). A matched prediction is listed here if it adds a"
        f" pattern of such a branch to its base query, or ran at least {AUDIT_SPEEDUP:g}x faster:"
        " read each one before the numbers are published, and record the verdict.",
        "",
        "| Run | Task | Seed | τ | Checked credit | Why | Official record | Verdict |",
        "|---|--:|--:|--:|--:|---|---|---|",
    ]
    listed, unread = 0, []
    for role in ROLES:
        run = inputs.runs.get(role)
        if run is None or run.table is None:
            continue
        if not run.queries:
            unread.append(role)
        for task, seeds in run.table.grades.items():
            for seed, grade in seeds.items():
                sql, base = run.queries.get((task, seed), ("", ""))
                reasons = audit_reasons(sql, base, grade.speedup) if grade.matched else []
                if not reasons:
                    continue
                listed += 1
                where = f"`{run.path / 'official' / f'seed-{seed}.jsonl'}`"
                cells = [role, task, seed, _num(grade.speedup, 2), grade.credit]
                lines.append(_row([*cells, "; ".join(reasons), where, "TODO"]))
    if not listed:
        lines[-2:] = ["No matched prediction needs a review."]
    if unread:
        lines += ["", f"No official records in {', '.join(unread)}: listed by speedup alone."]
    return _text(lines)


def _rsi(inputs: ReportInputs) -> str:
    lines = ["## RSI rounds", ""]
    rounds = _rounds(inputs.experiments)
    if not rounds:
        return _text([*lines, "No RSI experiment is in the ledger, so no round finished."])
    finished = sum(done for _, _, done in rounds)
    if not finished:
        lines += ["**No round finished**, so the final harness is v0.", ""]
    lines += [
        f"{finished} of {len(rounds)} rounds finished: every experiment in a finished round was"
        " accepted or rejected. An incomplete round's work stays in the tables below.",
        "",
        "| Round | Status | Experiments | Accepted | Best val R-VES after it |",
        "|--:|---|--:|--:|--:|",
    ]
    v0_val = inputs.table("v0-val")
    best = None
    if v0_val is not None:  # v0's val R-VES, the mean over its seeds, is where RSI starts.
        best = statistics.fmean(
            m.rves for m in seed_metrics(v0_val, scoring=Scoring.CHECKED).values()
        )
    series: list[tuple[str, float | None]] = [("v0", best)]
    for number, records, done in rounds:
        accepted = [r for r in records if r.decision == DECISION_ACCEPTED]
        for record in accepted:  # The best moves to each accepted candidate in turn.
            phase = record.confirmation if record.HasField("confirmation") else record.validation
            best = 100 * phase.candidate_score
        series.append((str(number), best))
        status = "finished" if done else "incomplete"
        lines.append(_row([number, status, len(records), len(accepted), _num(best)]))
    lines += ["", *_chart(series), "", *_experiments(inputs.experiments)]
    return _text(lines)


def _rounds(
    experiments: Sequence[ExperimentRecord],
) -> list[tuple[int, list[ExperimentRecord], bool]]:
    """Each round's experiments, and whether it finished: all of them accepted or rejected."""
    rounds: dict[int, list[ExperimentRecord]] = {}
    for record in experiments:
        rounds.setdefault(record.round, []).append(record)
    return [
        (number, records, all(r.decision in _DECIDED for r in records))
        for number, records in sorted(rounds.items())
    ]


def _chart(series: Sequence[tuple[str, float | None]]) -> list[str]:
    points = [(label, value) for label, value in series if value is not None]
    if not points:
        return ["(No val score to plot yet.)"]
    labels = ", ".join(f'"{label}"' for label, _ in points)
    values = ", ".join(f"{value:.1f}" for _, value in points)
    return [
        "```mermaid",
        "xychart-beta",
        '    title "Best val R-VES (checked) by round"',
        f"    x-axis [{labels}]",
        '    y-axis "R-VES" 0 --> 100',
        f"    line [{values}]",
        "```",
    ]


def _experiments(experiments: Sequence[ExperimentRecord]) -> list[str]:
    sources: dict[str, int] = {}
    lines = [
        "| Experiment | Round | Source | Change | Decision | Screen ΔR-VES | Val ΔR-VES"
        " | Cost (USD) |",
        "|---|--:|---|---|---|--:|--:|--:|",
    ]
    for record in experiments:
        source = _enum(HypothesisSource.Name(record.hypothesis.source), "HYPOTHESIS_SOURCE_")
        if record.decision == DECISION_ACCEPTED:
            sources[source] = sources.get(source, 0) + 1
        change = _enum(ChangeKind.Name(record.hypothesis.change_kind), "CHANGE_KIND_")
        text = " ".join(record.hypothesis.change.split())
        cost = sum(entry.usage.cost_usd for entry in record.usage_by_purpose)
        cells = [f'<a id="exp-{record.experiment_id}"></a>`{record.experiment_id}`', record.round]
        cells += [
            source,
            f"{change}: {text[:100]}",
            _enum(Decision.Name(record.decision), "DECISION_"),
        ]
        screen = record.screening if record.HasField("screening") else None
        val = record.validation if record.HasField("validation") else None
        cells += [_delta(screen), _delta(val), f"{cost:.2f}"]
        lines.append(_row(cells))
    counts = ", ".join(f"{count} from {source}" for source, count in sorted(sources.items()))
    return [*lines, "", f"Accepted changes by source: {counts or 'none'}."]


def _lessons(inputs: ReportInputs) -> str:
    lines = ["## Lessons", ""]
    if not inputs.lessons:
        return _text([*lines, "The ledger holds no lesson."])
    for lesson in inputs.lessons:
        evidence = ", ".join(f"[`{e}`](#exp-{e})" for e in lesson.experiment_ids) or "none"
        counts = f"helped {len(lesson.helped_by)}, hurt {len(lesson.hurt_by)}"
        lines.append(f"- {' '.join(lesson.text.split())} ({counts}; evidence: {evidence})")
    return _text(lines)


def _criteria(inputs: ReportInputs) -> str:
    experiments = inputs.experiments
    finished = sum(done for _, _, done in _rounds(experiments))
    regressions = sum(REJECTION_REASON_REGRESSION in r.rejection_reasons for r in experiments)
    final = inputs.runs.get("final-heldout")
    seeds = 0 if final is None else len(final.manifest.seeds)
    automation = "met" if finished >= 2 else "**at risk**" if finished == 1 else "**not met**"
    gain = "pending"
    if (pair := _heldout_pair(inputs)) is not None:
        delta = compare(*pair, scoring=Scoring.CHECKED).delta
        ci = delta_rves_ci(*pair, scoring=Scoring.CHECKED, bootstrap_seed=inputs.bootstrap_seed)
        gain = f"{100 * delta:+.1f} [{ci.low:+.1f}, {ci.high:+.1f}]"
    lessons = len(inputs.lessons)
    rows = [  # The PDF's criteria (spec §2): each with its status and where the report shows it.
        ("Score: start, final, gain on every benchmark", _AT_RISK, f"checked ΔR-VES {gain}"),
        ("Robustness: 3 or more runs", "met" if seeds >= 3 else "**pending**", f"{seeds} runs"),
        ("Generalization", "**not met**: one benchmark", "Committed splits; flags section"),
        ("Lessons from experiments", "met" if lessons else "**pending**", f"{lessons} lessons"),
        ("Efficiency: time, tokens and cost", "met", "Time, tokens and cost section"),
        ("Reproducibility: one command", "met", "`make final`; run manifests; uv.lock"),
        ("Automation: several rounds", automation, f"{finished} rounds finished"),
        ("Regression check", "met", f"best/effisql; {regressions} rejected for a regression"),
    ]
    lines = ["## Acceptance criteria", "", "| PDF criterion | Status | Evidence |", "|---|---|---|"]
    return _text([*lines, *(_row(row) for row in rows)])


def _heldout_pair(inputs: ReportInputs) -> tuple[GradeTable, GradeTable] | None:
    """v0's and the final's held-out tables, on the tasks both graded."""
    v0, final = inputs.table("v0-heldout"), inputs.table("final-heldout")
    if v0 is None or final is None:
        return None
    tasks = set(v0.tasks) & set(final.tasks)
    return v0.subset(tasks), final.subset(tasks)


def _tables(inputs: ReportInputs, split: str) -> list[GradeTable]:
    tables = (inputs.table(f"{version}-{split}") for version in ("v0", "final"))
    return [table for table in tables if table is not None]


_SECTIONS: dict[str, Callable[[ReportInputs], str]] = {
    "runs": _runs,
    "headline": _headline,
    "variation": _variation,
    "difficulty": _difficulty,
    "efficiency": _efficiency,
    "flags": _flags,
    "audit": _audit,
    "rsi": _rsi,
    "lessons": _lessons,
    "criteria": _criteria,
}


# Formatting.


def _row(cells: Sequence[object]) -> str:
    return "| " + " | ".join(str(cell).replace("|", r"\|") for cell in cells) + " |"


def _text(lines: Sequence[str]) -> str:
    return "\n".join(lines).rstrip("\n") + "\n"


def _num(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _enum(name: str, prefix: str) -> str:
    return name.removeprefix(prefix).lower()


def _pending(roles: str) -> str:
    return f"Pending: needs {roles}."


def _minutes(seconds: float | None) -> float | None:
    return None if seconds is None else seconds / 60


def _ratio(value: float | None, base: float | None) -> float | None:
    return None if value is None or not base else value / base


def _times(ratio: float | None) -> str:
    return "n/a" if ratio is None else f"{ratio:.2f}x v0's"


def _cap_verdict(cost: float | None, wall: float | None) -> str:
    if cost is None or wall is None:
        return "cannot be checked"
    return "within the cap" if max(cost, wall) <= EFFICIENCY_CAP else "**over the cap**"


def _delta(phase: PhaseResult | None) -> str:
    return "" if phase is None else f"{100 * phase.delta:+.1f}"
