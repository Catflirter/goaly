# goaly: final report

<!--
The narrative template of the final report (T23, spec §10). Write the prose here while the final
runs; every number comes from the generated sections. Each line of the form
"<!- - report:NAME - ->" (without the spaces) is replaced by a generated section, and each section
must be placed exactly once:

    uv run --locked python scripts/effisql/final_report.py \
        --template docs/results/final-report-template.md --out docs/results/final-report.md \
        --v0-heldout runs/baseline/heldout --v0-val runs/baseline/val \
        --final-heldout runs/final/heldout --final-val runs/final/val \
        --verdicts docs/results/audit-verdicts.json [--allow-partial]

Never type a number into the prose that a generated section does not show: every number must
trace to a run manifest or a ledger entry. Delete these comments and every TODO before
publishing.
-->

## Summary

TODO: two or three sentences. The starting score, the final score and the improvement on the
held-out set, in checked R-VES, with the interval; whether the gain holds in official R-VES too;
how many RSI rounds finished. Say plainly what is at risk: "gains on more than one benchmark"
(SWE on hold) and "several improvement rounds".

<!-- report:criteria -->

## What was built

TODO: one paragraph each.

- The harness: core (frozen), the Effi-SQL capability pack and the baseline v0 (spec §5).
- The evaluation: the isolated two-phase pipeline, with the solve container and the
  key-injecting proxy (spec §6.1); the grader and its checks against hard-coded results (§4.1).
- The RSI loop: analyst, implementer, guards and decision rules, run unattended (spec §8).

<!-- report:runs -->

## Results

TODO: interpret the headline tables. Official against checked numbers: any gap is a flagged
prediction (see Hard-coding flags).

<!-- report:headline -->

<!-- report:variation -->

TODO: is the variation between the final's runs small next to the improvement?

<!-- report:difficulty -->

TODO: where did the gains come from: easy, medium or hard tasks?

<!-- report:efficiency -->

TODO: what the final harness costs to run, against v0 and the 3x cap, and what the RSI
experiments cost in all.

<!-- report:flags -->

<!-- report:audit -->

TODO: read each listed prediction and record its verdict in docs/results/audit-verdicts.json,
as {"role", "task", "seed", "verdict": "constant branch" or "clean", "note"}, then regenerate. A
constant branch then gets no credit in every checked number above, and the audit table shows the
verdict. Say here how many were judged constant branches.

## Self-improvement

TODO: what the rounds tried, what was accepted and why. If no round finished, say so plainly,
and report what the partial round learned: its hypotheses, implementations, screening results
and lessons (spec §12).

<!-- report:rsi -->

<!-- report:lessons -->

## Deviations from the task

- **The 62 BIRD-Interact tasks are left out.** Their databases' scale dumps are not published, so
  only the 238 TPC-H tasks are used; the harness still runs such tasks given their database
  (spec §1.4, §4.1).
- **11 base plans were recorded at TPC-H SF10, not SF3** (tasks 2, 24, 57, 72, 80, 87, 94, 99,
  134, 137 and 146). The agent sees our own SF3 plans for them, and exclusions and grading use
  times measured on this machine, never the file's (spec §4.1).
- **SWE-fficiency is on hold** (2026-10-15), so Effi-SQL is the only benchmark scored and
  "gains on more than one benchmark" is not met (spec §2).
- **v0 has one held-out run, in official mode,** by the project lead's decision of 2026-10-15;
  its run-to-run variation comes from its 2 val seeds (spec §10).

## Reproducing

`make final` prepares the TPC-H database, the PostgreSQL instances and the solve sandbox image,
runs the final harness on held-out (3 runs, official mode) and val (3 runs, dev mode) through
the isolated two-phase pipeline, grades them and writes this report. It needs Docker, uv with
Python 3.12 and the executor API's key; there is no non-Docker fallback (spec §6.1, §11).
`make baseline` reruns v0 the same way. Every run's manifest records the commit, the harness
and config hashes, the dataset revision, the models and the machine.
