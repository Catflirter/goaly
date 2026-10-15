# goaly

goaly is a self-evolving agent harness. An RSI agent driven by GLM-5.3 improves the harness
(tools, prompts, agent loop and context management) that Qwen3.6-35B-A3B uses to solve
Effi-SQL tasks: rewriting slow PostgreSQL queries into faster ones. The SWE-fficiency lane is
on hold. The design is in
[docs/superpowers/specs/2026-10-14-rsi-harness-design.md](docs/superpowers/specs/2026-10-14-rsi-harness-design.md).

## Prerequisites

- [uv](https://docs.astral.sh/uv/). It installs Python 3.12 (pinned in `.python-version`) and
  every dependency at the version locked in `uv.lock`.
- GNU make.

## Configuration

| Variable | Required | Default |
|---|---|---|
| `GOALY_API_KEY` | yes | none |
| `GOALY_BASE_URL` | no | `https://api.goaly.ai/inference/v1` |

Set them as environment variables, or as lines in `~/.config/goaly/.env`
(keep that file private, e.g. `chmod 600`):

```sh
GOALY_API_KEY=<your key>
```

Environment variables take precedence over the file. Never commit secrets: `.env` files are
git-ignored. After `make setup`, a gitleaks pre-commit hook scans staged changes, and CI scans
the full history on every PR.

## Development

```sh
make setup             # create .venv from uv.lock and install the git hooks
make check             # as CI runs it: lint, type check, import contracts, generated code
                       # up to date, unit tests
make test              # unit tests with branch coverage
make test-integration  # tests that need Docker, a database or the real model API
make proto             # regenerate src/goaly/contracts/gen/ after editing proto/
```

Run `make` to list every target.

## Final evaluation

These need Docker, PostgreSQL 16 with TPC-H SF3 (which `make prepare` builds) and the executor
API's key. Every run takes the isolated pipeline: the LLM proxy, the solve container, then
grading on the host.

```sh
make freeze                 # print the plan to tag main, holding best/effisql's pack, as final
make freeze FREEZE_ARGS=--yes   # create the tag and push only the tag; it never merges
git checkout final
make final-plan             # print what make final would run; runs nothing
make final                  # prepare, then held-out x3 (official) and val x3 (dev), then the report
make baseline V0_REF=<v0>   # rerun v0 at its commit: train x1, val x2, then held-out x1
make report REPORT_ARGS=--allow-partial   # draft docs/results/final-report.md while runs grade
```

The report reads v0's runs from `runs/baseline/{heldout,val}`. If T12's archived runs live
elsewhere, pass their root, as in `make report BASELINE_RUNS=/path/to/baseline`. Record the
constant-branch audit's verdicts in `docs/results/audit-verdicts.json` before publishing.
Rerunning `make final` or `make baseline` continues where it stopped. Each phase's log is in
`runs/<final|baseline>/logs/`.
