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

To check that the LLM client and the real gateway still agree, run
`uv run --locked python scripts/llm_smoke.py`. It needs `GOALY_API_KEY`, makes four small calls
to Qwen and GLM, and prints only each call's model, status, tokens and cost. CI does not run it.
