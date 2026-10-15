# goaly

goaly is a self-evolving agent harness. An RSI agent driven by GLM-5.3 improves the harness
(tools, prompts, agent loop and context management) that Qwen3.6-35B-A3B uses to solve
SWE-fficiency and BIRD-CRITIC tasks. The design is in
[docs/superpowers/specs/2026-10-14-rsi-harness-design.md](docs/superpowers/specs/2026-10-14-rsi-harness-design.md).

## Prerequisites

- [uv](https://docs.astral.sh/uv/). It installs Python 3.12 (pinned in `.python-version`) and
  every dependency at the version locked in `uv.lock`.
- GNU make.
- Docker. Every real run's solve phase runs in a sandbox container, the organizers' official
  runs included, and there is no non-Docker fallback (design spec §6.1 and §11). Inside a
  nested container, Docker may work only with `--network=host`.
- A Linux host. The solve container reaches PostgreSQL and the LLM proxy only through Unix
  sockets on the host, and Docker Desktop on macOS or Windows cannot pass a host Unix socket
  into a container.
- PostgreSQL 16 on the same machine. `make final` sets up the TPC-H instances. For a database
  you provide, set `GOALY_PG_SOCKET_DIRS` to its Unix socket directory, create the agent role
  `effisql_agent`, and put this line in `pg_hba.conf` before any `peer` line. Stock Debian and
  Ubuntu packages ship `local all all peer`, under which the agent role's login fails:

  ```
  local   tpch   effisql_agent   scram-sha-256
  ```

## Configuration

| Variable | Required | Default |
|---|---|---|
| `GOALY_API_KEY` | yes | none |
| `GOALY_BASE_URL` | no | `https://api.goaly.ai/inference/v1` |
| `GOALY_PG_SOCKET_DIRS` | no | the solving instances that `make final` sets up |

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
make check             # lint, type check, import contracts and unit tests, as CI runs them
make test              # unit tests with branch coverage
make test-integration  # tests that need Docker, a database or the real model API
```

Run `make` to list every target.
