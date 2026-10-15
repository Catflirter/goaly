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
- Docker Engine on a Linux host, not Docker Desktop. Every real run's solve phase runs in a
  sandbox container, the organizers' official runs included, and there is no non-Docker
  fallback (design spec §6.1 and §11). The container reaches PostgreSQL and the LLM proxy only
  through Unix sockets on the host, and Docker Desktop, which runs the engine in a VM on every
  OS, cannot pass those in. Inside a nested container, Docker may work only with
  `--network=host`.

The items below come with the evaluation stack (T09, T11, T18 and T24) and do nothing on `main`
before it lands:

- At least 24 logical CPUs, with SMT siblings at N and N+12, for the default CPU sets (design
  spec §6.2). On a smaller machine, set the CPU sets in config.
- For `make final`'s database setup: git, a C compiler and make, to build TPC-H's dbgen;
  network access to GitHub; and about 50 GB of free disk. Its PostgreSQL 16 instances run in
  Docker, so you need no host PostgreSQL. Reserving their ports removes a start-up race:
  `sudo sysctl net.ipv4.ip_local_reserved_ports=55432-55439`.
- Or a PostgreSQL 16 database you provide, on the same machine: see below.

### A database you provide

It must be named `tpch` and hold TPC-H SF3. As a superuser, create the agent role:

```sql
CREATE ROLE effisql_agent LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS
  CONNECTION LIMIT 64 PASSWORD '<agent password>';
ALTER ROLE effisql_agent SET default_transaction_read_only = on;
ALTER ROLE effisql_agent SET temp_file_limit = '5GB';
ALTER ROLE effisql_agent SET statement_timeout = '130s';
ALTER ROLE effisql_agent SET client_connection_check_interval = '5s';
REVOKE ALL ON DATABASE tpch FROM PUBLIC;
GRANT CONNECT ON DATABASE tpch TO effisql_agent;
\connect tpch
REVOKE ALL ON SCHEMA public FROM PUBLIC;
GRANT USAGE ON SCHEMA public TO effisql_agent;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO effisql_agent;
```

Add this line to `pg_hba.conf`, before any `peer` line, then reload the server with
`SELECT pg_reload_conf();`. Stock Debian and Ubuntu packages ship `local all all peer`, under
which the agent role's login fails:

```
local   tpch   effisql_agent   scram-sha-256
```

Put the agent's password in `~/.config/goaly/pg.env` (`chmod 600`):

```sh
GOALY_PG_AGENT_PASSWORD=<agent password>
```

Then set `GOALY_PG_SOCKET_DIRS` to the server's socket directory and port, as `<dir>:<port>`.
The server's socket file is `<dir>/.s.PGSQL.<port>`, and the port is its `port` setting. For a
stock Debian or Ubuntu server, that is:

```sh
GOALY_PG_SOCKET_DIRS=/var/run/postgresql:5432
```

The variable is a comma-separated list of such entries. `make final`'s setup uses two, one per
solving instance; a database you provide uses one.

## Configuration

| Variable | Required | Default |
|---|---|---|
| `GOALY_API_KEY` | yes | none |
| `GOALY_BASE_URL` | no | `https://api.goaly.ai/inference/v1` |
| `GOALY_PG_SOCKET_DIRS` | no, and only with the evaluation stack | the solving instances that `make final` sets up |

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
