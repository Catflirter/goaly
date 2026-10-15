# goaly

goaly is a self-evolving agent harness. An RSI agent driven by GLM-5.3 improves the harness
(tools, prompts, agent loop and context management) that Qwen3.6-35B-A3B uses to solve
Effi-SQL tasks: rewriting slow PostgreSQL queries into equivalent, faster ones. The organizers
evaluate the frozen harness on a hidden held-out set with the executor model alone. The
SWE-fficiency lane is on hold. The design is in
[docs/superpowers/specs/2026-10-14-rsi-harness-design.md](docs/superpowers/specs/2026-10-14-rsi-harness-design.md)
("the spec" below).

- [How it works](#how-it-works)
- [Install](#install)
- [Reproduce the final result](#reproduce-the-final-result)
- [Run the organizers' task file](#run-the-organizers-task-file)
- [Rerun the baseline, v0](#rerun-the-baseline-v0)
- [Run RSI](#run-rsi)
- [Results](#results)
- [Limitations and residual risks](#limitations-and-residual-risks)
- [Development](#development)

## How it works

### Three layers

| Layer | Packages | Who may change it |
|---|---|---|
| Contracts and core | `goaly.contracts`: protobuf data contracts and Python Protocols. `goaly.harness.core`: the LLM client and proxy, tool executor, agent loop, context management, trajectories and runner. `goaly.bench`: data, environments, grader, metrics and the solve sandbox | People, through reviewed PRs. RSI cannot edit it, and its checksums are verified before every evaluation |
| Packs | `goaly.harness.packs.effisql`: `prompt.md`, `skills/`, `tools.py`, `policy.py`, `config.yaml` | The only code RSI may edit |
| RSI | `goaly.rsi`: analyst, implementer, guards, decision rules, ledger, lane and supervisor | Development only. Nothing on the evaluation path imports it |

`goaly.cli` is the composition root. import-linter enforces the layering in CI: `contracts`
imports nothing else from goaly, `harness` imports only `contracts`, `bench` adds `harness`,
`rsi` adds `bench`, and only `rsi` may import `minisweagent`. At runtime the harness calls only
`qwen3.6-35b-a3b`; GLM is used only by RSI (spec §3, §5).

### The benchmark: Effi-SQL

The dataset is `birdsql/Effi-SQL`, pinned by revision and sha256 and downloaded on first use. We
use its 238 tasks on TPC-H at scale factor 3, which we generate locally with dbgen. Its other 62
tasks need BIRD-Interact databases that are not published (spec §4.1). The agent sees the slow
query, the schema and the base plan, never the gold query. It may not create indexes or change
data: it connects with a `SELECT`-only role.

**Scoring** reimplements the official evaluator. A prediction matches if it returns the same
multiset of rows as `base_sql`. Its speedup τ is base time / prediction time, from medians of
interleaved `EXPLAIN ANALYZE` runs, capped at 1000. Per-task credit is 1.0 for τ ≥ 2, 0.75 for
τ ≥ 1.2, 0.5 for τ ≥ 1, 0.25 for τ ≥ 0.5, and 0 otherwise or when the prediction does not match.

| Metric | Meaning |
|---|---|
| **R-VES** (main) | 100 × mean credit over all tasks |
| Coverage@2x | 100 × the share of tasks matched with τ ≥ 2 |
| GMS | the geometric mean of τ over matched tasks |

Each metric is reported twice: **official**, as the evaluator computes it, and **checked**. Two
hard-coding checks run on every matched prediction: it must also match on a second TPC-H
database at another scale, and its plan must compute its output from the base query's tables. A
prediction that fails either check gets no credit in the checked numbers. RSI decides on the
checked numbers.

**Splits** (seed `20261014`, stratified by difficulty, committed in `src/goaly/bench/splits/`):
train 148, val 30, held-out 60. Tasks whose base query fails or times out on this machine, or
whose gold is not reliably faster, are excluded after splitting, and the list is committed with
a reason for each. Held-out is used only for v0's baseline and the final runs.

**Grading modes.** Dev mode, for RSI, caches the base query's side and grades on 3 dedicated
PostgreSQL instances. Official mode, for the reported held-out numbers, runs the exact
interleaved protocol with no cache.

### Two-phase isolation

Every real run takes the isolated pipeline (spec §6.1):

1. **Solve, in a container.** The pack's policy runs in a Docker sandbox: a read-only harness
   snapshot, no `.git`, no `~/.config`, no API key, no gold, and a seccomp profile that denies
   every IP socket. It reaches only Qwen, through the host's LLM proxy over a Unix socket, and
   the 2 solving PostgreSQL instances as the read-only agent role, over their Unix sockets. It
   can write only `<run>/solve/`.
2. **Proxy, on the host.** The proxy adds the real key, forwards only the executor model, clamps
   each request and records every call's tokens and cost. Decision costs come from these
   records, never from the container.
3. **Grade, on the host.** The grade phase never imports pack code. It takes the expected tasks,
   seeds and harness identity from the host, reads only predictions from `solve/`, grades on the
   grading instances with the grader role, and writes `results.jsonl` and the official
   prediction files.

`--in-process` runs both phases in one process, for tests and local debugging only. It refuses
the real key unless `--unsafe-in-process` is also given.

### The RSI loop

`goaly rsi run` runs rounds with no person or assistant in the loop (spec §8). Each round:

1. **Analyze.** One structured GLM call reads the ledger, the best version's train results and
   failed train trajectories, and proposes 3 hypotheses.
2. **Implement.** Each hypothesis is one run of mini-swe-agent's `DefaultAgent`, on GLM, in a
   Docker sandbox where only the pack's two directories are writable. It may smoke-test on train
   tasks with `goaly-smoke`, which the host answers. The host makes one commit per hypothesis.
3. **Guard.** A candidate is rejected if its diff leaves the pack, a core checksum changes, the
   hard-coding scan or the static import allowlist fails, or the pack's tests fail in a fresh
   sandbox.
4. **Screen** on up to 15 targeted train failures plus 5 solved train tasks. No net gain means
   rejection.
5. **Validate** on all 30 val tasks, then **decide**, then **confirm** on a second val seed.
6. **Record** every candidate in the ledger, with a lesson GLM writes.

**Decision rules.** All must hold. The thresholds are in `rsi/lane-effisql.toml`.

| Rule | Threshold |
|---|---|
| Net gain | ΔR-VES (checked, val) ≥ 5 points |
| Significance | a one-sided sign test on per-task credit gives p ≤ 0.1 |
| Regression | at most 1 task drops from credit ≥ 0.75 to ≤ 0.25 |
| Confirmation | ΔR-VES ≥ 3 points over both val seeds |
| Cost guardrail | over 1.5× the best's tokens or wall time per task doubles both Δ thresholds |
| Efficiency cap | over 3× v0's mean cost or wall time per task is rejected outright |
| Hard-coding | any flagged prediction fails the candidate |

The best version is the git tag `best/effisql`. A lane stops after 4 rounds in a row with no
accepted change, or at its absolute deadline. A supervisor restarts a crashed lane from the
ledger, and the lane opens "RSI results" PRs and sends ntfy.sh notifications itself. It never
merges.

## Install

### Requirements

- **Linux with Docker Engine**, not Docker Desktop. Every real run's solve phase runs in a sandbox
  container, the organizers' runs included, and there is no non-Docker fallback. The container
  reaches PostgreSQL and the LLM proxy only through host Unix sockets, which Docker Desktop's VM
  cannot pass in. Inside a nested container, Docker may work only with `--network=host`.
- [uv](https://docs.astral.sh/uv/). It installs Python 3.12 (pinned in `.python-version`) and
  every dependency at the version locked in `uv.lock`.
- GNU make, git, a C compiler, and network access to GitHub (for dbgen) and Hugging Face (for
  the dataset).
- At least 24 logical CPUs, with SMT siblings at N and N+12, for the default CPU sets (spec
  §6.2). On a smaller machine, set the CPU sets in config.
- About 50 GB of free disk for the PostgreSQL setup.
- A key for the model gateway.

### Python environment

```sh
git clone https://github.com/Catflirter/goaly.git && cd goaly
uv sync --locked               # to evaluate: contracts, harness, bench and the CLI
uv sync --locked --all-groups  # to run RSI too: adds the rsi group (mini-swe-agent)
```

`make setup` runs the second command and installs the git hooks, for development.

### The API key

Put the key in `~/.config/goaly/.env`, outside the repository, and keep the file private:

```sh
mkdir -p ~/.config/goaly
printf 'GOALY_API_KEY=%s\n' '<your key>' > ~/.config/goaly/.env
chmod 600 ~/.config/goaly/.env
```

Never commit a key. `.env` files are git-ignored, a gitleaks pre-commit hook scans staged
changes after `make setup`, and CI scans the full history on every PR. The key never enters a
sandbox: the host's proxy adds it to each request.

| Variable | Required | Default |
|---|---|---|
| `GOALY_API_KEY` | yes | none |
| `GOALY_BASE_URL` | no | `https://api.goaly.ai/inference/v1` |
| `GOALY_PG_SOCKET_DIRS` | no | the 2 solving instances `pgenv.py` sets up |
| `GOALY_NTFY_TOPIC` | no, RSI only | none: notifications are off |
| `GOALY_GITHUB_TOKEN` | no, RSI only | the Coder workspace's GitHub token |

Set them as environment variables or as lines in `~/.config/goaly/.env`. Environment variables
take precedence over the file.

### PostgreSQL, through pgenv.py

`make prepare`, which `make final` and `make baseline` run first, sets up the database and builds
the solve sandbox image, `goaly-rsi-sandbox:<uv.lock hash>`. Its database step is:

```sh
uv run --locked python scripts/effisql/pgenv.py setup   # build if needed, then start
```

`pgenv.py setup` builds the pinned dbgen, generates TPC-H SF3 and the alternate SF0.3 database,
loads one golden data directory and writes the schema snapshots. It then starts 5 PostgreSQL 16
instances in Docker, from the pinned `postgres:16` image: 2 solving and 3 grading, each pinned
to its own CPUs. No host PostgreSQL is needed. The data lives under `~/data/effisql/`. The build
creates `~/.config/goaly/pg.env` (mode 0600) with random passwords for the admin, agent and
grader roles, and no output prints them. Every command is idempotent, so a rerun reuses what
exists.

Optionally, reserve the instances' ports first, which removes a start-up race:
`sudo sysctl net.ipv4.ip_local_reserved_ports=55432-55439`.

Other commands: `pgenv.py status [--check]` shows the instances (`--check` also compares row
counts on every instance, so never run it during timed grading), `pgenv.py stop` shuts them down
cleanly, and `pgenv.py start` starts them again. `pgenv.py --help` lists the rest.

### A database you provide

For the solve phase, the organizers' entry point, you can use your own PostgreSQL 16 server on
the same machine instead. Its database must be named `tpch` and hold TPC-H SF3. As a superuser,
create the agent role:

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

The variable is a comma-separated list of such entries: `pgenv.py` sets up two, one per solving
instance, and a database you provide gives one. Grading always uses the grading instances that
`pgenv.py` sets up.

## Reproduce the final result

The final harness is the git tag `final`: core plus `best/effisql`'s pack, or v0's pack if no
RSI round accepted a change. Reproducing the final result is one command, on a clean checkout of
that tag:

```sh
git checkout final
make final-plan   # print what make final would run, and check the checkout; runs nothing
make final        # prepare, held-out x3 (official mode), val x3 (dev mode), then the report
```

`make final` does, in order:

1. `make prepare`: the TPC-H database, the 5 PostgreSQL instances and the solve sandbox image;
2. the final harness 3 times on held-out, graded in official mode, then 3 times on val, graded in
   dev mode. Each run is `goaly run --bench effisql` in its two phases: the solve phase in the
   container, then the grade phase on the host. One run's grading overlaps the next run's
   solving;
3. `make report`, which writes `docs/results/final-report.md`.

Runs go to `runs/final/{heldout,val}`, each phase's log to `runs/final/logs/`, and the host's
clock around every phase to `runs/final/stages.jsonl`. Rerunning `make final` continues where it
stopped: a graded run is skipped and a solve phase resumes. The final runs take hours: each
official held-out run grades for about 50–70 minutes.

The report reads v0's runs from `runs/baseline/{heldout,val}`. To use archived v0 runs that live
elsewhere, pass their root, as in `make report BASELINE_RUNS=/path/to/baseline`. To draft the
report while runs are still grading, run `make report REPORT_ARGS=--allow-partial`. Record the
constant-branch audit's verdicts in `docs/results/audit-verdicts.json` before publishing.

Maintainers create the tag with `make freeze`, which prints the plan, then
`make freeze FREEZE_ARGS=--yes`. It checks that main's pack equals `best/effisql`'s, tags main's
commit and pushes only the tag. It never merges.

Results are statistically reproducible, not token for token: sampling runs at T=0.6, and the
gateway ignores the `seed` parameter. Every run's manifest records the commit, the harness and
config hashes, the dataset revision, and the models with their sampling parameters. `uv.lock`
pins every dependency.

## Run the organizers' task file

The official entry point runs only the solve phase, on any task file in the dataset's format:

```sh
git checkout final
uv sync --locked
uv run --locked goaly run --bench effisql --instances heldout.jsonl --phase solve --out runs/heldout
```

It needs Docker and the PostgreSQL setup above, from `pgenv.py` or your own database. The
predictions land in `runs/heldout/official/seed-<k>.jsonl`, one file per seed (`--seeds N`,
default 1), written by the host. Each file holds every input record plus its `prediction`
field. `prediction` holds the rewritten SQL and a diff against `base_sql`, as two fenced blocks,
so the official runner can read it in both of its modes: `patch`, its default, and `end2end`.

A task whose plan is missing, or was recorded at another scale, gets our own `EXPLAIN` plan if
the plan cache holds one. To write the cache for your file first, run
`uv run --locked python scripts/effisql/pgenv.py plans --tasks heldout.jsonl`.
`goaly run --help` lists every option.

## Rerun the baseline, v0

v0 is the human-designed baseline (spec §5.4): two tools, `execute_sql` and `submit`, a minimal
prompt, thinking on, a budget of 30 steps or 10 minutes per task, and `base_sql` submitted
unchanged when the budget runs out. It is tagged `v0`. To rerun it as the baseline phase ran it:

```sh
git checkout v0
make baseline   # train x1 and val x2 in dev mode, then held-out x1 in official mode
```

Runs go to `runs/baseline/{train,val,heldout}`, where `make report` reads them. `V0_REF=<ref>`
evaluates another commit; the checkout must be clean and at that commit. `make final` does not
rerun v0, which would add its solving and about an hour of official grading.

## Run RSI

RSI needs the `rsi` dependency group, the PostgreSQL setup and the sandbox image
(`make prepare`), and the `v0` tag. The lane reads `rsi/lane-effisql.toml`: its round sizes, its
decision thresholds and its absolute `deadline`, set to the freeze time. Past the deadline the
lane only stops, so set a future deadline before a new run.

```sh
uv sync --locked --all-groups
uv run --locked goaly rsi run --detach   # start the supervisor in the background, and return
uv run --locked goaly rsi status         # print the lane's state as JSON; changes nothing
```

`goaly rsi run --detach` starts the supervisor in its own session, with no terminal input, and
prints its pid and its log, `runs/rsi/supervisor/supervisor.log`. A lock keeps it to one
instance, so running it again while it runs only reports that it is running. Before each lane
start, the supervisor checks Docker, the sandbox image, the gateway, PostgreSQL and free disk.
It restarts a crashed lane with backoff, and gives up after 5 crashes within an hour. Running
the command again after the machine restarts resumes the lane from the ledger.

The ledger is in `rsi/ledger/`, which git ignores: `experiments.jsonl`, `lessons.jsonl`, the
generated `lessons.md` and `lane_events.jsonl`. Implementer trajectories are under
`runs/rsi/effisql/<experiment_id>/`. Accepted changes reach `main` only through the "RSI
results" PRs that the lane opens from `rsi/effisql/results-<n>` branches, after review.

## Results

The final report is [docs/results/final-report.md](docs/results/), which `make report` generates
from the run manifests and the ledger. Every number here comes from it.

| Held-out (60 tasks, less exclusions) | v0 | Final | Δ |
|---|---|---|---|
| Checked R-VES | _TBD_ | _TBD_ | _TBD_ |
| Official R-VES | _TBD_ | _TBD_ | _TBD_ |
| Coverage@2x (checked) | _TBD_ | _TBD_ | _TBD_ |
| GMS (checked) | _TBD_ | _TBD_ | _TBD_ |
| Mean cost per task | _TBD_ | _TBD_ | _TBD_ |
| Mean wall time per task | _TBD_ | _TBD_ | _TBD_ |

- v0 is 1 official run, with a bootstrap 95% confidence interval over tasks. Final is the mean ±
  standard deviation of 3 official runs. Δ has a paired bootstrap interval over tasks. _TBD_
- Variation between runs, on val in dev mode, over v0's 2 seeds and the final's 3: _TBD_
- RSI: rounds finished, experiments, accepted changes, and the total experiment cost by
  purpose: _TBD_
- Lessons, each citing its experiments: _TBD_, in the report.

## Limitations and residual risks

From spec §12. The final report states each one that applies.

- **One benchmark.** SWE-fficiency is on hold, so gains are shown on Effi-SQL only, and the
  criterion "gains on more than one benchmark" is not met.
- **Few rounds.** The schedule planned 1 RSI round, with a second as upside. If no round
  finished, the final harness is v0, and the report says so.
- **62 Effi-SQL tasks unused.** Their BIRD-Interact databases are not published. The harness
  still runs such tasks if you provide their databases.
- **Data-gated constants.** A prediction that returns a precomputed single row only when the
  data looks like SF3, and computes the answer otherwise, passes both hard-coding checks. The
  project lead accepted this risk. Its mitigations are the hard-coding scan on pack code, the
  prompt rules, review of every results PR, and the report's audit of every final prediction
  whose plan has a constant output branch.
- **The static allowlist on pack code may be incomplete.** Isolation is the real boundary: pack
  code runs only in the solve container, which holds no key, no grader credentials and no gold,
  and reaches only Qwen, the read-only agent role and `<run>/solve/`.
- **Timing noise.** A τ near 1 can flip between credit tiers from one run to the next. Medians of
  3 runs on pinned instances, the sign test, the regression bound, the confirmation seed and 3
  final runs limit its effect.
- **Our TPC-H data may differ from the authors'.** We build SF3 with an open-source dbgen. Row
  counts are checked against the dataset's plans, and tasks that misbehave on this machine are
  excluded with a recorded reason. 11 of the dataset's base plans were recorded at SF10: the
  agent gets our SF3 plans for those, and grading uses times measured on this machine.
- **The organizers' evaluation mode is unknown.** Each prediction carries both the SQL and the
  diff, for `patch` and for `end2end`.
- **Docker is required.** There is no non-Docker fallback for official runs, and Docker Desktop
  does not work.
- **No disk quota on `<run>/solve/`.** Nothing bounds what the solve container writes there yet.
  The RSI supervisor's free-disk check is the backstop.

## Development

```sh
make setup             # create .venv from uv.lock, every group, and install the git hooks
make check             # as CI runs it: lint, type check, import contracts, generated code
                       # up to date, unit tests
make test              # unit tests with branch coverage
make test-integration  # tests that need Docker, a database or the real model API
make proto             # regenerate src/goaly/contracts/gen/ after editing proto/
```

Run `make` to list every target. To check that the LLM client and the real gateway still agree,
run `uv run --locked python scripts/llm_smoke.py`. It needs `GOALY_API_KEY`, makes four small
calls to Qwen and GLM, and prints only each call's model, status, tokens and cost. CI does not
run it.

**Add a tool** to a pack: write it in the pack's `tools.py` on the `Tool` protocol of
`goaly.contracts.interfaces`, with a pydantic argument model and an async `run` that returns a
`ToolResult`, and add it to the `ToolRegistry` in the pack's `policy.py`. core's tool executor
validates the arguments, enforces the timeout, truncates long output and turns exceptions into
structured errors.

**Add a benchmark**: implement the `Benchmark` and `Environment` protocols of
`goaly.contracts.interfaces` in `goaly.bench.<name>` (data, environment and grader), add a pack
in `goaly.harness.packs.<name>`, and register the harness and the benchmark's host side in
`goaly.cli.registry()`. The RSI machinery has no benchmark-specific code: a new lane is a pack
plus its `Benchmark` adapter (spec §3.3).
