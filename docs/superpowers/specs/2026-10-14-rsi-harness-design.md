# RSI Self-Evolving Harness: Design

- Date: 2026-10-14
- Status: approved 2026-10-14; amended 2026-10-15
- Linear: project P-CIR-12; this document is T01 = CIR-1633
- Deadline: 2026-10-15 17:00 Pacific (PDT) = 2026-10-16 00:00 UTC

## 1. Background and goals

### 1.1 Task

We are building a self-evolving harness system:

- The **improvement model**, GLM-5.3, drives an RSI agent. The agent creates and refines the executor's harness: tools, skills, system prompts, the agent loop and context management.
- The **executor model**, Qwen3.6-35B-A3B, solves tasks using that harness. The goal is to beat a human-designed baseline on Effi-SQL, which asks for a slow PostgreSQL query to be rewritten into an equivalent, faster one (§4.1).

We hand over the frozen harness. The organizers evaluate it on a hidden held-out set using only the executor model. The improvement model is not available after development ends.

**Amendment (2026-10-15).** The project lead made three decisions:

- Effi-SQL replaces the BIRD-CRITIC PG-530 benchmark.
- The SWE-fficiency lane is on hold (2026-10-15). Its design stays in this document, marked on hold, so the lane can resume without a redesign.
- The deadline is fixed at 2026-10-15 17:00 Pacific (PDT) = 2026-10-16 00:00 UTC.

### 1.2 Deliverables

1. **The frozen harness**: code, configs, prompts, skills and pinned dependencies. The organizers run it with the `goaly run` CLI. They pass in a task file in the benchmark's official format and get predictions back in the official format (§6.4).
2. **The RSI system**: code, an experiment ledger and lessons. It runs several improvement rounds on its own, with no assistant or person in the loop (§8.0).
3. **A report** that covers:
   - baseline and final scores;
   - how much the score varies across 3 repeated runs;
   - generalization to held-out tasks;
   - efficiency: time, tokens and cost, with experiment cost reported separately;
   - lessons, each backed by experimental evidence.

### 1.3 Constraints

| Item | Value |
|---|---|
| Time | One day; deadline 2026-10-15 17:00 Pacific (PDT) = 2026-10-16 00:00 UTC |
| Model gateway | `https://api.goaly.ai/inference/v1` (LiteLLM 1.92 + SGLang), OpenAI-compatible; key in `GOALY_API_KEY` |
| Executor model | `qwen3.6-35b-a3b`: FP8, 35B total / 3B active parameters; 262K native context, which the gateway stretches to 1.01M with YaRN; $0.15 input / $1.00 output per 1M tokens |
| Improvement model | `glm-5.3`: ~744B total / 40B active; thinking cannot be turned off; 1M context; $1.40 input / $4.40 output per 1M tokens |
| Budget | No practical limit on API spend, but efficiency is still graded |
| Machine | 12 cores / 24 threads, 125 GB RAM, ~800 GB disk, no GPU |
| Docker | Works only with `--network=host`, because runc 1.3.6 hits a procfs restriction inside this nested container |

### 1.4 Non-goals

- No training or fine-tuning.
- No development on Effi-SQL's 62 BIRD-Interact tasks, which run on `polar_equipment`, `robot_fault_prediction` and `solar_panel`. Their "scale" dumps are not published, so we use only the 238 TPC-H tasks (§4.1). This is a documented deviation: the harness still handles such tasks if the organizers' file contains them and they provide the database.
- No SWE-fficiency work while that lane is on hold (2026-10-15). Even after it resumes, there is no full 498-task run: this machine can't evaluate that many.

## 2. Acceptance criteria mapping

| PDF criterion | How the design meets it | Evidence |
|---|---|---|
| Benchmark score: starting and final scores, with improvement on every benchmark | Baseline v0 (§5.4) and final are compared on the same held-out set, graded in official mode (§4.1): v0 runs once there and the final three times. While SWE is on hold (2026-10-15), Effi-SQL is the only benchmark scored | T12 and T23 reports |
| Robustness: run the same test at least 3 times and report the variation; survive tool errors, timeouts and bad model output | The final harness runs 3 times on held-out in official mode, which meets the 3-run requirement; v0 runs once there, and its variation comes from its 2 val seeds (§10). Fault tolerance in core (§5.1); per-task exception isolation (§6.1); concurrency budgets and adaptive backoff (§6.2) | T23 report; T04 and T05 tests |
| Generalization: held-out tasks stay out of development; gains on more than one benchmark; no hard-coded answers | Three-way split (§7), a hard-coding scan (§8.5) and the grader's hard-coding checks (§4.1). The RSI machinery has no benchmark-specific code: a lane is a capability pack plus a `Benchmark` adapter (§3.3). **At risk:** while SWE is on hold, gains can be shown on one benchmark only, so "gains on more than one benchmark" is not met. The report says so plainly | Committed split ID lists; scan logs; held-out gains in the T23 report |
| Lessons learned: what was tried, what worked, what failed and why, backed by experiments | Experiment ledger and lessons (§8.6) | `experiments.jsonl`, `lessons.jsonl`, and the generated `lessons.md` |
| Efficiency: time, tokens, compute and cost reported alongside the score | Cost and tokens are recorded for every call (§9). Two limits in the decision rules (§8.3): a cost guardrail, and a hard cap that keeps the final harness at or below 3× v0's mean cost and wall time per task | Report tables, including the final harness's cost and time relative to v0 |
| Reproducibility: code, configs, prompts, tool versions and a single command | uv lockfile, run manifests, `make final` (§10) | T23, T24 |
| Automation: several improvement rounds with little human help | One command, `goaly rsi run`, runs every round with no person or AI assistant in the loop. A supervisor restarts a crashed lane, and the code opens the results PRs and sends the notifications itself (§8.0, §8.7). **At risk:** the schedule plans 1 round, with a second as upside, and no round may finish if the lane starts late (§13) | Ledger entries across rounds; the supervisor log |
| Regression check: keep the best harness; new changes must not break tasks that already pass | Best-version tags and a regression rule (§8.3) | Decision log |

## 3. Architecture

### 3.1 Layers

```
goaly.cli         composition root; the only module allowed to import every layer
   │
goaly.rsi         development only; uses GLM-5.3
   │ depends on
goaly.bench       data, environment implementations, grading, runner, metrics
   │ depends on
goaly.harness     uses only Qwen at runtime
   ├─ core/       infrastructure (RSI cannot edit it)
   └─ packs/      one capability pack per benchmark (the only part RSI can edit)
   │ depends on
goaly.contracts   shared data contracts (protobuf) and interfaces (Python Protocols); imports nothing else from goaly
```

import-linter checks these rules in CI:

- `contracts` imports nothing else from `goaly`.
- `harness` may import only `contracts`.
- `bench` may import `contracts` and `harness`.
- `rsi` may import `contracts`, `harness` and `bench`.
- `goaly.cli` may import every layer. Its `goaly rsi …` subcommands import `rsi` lazily, so `goaly run` never loads it.
- Only `rsi` may import `minisweagent`, the third-party agent loop the implementer runs on (§8.8). A `forbidden` contract checks this, with `include_external_packages = true`.

**What ships.** The organizers receive `contracts`, `harness`, `bench` and the CLI. `rsi` is only for development, and nothing on the evaluation path imports it. Its third-party dependencies, mini-swe-agent among them, live in the `rsi` dependency group, which a plain `uv sync` does not install (§11).

### 3.2 Repository layout

```
proto/goaly/v1/      shared data contracts: task.proto, environment.proto, trajectory.proto,
                     result.proto, manifest.proto, llm_call.proto, experiment.proto
src/goaly/
  contracts/
    gen/             Python code generated from proto/ (committed; CI checks it is up to date)
    interfaces.py    Environment, SqlEnvironment, ShellEnvironment, Policy, Tool, Benchmark
  harness/
    core/            llm.py, llm_proxy.py, tools.py, agent.py, policy.py, context.py, trajectory.py,
                     config.py
    packs/
      effisql/       prompt.md, skills/, tools.py, policy.py, config.yaml
      swefficiency/  (same files; on hold)
  bench/
    effisql/         data.py, env.py, grader.py, adapter.py, and the pipeline that launches
                     isolated runs (§6.1)
    swefficiency/    data.py, env.py, grader.py, adapter.py (on hold)
    runner.py, metrics.py, manifest.py
    sandbox.py       the shared container builder: the isolated solve phase (§6.1) and the implementer (§8.8)
    splits/          split ID lists
  rsi/
    ledger.py, workspace.py, guards.py, analyst.py, decide.py, lane.py,
    supervisor.py, results.py, notify.py
    implementer/     mini-swe-agent adapter: model.py (GLM through core's client), environment.py
                     (the sandbox as a mini-swe-agent environment, goaly-smoke), run.py (run, commit,
                     record), implementer.yaml (prompt and limits)
  cli.py             `goaly run …` and `goaly rsi …`
tests/               mirrors src/; tests that need Docker, a database or the real API are marked `integration`
docker/rsi-sandbox.Dockerfile     the sandbox image, for the solve phase (§6.1) and the implementer (§8.8)
docs/superpowers/specs/           design docs (the Linear tickets serve as the implementation plan)
docs/results/                     experiment results
runs/                             run artifacts (not committed)
```

**Names.** Effi-SQL's benchmark name, which is also its CLI value (§6.4), is `effisql`. Its RSI lane is the `LANE_EFFISQL` enum value, and the lane's best version carries the tag `best/effisql` (§8.1). SWE-fficiency, on hold, keeps `swefficiency` and `best/swe`.

### 3.3 Shared contracts

Every shared type lives in `goaly.contracts`. These types come in two kinds, and each kind uses a different technology.

**Data contracts (protobuf).** Data that is written to disk, read by another process, or has to stay readable across schema versions is defined once, in `proto/goaly/v1/`:

| Message | File | Purpose |
|---|---|---|
| `TaskInput`, `Prediction`, `Grade` | `task.proto` | What a policy receives, what it returns, and how the result was graded. Each has one variant per benchmark |
| `SqlResult`, `ShellResult` | `environment.proto` | What a task's environment returns for a SQL statement or a shell command |
| `Trajectory`, `Step`, `ToolResult` | `trajectory.proto` | Every step of a run: messages, tool calls and their results, usage and timing |
| `TaskResult` | `result.proto` | Per-task record: score, correctness, time, tokens, cost, steps and error type |
| `RunManifest` | `manifest.proto` | Commit, content and config hashes, dataset revision, model settings, timestamps and machine |
| `LlmCallRecord` | `llm_call.proto` | One LLM call: purpose, model, tokens, cost, latency and status |
| `ExperimentRecord`, `Hypothesis`, `Lesson`, `ImplementerRun`, `LaneEvent` | `experiment.proto` | The RSI ledger: experiments with their implementer runs (§8.8), lessons, and lane events such as restarts and results PRs (§8.7) |

- On disk, messages are stored as JSONL using protobuf's standard JSON mapping, keeping the original field names. People and the GLM analyst read trajectories directly, so a readable text format matters more than a compact binary one.
- Files written today must stay readable after the schema changes. Field numbers are never reused. On disk, field names are the JSON keys and enum values are written by name, so a field or an enum value is never renamed either, and when one is removed, its name is reserved as well as its number.
- The official submission formats (Effi-SQL's prediction records, SWE predictions) are not our contracts. Each adapter converts to them and keeps the official field names exactly.
- **Effi-SQL's prediction record** is the input task record plus a `prediction` field, because the official runner reads `base_sql` and `db` from the same record. The runner's default mode, `EVAL_SQL_MODE=patch`, applies the first diff-like block in `prediction` to `base_sql`. Plain SQL therefore fails as "Patch Apply Failed" and scores 0, and the README's command for `effi-sql-pg.jsonl` uses this mode. The other mode, `end2end`, skips diff blocks and takes the SQL. So `prediction` holds both forms, as two fenced blocks in this order and nothing else:
  1. A `sql` block with the full query: one statement beginning with `SELECT` or `WITH`, which is all `end2end` accepts. It must not look like a diff, so comments are stripped: no line may begin with `---`, `+++` or `@@`, and at most one with `-` or `+`.
  2. A `diff` block that replaces every line of the base with the new query. The applier drops one base line per `-` line without comparing their text, so the block has exactly one `-` line per base line. Its `-` and `+` lines may not begin with `---` or `+++`, which the applier reads as headers.

  Both blocks carry the same comment-stripped query, ending with `;`.
  - **Why `sql` comes first.** `end2end`'s plausibility test rejects some valid queries. 12 of the dataset's 300 base queries use `WITH x AS MATERIALIZED (`, which it refuses, and `end2end` then falls back to parsing the whole field. With the diff first, 7 of those 12 came back as garbage, which broke the identity fallback. With the `sql` block first and the `;`, every one of the 300 base and gold queries comes back exactly in both modes, as T11 verified.
  - Patch mode still applies the first diff-like block, and the `sql` block never looks like a diff.
  - A comment such as `-- final query` in either block would match the runner's final-answer marker (`FINAL_SQL_MARKER_RE`) and add a competing `end2end` candidate.

  The v0 fallback (§5.4) is encoded the same way, as `base_sql` plus an identity diff. A query the format cannot carry, such as several statements or a marker phrase inside a string literal, becomes an empty prediction with the problem stated. It is graded `PREDICTION_FAILED`, so our grade matches what the official runner sees; an empty prediction always fails. Which mode the organizers run is unknown (§12).
- Toolchain:
  - `grpcio-tools` bundles protoc, so no system package is needed.
  - `mypy-protobuf` generates type stubs, so the generated code passes `mypy --strict`.
  - `make proto` regenerates the code, and CI fails if the committed code is out of date.

**Behavioral interfaces (Python Protocols).** These are async method contracts for calls within a process. The signatures below are a draft; the contracts PR (T02a) fixes the final ones.

```python
class Environment(Protocol):
    """The runtime environment for one task."""
    async def close(self) -> None: ...

class SqlEnvironment(Environment, Protocol):
    async def execute(self, sql: str, *, timeout_s: float) -> SqlResult: ...

class ShellEnvironment(Environment, Protocol):
    async def exec(self, command: str, *, timeout_s: float) -> ShellResult: ...
    async def read_file(self, path: str) -> str: ...
    async def write_file(self, path: str, content: str) -> None: ...

class Policy(Protocol):
    """Implemented by a capability pack; solves one task."""
    async def solve(self, task: TaskInput, env: Environment, ctx: AgentContext) -> Prediction: ...

class Tool(Protocol):
    name: str
    description: str
    args_model: type[BaseModel]
    async def run(self, args: BaseModel, env: Environment) -> ToolResult: ...

class Benchmark(Protocol[InstanceT]):
    name: str
    def load_instances(self, split: Split) -> Sequence[InstanceT]: ...
    def provision(self, inst: InstanceT) -> AbstractAsyncContextManager[Environment]: ...
    def agent_input(self, inst: InstanceT) -> TaskInput: ...  # only what the agent may see
    async def grade(self, inst: InstanceT, prediction: Prediction) -> Grade: ...
    def to_official(self, inst: InstanceT, prediction: Prediction) -> dict[str, Any]: ...
```

- **Instances versus agent input.** `InstanceT` is the benchmark's own record and may hold gold answers. `agent_input` copies out only the fields the agent is allowed to see, so the type system itself keeps gold answers away from the agent.
- **harness never cares where an environment comes from.** bench provides the implementations: `SqlEnvironment` as a read-only session on a PostgreSQL solving instance (§4.1), `ShellEnvironment` as a Docker container. harness unit tests use fake environments and need neither a database nor a container.
- **Tool arguments stay as pydantic models.** The OpenAI tools API takes JSON Schema, which pydantic generates directly. GLM also edits tools during RSI, and with protobuf every edit would mean changing a `.proto` file and rerunning code generation.
- **gRPC-ready, but no gRPC yet.** Interface methods take and return only contract types (protobuf messages or plain values), never framework objects. That way a gRPC-backed implementation can later replace an in-process one without changing any caller. One example is running grading instances, or SWE's evaluation workers, on other machines, which would lift the CPU ceiling described in §6.2. We are not building gRPC now.

Adding a benchmark takes a `Benchmark` adapter and a capability pack. core does not change.

## 4. Benchmarks

### 4.1 Effi-SQL (making slow SQL faster)

**Data**

- We use the Hugging Face dataset `birdsql/Effi-SQL`, pinned at revision `86cbcc9efbd31db32a46897d47cbf865b1cba3a4`. It is licensed Apache-2.0 and is not gated.
- It is one file, `effi-sql-pg.jsonl`, with sha256 `2af3bdc9fe6c1197f23e31089660e7ce10fca6956b32e5970a3113f0c87e9d97`: 300 PostgreSQL tasks, released 2026-06-12.
- Fields:

  | Field | Meaning |
  |---|---|
  | `id` | int |
  | `db` | database name |
  | `base_sql` | the original slow query |
  | `optimized_sql` | the human-verified optimized query (gold) |
  | `base_time`, `fast_time` | milliseconds, measured on the authors' hardware |
  | `base_explain_analyze`, `optimized_explain_analyze` | JSON query plans |
  | `difficulty` | easy, medium or hard |

- Databases:
  - 238 tasks run on `tpch`, which is TPC-H at scale factor 3. Only its raw data is external, and we generate that locally with dbgen.
  - 62 tasks run on three BIRD-Interact databases: `polar_equipment` (14), `robot_fault_prediction` (22) and `solar_panel` (26). Their "scale" dumps are not published: the official code says to prepare them locally.
- **We use only the 238 TPC-H tasks.** The other 62 are a documented deviation (§1.4). The harness still handles such tasks if the organizers' file contains them and they provide the database.
- **What the agent sees:** `base_sql`; `db`; the database schema (DDL, row counts and indexes); the base query's plan; and `difficulty`. The plan is the file's `base_explain_analyze`, unless the file has no plan or recorded it at another scale. T09 detects that case and writes our own `EXPLAIN (FORMAT JSON)` on SF3 to `plans.json`, which T08 prefers over the file's plan.
- **What the agent never sees:** `optimized_sql`, `fast_time` and `optimized_explain_analyze`.
- **Plans in the dataset.** `base_explain_analyze` holds the plan's root node. For 154 of the 238 TPC-H tasks it is an `EXPLAIN ANALYZE` plan with actual row counts and times. For the other 84 it is a plain `EXPLAIN` plan, with estimates only.
- **Plans recorded at another scale.** 11 base plans were recorded on TPC-H SF10, not SF3: tasks 2, 24, 57, 72, 80, 87, 94, 99, 134, 137 and 146. T09's detector flags exactly these. Their tables have SF10's sizes (`lineitem` 59,986,052 rows, `orders` 15,000,000, `customer` 1,500,000), and the file's `base_time` for each comes from the same SF10 run. Task 80, for example, returns 17,995 rows on SF3, where its plan records 59,952. Some gold plans show SF10 sizes too, not always for the same tasks.

**Task**

- The agent rewrites `base_sql` into one query that returns the same rows, faster, and submits it.
- A valid query computes its rows from the tables. A query that embeds result values, such as the base query's output copied in as constants, is invalid, and the grader's hard-coding checks give it no credit (Grader, below).
- It may not create indexes or change data (Environment, below).

**Scoring** (the official evaluator's algorithm)

1. **Equivalence.** The prediction is executed fully and its rows are fetched in batches. Its result signature is the multiset of `sha256(str(row))`, plus the row count. The same is done for `base_sql`: the comparison is against the base query, not against `optimized_sql`. The two match only if the row counts and the multisets are equal.
2. **Timing.** `EXPLAIN (ANALYZE, FORMAT JSON)` is run, and the plan root's `"Execution Time"` is read in ms. There are 3 runs, and each run executes the prediction and the base in random order (interleaved). Timing stops after 2 runs if either side took 60 s or more. The time for each side is the median of its runs.
3. **Speedup.** τ = base_time / pred_time, capped at 1000. A pred time of 0 counts as 0.001 ms.
4. **Statement timeout:** 300 s.
5. **Per-task credit g(τ),** for matched tasks only:

   | τ | credit |
   |---|---|
   | ≥ 2 | 1.0 |
   | ≥ 1.2 | 0.75 |
   | ≥ 1 | 0.5 |
   | ≥ 0.5 | 0.25 |
   | otherwise | 0 |

   An unmatched or failed task gets 0. A task whose timing fails after it matched stays matched, with a speedup of 0, so its credit is 0 too.
6. **Metrics:**
   - **R-VES** = 100 × mean credit over all tasks.
   - **Coverage@2x** = 100 × the share of tasks that are matched with τ ≥ 2.
   - **GMS** = the geometric mean of τ over matched tasks. We compute it from the capped τ, over matched tasks with τ > 0.

   The official aggregator computes only the match rate and R-VES. Coverage@2x and GMS follow the definitions of the BIRD-CRITIC site's Effi-SQL metrics (https://bird-critic.github.io/).

**Environment**

- PostgreSQL 16, the same major version the official evaluator uses.
- TPC-H SF3 is generated locally with an open-source dbgen build. The schema has primary and foreign keys only, matching the official schema, and loading is followed by `ANALYZE`. The data is never committed.
- One data directory is copied into 5 instances, each pinned with cpuset (§6.2): 2 solving instances and 3 grading instances, 2 physical cores each. Solving and grading never share an instance.
- Every grading instance also holds an **alternate TPC-H database** at a smaller scale, such as SF0.3, built the same way. Its data differs from SF3's, which the robustness check (Grader, below) relies on.
- Agent sessions connect with a read-only role. It has only the `SELECT` privilege and no `CREATE`, so the agent cannot create indexes or change data: the official evaluation runs on the pristine database. The statement timeout is enforced by client-side cancellation, because a session can change its own `statement_timeout` and `default_transaction_read_only`.
- The grader runs predictions under a `SELECT`-only role too, in a read-only transaction, with client-side cancellation. Both roles default to `default_transaction_read_only = on`, as defense in depth; the privileges are what actually block writes.
- Every instance requires password authentication. Grading credentials never enter the repository or the implementer sandbox (§8.5).

**Grader**

- A clean reimplementation of the official algorithm (Scoring, above). The official evaluator is GitHub `L-nshpp/Diff-SQL`, file `diff_sql/evaluation/eval_execution_runner.py`. That repository has no license, so we reimplement its algorithm rather than vendor its code.
- Two modes:
  - **Dev mode,** for RSI screening and validation. The base query's result signature and its timing (the median of 3 runs) are measured once per task and cached. Each evaluation then runs only the prediction: one comparison run plus up to 3 timed runs. The 3 grading instances run in parallel. A prediction identical to `base_sql` after normalization is not re-timed: it gets the identity τ measured during setup, and that τ's credit. This saves about 110 s per such task and removes its tier-flip noise.
  - **Official mode,** for the reported held-out numbers: the exact interleaved protocol, with no base cache. It always measures, identity submissions included.
- **Result signatures.** Results can be large. At least 56 gradable tasks return more than 1M rows on SF3, counted from the plans that record actual rows at SF3. The largest SF3 results are around 18–20M rows: a full `lineitem` is 17,996,609 rows (task 1), and task 45's join returns 19,719,920. The setup measures the exact counts on SF3. The grader therefore reads rows through a server-side cursor and hashes them off the event loop. It keeps an order-independent compact digest, the sum of the row hashes mod 2^256 plus the row count, and never a dictionary of row hashes.
- **Hard-coding checks.** A prediction can return the base query's result as constants, for example `SELECT 12345.67 AS revenue` or `SELECT * FROM (VALUES …)`. Such a prediction matches and scores τ = 1000. Many tasks are exposed: of the 154 tasks whose plan records actual rows, 25 return a single row. Two checks therefore run on every matched prediction, in both grading modes:
  1. **Robustness check.** The prediction must also return `base_sql`'s result on the alternate database. Otherwise its credit is 0, and it is flagged `hard_coded` (contract field `EffiSqlGrade.alt_matched`).
  2. **Plan check.** `EXPLAIN (FORMAT JSON)` runs on every prediction before its comparison run, and a `ModifyTable` plan is refused there (below). The `literal_result` test applies only once the prediction has matched. Its credit is then 0, and it is flagged `literal_result`, if its plan scans none of the base query's relations, or if its output comes from a Values Scan or Result subtree with no relation scan beneath it. `VALUES` itself stays allowed: 3 golds use it, such as task 1, which unpivots with `VALUES (t.l_extendedprice), …`. This check closes the gap the first one leaves: `region` and `nation` have the same rows at every TPC-H scale, so a constant answer over them matches on both databases.

  Three rules govern how the checks apply:

  - **Matched predictions only.** A wrong rewrite also fails on the alternate database, so checking it would flag nearly every rejected candidate as hard-coded. An unmatched prediction's flags therefore stay unset.
  - **Flags do not stop timing.** A matched prediction is timed even when a check flags it.
  - **Data-modifying plans are refused, not flagged.** A plan with a `ModifyTable` node is refused at that first `EXPLAIN`, before the prediction runs, and the prediction counts as not matched (`PREDICTION_FAILED`).

  Gold mode must pass every gold that is not excluded, which calibrates both checks. In dev mode a flagged prediction gets credit 0, and in RSI it fails its candidate outright; the ledger and the lessons record the flag (§8.2). In official mode the official R-VES stays as the evaluator computes it, and the **checked R-VES** gives flagged predictions credit 0. The report gives both. In the grade contract, `credit` is the checked credit; the official credit is g(`speedup`) when matched. The checked Coverage@2x and GMS also leave out flagged tasks, because a hard-coded task stays matched with τ ≈ 1000 in the official numbers.
- **Self-checks.** The gold check alone is circular, because a gold that fails is excluded, and a matcher that always says "match" would pass both the gold and the identity check. The self-checks therefore include checks that can fail:
  - **Gold mode:** the gold SQL must match and speed up.
  - **Identity mode:** submitting `base_sql` must give τ ≈ 1.
  - **Negative checks:** SQL variants of `base_sql` that drop a row, duplicate a row, or reorder or perturb a column must not match. They run on a fixed sample of 10 non-excluded tasks whose results are small (under 10K rows), alongside matcher-level unit tests on synthetic rows, so the setup stays at about 14 instance-hours.
  - **Data check:** a task's base row count must equal the root `Actual Rows` of its `base_explain_analyze`, for the tasks whose plan records actual rows at SF3: 143 of the 154, once the 11 plans recorded at SF10 are set aside. For example, task 1 returns 17,996,609 rows. This catches a dbgen data mismatch. A mismatch on a plan recorded at another scale is a dataset anomaly, not a data mismatch: it neither fails the check nor counts toward the 10% stop rule.
  - **Exclusion bound:** if more than 10% of tasks are excluded, stop and investigate before RSI starts.
- **Excluded tasks.** A task is excluded if, on this machine, its base query errors or exceeds 300 s, or the gold `optimized_sql` does not match or is less than 1.2× faster. These are times we measure; the file's `base_time` is not used, since some were recorded at SF10. A task is also excluded if its gold fails the robustness check or the plan check and review finds the gold, not the check, at fault. On such a task either nothing can be scored or even the gold shows no reliable gain, so keeping it would only add noise to development and to accept/reject decisions. Exclusions are applied after splitting (§7), and the excluded tasks are committed with the reason for each. This affects only our own splits: the harness still handles any such task in the organizers' held-out set.
- **Noise.** Tier boundaries make a τ near 1 flip between credit 0.25 and 0.5 from one run to the next (§12). In dev mode, the fixed identity τ removes this noise for identity submissions.

**Reference numbers for the TPC-H subset**, from the dataset's own timings:

- Base time: median 11.8 s, p90 69 s, max 450 s. 5 tasks exceed 300 s, and one pass over every base query takes 2.13 h.
- Difficulty: easy 71, medium 111, hard 56.
- Gold scores: R-VES 88.3, Coverage@2x 53.4, GMS 3.55. This GMS uses the uncapped speedups; with τ capped at 1000, as the grader caps it, gold's GMS is 3.29.

These are the file's own timings. Some come from plans recorded at SF10, including 4 of the 5 tasks over 300 s and the 450 s maximum, so exclusions and grading use times measured on this machine.

### 4.2 SWE-fficiency (making repository code faster): on hold (2026-10-15)

**On hold (2026-10-15).** The project lead put this lane on hold. Its design stays below so the lane can resume without a redesign, but nothing in this section is being built now, and T13–T17 wait in the Linear backlog.

**Data**

- The main dataset is `swefficiency/swefficiency`: 498 tasks from 9 repositories. It was revised on 2026-03-30, and we pin that revision by commit hash.
- The Lite set (`swefficiency/swefficiency_lite`, 100 tasks) was last updated on 2025-11-21, before that revision. We use it only to pick task IDs. Every task's data (workload, tests, `test_cmd`, `rebuild_cmd`, image name) comes from the revised main dataset, which the organizers most likely use as well.
- Lite's image names point to `ghcr.io/swefficiency-anon/…`, which now refuses access. The images are available under the same tags at `ghcr.io/swefficiency/swefficiency-images`. T13 checks whether the revised main dataset already uses the new location; if it doesn't, the loader rewrites the registry prefix and records the rewrite in the run manifest.
- The agent never sees `patch`, `problem_statement` or `hints_text`.

**Task**

- The agent gets a repository at `/testbed` inside a container, plus a workload script.
- It changes the code so the workload runs faster, and outputs a git diff.
- It may not modify the workload or the tests, and may not install dependencies.

**Scoring**

- Each task scores SR_i = (T_pre / T_post_agent) / (T_pre / T_post_gold): the agent's speedup divided by the expert patch's speedup.
- The overall score is the harmonic mean of SR_i. No task scores below 0.001.
- If the patch doesn't apply, or any PASS_TO_PASS test fails, the task counts as no speedup: SR = 1 / gold speedup. That is the same as an empty patch.
- A correct patch that makes the code slower is scored as measured, with no clamping. It therefore scores below an empty patch.
- For reference, empty patches on the full set score about 0.024. Claude 4.5 Opus scores 0.225.

**What this scoring means for strategy**

1. Submit a patch only if the measured speedup clears a threshold and the relevant tests pass. Otherwise submit an empty patch.
2. The harmonic mean adds up the 1/SR terms. An empty patch makes a task's term equal to the expert's speedup. Partial progress on tasks with a large expert speedup is therefore worth the most, and any slowdown is amplified.

**Environment**

- Each task has an official image (linux/amd64, 1.4–3.6 GB compressed).
- Containers run with `--network=host`, cpuset pinning and a memory limit.

**Grader**

- We wrap the official `swefficiency eval/report` and adapt it in two ways:
  - it uses host networking;
  - CPU allocation is fitted to this machine: at most 4 evaluation workers, each on 2 physical cores.
- Gold is rerun on this machine to get T_post_gold for our hardware.
- Empty patches are scored without running an evaluation.
- Results are cached by (patch hash, task). T_pre and gold timings are cached too.
- **Excluded tasks.** A task is removed and replaced with another task from the same repository if, on this machine, its gold patch fails its tests or speeds the workload up by less than 1.5×. Expert speedups depend on hardware, and on such tasks no change can show a reliable gain. The removed tasks and the reason for each are committed. As with Effi-SQL, this affects only our own splits.
- **Open question: what the official evaluation runs.** Some tasks list a huge number of PASS_TO_PASS tests: one Lite pandas task has about 211K, alongside about 900 `covering_tests`. Whether evaluation runs every PASS_TO_PASS test or only `covering_tests` decides whether one task takes minutes or hours. T15 measures this first. Tasks that take more than 25 minutes to evaluate are swapped out (§7).

**Resource constraint:** evaluation is this lane's biggest bottleneck. Solving and evaluation run on separate CPU slots so one does not distort the other's timings.

**Known failure modes**

- Wrong hotspot: 68% of expert gains come from functions the models never edit.
- Satisficing: stopping after the first small speedup.
- Gaming: caching workload results or special-casing the workload.

## 5. Harness design

### 5.1 core (RSI cannot edit it)

**LLM client**

- Async, httpx-based, OpenAI-compatible.
- Each call times out after 300 s. 429s, 5xx errors, timeouts and connection errors get up to 5 retries with exponential backoff. A gateway 408 counts as a timeout. Concurrency limits and adaptive backoff follow §6.2.
- Reads usage, including reasoning tokens, and the `x-litellm-response-cost` response header.
- Writes one JSONL record per call, labeled with the purpose of the call.
- Turns context overflow and similar failures into specific exception types.

**Model allowlist.** The allowlist is per purpose, lives in core, and RSI cannot edit it:

- Executor calls (purpose `exec`) may use only `qwen3.6-35b-a3b`, so at runtime the harness calls nothing else.
- RSI calls (purposes `rsi-*`) use `glm-5.3`.

A call to a model its purpose does not allow raises an error. In isolated runs, the host's LLM proxy enforces the same rule for the solve container: it forwards only the executor model (§6.1).

**Who holds which client.** The runner hands the harness an LLM client bound to purpose `exec`. Harness and pack code cannot choose a purpose, and only `goaly.rsi` builds clients with `rsi-*` purposes. Pack code also may not import `settings` or the LLM client modules, nor reach private attributes such as the bound client's internals: the pack-code allowlist admits only an explicit list of `goaly.harness.core` modules and bans the known escapes (§8.5). An import-linter contract keeps those core modules from importing `settings` or the LLM client modules, so they cannot re-export the key or a client. That allowlist is a static check, not a sandbox. In every real run, pack code runs only in the solve container (§6.1, Isolated evaluation). There the client talks to the host's proxy with a per-run dummy token, so the key never enters pack code's process; only the one-process mode, for unit tests and the toy benchmark, runs pack code next to the client's real configuration. The allowlist stays as a second layer, a static defense in depth, and §12 names the residual risk.

**Model-specific handling**

- Qwen:
  - `chat_template_kwargs.enable_thinking` turns thinking on or off. `preserve_thinking` is optional.
  - The gateway rejects `reasoning_effort`. The client never sends a top-level `enable_thinking`, although a probe found that the gateway accepts it: thinking is switched only through `chat_template_kwargs.enable_thinking`.
  - Known issue: tool calls sometimes end up in `reasoning_content`. When that happens, the parser falls back to Qwen3-Coder XML (`<tool_call><function=…><parameter=…>`).
  - Known issue: argument JSON can be truncated or malformed. We repair it where possible and otherwise send the error back to the model.
- GLM (used only by RSI):
  - Thinking is always on. The gateway rejects the `thinking` parameter.
  - `reasoning_content` is sent back unchanged in the message history. Every GLM chat replays its reasoning unchanged: the analyst's, the implementer's and the lesson writer's (§8). The executor's replay rule below does not apply to them.
  - The RSI implementer runs on mini-swe-agent, which is synchronous, in worker threads (§8.8). Its model adapter schedules each call on the lane process's event loop with `asyncio.run_coroutine_threadsafe`. Implementer calls therefore get the same GLM profile, retries, concurrency limits and call records as every other call.

**Tool executor**

- Validates arguments with pydantic and enforces timeouts.
- Long output is cut down to its first and last 4K characters.
- When a tool raises, the model gets a structured error message.

**Agent loop**

- Budgets cap steps, wall-clock time, tokens and cost.
- After 3 consecutive errors the loop reminds the model of the tool schemas. After 5 it stops.
- Calling the same tool with the same arguments 3 times in a row triggers a hint.
- Context management: Qwen gets a 128K context budget. Past 96K, older tool outputs are folded into summaries. The system prompt, the task and the last 8 turns stay as they are.
- **Replay rule.** When the executor's chat is rebuilt for the next request:
  - XML tool-call markup is stripped from the replayed content and reasoning, whether or not a call was executed from it;
  - `reasoning_content` is replayed only within the last keep_turns = 8 steps. Earlier steps keep their content and drop their reasoning;
  - keep_turns is recorded with the trajectory, where the rebuild reads it.

  GLM chats replay their reasoning unchanged (model-specific handling, above).
- When a budget runs out, the policy's fallback submits a result.
- No exception can take down a batch.

**Trajectories** use the `Trajectory` and `Step` contracts (§3.3) and are written as JSONL. Each step records messages, tool calls and their results, token usage and latency. The trajectory ends with the final prediction. In the one-process mode it also carries the grade. In two-phase runs it carries none: readers take grades from `results.jsonl` (§6.1).

### 5.2 Capability packs (the only part RSI can edit)

| File | Purpose |
|---|---|
| `prompt.md` | System prompt template |
| `skills/*.md` | Curated skill guides. Ideally no more than 3, each focused; any of them can be removed |
| `tools.py` | Tools for this benchmark, built on core's Tool protocol |
| `policy.py` | How a task is solved: phases, self-checks, when to add guidance, best-of-n, fallback submission |
| `config.yaml` | Sampling parameters, thinking on/off, budgets, context thresholds |

**How skills reach the model** is up to the pack's policy, and core supports two options:

- put the skills straight into the system prompt;
- list them in an index that the model reads on demand through a `read_skill` tool.

The first option costs more tokens, although the gateway's prefix cache absorbs much of that. The second relies on the model remembering to read. Which works better is something RSI can test. v0 ships no skills.

### 5.3 Principles from prior work

- Self-checks use deterministic execution results only: running tests, timing, executing SQL. The executor never grades its own work.
- Guidance goes into tool results and mid-run messages, not a longer and longer system prompt.
- Keep skills few and focused. Tools and rules that don't help get removed.

### 5.4 Baseline v0 (the "simple tools" setup the PDF asks for)

- Effi-SQL has two tools:
  - `execute_sql` runs a query on a solving instance, with a 120 s timeout plus an executor backstop at the task's wall budget, 600 s for v0, and returns truncated rows or the error. It reports the execution time and, separately, the time the statement waited in the instance's queue (§6.2).
  - `submit` hands in the answer.

  The prompt is minimal. It states that the rewritten query must compute its result from the tables and return the same result as `base_sql` on any data, so embedding result values makes it invalid (§4.1).
- SWE (on hold, 2026-10-15) has two tools:
  - `bash` runs a command in the task container.
  - `submit` hands in the answer.

  The prompt is minimal but lists the official constraints.
- Both use the same settings:
  - Thinking is on. Sampling is T=0.6, top_p=0.95, top_k=20, with max_tokens 16K per call.
  - Solving budgets are 30 steps or 10 minutes for Effi-SQL, and 60 steps or 45 minutes for SWE.
- Fallback when a budget runs out before `submit`: v0 still submits something rather than nothing.
  - Effi-SQL submits `base_sql` unchanged. That gives τ ≈ 1, so a credit of 0.25–0.5. Its official record holds `base_sql` plus an identity diff (§3.3).
  - SWE submits the container's final git diff, as OpenHands and other common agents do.

  The SWE fallback does not check whether the agent's diff is any good, and the Effi-SQL fallback ignores any faster rewrite the agent found. Smarter policies, such as submitting the fastest rewrite the agent verified, or never submitting an unverified patch, are left for RSI to discover.
- v0 is frozen and tagged `baseline-v0`. Every comparison is made against it.

## 6. Evaluation framework (goaly.bench)

### 6.1 Runner and records

**Runner**

- Built on asyncio, with separate resource pools for: LLM concurrency, the solving and grading database instances, and, for SWE (on hold), solving CPU slots and evaluation workers (at most 4).
- An exception in one task never affects the others. Each task's result is written to disk as soon as it's done, so an interrupted run can resume.
- Evaluations are cached by (harness identity, task digest, seed, grading mode), on the host only (Isolated evaluation, below). Cache entries carry the grade.

**Isolated evaluation.** The project lead decided on 2026-10-15 to isolate candidate code. The static allowlist (§8.5) kept leaking, and pack code shared a process with the API key, the LLM client and the grader. One bypass could therefore steal the key, patch the grader in memory, or call GLM. The isolation closes this structurally, and the allowlist stays as a second layer.

1. **Two phases, two processes.** `goaly run` has a solve phase and a grade phase (`--phase solve|grade|all`, §6.4).
   - **The default command is the isolated pipeline.** `goaly run --bench effisql`, which runs `--phase all`, starts the proxy, then the solve container, then grades on the host in a fresh process.
   - **The launcher.** T11's `bench/effisql` pipeline is the host-side launcher that starts the proxy and the container, and T06's CLI calls it through a benchmark hook. Inside the container, the worker runs as `goaly run --phase solve --in-container`, an entry point that only the launcher uses.
   - **The solve phase** runs the pack's policy, inside the container.
   - **The grade phase** runs on the host, as a separate process, and grades with the grader role. It never imports pack code: only `bench` and `core` are loaded.
   - **Every real Effi-SQL run takes this path,** so v0 and candidates are measured the same way. That covers the T12 baseline, every RSI evaluation (smoke, screening, validation and confirmation) and the final runs.
   - **The one-process mode** needs an explicit `--in-process` flag. It refuses to run with the real gateway key unless `--unsafe-in-process` is also given. It is for unit tests, the toy benchmark and local debugging only, and no real experiment uses it.
2. **The trust boundary.** The solve container is untrusted, and the host never trusts a file the container wrote for anything that decides a score.
   - **The container writes only `<run>/solve/`:** its predictions, its trajectories, its ungraded result lines, and its own call log, which is informational only.
   - **Host-only files.** Everything else in the run directory is written only by the host and never mounted writable: the graded `results.jsonl`, `official/`, the manifest, the grade phase's state, the evaluation cache and the proxy's `llm_calls.jsonl`. After the solve phase, the host writes the official records to `official/seed-<k>.jsonl` from the predictions in `solve/`, one per input record; a task with no valid prediction gets an empty one, which always fails (§3.3).
   - **The grade phase takes what it expects from the host caller,** never from files in `solve/`:
     - the run's task IDs, seeds, split and grading mode;
     - the harness identity: the snapshot's commit and tree hash, plus the config hash. The config hash leaves out run-time knobs such as `max_tasks`, the phase, budget shares and LLM shares;
     - the task digest, computed on the host from the dataset.
   - **Every expected (task, seed) is graded.** A missing, duplicate or malformed solve record gives that task an error status with credit 0. It is never skipped, and never counted as already graded.
   - **Grades go to `results.jsonl`.** The grade phase appends a graded copy of each result there, and the last line for a task wins. A two-phase trajectory carries no grade, so readers take grades from `results.jsonl`.
   - **The evaluation cache** is read and written only by the host, keyed only by host-known identity. A cache hit is decided on the host before the container starts, and cached tasks are not sent to the container.
   - **Cost, tokens and time.** Decision-relevant cost and tokens come from the proxy's records, which the host writes; the project lead decided this. The host measures the run's wall time. Per-task wall time reported by the container is informational only. §8.3 says how the decision rules use these numbers.
3. **The solve container** runs with `--network=host`, made by the same builder as the implementer's sandbox (§8.8). The project lead decided that this builder lives in `goaly.bench`, so `goaly run` can use it without loading `goaly.rsi` (§3.1). T18 owns it, including the solve container's mount layout.
   - Mounts: the candidate's snapshot (§8.2 step 3), fully read-only and without `.git`; the gold-free runtime data file, read-only; and `<run>/solve/`, writable. Nothing else is mounted.
   - Environment: an explicit allowlist with a per-run dummy token; `GOALY_BASE_URL`, pointing at the local proxy (below); and the agent role's PostgreSQL password. That role is `SELECT`-only on `tpch`, and it can log in only on the 2 solving instances: its CONNECT on the grading instances is revoked explicitly, and it cannot connect to the alternate database (§4.1).
   - It runs on CPUs 0, 1, 12 and 13 with a 4 GB memory limit, like the implementer sandbox (§6.2, §8.8).
   - Never present: `~/.config`, the real API key, grader credentials and gold.
4. **A key-injecting LLM proxy runs on the host,** in the trusted parent process.
   - It listens on 127.0.0.1 and requires the per-run token.
   - It rebuilds the forwarded body from an allowlist of the keys that core's client sends. It refuses anything else with a 400, which is logged. It forces `stream=false` and `n=1`, and clamps `max_tokens` to the profile's limit.
   - It allows only the executor model, `qwen3.6-35b-a3b`. Any other model gets a 403, which is logged.
   - It forwards each request to the gateway with the real key, and passes the response back unchanged, `x-litellm-response-cost` included.
   - It writes one `LlmCallRecord` per request, refusals included, to the host-only `llm_calls.jsonl`. Each record holds the model, the status, the latency, the tokens from the response's `usage`, and the cost from `x-litellm-response-cost`. T21's decision rules and T07's metrics aggregate those records per run.
   - Inside the container, the client is core's LLM client (T03), configured with the proxy's URL and the dummy token, so throttling works as before. Its own call log is informational.
5. **What this closes.** Pack code can no longer see the key, reach GLM, or touch the grader, the grading data or the grades. A bypass inside the container reaches only three things:
   - the read-only agent role, on the solving instances;
   - Qwen, through the proxy;
   - `<run>/solve/`, which the grade phase reads only for predictions.
6. **The guards are unchanged.** Paths, checksums, the hard-coding scan, the AST allowlist and the unit-test guard still run first (§8.2 step 3).

The organizers' entry point (§6.4) needs only the solve phase, and it runs that phase in the same container. Reproducing our results therefore needs Docker, by the project lead's decision, and there is no non-Docker fallback for official runs (§11). Their database also needs the agent role. The environment setup creates it along with the PostgreSQL instances (`make final`, §10), and the README gives the statements that create it on a database they provide.

**Seeds and reproducibility.** In this document a *seed* means one independent replicate run, not a sampler seed.

- The gateway accepts a `seed` parameter but ignores it. A test on 2026-10-14 sent the same seed three times at T=0.6 and got three different outputs. Runs with T>0 therefore cannot be reproduced token for token.
- Greedy decoding (T=0) is deterministic on this gateway. However, Qwen's model card advises against greedy decoding in thinking mode because it causes repetition, so we keep sampling at T=0.6. T is a config value that RSI may tune.
- Reproducible here means statistically reproducible: the same code, config and data give consistent scores. Every score of the final harness is the mean ± standard deviation over 3 replicates. v0's held-out baseline is a single official run, by the project lead's decision (§10); its variation comes from its val seeds.

**Run manifest** (`RunManifest` contract): git commit, harness content hash, config hash, dataset revision, model names and sampling parameters, start and end times, machine info.

**Per-task records** (`TaskResult` contract): score, correctness, wall time, tokens (input, output and reasoning, counted separately), cost, steps and error type. In a two-phase run, the wall time, tokens and cost that the container wrote are informational; the decision rules use the host's numbers (§8.3).

### 6.2 Concurrency and rate limiting

**Gateway capacity (measured 2026-10-14).** We stepped load up gradually, sending 2c requests at each concurrency level c. Every prompt began with a random string so the server's prefix cache could not shortcut it. Qwen requests used about 3.8K input tokens and 1,024 output tokens with thinking on. GLM requests used about 2.5K input tokens and about 1K output tokens.

| Model | Concurrency | Errors | Latency p50 / p95 | Output tokens/s (all requests combined) |
|---|---|---|---|---|
| qwen3.6-35b-a3b | 1 | 0 | 2.3 s / 2.6 s | 444 |
| | 16 | 0 | 3.1 s / 3.2 s | 5,198 |
| | 32 | 0 | 3.9 s / 4.5 s | 7,022 |
| | 64 | 0 | 6.9 s / 8.2 s | 8,599 |
| | 128 | 1 × 502 | 13.8 s / 16.2 s | 8,555 |
| glm-5.3 | 1 | 0 | 3.8 s / 4.4 s | 261 |
| | 16 | 0 | 12.0 s / 15.7 s | 1,135 |
| | 32 | 0 | 19.8 s / 26.9 s | 1,455 |

What the numbers show:

- We saw no 429s and no rate-limit headers, so the key has no request or token rate limit within the range we tested.
- The real ceiling is backend capacity:
  - Qwen output throughput levels off at about 8.5K tokens/s from concurrency 64. Above that, extra requests just wait in a queue and latency grows linearly. The first 5xx showed up at 128.
  - GLM's total throughput is roughly one sixth of Qwen's.
- Other teams probably share this gateway, so available capacity can drop at any moment. The limits below therefore adapt instead of assuming these numbers hold.

**Concurrency budgets.** A model's budget is the total number of requests in flight across all of our processes:

- Qwen: 48. That is about 75% of the point where throughput stops growing, and keeps p50 latency near 5 s.
- GLM: 16.

Config divides each budget among the processes that run at the same time, and the division changes by phase. Each process enforces its share with one asyncio semaphore per model.

The budgets now cover one active lane, Effi-SQL:

| Phase | Qwen (48 total) | GLM (16 total) | Notes |
|---|---|---|---|
| Baseline (T12) | Effi-SQL 40, background 8 | unused | v0 runs on train and val first. Its held-out run, 1 seed in official mode, starts as soon as those are done |
| RSI (T22) | Effi-SQL lane 32, background 8; 8 spare | Effi-SQL lane 8; the other 8 are headroom | Background covers confirmation reruns and v0's held-out run if it is still pending |
| Final evaluation (T23) | Effi-SQL 40; 8 spare | unused | GLM is no longer available |

The SWE lane is on hold (2026-10-15) and has no share. If it resumes, the shares and the CPU partition (below) have to be planned again.

An agent never has more than one LLM request in flight, so a process runs at most as many tasks at once as its Qwen share. Each share is an upper limit: adaptive limiting (below) lowers it when the gateway is busy. Shares are config values, so changing them does not require a code change. In the Effi-SQL lane the binding limit is lower still: it is task concurrency, set by the solving instances (Throughput planning, below).

In the RSI phase a lane's GLM share covers its analyst, its implementers and its lesson writer. A lane runs at most k implementers at once (k as in §8.2), each with at most one call in flight, so k stays well below the share.

**Adaptive limiting** (in the LLM client, T03)

- On a 429 or 503, wait for `Retry-After` if the response has one, up to the 60 s backoff cap; every attempt still records the server's own value. Otherwise back off exponentially with full jitter: 1 s base, 60 s cap.
- A gateway 408, which LiteLLM returns when its upstream times out, counts as a timeout and is retried.
- AIMD:
  - the process halves its concurrency for that model once per burst of 429s, 502s, 503s, 504s or timeouts. Signals from requests sent before the last halving count toward that same halving, so one batch of failures halves the limit only once. 500s and connection errors do not lower it; they count only toward the circuit breaker;
  - after every 20 consecutive successes, it adds 1 back. Any failure resets the run of successes;
  - concurrency never exceeds the process's configured share.
- Circuit breaker: if more than half of a model's calls fail within 2 minutes, the process stops starting new tasks for 60 s and logs a warning. Calls already in flight finish or time out as usual. Attempts sent before the breaker opened do not count toward its next judgement.
- Every throttling event is logged with the call's purpose, so the report can show how much time the gateway cost us.

**Throughput planning.** With one active lane, the scarce resources are the solving instances, which bound how many tasks can run at once, and time on the grading instances. Qwen output tokens do not bind.

- **Tokens.** An Effi-SQL val pass is 30 tasks × 1 seed. At about 15 calls per task and about 1.5K output tokens per call, that is about 0.7M output tokens, only a few minutes of gateway time. Most of those tokens are spent on thinking, so a policy that uses fewer tokens also speeds up RSI iterations.
- **Solving capacity.** A typical task runs the base once and then 1–2 rewrites: about 50–75 core-seconds on a solving instance.
  - Each solving instance runs at most 2 statements at once, and the others queue. `execute_sql` reports the queue wait separately from the execution time, so the timing the agent sees stays meaningful.
  - The agent's statement timeout is 120 s. The base median is 12 s and the p90 is 69 s. By the file's timings, 12 base queries exceed 120 s, 7 of them gradable, including task 1 at 130.8 s. Half of the 12 were timed at SF10 (§4.1), so the setup's own measurements decide which tasks these are. For such a task the agent cannot time the base and works from its plan.
  - The Effi-SQL lane therefore runs about 16 tasks at once. That task concurrency is the binding limit, set by the solving instances rather than by tokens; the Qwen shares above remain upper bounds on requests. Background runs use the same solving instances.
  - T12 measures the queue wait.
- **Fetching and hashing.** Every comparison run fetches and hashes the whole result, at a few microseconds per row. At least 56 gradable tasks return more than 1M rows on SF3, and each of the largest SF3 results, around 18–20M rows, takes over a minute. The setup measures the exact counts (§4.1), and the estimates below include this time.
- **Grading time per candidate** (dev mode, §4.1). A prediction costs one comparison run, up to 3 timed runs and the robustness check on the alternate database. From the dataset's timings, leaving out the 5 tasks over 300 s, that is about 55 s per task at gold speed and about 110 s at τ ≈ 1; an identity submission costs nothing. On the 3 grading instances, screening a candidate (20 tasks) takes about 6–12 minutes and validating it (30 tasks) about 9–18 minutes. A round of 3 candidates spends about 40–110 minutes on grading, depending on how many reach validation.
- **Official mode** times both sides with no cache: about 2.5–3.5 minutes per task, so a 60-task held-out run takes about 50–70 minutes on 3 instances. v0's single held-out run therefore takes about an hour of grading.
- **One-time setup.** The base cache, the identity τ, the alternate-database signatures, the gold check and the exclusions take about 14 instance-hours: about 3 hours on all 5 instances.
- **Rounds.** An Effi-SQL round takes about 85–190 minutes end to end, the fast and slow cases of §13: analysis and implementation, then screening, validation and one confirmation rerun, with solving and grading overlapping. The schedule (§13) plans 1 round, with a second as upside, and gives the conditions for each.
- These figures are estimates from the file's timings, which come from the authors' hardware and partly from SF10 (§4.1), so they lean high. T10 and T12 measure the real times on this machine, and these numbers will be updated from those measurements.
- **SWE (on hold, 2026-10-15).** The SWE lane is limited by CPU and evaluation, not by the gateway. An SWE round takes about 2.5 hours. Solving runs on 8 slots and is capped at 45 minutes per task; evaluation runs on 4 workers. That allows 3–4 rounds overnight. Config can fit in more rounds by using one candidate per round or by capping solving at 30 minutes. The SWE figures are estimates. T14 and T15 measure actual solving and evaluation times, and these numbers will be updated from those measurements.

**Grading-instance plan.** Grading capacity binds the whole day, so every phase has a fixed share of the grading instances and a fixed priority. The times follow the estimates above and the schedule in §13.

| Phase | Starts | Grading instances | Priority | Load |
|---|---|---|---|---|
| Setup and exclusions (T10) | When T09 and T10 land | All 5: nothing solves before T12, so the solving instances grade too | First | About 14 instance-hours, about 3 h |
| v0's dev runs: train with 1 seed, val with 2 (T12) | When T11 lands | 3 | First: RSI starts from these results | Up to about 6 instance-hours; identity submissions are free |
| v0's official held-out run, 1 seed (T12) | As soon as v0's dev runs are done | 3 | Ahead of the lane, because the final comparison needs it | About 2.5–3.5 instance-hours, about 1–1.2 h |
| RSI rounds (T22) | 16:00 | What v0's held-out run leaves, then all 3 | After v0's held-out run | About 2–5 instance-hours per round |
| Final: held-out 3 times in official mode, val 3 times in dev mode (T23) | At the freeze | 3 while final solving runs, then all 5 | The only phase | About 10–14 instance-hours, about 2.5–3.5 h |

A solving instance that grades, in setup or after final solving, also holds the alternate database. The agent's role cannot connect to it.

**`goaly-smoke` gradings go first.** They jump the queue ahead of every phase above. There are at most 2 per hypothesis, graded in dev mode, and each waits at most for the evaluation already running on its instance. The grading-instance pool (T09's API) carries these priorities, and T10's scheduler drives it.

**Inside a process**

- Each process runs one asyncio event loop.
- Blocking work (Docker SDK calls, subprocesses, file I/O, git) goes through async APIs or `asyncio.to_thread`. Threads never share mutable state.
- The HTTP client keeps one connection pool per model, sized to the process's share.
- Implementers are mini-swe-agent agents, which are synchronous. Each lane runs them on its own thread pool of k threads. A thread reaches the LLM client only through the event loop, so the client's state never crosses threads, and no code on the event-loop thread may call an implementer directly.

**CPU partition.** The machine has 12 physical cores (24 hardware threads).

| Physical cores | Logical CPUs | Used for |
|---|---|---|
| 0–1 | 0, 1, 12, 13 | system, dockerd and our Python processes, which are confined to these CPUs, plus the implementer sandboxes (§8.8), with a 4 GB memory limit each |
| 2–5 | 2–5, 14–17 | 2 solving instances, 2 physical cores each |
| 6–11 | 6–11, 18–23 | 3 grading instances, 2 physical cores each |

Each PostgreSQL instance is pinned with cpuset to both SMT siblings of its physical cores: physical core N is logical CPUs N and N+12. Solving and grading never share an instance, so no SWE-style waves are needed. cpuset does not isolate the L3 cache, memory bandwidth or the disk, though, so heavy solving can still slow grading a little. T10 therefore measures identity τ with the solving instances idle and under load, and runs periodic identity probes during grading.

**SWE (on hold, 2026-10-15).** The SWE plan used cores 4–11 in waves: each wave either solved (8 slots, 1 core each) or ran official evaluation (4 workers, 2 cores each), never both at once. Solving agents run tests and workloads that keep their cores fully busy; if official timing ran next to them, the measured speedups would be distorted, and so would every accept/reject decision based on them. Effi-SQL now uses those cores, so the partition has to be planned again if SWE resumes.

**Other shared resources**

| Resource | Limit | How it is enforced |
|---|---|---|
| Solving instances | 2, each running at most 2 statements at once; every agent session reaches them through the read-only role | psycopg async pools with a per-instance statement queue (T09). Client-side cancellation enforces the 120 s statement timeout |
| Grading instances | 3, each running one evaluation at a time, so evaluations never disturb each other's timings. Each holds the main and the alternate database | The grading-instance pool (T09's API), which serves work by priority, `goaly-smoke` gradings first. T10's scheduler drives it and checks each instance's catalog fingerprint before every batch (§8.5) |
| SWE solving and evaluation (on hold) | 8 solving slots or 4 evaluation workers, never both at once | The slot scheduler (T14) pins containers with cpuset and switches mode between waves |
| Disk (SWE, on hold) | Image budget | Images are pruned in batches (T13) |

**Concurrent writes**

- Each per-run JSONL file (results, trajectories, call logs) has a single writer task that reads from an asyncio queue. Lines are flushed as they are written, so a crash loses at most the line being written.
- Cache entries are written to a temporary file and then renamed into place, so readers never see a partial entry.
- Lanes run as separate processes and share `experiments.jsonl` and `lessons.jsonl`. Only one lane is active while SWE is on hold, but every write still takes an exclusive `fcntl` lock.
- A repository-level file lock serializes git operations on the main repository, such as adding or removing worktrees and moving tags. Each candidate edits only its own worktree.

**Tests**

- A FakeLLM that injects 429s, latency and timeouts drives 500 simulated tasks. The tests check that no process exceeds its share, that limits back off and recover, that nothing deadlocks, and that every result is recorded exactly once.
- Concurrency tests cover the JSONL writer, the cache and the ledger lock.

### 6.3 Metrics

- Main metric: Effi-SQL R-VES. Coverage@2x and GMS are secondary (§4.1). Each is reported twice: as the official evaluator computes it, and checked, with flagged hard-coded predictions given no credit (§4.1). RSI decides on the checked numbers. SWE (on hold, 2026-10-15): harmonic-mean SR.
- With several seeds, we report mean ± standard deviation and bootstrap 95% confidence intervals. v0's single held-out run gets a bootstrap 95% confidence interval over tasks, and the improvement from v0 to the final gets a paired bootstrap over tasks.
- Two versions are compared task by task, on per-task credit for Effi-SQL. We count the tasks each version wins and run a one-sided sign test. SWE uses a paired bootstrap.
- An Effi-SQL task counts as solved when its credit is at least 0.75 (matched, with τ ≥ 1.2), the bar the regression rule uses (§8.3). Any other task is a failure. The stably solved set is the tasks with credit ≥ 0.75 in every seed.
- Results are broken down by difficulty, and for SWE by repository.

### 6.4 CLI

- `goaly run --bench {effisql,swefficiency} --split {train,val,heldout} --harness <path|tag> --seeds N --phase {solve,grade,all} --out runs/<id>`
  - `--phase all`, the default, is the isolated pipeline: the host starts the proxy, then the solve container, then grades in a fresh process (§6.1).
  - `--phase solve` stops after the solve container. The host writes the official records from the container's predictions, and the results stay ungraded.
  - `--phase grade` grades a run directory on the host. It takes the tasks, seeds, mode and harness identity from its arguments and the host-written manifest, never from `solve/` (§6.1).
  - `--out` names a run directory, never a single file. Grades go to `results.jsonl` in it, where the grade phase appends a graded copy of each result and the last line wins (§6.1).
- **Who starts what.** T06's CLI calls the benchmark's launcher through a hook. For Effi-SQL the launcher is T11's `bench/effisql` pipeline, which starts the proxy and the solve container on the host.
- `--phase solve --in-container` is the worker's entry point inside the container. Only the launcher passes it.
- `--in-process` runs both phases in one process, with no container and no proxy. It refuses to run with the real gateway key unless `--unsafe-in-process` is also given. It is for unit tests, the toy benchmark and local debugging only; no real experiment uses it.
- `--instances <official-format jsonl>` runs any task file. The organizers use this for their held-out set, and the README documents it as the official entry point. They need only the solve phase:

  ```
  goaly run --bench effisql --instances heldout.jsonl --harness final --phase solve --out runs/heldout
  ```

  The official predictions land in the run directory as `official/seed-<k>.jsonl`, one file per seed, written by the host. Each is the official prediction file: every input record plus its `prediction` field, which holds both the SQL and the diff, in that order (§3.3).

  This needs the executor API, a PostgreSQL database for Effi-SQL with the agent role (§6.1), and Docker. The solve phase always runs in its container (§6.1), so there is no non-Docker fallback for official runs (§11). SWE (on hold) would need Docker for its task containers too.

## 7. Data splits

| | train | val | held-out |
|---|---|---|---|
| Effi-SQL (the 238 TPC-H tasks, stratified by difficulty) | 148 | 30 | 60 |
| SWE (on hold, 2026-10-15; IDs from the 100-task Lite set, stratified by repository; task data from the revised main dataset) | 12 | 12 | 12 |

- The random seed is fixed at `20261014`. The split ID lists are committed to `src/goaly/bench/splits/`.
- Effi-SQL's train split is all the tasks not drawn for val or held-out.
- Each split has one job:
  - **train:** the analyst may see full trajectories, grader feedback and gold answers.
  - **val:** results go to the decision step and to the final report. The analyst never sees per-task val results or val trajectories. Its ledger summary keeps each experiment's decision and aggregate result with the per-task val results stripped, and lessons cite only the aggregate decision.
  - **held-out:** used only for v0's baseline run and the final runs. No other version runs on it.
- SWE (on hold): if an SWE task takes more than 25 minutes to evaluate, it is swapped for another task from the same repository, and the swap is recorded.
- Exclusions (§4.1, §4.2) are applied after splitting, which keeps the split itself reproducible. An excluded Effi-SQL task is simply dropped from its split, so that split ends up slightly smaller. An excluded SWE task (on hold) is replaced with another task from the same repository, so every SWE split keeps 12 tasks.

## 8. RSI system (goaly.rsi)

### 8.0 Autonomy

Once `goaly rsi run` starts, the loop runs as a program until every lane stops. Only GLM-5.3, called through core's LLM client, and goaly's own code act. No Claude Code, no other assistant and no person takes part in any step: Claude Code only builds the system and starts it, and people review the results PRs afterwards.

| Step | Done by |
|---|---|
| Analysis and hypotheses | GLM, one structured call per round (§8.2) |
| Implementation | GLM, in mini-swe-agent's non-interactive agent (§8.8) |
| Guards, screening, validation and decisions | goaly code |
| Ledger and lessons | goaly code; GLM writes the lesson text in one structured call |
| Results PRs and notifications | goaly code, through the GitHub REST API and ntfy.sh (§8.7) |
| Crash recovery and stopping | goaly's supervisor and the stopping rules (§8.4, §8.7) |

Every external call has a timeout. Every agent run has step, cost and time limits. The lane processes run with no terminal input, so nothing can wait for an answer. When a step fails, the code decides what follows: retry, reject, abort the experiment, or stop the lane and notify.

### 8.1 Lanes

- One lane is active: Effi-SQL. The SWE lane is on hold (2026-10-15). Its design below stays, so it can resume, and the two lanes would then run in parallel.
- Each lane tracks its best version with a git tag (`best/effisql`, `best/swe`) and edits only its own pack.
- core and bench are frozen while RSI runs.

### 8.2 One round

1. **Analyze** (GLM, purpose `rsi-analyst`). One structured call through core's client, with no tools and no agent loop. Invalid output is retried.
   - Inputs:
     - a summary of the ledger, rejected attempts included, with the per-task val results stripped (§7);
     - the current best's per-task results on train;
     - summaries of failed trajectories, with grader feedback and gold answers;
     - an explanation of the benchmark's scoring rules;
     - an idea bank of techniques that prior work found effective, for reference. Examples: profiling before optimizing, keeping the fastest verified patch, execution-based self-consistency for SQL, inspecting the schema first. GLM decides whether and how to use any of them.
   - Output: structured JSON that groups failures by pattern and proposes 3 hypotheses. Each hypothesis states:
     - which failure pattern it targets;
     - the kind of change: prompt, skill, tool, policy, config or removal;
     - exactly what to change;
     - the expected effect and the risk;
     - where the idea came from: the idea bank, or GLM's own analysis.
2. **Implement** (GLM coding agent, purpose `rsi-impl`)
   - The top k hypotheses (k=3 for Effi-SQL, k=2 for SWE, on hold) are built in parallel. Each gets its own git worktree, on the branch `rsi/<lane>/<experiment_id>` from the lane's best.
   - Each hypothesis is one run of the implementer agent (§8.8): mini-swe-agent's `DefaultAgent` in a Docker sandbox. The agent's one tool is bash, run inside the sandbox. With it the agent can:
     - read anything in the repository;
     - write only to `src/goaly/harness/packs/<lane>/` and `tests/harness/packs/<lane>/`. Every other path is mounted read-only, so the kernel refuses the write;
     - run the pack's unit tests in the sandbox;
     - smoke-test on 1–2 train tasks with `goaly-smoke <instance_id>`, the one command the host runs on the agent's behalf.
   - Each hypothesis ends as one commit, which goaly.rsi makes on the host after the agent finishes, staging only the pack paths. A run that ends any other way, or that changed nothing, makes no commit, and the candidate is rejected as not implemented.
3. **Guards.** The candidate is rejected if any of these fails:
   - the diff stays inside the allowed paths;
   - the `bench/`, `harness/core/` and `contracts/` checksums are unchanged;
   - the hard-coding scan passes (§8.5);
   - the static AST allowlist passes on the candidate's pack code (§8.5). As defense in depth, it bans the known ways for pack code to reach the key or build an LLM client (§5.1);
   - the pack's unit tests pass. They run in a fresh sandbox with the whole worktree read-only, never on the host, because the agent may have written them.

   The guards and every evaluation run on a **snapshot**: a read-only copy of the candidate commit's files, with no `.git`, never the live worktree. The commit and its tree hash identify the harness (§6.1).
4. **Screen.** The candidate's results are compared with the best's cached results on the same tasks. No net gain means rejection. In screening, validation and confirmation alike, a candidate with any prediction flagged `hard_coded` or `literal_result` (§4.1) fails outright. Every candidate evaluation, smoke tests included, takes the isolated two-phase path (§6.1).
   - Effi-SQL: up to 15 train failures the hypothesis targets, plus 5 train tasks the best already solves, 1 seed. Solved and failed are as defined in §6.3.
   - SWE (on hold): the 3 train tasks the hypothesis targets.
5. **Validate.** Effi-SQL runs all 30 val tasks with 1 seed. SWE (on hold) runs all 12 val tasks with 1 seed.
6. **Decide** using the rules in §8.3. If more than one candidate passes, the one with the largest Δ is accepted. The other passing hypotheses go back in the queue and are rebuilt on the new best next round.
7. **Confirm** (Effi-SQL only). The accepted candidate runs on val with 1 more seed. The best changes only if ΔR-VES over both seeds is still at least 3 points (§8.3).
8. **Record.** Every candidate goes into the ledger, with any hard-coding flags. For each one, GLM writes a lesson in one structured call (purpose `rsi-lesson`): what was tried, what happened and why, citing the experiment ID. A lesson about a flagged candidate says what was flagged.

### 8.3 Decision rules

**Effi-SQL.** Val is 30 tasks with 1 seed, and a confirmation runs on 1 more seed. A task's credit is its mean over that version's val seeds; R-VES and the regression rule use those means. All four rules must hold:

- Net gain: ΔR-VES = candidate R-VES − best R-VES ≥ 5 points.
- Significance: a one-sided sign test on per-task credit gives p ≤ 0.1.
- Regression: at most 1 task whose credit is ≥ 0.75 under the best and ≤ 0.25 under the candidate.
- Confirmation: ΔR-VES ≥ 3 points over both seeds.

Cost guardrail: if the candidate uses more than 1.5× the best's mean tokens or wall time per task, both Δ thresholds double, to 10 and 6 points.

**SWE (on hold, 2026-10-15).** All three conditions must hold:

- Gain: harmonic-mean SR on val improves by at least 10% relative, and a paired bootstrap gives P(Δ>0) ≥ 0.8.
- Regression: at most 1 task that has a real speedup under the best (SR > 1/gold) and drops to SR ≤ 1/gold under the candidate.
- Cost guardrail: same as Effi-SQL.

**Every lane: efficiency cap.** A candidate is rejected outright, whatever its gain, if on the same split its mean cost or mean wall time per task is more than 3× that of baseline v0. Without this cap, many small accepted changes could add up to a large cost increase; the cap keeps gains from depending on unlimited extra compute. The report states the cap and where the final harness stands against it.

**Where the cost and time numbers come from** (§6.1). Tokens and cost come from the proxy's records, and wall time from the host's clock around each solve run. Each run's totals are spread evenly over the tasks it solved. Wall time is also multiplied by the run's task concurrency, capped at its number of tasks, so runs with different shares compare fairly. A cache entry keeps these per-task numbers with the grade. Numbers that the container reports per task are informational only.

**Why these thresholds.** A higher bar accepts fewer changes, but the ones it accepts are more trustworthy. A lower bar accepts more changes, but more of them are noise mistaken for gains.

- Effi-SQL: with 30 val tasks, each task carries up to 3.3 points of R-VES, and one credit tier (0.25) is worth about 0.8 points. A 5-point gain therefore needs at least two tasks to improve, so one noisy flip at a tier boundary (§4.1) cannot pass on its own, and the confirmation seed checks that the gain holds.
- SWE: with only 12 tasks the noise is larger, so SWE requires a larger relative gain.

All thresholds are set in config, and every decision records the thresholds it used.

### 8.4 Stopping

A lane stops after 4 rounds in a row with no accepted change, or at its deadline. The deadline is an absolute time in config, set to the freeze time (§13), so a restart never moves it.

When the deadline is reached, the lane:

1. aborts the running round's in-flight experiments and records them as aborted;
2. keeps the best;
3. writes the stop record;
4. opens the final results PR;
5. notifies the project lead;
6. exits 0.

A lane that stops after 4 rounds with no accepted change does steps 3–6. The supervisor does not restart a lane that exits 0, and it exits once every lane has stopped (§8.7).

### 8.5 Anti-gaming and safety boundaries

| Risk | Defense |
|---|---|
| Editing the grader, core or contracts | **Prevented.** The implementer works in a Docker sandbox in which only its lane's two pack directories are writable. Every other write fails with a read-only filesystem error, and the sandbox cannot commit. **Detected.** A candidate whose commit touches any other path is rejected. The `bench/`, `harness/core/` and `contracts/` checksums are verified before every evaluation |
| Reading val or held-out data, or gold answers | The sandbox mounts only the candidate's worktree, which holds split ID lists but no gold answers: benchmark files with gold answers are never committed. `goaly-smoke` accepts train tasks only. A candidate's evaluation loads only the gold-free runtime file, which holds agent-visible and grading fields; Effi-SQL grading compares against `base_sql`, so it needs no gold at all. The full dataset, gold included, is read only by the setup and exclusion tools, and by the analyst for train only |
| Leaking or misusing the API key | The key never enters a sandbox: no host environment variable is passed in, and `~/.config` is not mounted. The implementer's model calls are made by the lane process, outside the sandbox. In the solve container, the client holds only a per-run dummy token for the host's proxy, which adds the real key (§6.1). mini-swe-agent's local environment, which passes on the parent's environment and protects nothing, is never used for the implementer |
| Running code the agent wrote | Agent-written tests run only in sandboxes, during implementation and in the unit-test guard. Candidate pack code runs only in the solve phase, inside the sandbox container (§6.1, Isolated evaluation), which holds no key, no grader credentials and no gold. A static AST allowlist checks it first, as a second layer: a static check, not a sandbox, so defense in depth, with the residual risk named in §12. **Allowed imports:** relative imports; the lane's own pack, `goaly.harness.packs.<lane>`; `goaly.contracts`; an explicit list of `goaly.harness.core` modules, which excludes `settings` and the LLM client modules (§5.1); pydantic; `asyncio`, for its concurrency primitives; and exactly these stdlib modules: json, re, typing, dataclasses, math, statistics, collections, itertools, functools, textwrap and enum. **Banned:** the builtins `open`, `eval`, `exec`, `compile`, `__import__`, `getattr`, `setattr`, `delattr`, `vars`, `globals`, `locals`, `dir`, `type`, `breakpoint`, `input` and `memoryview`; the modules `importlib`, `subprocess`, `os`, `sys`, `socket`, `pathlib`, `shutil` and `ctypes`, and `__future__`, whose `annotations` feature turns every annotation into a string; every attribute name with a leading `_`, on any object, `self` included, because an exemption by name could be claimed by any parameter called `self`. The one exception is a name-mangled private attribute, `self.__x`, inside a class the pack defines, and dunders are always banned. Also banned: the attribute names `format` and `format_map`, anywhere, since they can reach attributes from inside a string; the frame and coroutine attributes `f_*`, `gi_*`, `cr_*`, `ag_*` and `tb_*`; any attribute of an imported module that is itself a module, resolved at check time, such as `dataclasses.inspect`; and attribute names that start processes, open sockets or pipes, or reach modules. Those are `sys`, `os`, `modules`, `builtins`, `__builtins__` and `subprocess`, and asyncio's `create_subprocess_exec`, `create_subprocess_shell`, `open_connection`, `open_unix_connection`, `start_server`, `start_unix_server`, `subprocess_exec`, `subprocess_shell`, `create_connection`, `create_server`, `create_unix_connection`, `create_unix_server`, `create_datagram_endpoint`, `connect_read_pipe`, `connect_write_pipe` and `sock_*`. **String evaluation is banned too:** any `str` constant in an annotation position, `typing.get_type_hints`, `typing.ForwardRef`, `pydantic.create_model`, and `TypeAdapter` called with a string, since each of them evaluates a string as code. Every attribute-name rule also applies to the keyword attributes of a `match` class pattern (`MatchClass.kwd_attrs`), such as `case object(_client=c)`, which read attributes without an attribute node. **Re-exported names:** at check time, a name imported from an allowed core module is rejected if its `__module__` is `settings` or an `llm*` module, and an import-linter contract keeps the allowed core modules from importing those modules at all |
| The sandboxes reaching other services | Host networking is the only mode that works here, so the sandboxes can reach localhost and the internet. The implementer sandbox holds no secret: PostgreSQL requires a password it never gets, and the gateway is useless without the key. The solve container holds only a per-run proxy token and the agent role's password, for a `SELECT`-only role on `tpch` (§6.1). The Docker socket is never mounted in either |
| Hard-coded answers | A scan of the diff rejects it if it contains a gold-only fragment of any split's gold answer, at least 40 characters after normalization: gold `optimized_sql` for Effi-SQL, the gold patch for SWE (on hold). A gold-only fragment is a window of the gold that occurs in no `base_sql` and not in the schema DDL. They are built from the full dataset and the schema DDL and stored outside the repository, as hashes, so the sandbox never sees them. Scanning every gold window would reject legitimate skills: 147 of the 238 golds share such a window with their own `base_sql`, and standard TPC-H joins recur across many golds. The scan also rejects pack changes that embed large literal lists, above a size set in config. Effi-SQL instance ids are not scanned: they are small integers, so a match would mean nothing. There is no database-name rule either, since every task runs on the one `tpch` database. The SWE scan (on hold) also rejects an instance_id from any split |
| Hard-coded results: a prediction that returns the base query's result as constants | The grader's robustness check, on the alternate database, and its plan check give such a prediction credit 0 and flag it (§4.1). A flagged prediction fails its candidate outright, and the ledger and the lessons record it. The report gives both the official and the checked R-VES |
| Changing the database: indexes or data | Agent sessions use a role with only the `SELECT` privilege and no `CREATE` (§4.1), and client-side cancellation enforces their timeout. Grading runs on its own instances, and the official evaluation runs on the pristine database |
| Tampering with the grading instances | Before each grading batch, the grader verifies the instance's catalog fingerprint: its indexes, its relation sizes (`relpages`) and its write counters. Every instance requires password authentication. Grading credentials never enter the repository or a sandbox. The implementer sandbox has host networking but no credentials, and the solve container has only the agent role, which cannot connect to the alternate database (§6.1) |
| Relying on the improvement model | The per-purpose model allowlist lives in core, which RSI cannot edit: executor calls may use only Qwen (§5.1). In the solve container, the client reaches the gateway only through the host's proxy, which forwards only the executor model and refuses anything else with a 403 (§6.1). The static allowlist, which bans importing the client modules, loading the key and reaching the client's internals, is a second layer: defense in depth, not a sandbox (§12) |
| Special-casing the workload or caching between timing runs (SWE, on hold) | The official grader times each repetition in a fresh process and checks for stack inspection; the prompt and skills forbid both; held-out results expose this kind of overfitting |
| Editing test files (SWE, on hold) | The official grader resets test files before it runs them |
| Overfitting to val | The analyst never sees per-task val results or val trajectories, and lessons cite only the aggregate decision (§7). Held-out is used only for v0's baseline run and the final runs |

### 8.6 Ledger and lessons

- `experiments.jsonl` (`ExperimentRecord` contract): each record holds the experiment ID, lane, parent version, hypothesis and where it came from (idea bank or GLM's own analysis), kind of change, diff stats, screening and validation metrics, decision and reasoning, and cost. The report states how many accepted changes came from each source.
- `lessons.jsonl` (`Lesson` contract) is the source of truth for lessons. Entries are added one at a time, as in ACE, and each entry counts how often it helped or hurt. GLM maintains the file and can delete entries, but never rewrites it in one go, so nothing is lost by accident.
- `lessons.md` is generated from `lessons.jsonl` for human readers. Counters embedded in hand-edited markdown could not be updated reliably.
- Each experiment record holds the implementer's run (`ImplementerRun`): how it ended, its step count and wall time, its smoke tests, its closing summary, and a link to its trajectory, which stays in mini-swe-agent's own format under `runs/` (§8.8).
- `lane_events.jsonl` (`LaneEvent` contract) records what happens to a lane outside any one experiment: starts, restarts, a failed recovery, stops, results PRs and notifications (§8.7).

### 8.7 Operations

**Supervisor.** This machine has no systemd and no cron, so goaly supervises itself. `goaly rsi run --detach` starts one supervisor process in its own session, with no terminal input, and returns. A lock file keeps it to one instance, and a minimal shell loop restarts the supervisor itself if it crashes.

- The supervisor runs each lane as its own process. Each lane writes a heartbeat every 30 s; a lane whose heartbeat is 10 minutes old is killed and counted as crashed.
- A crashed lane is restarted from the ledger after a backoff of 30 s, 1, 2 and 5 minutes, then every 10 minutes. After 5 crashes within an hour, automatic recovery has failed: the lane is stopped and the project lead is notified.
- Before every start the supervisor checks Docker, PostgreSQL, the sandbox image, the gateway and free disk. A start that fails these checks counts as a crash.
- Monitoring means only these health checks. No assistant monitors the run, and nothing in it waits for a person; the project lead may read the logs, but the run never depends on it.

**Resuming.** Every step is recorded in the ledger, so a restarted lane picks up where it stopped:

- an experiment interrupted before its commit is aborted, and its hypothesis goes back in the queue;
- one interrupted after its commit resumes at its next phase, and cached evaluations make finished tasks free;
- the round number and the rounds without an accepted change are recomputed from the ledger. The deadline is absolute (§8.4), so a restart never extends it;
- leftover sandbox containers and worktrees are removed.

The project lead manages the Coder workspace's auto-stop. If the workspace stops, every process stops with it, and running `goaly rsi run` again after the restart resumes every lane from the ledger.

**Results PRs.** A lane opens an "RSI results" PR when it has 3 accepted changes since its last PR, or when 3 hours have passed since then, whichever happens first, and a final one when it stops. goaly.rsi does this itself:

1. it builds an `rsi/<lane>/results-<n>` branch from `main` with the accepted pack commits, the ledger entries and the lessons;
2. it pushes only that branch;
3. it opens the PR through the GitHub REST API, with a body generated from the ledger.

The PR number goes into the ledger, so a restart never opens a duplicate. The GitHub token is read at run time and never appears in a URL, an argument, a log or the sandbox. The code never merges, approves or pushes outside `rsi/*`. Results PRs wait for the project lead's review, and the lane does not pause for them.

**Notifications.** goaly.rsi sends them itself, over HTTPS to an ntfy.sh topic. The topic name is a secret, read from `~/.config/goaly/.env` as `GOALY_NTFY_TOPIC`, and the project lead subscribes to it in the ntfy app. A notification goes out when:

- a lane crashes and automatic recovery fails;
- the gateway circuit breaker trips repeatedly;
- a results PR is opened;
- a lane stops.

Every notification is also written to the lane's event log and listed in the next results PR. A failed delivery is retried and never blocks the loop, and a notification never carries a secret or task data.

### 8.8 Implementer agent

The implementer is mini-swe-agent's `DefaultAgent` (`mini-swe-agent==2.4.6`, MIT), used as a library and unchanged. goaly.rsi supplies its model and its environment, and owns everything around the run. `DefaultAgent` is mini-swe-agent's non-interactive agent: the `mini` CLI's interactive agent, which asks a person to confirm each step, is never used.

**Model adapter.** A goaly `Model` implements mini-swe-agent's model protocol on top of core's LLM client (T03):

- Each step is one client call to `glm-5.3` with purpose `rsi-impl`, and the experiment ID as the run ID. The client applies its accounting, retries, `Retry-After`, AIMD, the circuit breaker and the lane's GLM share, and writes the `LlmCallRecord`.
- The assistant message goes into the history exactly as returned, `reasoning_content` and `tool_calls` included.
- A step's cost is the gateway's `x-litellm-response-cost`, so the agent's cost limit counts real spend. mini-swe-agent's litellm model is not used, and litellm is not installed; its price table has no entry for `glm-5.3` anyway.
- Each step records its call ID, which joins it to its `LlmCallRecord`.
- **Fallback.** If T03's client is late, a small subclass of mini-swe-agent's `LitellmModel` that reads the gateway's cost header stands in. Switching back to core's client changes one class. While the fallback is in use, three claims in this section and §11 do not hold:
  - litellm stays out of the process that holds the key;
  - GLM calls go through core's client;
  - core records every call and enforces the lane's GLM share.

  litellm is then installed only for the fallback, pinned exactly to a vetted version, never 1.82.7 or 1.82.8.

**Sandbox.** One Docker container per hypothesis, with `--network=host`, the only mode that works on this machine (§1.3):

- The image is `goaly-rsi-sandbox:<uv.lock hash>`: Python 3.12, git, ripgrep and the locked dev dependencies.
- Mounts: the worktree, read-only, at its own path; the lane's two pack directories, `src/goaly/harness/packs/<lane>/` and `tests/harness/packs/<lane>/`, read-write; and the main repository's `.git`, read-only, so `git diff` works but nothing can be committed. Nothing else from the host is mounted: no `~/data`, no `runs/`, no other worktree and no `~/.config`.
- The container runs as the host user's uid and gid, with a read-only root filesystem, a tmpfs `/tmp`, no capabilities, `no-new-privileges`, and process, memory and CPU limits, on cores 0–1 (§6.2).
- No host environment variable is passed in, and no key reaches the sandbox.
- Every command runs under `timeout` inside the container, because a timed-out `docker exec` leaves its command running. Containers carry the experiment ID as a label, and a lane removes any it finds left over when it starts.
- The shared builder in `goaly.bench` makes this container and the solve-phase container of every evaluation (§6.1, Isolated evaluation). The solve container differs in three ways: an extra writable mount for the run directory, the gold-free runtime file mounted read-only, and an explicit environment allowlist.

**`goaly-smoke`.** `goaly-smoke <instance_id>` is the only command the sandbox does not run itself. The host:

1. refuses any instance that is not in train, and any call after the second;
2. runs the step-3 guards on the worktree as it stands;
3. runs the candidate on that task through the isolated two-phase path (§6.1), with one seed. Its grading goes ahead of every other phase on the grading instances (§6.2);
4. returns the score, the stop reason, the step count, the cost and the last tool error.

**Limits.** These are config values. Our adapter enforces 60 steps, $3 of GLM and 30 minutes per hypothesis; the agent also stops after 3 format errors in a row, and each command times out after 120 s.

**The prompt** (`implementer.yaml`) gives the hypothesis, the pack layout (§5.2) and the allowed paths, how to run the unit tests and smoke tests, the hard-coding rules and the AST allowlist (§8.5), and how to finish.

**Finishing.** The agent finishes by running `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` with a short summary. goaly.rsi then, on the host:

1. checks that every changed path is inside the pack directories;
2. stages only those directories;
3. commits once, as `rsi(<lane>): <summary>`, with `Experiment-Id` and `Hypothesis-Id` trailers.

**Endings without a commit.**

- A limit, repeated format errors, or a finish with no change: the candidate is rejected as not implemented.
- A gateway or Docker failure: the experiment is aborted, not rejected.

**Trajectory.** mini-swe-agent saves it after every step, in its own format, to `runs/rsi/<lane>/<experiment_id>/implementer.traj.json`. The experiment record links to it (§8.6).

**The analyst and the lesson writer** stay single structured GLM calls through core's client (§8.2). They are not agents: their input is already summarized, and their output is JSON that pydantic validates. When one of their chats is replayed, for example to retry invalid output, its reasoning goes back unchanged, as in every GLM chat (§5.1).

**Why mini-swe-agent.** The implementer needs a bash coding agent, not the executor's loop. mini-swe-agent provides one that is small, MIT-licensed, widely used on SWE-bench, and runs in a Docker sandbox. Keeping it apart from core also means that changes to the executor's loop cannot break RSI, and the implementer no longer waits for T04 or T05.

## 9. Cost and efficiency accounting

- Every LLM call records its purpose (exec, rsi-analyst, rsi-impl or rsi-lesson), model, tokens, cost from the response header, and latency. In isolated runs, the host's proxy writes one record per request, refusals included, to the host-only `llm_calls.jsonl` (§6.1). The decision rules and the report use those records, because files written in the solve container are untrusted; the container's own call log is informational.
- The report shows two costs separately:
  - what the final harness costs, in money and time, to run on the held-out set;
  - what the RSI experiments cost in total, by lane and by purpose.
- Implementer calls are recorded by core's LLM client like every other call, with purpose `rsi-impl` and the experiment ID as the run ID. mini-swe-agent's own cost tracking is not used. A smoke test's Qwen calls are `exec` calls in a smoke run attributed to the experiment, and they count toward its cost.
- On the machine side, the host records each run's wall time, how long each Effi-SQL evaluation holds its grading instance, and the CPU time used by SWE evaluation (on hold). Per-task wall time follows §8.3; the container's own per-task times are informational.
- The report compares the final harness's mean cost and wall time per task with v0's, against the 3× efficiency cap (§8.3).

## 10. Final evaluation and reproducibility

- **Freeze:** core and `best/effisql` are merged and tagged `final`. `best/swe` would join them only if the SWE lane resumes.
- **Runs:**
  - final runs 3 times on held-out in official mode (§4.1), plus 3 times on val in dev mode;
  - once final solving finishes, the 2 solving instances are converted into grading instances, so final grading uses all 5 (§6.2);
  - v0's held-out run, 1 seed in official mode, was already done during the baseline phase. The project lead decided on 1 seed (2026-10-15) to free grading time for RSI. v0 and the final are compared in the same mode, and the final's 3 runs meet the 3-run requirement (§2).
- **The report includes:**
  - starting score, final score and the improvement, with confidence intervals, for both the official and the checked R-VES (§4.1). The final's score is the mean ± standard deviation of its 3 held-out runs. v0's is its single run, with a bootstrap 95% confidence interval over tasks, and the improvement's interval is a paired bootstrap over tasks;
  - how much scores vary between runs: the final's 3 held-out runs, and v0's 2 val seeds in dev mode, since v0 has only 1 held-out run;
  - results by difficulty, and by repository if SWE resumes;
  - efficiency and experiment cost;
  - a plot of score by round;
  - lessons, each citing experiment IDs.
- **Reproduction:** `make final` does the following in order:
  1. prepares environments: the TPC-H database and its PostgreSQL instances, and the sandbox image for the solve phase (§6.1). SWE images are built only if that lane resumes;
  2. runs final;
  3. grades the results;
  4. generates the report.

  The report reads v0's held-out results from the runs archived during the baseline phase. A separate `make baseline` reruns v0 for anyone who wants to reproduce it too. We keep the two commands apart because rerunning v0 as part of `make final` would add its solving and about an hour of official-mode grading (§6.2).

  Dependencies are pinned in `uv.lock`, and the manifests record every version.

## 11. Engineering standards and workflow

**Stack and tests**

- Python 3.12 + uv, ruff, mypy --strict, import-linter, pre-commit and GitHub Actions.
- Protobuf for the shared data contracts (§3.3): `grpcio-tools` and `mypy-protobuf`, regenerated with `make proto`. CI fails if the generated code is out of date.
- pytest, written test-first. Unit tests never touch the network. Tests that need Docker, a database or the real API are marked `integration` and are skipped in CI.

**Requirements for running goaly**

- Docker is required, the organizers' reproduction included. Every real run's solve phase runs in the sandbox container (§6.1), and there is no non-Docker fallback for official runs; the project lead decided this. The README lists Docker among its prerequisites.
- On this machine Docker works only with `--network=host` (§1.3), which the container also needs to reach the proxy and PostgreSQL on localhost.
- Also required: uv with Python 3.12, PostgreSQL 16 with TPC-H SF3 (which `make final` prepares), and the executor API's key.

**Config and logging**

- Configs are validated with pydantic.
- Secrets are read only from environment variables or `~/.config/goaly/.env`.
- Logs are structured JSON.

**Third-party agent framework (RSI only)**

- `mini-swe-agent==2.4.6` (MIT) is pinned exactly, in a separate `rsi` dependency group. The adapter relies on its message format and helper functions, so every upgrade is a deliberate PR. 2.4.6 was released on 2026-07-23, so it satisfies the lock's `exclude-newer` cutoff of 2026-10-11.
- A `[tool.uv] exclude-dependencies` entry scoped to mini-swe-agent drops its unused heavy dependencies: `litellm`, `openai`, `datasets`, `textual` and `prompt-toolkit`. The group then adds about 12 small packages, and the classes we use never import the excluded ones. This also keeps litellm, whose 1.82.7 and 1.82.8 releases were compromised in March 2026, out of the process that holds the API key, except while the §8.8 fallback is in use.
- `make setup` and CI install every group. The organizers' plain `uv sync --locked` leaves the `rsi` group out.
- An import-linter contract lets only `goaly.rsi` import `minisweagent` (§3.1).
- Before the first import, goaly sets `MSWEA_SILENT_STARTUP=1` and points `MSWEA_GLOBAL_CONFIG_DIR` at an empty directory it owns. mini-swe-agent then neither prints a banner nor loads a `.env` file into the environment. Its logger is routed into our JSON logs.
- mini-swe-agent ships `py.typed` but leaves many functions unannotated, so one adapter module wraps every call into it, under a mypy override scoped to that module.
- Unit tests drive the implementer with mini-swe-agent's `DeterministicToolcallModel`, or with a FakeLLM behind core's client, and a fake environment. Tests that start a sandbox are marked `integration`.

**Language:** everything in the repository is written in English, including code, comments, design docs, the README and reports.

**Collaboration**

- Every PR stays small: roughly 400 lines or less of non-test code. Generated code, such as the protobuf output, does not count toward that limit.
- A ticket usually maps to one PR. A larger ticket is split into a short series of stacked PRs, and each PR in the series references the ticket. Tickets themselves are not split.
- PRs that depend on each other are stacked.
- The project lead reviews and merges every PR. Review feedback is fixed in new commits, never by force-pushing.
- Changes that RSI makes on its own go only to `rsi/*` branches. goaly.rsi pushes them and opens the "RSI results" PRs itself (§8.7). It never merges.

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Grading time. Grading capacity binds the whole day: the setup takes about 14 instance-hours, each official held-out run about 50–70 minutes, and dev mode about 40–110 minutes per round (§6.2). A slow prediction can hold an instance for up to the 300 s statement timeout on each run | The grading-instance plan (§6.2) fixes each phase's instances and priority. Dev mode caches the base side and grades identity submissions for free, and the setup and the final also use the 2 solving instances. v0's held-out run, 1 seed, starts as soon as v0's train and val dev runs are done (T12). T10 measures the real times first, and the schedule (§13) plans 1 RSI round, with a second as upside |
| RSI completes no round before the freeze: a late T11 or a late lane start, with slow rounds (§13) | Report it honestly: the final harness is then v0, and the report says that no round completed. Keep what the partial round learned: its hypotheses, implementations, screening results and lessons are in the ledger and go into the report |
| Solving capacity. Every agent statement runs on the 2 solving instances, at about 50–75 core-seconds per task | At most 2 statements in flight per solving instance, with the queue wait reported to the agent separately; a 120 s statement timeout; about 16 lane tasks at once (§6.2). T12 measures the queue wait |
| Hard-coded results: a prediction that returns the base result as constants matches and scores τ = 1000 | The robustness check on the alternate database and the plan check (§4.1). A flagged candidate fails outright, and the report gives the checked R-VES next to the official one |
| The organizers' evaluation mode is unknown: patch, the runner's default, or end2end | Every prediction carries both forms (§3.3), and T11 tests both extraction paths against a reimplemented extractor. The project lead may ask the organizers |
| Generating TPC-H. Our dbgen build and load may not match the authors' setup, and generating and loading SF3 and the alternate database takes time | T09 does it first, in the background: the official schema (primary and foreign keys only), then `ANALYZE`, then one data directory copied into the 5 instances. The data check compares base row counts with the dataset's plans for the tasks whose plans record actual rows at SF3, and more than 10% of tasks excluded stops the run for investigation (T10), so a mismatch shows up before RSI starts. T09's first build matched all 8 SF3 table counts exactly |
| The dataset's recorded plans mix scales: 11 base plans, with the base times recorded alongside them, come from TPC-H SF10 instead of SF3, and so do some gold plans (§4.1) | T09 detects such plans and writes our own SF3 plans to `plans.json`, which the agent sees instead. The data check sets them aside as dataset anomalies, and exclusions and grading use times measured on this machine, never the file's |
| Timing noise at tier boundaries: a τ near 1 can flip between credit 0.25 and 0.5 from one run to the next | Medians of 3 runs on dedicated, pinned grading instances, with periodic identity probes (§6.2). Dev mode gives identity submissions a fixed τ (§4.1). The decision rules add a sign test, a regression bound and a confirmation seed (§8.3). Final scores are reported as mean ± standard deviation over 3 runs |
| The official evaluator's repository has no license | We reimplement its algorithm (§4.1) rather than vendor its code. Negative self-checks, which a broken matcher would fail, test the reimplementation together with the gold, identity and data checks |
| SWE on hold: the PDF's "gains on more than one benchmark" is at risk (§2) | Nothing within this deadline; the report states the gap plainly. The SWE design stays in this document and the RSI machinery has no benchmark-specific code, so the lane can resume without a redesign |
| Tight schedule: 21 hours from 2026-10-15 03:00 UTC to the deadline | One active lane. Contracts come first, then component PRs in parallel; stacked PRs keep review from blocking work. RSI must be running by about 16:00 UTC, and the plan counts on 1 round, with a second as upside (§13) |
| The implementer writes outside its pack: bash can write anywhere | The Docker sandbox with read-only mounts prevents it (§8.8), and the guards and checksums detect it. A unit-tested function builds the mount arguments, and an integration test asserts the read-only refusal |
| Candidate pack code escapes the static allowlist. Each review round has found new escapes, so the list may still be incomplete | Isolation closes the structural gap (§6.1, Isolated evaluation). Pack code runs only in the solve container, which holds no key, no grader credentials and no gold, and the grader never imports pack code. The residual risk inside the container is limited to Qwen calls through the proxy and the read-only agent role, plus writing the run directory, which the grader reads only as data. The allowlist (§8.5) stays as a second layer, a static defense in depth that T18 tests against every known escape. Those are private attributes on any object, reflection builtins, `format` attributes, frame and coroutine attributes, module-valued attributes, string annotations and the functions that evaluate them, `match` class patterns, and names that allowed core modules re-export. Checksums are verified before every evaluation, and the project lead reviews every results PR before it merges |
| The solve container adds start-up time to every run | About 2 s per run, which is negligible next to solving and grading |
| mini-swe-agent changes or breaks | An exact pin. The adapter uses only `DefaultAgent`, `DockerEnvironment`, its exceptions and two helper functions. Vendoring them, about 450 lines under MIT, is the fallback |
| T03's client is late for the implementer | The fallback model, a small `LitellmModel` subclass that reads the gateway's cost header (§8.8). While it is used, litellm runs in the process that holds the key, with a vetted version pinned, and core neither sees the GLM calls nor enforces the lane's GLM share. Switching back to core's client later changes one class |
| A lane hangs or crashes during the unattended run | Heartbeats, bounded restarts with backoff, and resuming from the ledger (§8.7). A notification when recovery fails |
| The Coder workspace stops during the run | The project lead manages its auto-stop. Running `goaly rsi run` again after a restart resumes every lane from the ledger |
| GitHub or ntfy.sh is down | Retries, and nothing in the loop waits for them. Notifications are also logged and listed in the next results PR |
| Upstream interfaces change during review | Settle the interfaces first, in T05 and T06. Downstream PRs rebase |
| Malformed Qwen tool calls | Fallback parsing and fault tolerance in core (T04) |
| The gateway gets overloaded, possibly because other teams share it | Concurrency budgets at about 75% of measured capacity; adaptive limiting and a circuit breaker (§6.2) |
| Noisy evaluations | Paired comparisons, multiple seeds and confirmation reruns |
| SWE evaluation is too slow (on hold) | T15 measures it first. If it's too slow, shrink the subset. Empty patches skip evaluation, results are cached, and v0's held-out runs start early in the background |
| SWE images use too much disk (on hold) | Pull only the images the subsets need, and prune in batches |

## 13. Schedule (critical path)

The schedule runs from now, 2026-10-15 03:00 UTC, to the deadline, 2026-10-16 00:00 UTC (2026-10-15 17:00 PDT). All times are UTC. T02 is done, and the SWE track (T13–T17) is on hold.

| Time (UTC) | Work |
|---|---|
| 03:00–05:00 | T01 (this amendment and its fixes); T02a's contracts reworked for Effi-SQL |
| 05:00–07:30 | T03, T04, T08 and T09 in parallel. T09 generates and loads TPC-H SF3 and the alternate database in the background |
| 07:30–09:30 | T05 and T10 |
| 09:30–12:30 | T06, then T07 and T18. As soon as T10 lands, the setup runs on all 5 instances: the base cache, the identity τ, the alternate-database signatures, the self-checks and the exclusion list, about 3 h |
| 12:30–14:30 | T11, T19 and T20; T20 needs only T03 and T18, not T04 or T05. v0's dev runs (T12) start as soon as T11 lands. T11 is on the grading instances' critical path |
| 14:30–16:00 | T21. v0's held-out run, 1 seed in official mode, starts as soon as its train and val dev runs are done, and takes about 1–1.5 h with its solving. The project lead reviews the stack the lane runs on |
| 16:00–20:00 | T22: the Effi-SQL lane runs under the supervisor, with no assistant in the loop (§8.0). Its grading queues behind v0's held-out run (§6.2). Plan for 1 round, with a second as upside (below). It must be running by about 16:00 |
| 20:00–23:30 | T23: the freeze, then final solving and grading on all 5 instances, about 2.5–3.5 h in all. A 3.5 h final leaves no time for the report afterwards, so the report is drafted while the final runs |
| 23:30–00:00 | The final numbers go into the report; T24; buffer before the deadline |

**RSI rounds.** The hours freed by v0's single held-out run support 1 round, with a second as upside, but not more. **Target: T11 lands by about 13:30.**

The end times below make three assumptions:

- the setup has finished when T11 lands, which needs T10 by about 09:30;
- the lane starts at 16:00;
- the estimates in §6.2 hold.

The fast case assumes 4-minute tasks, 55 s of dev grading per task, 1 candidate reaching validation, and 20 minutes of analysis and implementation. The slow case assumes 8 minutes, 107 s, 2 candidates and 40 minutes. A round then takes about 85 minutes in the fast case and about 190 in the slow case.

| T11 lands | v0's held-out run ends | Round 1 ends | Round 2 ends |
|---|---|---|---|
| 12:30 | about 14:45 (fast) to 16:15 (slow) | about 17:25 to 19:10 | about 18:50 to 22:15 |
| 13:30 | about 15:45 to 17:15 | about 17:25 to 19:10 | about 18:50 to 22:15 |
| 14:30 | about 16:45 to 18:15 | about 17:35 to 20:10 | about 19:00 to 23:15 |

- Round 1 finishes before the 20:00 freeze if T11 lands by about 13:30. If T11 lands at 14:30 and the rounds are slow, round 1 ends just after the freeze.
- A second round fits only when the rounds run at the fast end, whenever T11 lands. At the slow end, round 2 ends after 22:00.
- If the lane starts after 16:00, every time shifts by the delay. No round may then finish, and the §12 risk "RSI completes no round before the freeze" applies.
- The freeze does not wait for a round that is still running. That round is reported as incomplete, and the best stays as it was before it.

The project lead's review latency is budgeted. PRs stack, so no slot waits for a review, and the 14:30–16:00 slot leaves time to review the code the lane runs on before it starts.

Isolated evaluation (§6.1) adds about 2–3 hours of work across T03 (the proxy), T06 (the phases), T11 (the Effi-SQL pipeline), T18 (the solve-mode sandbox) and T21 (using it for every candidate evaluation). That work has already started, inside the slots above.

## Appendix A: Ticket mapping

T01–T24 correspond, in order, to CIR-1633 through CIR-1656, so T_n = CIR-(1632+n). T02a (shared contracts) was added during spec review and is tracked separately in Linear. The dependencies here match the blocked-by relations in Linear.

**Tracks after the 2026-10-15 amendment**

- T08–T12 (CIR-1640 to CIR-1644) are now the Effi-SQL track:

  | Ticket | Linear | Work |
  |---|---|---|
  | T08 | CIR-1640 | Effi-SQL data, splits and exclusions (§4.1, §7) |
  | T09 | CIR-1641 | PostgreSQL 16 environment: TPC-H SF3, the alternate database and the 5 pinned instances (§4.1, §6.2) |
  | T10 | CIR-1642 | Grader: dev and official modes, hard-coding checks, self-checks and the exclusion run (§4.1) |
  | T11 | CIR-1643 | Baseline pack v0 and `goaly run --bench effisql` (§5.4, §6.4) |
  | T12 | CIR-1644 | Baseline experiment (§6.2, §10) |

- T13–T17 (CIR-1645 to CIR-1649), the SWE-fficiency track, are on hold (2026-10-15).
