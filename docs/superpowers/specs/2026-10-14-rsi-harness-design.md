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
2. **The RSI system**: code, an experiment ledger and lessons. It runs several improvement rounds on its own.
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
| Benchmark score: starting and final scores, with improvement on every benchmark | Baseline v0 (§5.4) and final are compared on the same held-out set, graded in official mode (§4.1). While SWE is on hold (2026-10-15), Effi-SQL is the only benchmark scored | T12 and T23 reports |
| Robustness: run the same test at least 3 times and report the variation; survive tool errors, timeouts and bad model output | Three held-out runs; fault tolerance in core (§5.1); per-task exception isolation (§6.1); concurrency budgets and adaptive backoff (§6.2) | T23 report; T04 and T05 tests |
| Generalization: held-out tasks stay out of development; gains on more than one benchmark; no hard-coded answers | Three-way split (§7) and a hard-coding scan (§8.5). The RSI machinery has no benchmark-specific code: a lane is a capability pack plus a `Benchmark` adapter (§3.3). **At risk:** while SWE is on hold, gains can be shown on one benchmark only, so "gains on more than one benchmark" is not met. The report says so plainly | Committed split ID lists; scan logs; held-out gains in the T23 report |
| Lessons learned: what was tried, what worked, what failed and why, backed by experiments | Experiment ledger and lessons (§8.6) | `experiments.jsonl`, `lessons.jsonl`, and the generated `lessons.md` |
| Efficiency: time, tokens, compute and cost reported alongside the score | Cost and tokens are recorded for every call (§9). Two limits in the decision rules (§8.3): a cost guardrail, and a hard cap that keeps the final harness at or below 3× v0's mean cost and wall time per task | Report tables, including the final harness's cost and time relative to v0 |
| Reproducibility: code, configs, prompts, tool versions and a single command | uv lockfile, run manifests, `make final` (§10) | T23, T24 |
| Automation: several improvement rounds with little human help | `goaly rsi run` (§8) | Ledger entries across rounds |
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

**What ships.** The organizers receive `contracts`, `harness`, `bench` and the CLI. `rsi` is only for development, and nothing on the evaluation path imports it.

### 3.2 Repository layout

```
proto/goaly/v1/      shared data contracts: task.proto, environment.proto, trajectory.proto,
                     result.proto, manifest.proto, llm_call.proto, experiment.proto
src/goaly/
  contracts/
    gen/             Python code generated from proto/ (committed; CI checks it is up to date)
    interfaces.py    Environment, SqlEnvironment, ShellEnvironment, Policy, Tool, Benchmark
  harness/
    core/            llm.py, tools.py, agent.py, policy.py, context.py, trajectory.py, config.py
    packs/
      effisql/       prompt.md, skills/, tools.py, policy.py, config.yaml
      swefficiency/  (same files; on hold)
  bench/
    effisql/         data.py, env.py, grader.py, adapter.py
    swefficiency/    data.py, env.py, grader.py, adapter.py (on hold)
    runner.py, metrics.py, manifest.py
    splits/          split ID lists
  rsi/
    ledger.py, workspace.py, guards.py, analyst.py, implementer.py, decide.py, lane.py
  cli.py             `goaly run …` and `goaly rsi …`
tests/               mirrors src/; tests that need Docker, a database or the real API are marked `integration`
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
| `ExperimentRecord`, `Hypothesis`, `Lesson` | `experiment.proto` | The RSI ledger |

- On disk, messages are stored as JSONL using protobuf's standard JSON mapping, keeping the original field names. People and the GLM analyst read trajectories directly, so a readable text format matters more than a compact binary one.
- Files written today must stay readable after the schema changes. Field numbers are never reused. On disk, field names are the JSON keys and enum values are written by name, so a field or an enum value is never renamed either, and when one is removed, its name is reserved as well as its number.
- The official submission formats (Effi-SQL's prediction file, SWE predictions) are not our contracts. Each adapter converts to them and keeps the official field names exactly.
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
- **What the agent sees:** `base_sql`; the database schema (DDL, row counts and indexes); the base query's plan; and `difficulty`.
- **What the agent never sees:** `optimized_sql`, `fast_time` and `optimized_explain_analyze`.

**Task**

- The agent rewrites `base_sql` into one query that returns the same rows, faster, and submits it.
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

   An unmatched or failed task gets 0.
6. **Metrics:**
   - **R-VES** = 100 × mean credit over all tasks.
   - **Coverage@2x** = 100 × the share of tasks that are matched with τ ≥ 2.
   - **GMS** = the geometric mean of τ over matched tasks.

**Environment**

- PostgreSQL 16, the same major version the official evaluator uses.
- TPC-H SF3 is generated locally with an open-source dbgen build. The schema has primary and foreign keys only, matching the official schema, and loading is followed by `ANALYZE`. The data is never committed.
- One data directory is copied into 5 instances, each pinned with cpuset (§6.2): 2 solving instances and 3 grading instances, 2 physical cores each. Solving and grading never share an instance.
- Agent sessions connect with a read-only role, plus a `statement_timeout`. The agent cannot create indexes or change data: the official evaluation runs on the pristine database.

**Grader**

- A clean reimplementation of the official algorithm (Scoring, above). The official evaluator is GitHub `L-nshpp/Diff-SQL`, file `diff_sql/evaluation/eval_execution_runner.py`. That repository has no license, so we reimplement its algorithm rather than vendor its code.
- Two modes:
  - **Dev mode,** for RSI screening and validation. The base query's result signature and its timing (the median of 3 runs) are measured once per task and cached. Each evaluation then runs only the prediction: one comparison run plus up to 3 timed runs. The 3 grading instances run in parallel.
  - **Official mode,** for the reported held-out numbers: the exact interleaved protocol, with no base cache.
- Self-checks:
  - In gold mode, the gold SQL must match and speed up.
  - In identity mode, submitting `base_sql` must give τ ≈ 1.
- **Excluded tasks.** A task is excluded if, on this machine, its base query errors or exceeds 300 s, or the gold `optimized_sql` does not match or is less than 1.2× faster. On such a task either nothing can be scored or even the gold shows no reliable gain, so keeping it would only add noise to development and to accept/reject decisions. Exclusions are applied after splitting (§7), and the excluded tasks are committed with the reason for each. This affects only our own splits: the harness still handles any such task in the organizers' held-out set.
- **Noise.** Tier boundaries make a τ near 1 flip between credit 0.25 and 0.5 from one run to the next (§12).

**Reference numbers for the TPC-H subset**, from the dataset's own timings:

- Base time: median 11.8 s, p90 69 s, max 450 s. 5 tasks exceed 300 s, and one pass over every base query takes 2.13 h.
- Difficulty: easy 71, medium 111, hard 56.
- Gold scores: R-VES 88.3, Coverage@2x 53.4, GMS 3.55. This GMS uses the uncapped speedups; with τ capped at 1000, as the grader caps it, gold's GMS is 3.29.

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
- Each call times out after 300 s. 429s, 5xx errors, timeouts and connection errors get up to 5 retries with exponential backoff. Concurrency limits and adaptive backoff follow §6.2.
- Reads usage, including reasoning tokens, and the `x-litellm-response-cost` response header.
- Writes one JSONL record per call, labeled with the purpose of the call.
- Turns context overflow and similar failures into specific exception types.

**Model allowlist:** at runtime the harness client may call only the configured executor model (`qwen3.6-35b-a3b`). Calling any other model raises an error.

**Model-specific handling**

- Qwen:
  - `chat_template_kwargs.enable_thinking` turns thinking on or off. `preserve_thinking` is optional.
  - The gateway rejects `reasoning_effort` and a top-level `enable_thinking`.
  - Known issue: tool calls sometimes end up in `reasoning_content`. When that happens, the parser falls back to Qwen3-Coder XML (`<tool_call><function=…><parameter=…>`).
  - Known issue: argument JSON can be truncated or malformed. We repair it where possible and otherwise send the error back to the model.
- GLM (used only by RSI):
  - Thinking is always on. The gateway rejects the `thinking` parameter.
  - `reasoning_content` is sent back unchanged in the message history.

**Tool executor**

- Validates arguments with pydantic and enforces timeouts.
- Long output is cut down to its first and last 4K characters.
- When a tool raises, the model gets a structured error message.

**Agent loop**

- Budgets cap steps, wall-clock time, tokens and cost.
- After 3 consecutive errors the loop reminds the model of the tool schemas. After 5 it stops.
- Calling the same tool with the same arguments 3 times in a row triggers a hint.
- Context management: Qwen gets a 128K context budget. Past 96K, older tool outputs are folded into summaries. The system prompt, the task and the last 8 turns stay as they are.
- When a budget runs out, the policy's fallback submits a result.
- No exception can take down a batch.

**Trajectories** use the `Trajectory` and `Step` contracts (§3.3) and are written as JSONL. Each step records messages, tool calls and their results, token usage and latency. The trajectory ends with the final prediction and its grade.

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
  - `execute_sql` runs a query on a solving instance, with a timeout, and returns truncated rows or the error, plus the elapsed time.
  - `submit` hands in the answer.

  The prompt is minimal.
- SWE (on hold, 2026-10-15) has two tools:
  - `bash` runs a command in the task container.
  - `submit` hands in the answer.

  The prompt is minimal but lists the official constraints.
- Both use the same settings:
  - Thinking is on. Sampling is T=0.6, top_p=0.95, top_k=20, with max_tokens 16K per call.
  - Solving budgets are 30 steps or 10 minutes for Effi-SQL, and 60 steps or 45 minutes for SWE.
- Fallback when a budget runs out before `submit`: v0 still submits something rather than nothing.
  - Effi-SQL submits `base_sql` unchanged. That gives τ ≈ 1, so a credit of 0.25–0.5.
  - SWE submits the container's final git diff, as OpenHands and other common agents do.

  Neither fallback makes use of what the agent found before its budget ran out. Smarter policies, such as submitting the fastest rewrite the agent verified, or never submitting an unverified patch, are left for RSI to discover.
- v0 is frozen and tagged `baseline-v0`. Every comparison is made against it.

## 6. Evaluation framework (goaly.bench)

### 6.1 Runner and records

**Runner**

- Built on asyncio, with separate resource pools for: LLM concurrency, the solving and grading database instances, and, for SWE (on hold), solving CPU slots and evaluation workers (at most 4).
- An exception in one task never affects the others. Each task's result is written to disk as soon as it's done, so an interrupted run can resume.
- Evaluations are cached by (harness content hash, task, seed).

**Seeds and reproducibility.** In this document a *seed* means one independent replicate run, not a sampler seed.

- The gateway accepts a `seed` parameter but ignores it. A test on 2026-10-14 sent the same seed three times at T=0.6 and got three different outputs. Runs with T>0 therefore cannot be reproduced token for token.
- Greedy decoding (T=0) is deterministic on this gateway. However, Qwen's model card advises against greedy decoding in thinking mode because it causes repetition, so we keep sampling at T=0.6. T is a config value that RSI may tune.
- Reproducible here means statistically reproducible: the same code, config and data give consistent scores. Every reported score is the mean ± standard deviation over at least 3 replicates.

**Run manifest** (`RunManifest` contract): git commit, harness content hash, config hash, dataset revision, model names and sampling parameters, start and end times, machine info.

**Per-task records** (`TaskResult` contract): score, correctness, wall time, tokens (input, output and reasoning, counted separately), cost, steps and error type.

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
| Baseline (T12) | Effi-SQL 40, background 8 | unused | v0 runs on train and val; its three held-out runs go in the background |
| RSI (T22) | Effi-SQL lane 32, background 8; 8 spare | Effi-SQL lane 8; the other 8 are headroom | Background covers confirmation reruns and any v0 held-out runs still pending |
| Final evaluation (T23) | Effi-SQL 40; 8 spare | unused | GLM is no longer available |

The SWE lane is on hold (2026-10-15) and has no share. If it resumes, the shares and the CPU partition (below) have to be planned again.

An agent never has more than one LLM request in flight, so a process runs about as many tasks at once as its Qwen share. Each share is an upper limit: adaptive limiting (below) lowers it when the gateway is busy. Shares are config values, so changing them does not require a code change.

**Adaptive limiting** (in the LLM client, T03)

- On a 429 or 503, wait for `Retry-After` if the response has one. Otherwise back off exponentially with full jitter: 1 s base, 60 s cap.
- AIMD:
  - after a burst of 429s, 503s or timeouts, the process halves its concurrency for that model;
  - after every 20 consecutive successes, it adds 1 back;
  - concurrency never exceeds the process's configured share.
- Circuit breaker: if more than half of a model's calls fail within 2 minutes, the process stops starting new tasks for 60 s and logs a warning. Calls already in flight finish or time out as usual.
- Every throttling event is logged with the call's purpose, so the report can show how much time the gateway cost us.

**Throughput planning.** With one active lane, the scarce resources are Qwen output tokens and time on the grading instances.

- **Tokens.** An Effi-SQL val pass is 30 tasks × 1 seed. At about 15 calls per task and about 1.5K output tokens per call, that is about 0.7M output tokens, only a few minutes of gateway time, so the 10-minute task budget bounds a pass instead. Most of those tokens are spent on thinking. A policy that uses fewer tokens therefore also speeds up RSI iterations.
- **Grading time per candidate** (dev mode, §4.1). A prediction costs one comparison run plus up to 3 timed runs. From the dataset's timings, leaving out the 5 tasks over 300 s, that is about 40 s per task at gold speed and about 90 s at τ ≈ 1. On the 3 grading instances, screening a candidate (20 tasks) takes about 4–10 minutes and validating it (30 tasks) about 7–15 minutes. A round of 3 candidates spends about 30–90 minutes on grading, depending on how many reach validation.
- **Official mode** times both sides with no cache: about 2–3 minutes per task, so a 60-task held-out run takes about 40–60 minutes. v0's three held-out runs therefore take about 3 hours of grading, and the final three about 2–3 hours.
- **One-time setup.** Measuring every task's base query (for the cache and the exclusions) and its gold (for the gold check) takes about 3 hours of grading-instance time.
- **Rounds.** An Effi-SQL round takes about 1–2 hours: analysis and implementation, then screening, validation and one confirmation rerun, with solving and grading overlapping. The schedule (§13) gives the lane about 4 hours, so expect about 2–4 rounds.
- **Solving.** Every agent session shares the 2 solving instances: up to 48 sessions on 4 cores. Their queries therefore run slower than in grading, and the elapsed times the agent sees are only a rough guide. Grading never uses them.
- These figures are estimates from the authors' hardware. T10 measures the real times on this machine, and these numbers will be updated from those measurements.
- **SWE (on hold, 2026-10-15).** The SWE lane is limited by CPU and evaluation, not by the gateway. An SWE round takes about 2.5 hours. Solving runs on 8 slots and is capped at 45 minutes per task; evaluation runs on 4 workers. That allows 3–4 rounds overnight. Config can fit in more rounds by using one candidate per round or by capping solving at 30 minutes. The SWE figures are estimates. T14 and T15 measure actual solving and evaluation times, and these numbers will be updated from those measurements.

**Inside a process**

- Each process runs one asyncio event loop.
- Blocking work (Docker SDK calls, subprocesses, file I/O, git) goes through async APIs or `asyncio.to_thread`. Threads never share mutable state.
- The HTTP client keeps one connection pool per model, sized to the process's share.

**CPU partition.** The machine has 12 physical cores (24 hardware threads).

| Physical cores | Used for |
|---|---|
| 0–1 | system, dockerd and our Python processes |
| 2–5 | 2 solving instances, 2 cores each |
| 6–11 | 3 grading instances, 2 cores each |

Each PostgreSQL instance is pinned with cpuset. Solving and grading never share an instance, so solving agents can never distort the timings that accept/reject decisions rest on, and no SWE-style waves are needed.

**SWE (on hold, 2026-10-15).** The SWE plan used cores 4–11 in waves: each wave either solved (8 slots, 1 core each) or ran official evaluation (4 workers, 2 cores each), never both at once. Solving agents run tests and workloads that keep their cores fully busy; if official timing ran next to them, the measured speedups would be distorted, and so would every accept/reject decision based on them. Effi-SQL now uses those cores, so the partition has to be planned again if SWE resumes.

**Other shared resources**

| Resource | Limit | How it is enforced |
|---|---|---|
| Solving instances | 2, shared by every agent session through a read-only role with a `statement_timeout` | psycopg async pools (T09) |
| Grading instances | 3, each running one evaluation at a time, so evaluations never disturb each other's timings | The grader's instance pool (T10) |
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

- Main metric: Effi-SQL R-VES. Coverage@2x and GMS are secondary (§4.1). SWE (on hold, 2026-10-15): harmonic-mean SR.
- With several seeds, we report mean ± standard deviation and bootstrap 95% confidence intervals.
- Two versions are compared task by task, on per-task credit for Effi-SQL. We count the tasks each version wins and run a one-sided sign test. SWE uses a paired bootstrap.
- An Effi-SQL task counts as solved when its credit is at least 0.75 (matched, with τ ≥ 1.2), the bar the regression rule uses (§8.3). Any other task is a failure. The stably solved set is the tasks solved in every seed.
- Results are broken down by difficulty, and for SWE by repository.

### 6.4 CLI

- `goaly run --bench {effisql,swefficiency} --split {train,val,heldout} --harness <path|tag> --seeds N --out runs/<id>`
- `--instances <official-format jsonl>` runs any task file. The organizers use this for their held-out set, and the README documents it as the official entry point:

  ```
  goaly run --bench effisql --instances heldout.jsonl --harness final --out predictions.jsonl
  ```

  This needs only the executor API and the matching environment: a PostgreSQL database for Effi-SQL, Docker for SWE.

## 7. Data splits

| | train | val | held-out |
|---|---|---|---|
| Effi-SQL (the 238 TPC-H tasks, stratified by difficulty) | 148 | 30 | 60 |
| SWE (on hold, 2026-10-15; IDs from the 100-task Lite set, stratified by repository; task data from the revised main dataset) | 12 | 12 | 12 |

- The random seed is fixed at `20261014`. The split ID lists are committed to `src/goaly/bench/splits/`.
- Effi-SQL's train split is all the tasks not drawn for val or held-out.
- Each split has one job:
  - **train:** the analyst may see full trajectories, grader feedback and gold answers.
  - **val:** results go to the decision step and to the final report; the analyst never sees them.
  - **held-out:** used only for v0's baseline runs and the final runs. No other version runs on it.
- If an SWE task takes more than 25 minutes to evaluate, it is swapped for another task from the same repository, and the swap is recorded.
- Exclusions (§4.1, §4.2) are applied after splitting, which keeps the split itself reproducible. An excluded Effi-SQL task is simply dropped from its split, so that split ends up slightly smaller. An excluded SWE task is replaced with another task from the same repository, so every SWE split keeps 12 tasks.

## 8. RSI system (goaly.rsi)

### 8.1 Lanes

- One lane is active: Effi-SQL. The SWE lane is on hold (2026-10-15). Its design below stays, so it can resume, and the two lanes would then run in parallel.
- Each lane tracks its best version with a git tag (`best/effisql`, `best/swe`) and edits only its own pack.
- core and bench are frozen while RSI runs.

### 8.2 One round

1. **Analyze** (GLM, purpose `rsi-analyst`)
   - Inputs:
     - a summary of the ledger, rejected attempts included, but without val results (§7);
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
   - The top k hypotheses (k=3 for Effi-SQL, k=2 for SWE) are built in parallel, each in its own git worktree.
   - The agent may:
     - read anything in the repository;
     - write only to `src/goaly/harness/packs/<lane>/` and `tests/harness/packs/<lane>/`;
     - run the pack's unit tests;
     - smoke-test on 1–2 tasks.
   - Each hypothesis ends as one commit.
3. **Guards.** The candidate is rejected if any of these fails:
   - the diff stays inside the allowed paths;
   - the `bench/` and `core/` checksums are unchanged;
   - the hard-coding scan passes;
   - the pack's unit tests pass.
4. **Screen.** The candidate's results are compared with the best's cached results on the same tasks. No net gain means rejection.
   - Effi-SQL: up to 15 train failures the hypothesis targets, plus 5 train tasks the best already solves, 1 seed. Solved and failed are as defined in §6.3.
   - SWE (on hold): the 3 train tasks the hypothesis targets.
5. **Validate.** Effi-SQL runs all 30 val tasks with 1 seed. SWE (on hold) runs all 12 val tasks with 1 seed.
6. **Decide** using the rules in §8.3. If more than one candidate passes, the one with the largest Δ is accepted. The other passing hypotheses go back in the queue and are rebuilt on the new best next round.
7. **Confirm** (Effi-SQL only). The accepted candidate runs on val with 1 more seed. The best changes only if ΔR-VES over both seeds is still at least 3 points (§8.3).
8. **Record.** Every candidate goes into the ledger. For each one, GLM writes a lesson: what was tried, what happened and why, citing the experiment ID.

### 8.3 Decision rules

**Effi-SQL.** Val is 30 tasks with 1 seed, and a confirmation runs on 1 more seed. All four rules must hold:

- Net gain: ΔR-VES = candidate R-VES − best R-VES ≥ 5 points.
- Significance: a one-sided sign test on per-task credit gives p ≤ 0.1.
- Regression: at most 1 task whose credit is ≥ 0.75 under the best and ≤ 0.25 under the candidate.
- Confirmation: ΔR-VES ≥ 3 points over both seeds.

Cost guardrail: if the candidate uses more than 1.5× the best's mean tokens or wall time per task, both Δ thresholds double.

**SWE (on hold, 2026-10-15).** All three conditions must hold:

- Gain: harmonic-mean SR on val improves by at least 10% relative, and a paired bootstrap gives P(Δ>0) ≥ 0.8.
- Regression: at most 1 task that has a real speedup under the best (SR > 1/gold) and drops to SR ≤ 1/gold under the candidate.
- Cost guardrail: same as Effi-SQL.

**Every lane: efficiency cap.** A candidate is rejected outright, whatever its gain, if on the same split its mean cost or mean wall time per task is more than 3× that of baseline v0. Without this cap, many small accepted changes could add up to a large cost increase; the cap keeps gains from depending on unlimited extra compute. The report states the cap and where the final harness stands against it.

**Why these thresholds.** A higher bar accepts fewer changes, but the ones it accepts are more trustworthy. A lower bar accepts more changes, but more of them are noise mistaken for gains.

- Effi-SQL: with 30 val tasks, each task carries up to 3.3 points of R-VES, and one credit tier (0.25) is worth about 0.8 points. A 5-point gain therefore needs at least two tasks to improve, so one noisy flip at a tier boundary (§4.1) cannot pass on its own, and the confirmation seed checks that the gain holds.
- SWE: with only 12 tasks the noise is larger, so SWE requires a larger relative gain.

All thresholds are set in config, and every decision records the thresholds it used.

### 8.4 Stopping

A lane stops after 4 rounds in a row with no accepted change, or when its time budget runs out.

### 8.5 Anti-gaming and safety boundaries

| Risk | Defense |
|---|---|
| Editing the grader, core or contracts | The implementer can write only to its lane's directories, and a candidate whose diff touches any other path is rejected. The `bench/`, `harness/core/` and `contracts/` checksums are verified before every evaluation |
| Hard-coded answers | A scan of the diff rejects it if it contains a fragment of any split's gold answer (≥ 40 characters after normalization): gold `optimized_sql` for Effi-SQL, the gold patch for SWE (on hold). Effi-SQL instance ids are not scanned: they are small integers, so a match would mean nothing. There is no database-name rule either, since every task runs on the one `tpch` database. The SWE scan (on hold) also rejects an instance_id from any split |
| Changing the database: indexes or data | Agent sessions connect with a read-only role (§4.1). Grading runs on its own instances, and the official evaluation runs on the pristine database |
| Relying on the improvement model | The model allowlist lives in core, which RSI cannot edit |
| Special-casing the workload or caching between timing runs (SWE, on hold) | The official grader times each repetition in a fresh process and checks for stack inspection; the prompt and skills forbid both; held-out results expose this kind of overfitting |
| Editing test files (SWE, on hold) | The official grader resets test files before it runs them |
| Overfitting to val | The analyst never sees val results or trajectories. Held-out is used only for v0's baseline runs and the final runs |

### 8.6 Ledger and lessons

- `experiments.jsonl` (`ExperimentRecord` contract): each record holds the experiment ID, lane, parent version, hypothesis and where it came from (idea bank or GLM's own analysis), kind of change, diff stats, screening and validation metrics, decision and reasoning, and cost. The report states how many accepted changes came from each source.
- `lessons.jsonl` (`Lesson` contract) is the source of truth for lessons. Entries are added one at a time, as in ACE, and each entry counts how often it helped or hurt. GLM maintains the file and can delete entries, but never rewrites it in one go, so nothing is lost by accident.
- `lessons.md` is generated from `lessons.jsonl` for human readers. Counters embedded in hand-edited markdown could not be updated reliably.

### 8.7 Operations

- **Processes.** Each lane runs as its own background process. Every step is recorded in the ledger, so a lane that crashes restarts from the ledger and picks up where it stopped.
- **Results PRs.** A lane opens an "RSI results" PR when it has 3 accepted changes since its last PR, or when 3 hours have passed since then, whichever happens first. A final results PR is opened before the freeze. Each PR contains the accepted pack changes, the ledger entries and the lessons. Results PRs wait for the project lead's review; the lane does not pause for them.
- **Notifications.** The project lead receives a push notification when:
  - a lane crashes and automatic recovery fails;
  - the gateway circuit breaker trips repeatedly;
  - a results PR is opened;
  - a lane stops because 4 rounds in a row accepted nothing.

## 9. Cost and efficiency accounting

- Every LLM call records its purpose (exec, rsi-analyst, rsi-impl or rsi-lesson), model, tokens, cost from the response header, and latency.
- The report shows two costs separately:
  - what the final harness costs, in money and time, to run on the held-out set;
  - what the RSI experiments cost in total, by lane and by purpose.
- On the machine side, we record wall time per task, how long each Effi-SQL evaluation holds its grading instance, and the CPU time used by SWE evaluation (on hold).
- The report compares the final harness's mean cost and wall time per task with v0's, against the 3× efficiency cap (§8.3).

## 10. Final evaluation and reproducibility

- **Freeze:** core and `best/effisql` are merged and tagged `final`. `best/swe` would join them only if the SWE lane resumes.
- **Runs:**
  - final runs 3 times on held-out in official mode (§4.1), plus 3 times on val;
  - v0's three held-out runs, also in official mode, were already done during the baseline phase.
- **The report includes:**
  - starting score, final score and the improvement, with mean ± standard deviation and confidence intervals;
  - how much scores vary between runs;
  - results by difficulty, and by repository if SWE resumes;
  - efficiency and experiment cost;
  - a plot of score by round;
  - lessons, each citing experiment IDs.
- **Reproduction:** `make final` does the following in order:
  1. prepares environments (the TPC-H database and its PostgreSQL instances; SWE images only if that lane resumes);
  2. runs final;
  3. grades the results;
  4. generates the report.

  The report reads v0's held-out results from the runs archived during the baseline phase. A separate `make baseline` reruns v0 for anyone who wants to reproduce it too. We keep the two commands apart because rerunning v0 as part of `make final` would add about 3 hours of official-mode grading (§6.2).

  Dependencies are pinned in `uv.lock`, and the manifests record every version.

## 11. Engineering standards and workflow

**Stack and tests**

- Python 3.12 + uv, ruff, mypy --strict, import-linter, pre-commit and GitHub Actions.
- Protobuf for the shared data contracts (§3.3): `grpcio-tools` and `mypy-protobuf`, regenerated with `make proto`. CI fails if the generated code is out of date.
- pytest, written test-first. Unit tests never touch the network. Tests that need Docker, a database or the real API are marked `integration` and are skipped in CI.

**Config and logging**

- Configs are validated with pydantic.
- Secrets are read only from environment variables or `~/.config/goaly/.env`.
- Logs are structured JSON.

**Language:** everything in the repository is written in English, including code, comments, design docs, the README and reports.

**Collaboration**

- Every PR stays small: roughly 400 lines or less of non-test code. Generated code, such as the protobuf output, does not count toward that limit.
- A ticket usually maps to one PR. A larger ticket is split into a short series of stacked PRs, and each PR in the series references the ticket. Tickets themselves are not split.
- PRs that depend on each other are stacked.
- The project lead reviews and merges every PR. Review feedback is fixed in new commits, never by force-pushing.
- Changes that RSI makes on its own go only to `rsi/*` branches. They are bundled into "RSI results" PRs from time to time.

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Grading time. Dev mode spends about 30–90 minutes per round on grading, each official held-out run takes about 40–60 minutes, and the one-time setup about 3 hours (§6.2). v0's held-out runs share the 3 grading instances with the RSI lane, and a slow prediction can hold an instance for up to the 300 s statement timeout on each run | Dev mode caches the base side, and the 3 grading instances run in parallel. The setup starts as soon as TPC-H is loaded and the grader lands (T10), and v0's held-out runs start in the background as soon as v0 exists (T12). T10 measures the real times first, and the schedule (§13) is adjusted from them |
| Generating TPC-H. Our dbgen build and load may not match the authors' setup, and generating and loading SF3 takes time | T09 does it first, in the background: the official schema (primary and foreign keys only), then `ANALYZE`, then one data directory copied into the 5 instances. Exclusions and the gold and identity self-checks run on this machine's data (T10), so a mismatch shows up before RSI starts |
| Timing noise at tier boundaries: a τ near 1 can flip between credit 0.25 and 0.5 from one run to the next | Medians of 3 runs on dedicated, pinned grading instances. The decision rules add a sign test, a regression bound and a confirmation seed (§8.3). Final scores are reported as mean ± standard deviation over 3 runs |
| The official evaluator's repository has no license | We reimplement its algorithm (§4.1) rather than vendor its code. The gold and identity self-checks confirm that the reimplementation behaves as expected |
| SWE on hold: the PDF's "gains on more than one benchmark" is at risk (§2) | Nothing within this deadline; the report states the gap plainly. The SWE design stays in this document and the RSI machinery has no benchmark-specific code, so the lane can resume without a redesign |
| Tight schedule: 21 hours from 2026-10-15 03:00 UTC to the deadline | One active lane. Contracts come first, then component PRs in parallel; stacked PRs keep review from blocking work. RSI must be running by about 16:00 UTC (§13) |
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
| 03:00–05:00 | T01 (this amendment); T02a's contracts reworked for Effi-SQL |
| 05:00–07:30 | T03, T04, T08 and T09 in parallel; T09 generates and loads TPC-H SF3 in the background |
| 07:30–09:30 | T05 and T10. Once T10 lands, the setup runs in the background: the base cache, the self-checks and the exclusion list |
| 09:30–12:30 | T06, then T07 and T18 |
| 12:30–14:30 | T11, T19 and T20. T12 starts as soon as T11 lands: v0 on train and val first, then its three held-out runs in the background |
| 14:30–16:00 | T21 |
| 16:00–20:00 | T22: the Effi-SQL lane runs, about 2–4 rounds. It must be running by about 16:00 |
| 20:00–23:00 | T23: freeze, the final runs (3 on held-out in official mode, 3 on val) and the report |
| 23:00–00:00 | T24, and buffer before the deadline |

## Appendix A: Ticket mapping

T01–T24 correspond, in order, to CIR-1633 through CIR-1656, so T_n = CIR-(1632+n). T02a (shared contracts) was added during spec review and is tracked separately in Linear. The dependencies here match the blocked-by relations in Linear.

**Tracks after the 2026-10-15 amendment**

- T08–T12 (CIR-1640 to CIR-1644) are now the Effi-SQL track:

  | Ticket | Linear | Work |
  |---|---|---|
  | T08 | CIR-1640 | Effi-SQL data, splits and exclusions (§4.1, §7) |
  | T09 | CIR-1641 | PostgreSQL 16 environment: TPC-H SF3 and the 5 pinned instances (§4.1, §6.2) |
  | T10 | CIR-1642 | Grader: dev and official modes, self-checks and the exclusion run (§4.1) |
  | T11 | CIR-1643 | Baseline pack v0 and `goaly run --bench effisql` (§5.4, §6.4) |
  | T12 | CIR-1644 | Baseline experiment (§6.2, §10) |

- T13–T17 (CIR-1645 to CIR-1649), the SWE-fficiency track, are on hold (2026-10-15).
