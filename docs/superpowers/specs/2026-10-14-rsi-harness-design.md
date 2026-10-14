# RSI Self-Evolving Harness: Design

- Date: 2026-10-14
- Status: in review
- Linear: project P-CIR-12; this document is T01 = CIR-1633
- Deadline: 2026-10-15

## 1. Background and goals

### 1.1 Task

We are building a self-evolving harness system:

- The **improvement model**, GLM-5.3, drives an RSI agent. The agent creates and refines the executor's harness: tools, skills, system prompts, the agent loop and context management.
- The **executor model**, Qwen3.6-35B-A3B, solves tasks using that harness. The goal is to beat a human-designed baseline on both SWE-fficiency and BIRD-CRITIC.

We hand over the frozen harness. The organizers evaluate it on a hidden held-out set using only the executor model. The improvement model is not available after development ends.

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
| Time | One day; deadline 2026-10-15 |
| Model gateway | `https://api.goaly.ai/inference/v1` (LiteLLM 1.92 + SGLang), OpenAI-compatible; key in `GOALY_API_KEY` |
| Executor model | `qwen3.6-35b-a3b`: FP8, 35B total / 3B active parameters; 262K native context, which the gateway stretches to 1.01M with YaRN; $0.15 input / $1.00 output per 1M tokens |
| Improvement model | `glm-5.3`: ~744B total / 40B active; thinking cannot be turned off; 1M context; $1.40 input / $4.40 output per 1M tokens |
| Budget | No practical limit on API spend, but efficiency is still graded |
| Machine | 12 cores / 24 threads, 125 GB RAM, ~800 GB disk, no GPU |
| Docker | Works only with `--network=host`, because runc 1.3.6 hits a procfs restriction inside this nested container |

### 1.4 Non-goals

- No training or fine-tuning.
- No BIRD-CRITIC MySQL, SQL Server or Oracle support. The interfaces leave room for these dialects.
- No full 498-task SWE-fficiency run. This machine can't evaluate that many.

## 2. Acceptance criteria mapping

| PDF criterion | How the design meets it | Evidence |
|---|---|---|
| Benchmark score: starting and final scores, with improvement on every benchmark | Baseline v0 (§5.4) and final are compared on the same held-out set | T12, T17 and T23 reports |
| Robustness: run the same test at least 3 times and report the variation; survive tool errors, timeouts and bad model output | Three held-out runs; fault tolerance in core (§5.1); per-task exception isolation (§6.1); concurrency budgets and adaptive backoff (§6.2) | T23 report; T04 and T05 tests |
| Generalization: held-out tasks stay out of development; gains on more than one benchmark; no hard-coded answers | Three-way split (§7) and a hard-coding scan (§8.5). Each benchmark gets its own lane, and both lanes run the same RSI machinery with no benchmark-specific code, so a gain on two very different tasks shows that the method generalizes | Committed split ID lists; scan logs; per-benchmark gains in the T23 report |
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
proto/goaly/v1/      shared data contracts: task.proto, trajectory.proto, result.proto,
                     manifest.proto, llm_call.proto, experiment.proto
src/goaly/
  contracts/
    gen/             Python code generated from proto/ (committed; CI checks it is up to date)
    interfaces.py    Environment, SqlEnvironment, ShellEnvironment, Policy, Tool, Benchmark
  harness/
    core/            llm.py, tools.py, agent.py, policy.py, context.py, trajectory.py, config.py
    packs/
      birdcritic/    prompt.md, skills/, tools.py, policy.py, config.yaml
      swefficiency/  (same files)
  bench/
    birdcritic/      data.py, env.py, grader.py, adapter.py
    swefficiency/    data.py, env.py, grader.py, adapter.py
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

### 3.3 Shared contracts

Every shared type lives in `goaly.contracts`. These types come in two kinds, and each kind uses a different technology.

**Data contracts (protobuf).** Data that is written to disk, read by another process, or has to stay readable across schema versions is defined once, in `proto/goaly/v1/`:

| Message | Purpose |
|---|---|
| `TaskInput`, `Prediction`, `Grade` | What a policy receives, what it returns, and how the result was graded. Each has one variant per benchmark |
| `Trajectory`, `Step` | Every step of a run: messages, tool calls, results, usage and timing |
| `TaskResult` | Per-task record: score, correctness, time, tokens, cost, steps and error type |
| `RunManifest` | Commit, content and config hashes, dataset revision, model settings, timestamps and machine |
| `LlmCallRecord` | One LLM call: purpose, model, tokens, cost, latency and status |
| `ExperimentRecord`, `Hypothesis`, `Lesson` | The RSI ledger |

- Field numbers are never reused, and removed fields are marked `reserved`, so files written today stay readable after the schema changes.
- On disk, messages are stored as JSONL using protobuf's standard JSON mapping, keeping the original field names. People and the GLM analyst read trajectories directly, so a readable text format matters more than a compact binary one.
- The official submission formats (BIRD `pred_sqls`, SWE predictions) are not our contracts. Each adapter converts to them and keeps the official field names exactly.
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
- **harness never cares where an environment comes from.** bench provides the implementations: `SqlEnvironment` as a PostgreSQL sandbox, `ShellEnvironment` as a Docker container. harness unit tests use fake environments and need neither a database nor a container.
- **Tool arguments stay as pydantic models.** The OpenAI tools API takes JSON Schema, which pydantic generates directly. GLM also edits tools during RSI, and with protobuf every edit would mean changing a `.proto` file and rerunning code generation.
- **gRPC-ready, but no gRPC yet.** Interface methods take and return only contract types (protobuf messages or plain values), never framework objects. That way a gRPC-backed implementation can later replace an in-process one without changing any caller. One example is running SWE evaluation workers on other machines, which would lift the CPU ceiling described in §6.2. We are not building gRPC now.

Adding a benchmark takes a `Benchmark` adapter and a capability pack. core does not change.

## 4. Benchmarks

### 4.1 BIRD-CRITIC (fixing broken SQL)

**Data**

- We use `birdsql/bird-critic-1.0-postgresql`: 530 PostgreSQL tasks, pinned to one HF revision.
- Public fields: instance_id, db_id, query (the user's question), issue_sql, preprocess_sql, clean_up_sql, category, and schema text for each task.
- The gold `sol_sql` and `test_cases` are not public. We request them by email: write to bird.bench25@gmail.com with `[bird-critic-1 GT&Test Cases]` in the subject. GT files are never committed.

**Scoring**

A task counts as solved when the prediction runs with no error or timeout and passes every test case for that task. The score is the share of solved tasks. Each of the four categories checks something different:

| Category | Check |
|---|---|
| Query | `ex_base` compares result sets |
| Personalization | extra constraint checks |
| Management | checks the database state after execution |
| Efficiency | EXPLAIN cost must be lower than for issue_sql |

**Grading details** (these must match the official harness exactly)

- Results are compared as sets of rows, so row order and duplicate rows don't matter, but column order does.
- Dates are truncated to the day. Floats must match exactly.
- An empty result fails.
- Only the last statement's result counts.
- The predicted SQL runs twice.
- Grading limits are 60 s per statement, 300 s per task and 10,000 rows. The agent's own solving budget is set separately (§5.4).

**Environment**

- PostgreSQL 14.12 + PostGIS, taken from the official compose setup, with host networking and the official time zone.
- The official dump is restored into template databases.
- Each task gets its own copy via `CREATE DATABASE … TEMPLATE`. preprocess_sql runs on that copy, and the copy is dropped afterwards.
- Solving and grading never share a copy.

**Grader**

- We vendor the official test_case helpers (MIT license, attribution kept). They run on our template-clone sandboxes, so many tasks can be graded at once.
- Two checks show that we match the official harness:
  - In gold mode, the gold answers should pass every task. Tasks with known-bad official answers are listed separately.
  - On a sample of tasks, results are compared one by one against the official harness.
- **Excluded tasks.** A task whose gold answer fails in gold mode is removed from our splits, and the list of removed tasks is committed along with the reason for each. Such a task cannot be scored, so keeping it would only add noise to development and to accept/reject decisions. This applies only to our development sets: the harness still handles any such task that appears in the organizers' held-out set.

**Reference scores (PG-530):** ~30B open models 28–32%; Claude Opus 4.6 43.4%; leaderboard best 56.8%; humans 78.9%.

### 4.2 SWE-fficiency (making repository code faster)

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
- **Excluded tasks.** A task is removed and replaced with another task from the same repository if, on this machine, its gold patch fails its tests or speeds the workload up by less than 1.5×. Expert speedups depend on hardware, and on such tasks no change can show a reliable gain. The removed tasks and the reason for each are committed. As with BIRD, this affects only our development sets.
- **Open question: what the official evaluation runs.** Some tasks list a huge number of PASS_TO_PASS tests: one Lite pandas task has about 211K, alongside about 900 `covering_tests`. Whether evaluation runs every PASS_TO_PASS test or only `covering_tests` decides whether one task takes minutes or hours. T15 measures this first. Tasks that take more than 25 minutes to evaluate are swapped out (§7).

**Resource constraint:** evaluation is the project's biggest bottleneck. Solving and evaluation run on separate CPU slots so one does not distort the other's timings.

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

- BIRD has two tools:
  - `execute_sql` runs a query in the task sandbox and returns truncated results or the error.
  - `submit` hands in the answer.

  The prompt is minimal.
- SWE has two tools:
  - `bash` runs a command in the task container.
  - `submit` hands in the answer.

  The prompt is minimal but lists the official constraints.
- Both use the same settings:
  - Thinking is on. Sampling is T=0.6, top_p=0.95, top_k=20, with max_tokens 16K per call.
  - Solving budgets are 30 steps or 10 minutes for BIRD, and 60 steps or 45 minutes for SWE.
- Fallback when a budget runs out before `submit`: v0 still submits something rather than nothing.
  - BIRD submits the last SQL that executed successfully. If none did, it submits the original issue_sql.
  - SWE submits the container's final git diff, as OpenHands and other common agents do.

  Neither fallback checks whether the result is any good. Smarter policies, such as never submitting an unverified patch, are left for RSI to discover.
- v0 is frozen and tagged `baseline-v0`. Every comparison is made against it.

## 6. Evaluation framework (goaly.bench)

### 6.1 Runner and records

**Runner**

- Built on asyncio, with separate resource pools for: LLM concurrency, database slots, SWE solving CPU slots and SWE evaluation workers (at most 4).
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

| Phase | Qwen (48 total) | GLM (16 total) | Notes |
|---|---|---|---|
| Baseline (T12, T17) | BIRD 36, SWE 12 | unused | Run train and val first, then v0's three held-out runs |
| RSI (T22) | BIRD lane 28, SWE lane 12, background 8 | BIRD lane 8, SWE lane 8 | Background covers confirmation reruns and any v0 held-out runs still pending |
| Final evaluation (T23) | BIRD 40, SWE 8 | unused | GLM is no longer available |

An agent never has more than one LLM request in flight, so a process runs about as many tasks at once as its Qwen share. Each share is an upper limit: adaptive limiting (below) lowers it when the gateway is busy. Shares are config values, so changing them does not require a code change.

**Adaptive limiting** (in the LLM client, T03)

- On a 429 or 503, wait for `Retry-After` if the response has one. Otherwise back off exponentially with full jitter: 1 s base, 60 s cap.
- AIMD:
  - after a burst of 429s, 503s or timeouts, the process halves its concurrency for that model;
  - after every 20 consecutive successes, it adds 1 back;
  - concurrency never exceeds the process's configured share.
- Circuit breaker: if more than half of a model's calls fail within 2 minutes, the process stops starting new tasks for 60 s and logs a warning. Calls already in flight finish or time out as usual.
- Every throttling event is logged with the call's purpose, so the report can show how much time the gateway cost us.

**Throughput planning.** Qwen output tokens are the scarce resource.

- A BIRD val pass is 100 tasks × 2 seeds, with about 15 calls per task and about 1.5K output tokens per call. That comes to about 4.5M output tokens, or roughly 12–16 minutes depending on what else is running.
- Most of those tokens are spent on thinking. A policy that uses fewer tokens therefore also speeds up RSI iterations.
- A BIRD RSI round takes about 45–60 minutes: 3 candidates, screening, validation and one confirmation rerun. That gives the BIRD lane about 8–10 rounds overnight.
- The SWE lane is limited by CPU and evaluation, not by the gateway. An SWE round takes about 2.5 hours. Solving runs on 8 slots and is capped at 45 minutes per task; evaluation runs on 4 workers. That allows 3–4 rounds overnight. Config can fit in more rounds by using one candidate per round or by capping solving at 30 minutes.
- The SWE figures are estimates. T14 and T15 measure actual solving and evaluation times, and these numbers will be updated from those measurements.

**Inside a process**

- Each process runs one asyncio event loop.
- Blocking work (Docker SDK calls, subprocesses, file I/O, git) goes through async APIs or `asyncio.to_thread`. Threads never share mutable state.
- The HTTP client keeps one connection pool per model, sized to the process's share.

**CPU partition.** The machine has 12 physical cores (24 hardware threads).

| Physical cores | Used for |
|---|---|
| 0–1 | OS, dockerd and our Python processes |
| 2–3 | PostgreSQL for BIRD |
| 4–11 (8 cores) | SWE only. Each wave uses one of two modes: solving (8 slots, 1 core each) or official evaluation (4 workers, 2 cores each) |

SWE solving and SWE evaluation never run at the same time. Solving agents run tests and workloads that keep their cores fully busy. If official timing ran next to them, the measured speedups would be distorted, and so would every accept/reject decision based on them. The SWE lane therefore works in waves: it solves a batch, then evaluates that batch. BIRD is not affected: its database has its own cores, and its bottleneck is LLM throughput.

**Other shared resources**

| Resource | Limit | How it is enforced |
|---|---|---|
| PostgreSQL connections | `max_connections` 200. Pools: 32 for the BIRD lane, 12 for background runs, 12 for the grader | psycopg async pools; each task gets its own sandbox database |
| Template clones | One clone at a time per template database | PostgreSQL refuses to clone a template while it has other connections. The databases are small, so serializing clones costs little |
| SWE solving and evaluation | 8 solving slots or 4 evaluation workers, never both at once | The slot scheduler (T14) pins containers with cpuset and switches mode between waves |
| Disk | Image budget | Images are pruned in batches (T13) |

**Concurrent writes**

- Each per-run JSONL file (results, trajectories, call logs) has a single writer task that reads from an asyncio queue. Lines are flushed as they are written, so a crash loses at most the line being written.
- Cache entries are written to a temporary file and then renamed into place, so readers never see a partial entry.
- Both lanes write `experiments.jsonl` and `lessons.jsonl`, and the lanes run as separate processes. Every write takes an exclusive `fcntl` lock.
- A repository-level file lock serializes git operations on the main repository, such as adding or removing worktrees and moving tags. Each candidate edits only its own worktree.

**Tests**

- A FakeLLM that injects 429s, latency and timeouts drives 500 simulated tasks. The tests check that no process exceeds its share, that limits back off and recover, that nothing deadlocks, and that every result is recorded exactly once.
- Concurrency tests cover the JSONL writer, the cache and the ledger lock.

### 6.3 Metrics

- Main metrics: BIRD success rate and SWE harmonic-mean SR.
- With several seeds, we report mean ± standard deviation and bootstrap 95% confidence intervals.
- Two versions are compared task by task. We count the tasks each version wins and run a one-sided sign test. SWE uses a paired bootstrap.
- The stably solved set is the tasks solved in every seed.
- Results are broken down by category and by repository.

### 6.4 CLI

- `goaly run --bench {birdcritic,swefficiency} --split {train,val,heldout} --harness <path|tag> --seeds N --out runs/<id>`
- `--instances <official-format jsonl>` runs any task file. The organizers use this for their held-out set, and the README documents it as the official entry point:

  ```
  goaly run --bench birdcritic --instances heldout.jsonl --harness final --out predictions.jsonl
  ```

  This needs only the executor API and the matching environment: a database for BIRD, Docker for SWE.

## 7. Data splits

| | train | val | held-out |
|---|---|---|---|
| BIRD (PG-530, stratified by category) | 150 | 100 | 280 |
| SWE (IDs from the 100-task Lite set, stratified by repository; task data from the revised main dataset) | 12 | 12 | 12 |

- The random seed is fixed at `20261014`. The split ID lists are committed to `src/goaly/bench/splits/`.
- Each split has one job:
  - **train:** the analyst may see full trajectories, grader feedback and gold answers.
  - **val:** results go only to the decision step; the analyst never sees them.
  - **held-out:** no modified version runs on it during development. The one exception is v0, whose held-out results are the baseline for the final comparison.
- If an SWE task takes more than 25 minutes to evaluate, it is swapped for another task from the same repository, and the swap is recorded.
- Exclusions (§4.1, §4.2) are applied after splitting, which keeps the split itself reproducible. An excluded BIRD task is simply dropped from its split, so that split ends up slightly smaller. An excluded SWE task is replaced with another task from the same repository, so every SWE split keeps 12 tasks.

## 8. RSI system (goaly.rsi)

### 8.1 Lanes

- BIRD and SWE each get their own lane, and the two run in parallel.
- Each lane tracks its best version with a git tag (`best/bird`, `best/swe`) and edits only its own pack.
- core and bench are frozen while RSI runs.

### 8.2 One round

1. **Analyze** (GLM, purpose `rsi-analyst`)
   - Inputs:
     - a summary of the ledger, rejected attempts included;
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
   - The top k hypotheses (k=3 for BIRD, k=2 for SWE) are built in parallel, each in its own git worktree.
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
   - BIRD: 1 seed on up to 15 train failures the hypothesis targets, plus 5 train tasks the best already solves.
   - SWE: the 3 train tasks the hypothesis targets.
5. **Validate.** BIRD runs all 100 val tasks with 2 seeds. SWE runs all 12 val tasks with 1 seed.
6. **Decide** using the rules in §8.3. If more than one candidate passes, the one with the largest Δ is accepted. The other passing hypotheses go back in the queue and are rebuilt on the new best next round.
7. **Confirm** (BIRD only). The accepted candidate runs on val with one more seed. The best changes only if Δ over all 3 seeds is still at least 2pp.
8. **Record.** Every candidate goes into the ledger. For each one, GLM writes a lesson: what was tried, what happened and why, citing the experiment ID.

### 8.3 Decision rules

**BIRD.** A task's solve rate is its average over seeds. All four conditions must hold:

- Net gain: Δ = candidate mean success rate − best mean success rate ≥ 3pp.
- Significance: the candidate wins more tasks than it loses (a win is a task where its solve rate is higher than the best's), and a one-sided sign test gives p ≤ 0.1.
- Regression: at most 1 task that the best solves in every seed and the candidate fails in every seed.
- Cost guardrail: if the candidate uses more than 1.5× the best's mean tokens or wall time per task, the Δ threshold doubles.

**SWE.** All three conditions must hold:

- Gain: harmonic-mean SR on val improves by at least 10% relative, and a paired bootstrap gives P(Δ>0) ≥ 0.8.
- Regression: at most 1 task that has a real speedup under the best (SR > 1/gold) and drops to SR ≤ 1/gold under the candidate.
- Cost guardrail: same as BIRD.

**Both lanes: efficiency cap.** A candidate is rejected outright, whatever its gain, if on the same split its mean cost or mean wall time per task is more than 3× that of baseline v0. Without this cap, many small accepted changes could add up to a large cost increase; the cap keeps gains from depending on unlimited extra compute. The report states the cap and where the final harness stands against it.

**Why these thresholds.** A higher bar accepts fewer changes, but the ones it accepts are more trustworthy. A lower bar accepts more changes, but more of them are noise mistaken for gains.

- BIRD: with 100 val tasks and 2 seeds, 3pp is roughly the smallest gain that clears noise.
- SWE: with only 12 tasks the noise is larger, so SWE requires a larger relative gain.

All thresholds are set in config, and every decision records the thresholds it used.

### 8.4 Stopping

A lane stops after 4 rounds in a row with no accepted change, or when its time budget runs out.

### 8.5 Anti-gaming and safety boundaries

| Risk | Defense |
|---|---|
| Editing the grader, core or contracts | The implementer can write only to its lane's directories, and a candidate whose diff touches any other path is rejected. The `bench/`, `harness/core/` and `contracts/` checksums are verified before every evaluation |
| Hard-coded answers | A scan of the diff rejects it if it contains any of: an instance_id from any split; part of any split's gold SQL or gold patch (≥ 40 characters after normalization); a db_id written as a full identifier. A db_id that is also an ordinary English word only raises a warning |
| Relying on the improvement model | The model allowlist lives in core, which RSI cannot edit |
| Special-casing the workload or caching between timing runs | The official grader times each repetition in a fresh process and checks for stack inspection; the prompt and skills forbid both; held-out results expose this kind of overfitting |
| Editing test files | The official grader resets test files before it runs them |
| Overfitting to val | The analyst never sees val trajectories, and held-out is used only once, at the end |

### 8.6 Ledger and lessons

- `experiments.jsonl` (`ExperimentRecord` contract): each record holds the experiment ID, lane, parent version, hypothesis and where it came from (idea bank or GLM's own analysis), kind of change, diff stats, screening and validation metrics, decision and reasoning, and cost. The report states how many accepted changes came from each source.
- `lessons.jsonl` (`Lesson` contract) is the source of truth for lessons. Entries are added one at a time, as in ACE, and each entry counts how often it helped or hurt. GLM maintains the file and can delete entries, but never rewrites it in one go, so nothing is lost by accident.
- `lessons.md` is generated from `lessons.jsonl` for human readers. Counters embedded in hand-edited markdown could not be updated reliably.

### 8.7 Operations

- **Processes.** Each lane runs as its own background process. Every step is recorded in the ledger, so a lane that crashes restarts from the ledger and picks up where it stopped.
- **Results PRs.** A lane opens an "RSI results" PR when it has 3 accepted changes since its last PR, or when 3 hours have passed since then, whichever happens first. A final results PR is opened before the freeze. Each PR contains the accepted pack changes, the ledger entries and the lessons. PRs opened overnight wait for the project lead's review.
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
- On the machine side, we record wall time per task and the CPU time used by SWE evaluation.
- The report compares the final harness's mean cost and wall time per task with v0's, against the 3× efficiency cap (§8.3).

## 10. Final evaluation and reproducibility

- **Freeze:** core, `best/bird` and `best/swe` are merged and tagged `final`.
- **Runs:**
  - final runs 3 times on each benchmark's held-out set and 3 times on val;
  - v0's three held-out runs were already done during the baseline phase.
- **The report includes:**
  - starting score, final score and the improvement, with mean ± standard deviation and confidence intervals;
  - how much scores vary between runs;
  - results by category and by repository;
  - efficiency and experiment cost;
  - a plot of score by round;
  - lessons, each citing experiment IDs.
- **Reproduction:** `make final` does the following in order:
  1. prepares environments (images and databases);
  2. runs final;
  3. grades the results;
  4. generates the report.

  The report reads v0's held-out results from the runs archived during the baseline phase. A separate `make baseline` reruns v0 for anyone who wants to reproduce it too. We keep the two commands apart because rerunning v0 as part of `make final` would double the time; SWE evaluation alone would add several hours.

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
| SWE evaluation is too slow | T15 measures it first. If it's too slow, shrink the subset. Empty patches skip evaluation, results are cached, and v0's held-out runs start early in the background |
| BIRD gold answers arrive late | Build everything else first. If needed, use public Six-Gym-PG data for train in the meantime |
| Tight schedule (critical path ≈ 20 h) | BIRD and SWE tracks run in parallel. Stacked PRs keep review from blocking work. RSI runs overnight |
| Upstream interfaces change during review | Settle the interfaces first, in T05 and T06. Downstream PRs rebase |
| Malformed Qwen tool calls | Fallback parsing and fault tolerance in core (T04) |
| The gateway gets overloaded, possibly because other teams share it | Concurrency budgets at about 75% of measured capacity; adaptive limiting and a circuit breaker (§6.2) |
| Noisy evaluations | Paired comparisons, multiple seeds and confirmation reruns |
| SWE images use too much disk | Pull only the images the subsets need, and prune in batches |

## 13. Schedule (critical path)

| Hours from start | Work |
|---|---|
| 0–1 | T01, T02 |
| 1–1.5 | T02a (shared contracts) |
| 1.5–3 | T03, T04, T08, T09, T13 and T14 in parallel; images and database dumps download in the background |
| 3–5 | T05, T06, T10, T15 |
| 5–7 | T07, T11, T16, T18; baseline experiments start (T12, T17) |
| 7–9 | T19, T20, T21 |
| 9–17 | T22: both lanes running |
| 17–20 | T23, T24 |

## Appendix A: Ticket mapping

T01–T24 correspond, in order, to CIR-1633 through CIR-1656, so T_n = CIR-(1632+n). T02a (shared contracts) was added during spec review and is tracked separately in Linear. The dependencies here match the blocked-by relations in Linear.
