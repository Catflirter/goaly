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

1. **The frozen harness**: code, configs, prompts, skills and pinned dependencies. A single command runs it on tasks in the official formats.
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
| Robustness: run the same test at least 3 times and report the variation; survive tool errors, timeouts and bad model output | Three held-out runs; fault tolerance in core (§5.1); per-task exception isolation (§6) | T23 report; T04 and T05 tests |
| Generalization: held-out tasks stay out of development; gains on more than one benchmark; no hard-coded answers | Three-way split (§7); one lane per benchmark; hard-coding scan (§8.5) | Committed split ID lists; scan logs |
| Lessons learned: what was tried, what worked, what failed and why, backed by experiments | Experiment ledger and lessons (§8.6) | `experiments.jsonl`, `lessons.md` |
| Efficiency: time, tokens, compute and cost reported alongside the score | Cost and tokens recorded for every call (§9); cost guardrail (§8.3) | Report tables |
| Reproducibility: code, configs, prompts, tool versions and a single command | uv lockfile, run manifests, `make final` (§10) | T23, T24 |
| Automation: several improvement rounds with little human help | `goaly rsi run` (§8) | Ledger entries across rounds |
| Regression check: keep the best harness; new changes must not break tasks that already pass | Best-version tags and a regression rule (§8.3) | Decision log |

## 3. Architecture

### 3.1 Layers

```
goaly.rsi       development only; uses GLM-5.3
   │ depends on
goaly.bench     data, environments, grading, runner, metrics
   │ depends on
goaly.harness   the deliverable; uses only Qwen at runtime
   ├─ core/     infrastructure (RSI cannot edit it)
   └─ packs/    one capability pack per benchmark (the only part RSI can edit)
```

import-linter enforces these dependencies in CI. `harness` must not import `bench` or `rsi`. `bench` must not import `rsi`.

### 3.2 Repository layout

```
src/goaly/
  harness/
    core/            llm.py, tools.py, agent.py, policy.py, context.py, trajectory.py, config.py
    packs/
      birdcritic/    prompt.md, skills/, tools.py, policy.py, config.yaml
      swefficiency/  (same files)
    cli.py
  bench/
    base.py          Benchmark / Environment protocols
    birdcritic/      data.py, env.py, grader.py, adapter.py
    swefficiency/    data.py, env.py, grader.py, adapter.py
    runner.py, metrics.py, manifest.py
    splits/          split ID lists
  rsi/
    ledger.py, workspace.py, guards.py, analyst.py, implementer.py, decide.py, lane.py, cli.py
tests/               mirrors src/; tests that need Docker, a database or the real API are marked `integration`
docs/superpowers/{specs,plans}/   design and implementation plans
docs/results/                     experiment results
runs/                             run artifacts (not committed)
```

### 3.3 Key interfaces

This is a draft. The implementation PRs define the final signatures.

```python
class Environment(Protocol):
    """Runtime environment for one task, such as a database sandbox or a container."""
    async def close(self) -> None: ...

class Benchmark(Protocol):
    name: str
    def load_instances(self, split: Split) -> Sequence[Instance]: ...
    def provision(self, inst: Instance) -> AbstractAsyncContextManager[Environment]: ...
    def agent_input(self, inst: Instance) -> TaskInput: ...
    async def grade(self, inst: Instance, prediction: Prediction) -> Grade: ...
    def to_official(self, inst: Instance, prediction: Prediction) -> dict[str, Any]: ...

class Policy(Protocol):
    """Implemented by a capability pack; drives the solution of one task."""
    async def solve(self, task: TaskInput, env: Environment, ctx: AgentContext) -> Prediction: ...

class Tool(Protocol):
    name: str
    description: str
    args_model: type[BaseModel]
    async def run(self, args: BaseModel, env: Environment) -> ToolResult: ...
```

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

**Reference scores (PG-530):** ~30B open models 28–32%; Claude Opus 4.6 43.4%; leaderboard best 56.8%; humans 78.9%.

### 4.2 SWE-fficiency (making repository code faster)

**Data**

- We use `swefficiency/swefficiency`: 498 tasks from 9 repositories, pinned to one HF revision (the dataset was revised on 2026-03-30).
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

**Resource constraint:** evaluation is the project's biggest bottleneck. Solving and evaluation run on separate CPU slots so one does not distort the other's timings.

**Known failure modes**

- Wrong hotspot: 68% of expert gains come from functions the models never edit.
- Satisficing: stopping after the first small speedup.
- Gaming: caching workload results or special-casing the workload.

## 5. Harness design

### 5.1 core (RSI cannot edit it)

**LLM client**

- Async, httpx-based, OpenAI-compatible.
- Each call times out after 300 s. 429s, 5xx errors, timeouts and connection errors get up to 5 retries with exponential backoff.
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

**Trajectories** are versioned JSONL (`schema_version`). Each step records messages, tool calls and their results, token usage and latency. The trajectory ends with the final prediction and its grade.

### 5.2 Capability packs (the only part RSI can edit)

| File | Purpose |
|---|---|
| `prompt.md` | System prompt template |
| `skills/*.md` | Curated skill guides. Ideally no more than 3, each focused; any of them can be removed |
| `tools.py` | Tools for this benchmark, built on core's Tool protocol |
| `policy.py` | How a task is solved: phases, self-checks, when to add guidance, best-of-n, fallback submission |
| `config.yaml` | Sampling parameters, thinking on/off, budgets, context thresholds |

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
- v0 is frozen and tagged `baseline-v0`. Every comparison is made against it.

## 6. Evaluation framework (goaly.bench)

**Runner**

- Built on asyncio, with separate resource pools for: LLM concurrency, database slots, SWE solving CPU slots and SWE evaluation workers (at most 4).
- An exception in one task never affects the others. Each task's result is written to disk as soon as it's done, so an interrupted run can resume.
- Evaluations are cached by (harness content hash, task, seed).

**Run manifest:** git commit, harness content hash, config hash, dataset revision, model names and sampling parameters, start and end times, machine info.

**Per-task records:** score, correctness, wall time, tokens (input, output and reasoning, counted separately), cost, steps and error type.

**Metrics**

- Main metrics: BIRD success rate and SWE harmonic-mean SR.
- With several seeds, we report mean ± standard deviation and bootstrap 95% confidence intervals.
- Two versions are compared task by task. We count the tasks each version wins and run a one-sided sign test. SWE uses a paired bootstrap.
- The stably solved set is the tasks solved in every seed.
- Results are broken down by category and by repository.

**CLI**

- `goaly run --bench {birdcritic,swefficiency} --split {train,val,heldout} --harness <path|tag> --seeds N --out runs/<id>`
- `--instances <official-format jsonl>` runs any task file. The organizers use this for their held-out set. It needs only the executor API and the matching environment: a database for BIRD, Docker for SWE.

## 7. Data splits

| | train | val | held-out |
|---|---|---|---|
| BIRD (PG-530, stratified by category) | 150 | 100 | 280 |
| SWE (drawn from the 100-task Lite set, stratified by repository) | 12 | 12 | 12 |

- The random seed is fixed at `20261014`. The split ID lists are committed to `src/goaly/bench/splits/`.
- Each split has one job:
  - **train:** the analyst may see full trajectories, grader feedback and gold answers.
  - **val:** results go only to the decision step; the analyst never sees them.
  - **held-out:** no modified version runs on it during development. The one exception is v0, whose held-out results are the baseline for the final comparison.
- If an SWE task takes more than 25 minutes to evaluate, it is swapped for another task from the same repository, and the swap is recorded.

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
     - an explanation of the benchmark's scoring rules.
   - Output: structured JSON that groups failures by pattern and proposes 3 hypotheses. Each hypothesis states:
     - which failure pattern it targets;
     - the kind of change: prompt, skill, tool, policy, config or removal;
     - exactly what to change;
     - the expected effect and the risk.
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

All thresholds are set in config, and every decision records the thresholds it used.

### 8.4 Stopping

A lane stops after 4 rounds in a row with no accepted change, or when its time budget runs out.

### 8.5 Anti-gaming and safety boundaries

| Risk | Defense |
|---|---|
| Editing the grader or core | The implementer can write only to its lane's directories. The `bench/` and `core/` checksums are verified before every evaluation |
| Hard-coded answers | A scan of the diff rejects it if it contains any of: an instance_id from any split; part of any split's gold SQL or gold patch (≥ 40 characters after normalization); a db_id written as a full identifier. A db_id that is also an ordinary English word only raises a warning |
| Relying on the improvement model | The model allowlist lives in core, which RSI cannot edit |
| Special-casing the workload or caching between timing runs | The official grader times each repetition in a fresh process and checks for stack inspection; the prompt and skills forbid both; held-out results expose this kind of overfitting |
| Editing test files | The official grader resets test files before it runs them |
| Overfitting to val | The analyst never sees val trajectories, and held-out is used only once, at the end |

### 8.6 Ledger and lessons

- `experiments.jsonl`: each record holds the experiment ID, lane, parent version, hypothesis, kind of change, diff stats, screening and validation metrics, decision and reasoning, and cost.
- `lessons.md`: entries are added one at a time (as in ACE), and each counts how often it helped or hurt. GLM maintains the file and can delete entries. It is never rewritten in one go, so nothing is lost by accident.

## 9. Cost and efficiency accounting

- Every LLM call records its purpose (exec, rsi-analyst, rsi-impl or rsi-lesson), model, tokens, cost from the response header, and latency.
- The report shows two costs separately:
  - what the final harness costs, in money and time, to run on the held-out set;
  - what the RSI experiments cost in total, by lane and by purpose.
- On the machine side, we record wall time per task and the CPU time used by SWE evaluation.

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

  Dependencies are pinned in `uv.lock`, and the manifests record every version.

## 11. Engineering standards and workflow

**Stack and tests**

- Python 3.12 + uv, ruff, mypy --strict, import-linter, pre-commit and GitHub Actions.
- pytest, written test-first. Unit tests never touch the network. Tests that need Docker, a database or the real API are marked `integration` and are skipped in CI.

**Config and logging**

- Configs are validated with pydantic.
- Secrets are read only from environment variables or `~/.config/goaly/.env`.
- Logs are structured JSON.

**Language:** everything in the repository is written in English, including code, comments, design docs, plans, the README and reports.

**Collaboration**

- One Linear ticket maps to one branch and one small PR, with roughly 400 lines or less of non-test code.
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
| Noisy evaluations | Paired comparisons, multiple seeds and confirmation reruns |
| SWE images use too much disk | Pull only the images the subsets need, and prune in batches |

## 13. Schedule (critical path)

| Hours from start | Work |
|---|---|
| 0–1 | T01, T02 |
| 1–3 | T03, T04, T08, T09, T13 and T14 in parallel; images and database dumps download in the background |
| 3–5 | T05, T06, T10, T15 |
| 5–7 | T07, T11, T16, T18; baseline experiments start (T12, T17) |
| 7–9 | T19, T20, T21 |
| 9–17 | T22: both lanes running |
| 17–20 | T23, T24 |

## Appendix A: Ticket mapping

T01–T24 correspond, in order, to CIR-1633 through CIR-1656, so T_n = CIR-(1632+n). The dependencies here match the blocked-by relations in Linear.
