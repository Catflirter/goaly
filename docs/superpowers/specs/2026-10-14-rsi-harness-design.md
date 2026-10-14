# RSI Self-Evolving Harness 设计文档

- 日期：2026-10-14
- 状态：待审阅
- Linear：项目 P-CIR-12，本文档对应 T01 = CIR-1633
- 截止：2026-10-15

## 1. 背景与目标

### 1.1 任务

我们要做一个能自我进化的 harness 系统：

- **改进模型** GLM-5.3 驱动 RSI agent。它不断创建和改进执行模型的 harness，包括工具、技能、system prompt、agent 循环和上下文管理。
- **执行模型** Qwen3.6-35B-A3B 用这套 harness 去解题，目标是在 SWE-fficiency 和 BIRD-CRITIC 上都超过人工设计的 baseline。

最终交付冻结后的 harness。主办方只会用执行模型，在隐藏的 held-out 集上评测；开发期结束后，改进模型就不能再用了。

### 1.2 交付物

1. **冻结的 harness**：代码、配置、prompt、技能和依赖版本。用一条命令就能在官方格式的任务上运行。
2. **RSI 系统**：代码、实验账本和 lessons，能自动完成多轮改进。
3. **报告**，包括：
   - baseline 与最终版本的分数；
   - 同一测试跑 3 次的分数波动；
   - 在 held-out 上的泛化表现；
   - 效率，即时间、token 和成本（实验成本单独列）；
   - 有实验证据支撑的 lessons。

### 1.3 约束

| 项 | 值 |
|---|---|
| 时间 | 1 天，截止 2026-10-15 |
| 模型网关 | `https://api.goaly.ai/inference/v1`，底层是 LiteLLM 1.92 + SGLang，接口兼容 OpenAI；密钥放在 `GOALY_API_KEY` |
| 执行模型 | `qwen3.6-35b-a3b`：FP8，35B 总参数、3B 激活；原生上下文 262K，网关用 YaRN 扩到 1.01M；每百万 token 输入 $0.15、输出 $1.00 |
| 改进模型 | `glm-5.3`：约 744B 总参数、40B 激活；思考模式不能关；上下文 1M；每百万 token 输入 $1.40、输出 $4.40 |
| 预算 | API 视为不限量，但效率仍然是评分项 |
| 机器 | 12 核 24 线程，125GB 内存，约 800GB 磁盘，没有 GPU |
| Docker | 能用，但只能用 `--network=host`（嵌套容器里 runc 1.3.6 的 procfs 限制） |

### 1.4 不做的事

- 不训练、不微调模型。
- BIRD-CRITIC 的 MySQL / SQL Server / Oracle 方言：接口留好，但不实现。
- 不评测 SWE-fficiency 全部 498 题，算力不够。

## 2. 验收标准映射

| PDF 验收项 | 设计中怎么落实 | 证据 |
|---|---|---|
| Benchmark score：给出起点和终点分数，每个 benchmark 都要提升 | 在同一个 held-out 上对比 baseline v0（§5.4）和 final | T12、T17、T23 的报告 |
| Robustness：同一测试至少跑 3 次并报告波动；工具出错、超时、坏输出都不能让系统崩溃 | held-out 跑 3 次；core 层容错（§5.1）；单题异常隔离（§6） | T23 报告；T04、T05 的测试 |
| Generalization：held-out 不参与开发；跨 benchmark 都要提升；不硬编码答案 | 三段式数据切分（§7）；两条车道各自提升；防硬编码扫描（§8.5） | 入库的切分 ID 列表；扫描日志 |
| Lessons learned：记录试过什么、什么有效、什么无效、为什么，并有实验支撑 | 实验账本 + lessons（§8.6） | `experiments.jsonl`、`lessons.md` |
| Efficiency：时间、token、计算量、成本要和分数一起报告 | 每次调用都记录成本和 token（§9）；成本护栏（§8.3） | 报告中的表格 |
| Reproducibility：代码、配置、prompt、工具版本齐全，一条命令复现 | uv 锁定依赖、run manifest、`make final`（§10） | T23、T24 |
| Automation：能自动完成多轮改进 | `goaly rsi run`（§8） | 账本中的多轮记录 |
| Regression check：保留最优版本，新改动不能破坏已经做对的题 | best tag + 回归规则（§8.3） | 决策日志 |

## 3. 总体架构

### 3.1 分层

```
goaly.rsi       开发期运行，使用 GLM-5.3
   │ 依赖
goaly.bench     数据、环境、评分、runner、指标
   │ 依赖
goaly.harness   交付物，运行时只使用 Qwen
   ├─ core/     基础设施（RSI 不可改）
   └─ packs/    每个 benchmark 的能力包（RSI 唯一可以改的地方）
```

依赖方向由 import-linter 在 CI 中强制检查：`harness` 不能导入 `bench` 和 `rsi`，`bench` 不能导入 `rsi`。

### 3.2 目录结构

```
src/goaly/
  harness/
    core/            llm.py, tools.py, agent.py, policy.py, context.py, trajectory.py, config.py
    packs/
      birdcritic/    prompt.md, skills/, tools.py, policy.py, config.yaml
      swefficiency/  （同上）
    cli.py
  bench/
    base.py          Benchmark / Environment 协议
    birdcritic/      data.py, env.py, grader.py, adapter.py
    swefficiency/    data.py, env.py, grader.py, adapter.py
    runner.py, metrics.py, manifest.py
    splits/          各 split 的 ID 列表
  rsi/
    ledger.py, workspace.py, guards.py, analyst.py, implementer.py, decide.py, lane.py, cli.py
tests/               结构与 src 对应；需要 docker、数据库或真实 API 的测试打 integration 标记
docs/superpowers/{specs,plans}/   设计文档与实施计划
docs/results/                     实验结果
runs/                             运行产物，不入库
```

### 3.3 关键接口

下面是草案，最终以实现 PR 为准。

```python
class Environment(Protocol):
    """一道题的运行环境，例如数据库沙箱或容器。"""
    async def close(self) -> None: ...

class Benchmark(Protocol):
    name: str
    def load_instances(self, split: Split) -> Sequence[Instance]: ...
    def provision(self, inst: Instance) -> AbstractAsyncContextManager[Environment]: ...
    def agent_input(self, inst: Instance) -> TaskInput: ...
    async def grade(self, inst: Instance, prediction: Prediction) -> Grade: ...
    def to_official(self, inst: Instance, prediction: Prediction) -> dict[str, Any]: ...

class Policy(Protocol):
    """由能力包实现，负责编排一道题的求解过程。"""
    async def solve(self, task: TaskInput, env: Environment, ctx: AgentContext) -> Prediction: ...

class Tool(Protocol):
    name: str
    description: str
    args_model: type[BaseModel]
    async def run(self, args: BaseModel, env: Environment) -> ToolResult: ...
```

要新增一个 benchmark，只需实现一个 `Benchmark` 适配器和一个能力包，不用改 core。

## 4. 两个 benchmark

### 4.1 BIRD-CRITIC（修复有问题的 SQL）

**数据**
- 使用 `birdsql/bird-critic-1.0-postgresql`，共 530 题，PostgreSQL 方言，固定 HF revision。
- 公开字段：instance_id、db_id、query（用户的问题）、issue_sql、preprocess_sql、clean_up_sql、category，以及每题的 schema 文本。
- 标准答案 `sol_sql` 和测试用例 `test_cases` 不公开，需要发邮件申请：收件人 bird.bench25@gmail.com，标题里带上 `[bird-critic-1 GT&Test Cases]`。GT 文件不入库。

**计分**
- 成功率 = 满足以下全部条件的题目占比：执行不报错、不超时、通过该题全部测试用例。
- 题目分四类，各自的判定方式：

| 类别 | 判定方式 |
|---|---|
| Query | `ex_base`，比较结果集 |
| Personalization | 额外的约束检查 |
| Management | 检查执行后的数据库状态 |
| Efficiency | EXPLAIN 代价要低于 issue_sql |

**评分细节**（必须和官方完全一致）
- 结果按"行的集合"比较：不管行序和重复行，但列序必须对。
- 日期截断到天；浮点数精确比较。
- 空结果直接判错。
- 只看最后一条语句的结果。
- 预测的 SQL 会被执行两次。
- 评分超时：单条语句 60s，单题 300s。结果最多 10,000 行。这里的超时只管评分；agent 求解的预算另算（见 §5.4）。

**环境**
- PostgreSQL 14.12 + PostGIS，沿用官方的 compose 配置，用 host 网络，时区和官方一致。
- 官方 dump 先恢复成模板库。每道题用 `CREATE DATABASE … TEMPLATE` 克隆一个独立副本，执行 preprocess_sql，用完就删。
- agent 求解和评分各用一个副本，互不影响。

**评分器实现**
- vendored 官方的 test_case helper（MIT 许可，保留署名），放在我们自己的模板克隆沙箱上跑，这样可以高并发。
- 怎么证明和官方一致：
  - gold 模式下，GT 应该全部通过；官方已知有错的题单独列出来。
  - 抽样题目，和官方 harness 的结果逐一比对。

**参考成绩（PG-530）**：~30B 的开源模型 28–32%，Claude Opus 4.6 43.4%，榜首 56.8%，人类 78.9%。

### 4.2 SWE-fficiency（让仓库代码跑得更快）

**数据**
- 使用 `swefficiency/swefficiency`，共 498 题，来自 9 个仓库，固定 HF revision（数据集在 2026-03-30 修订过）。
- `patch`、`problem_statement`、`hints_text` 三个字段对 agent 隐藏。

**任务**
- 给 agent 的是：容器里 `/testbed` 下的仓库，加一个 workload 脚本。
- agent 要改代码让 workload 变快，最后输出 git diff。
- 不准修改 workload，不准修改测试，不准安装依赖。

**计分**
- 单题分数：SR_i = (T_pre / T_post_agent) / (T_pre / T_post_gold)，也就是 agent 的加速比除以专家补丁的加速比。
- 总分取各题 SR 的调和平均，单题 SR 下限 0.001。
- 几种特殊情况：
  - 补丁打不上，或者任意一个 PASS_TO_PASS 测试失败：按"没加速"算，SR = 1/gold 加速比，和交空补丁一样。
  - 补丁正确但让代码变慢：照实计分，不截断，所以分数会比交空补丁还低。
- 参考：全集都交空补丁约得 0.024；Claude 4.5 Opus 是 0.225。

**这套计分对策略意味着什么**
1. 只有实测加速达到阈值、相关测试也通过时才提交补丁，否则提交空补丁。
2. 调和平均相当于把各题的 1/SR 加起来。交空补丁时，某题这一项就等于专家的加速倍数。所以在专家加速很大的题上哪怕只做出一部分进展，收益也最大；反过来，任何一题变慢都会被放大。

**环境**
- 每题一个官方镜像，linux/amd64，压缩后 1.4–3.6GB。
- 容器用 `--network=host`，cpuset 绑核，限制内存。

**评分器**
- 封装官方的 `swefficiency eval/report`，做以下适配：
  - 改成 host 网络。
  - CPU 分配适配本机：最多 4 个评测 worker，每个 2 个物理核。
- 在本机重跑一遍 gold，得到 T_post_gold。
- 空补丁不送评测，直接计分。
- 缓存：评测结果按 (patch 哈希, 题目) 缓存；T_pre 和 gold 的计时也缓存。

**资源约束**：评测是整个项目最大的瓶颈。agent 求解和评测使用不同的 CPU 槽位，避免互相干扰计时。

**已知的失败模式**
- 没找对热点：68% 的专家收益来自模型根本没碰过的函数。
- 见好就收：拿到一点加速就停手。
- 投机取巧：缓存 workload 的结果，或者针对 workload 写特判。

## 5. Harness 设计

### 5.1 core（RSI 不可改）

**LLM 客户端**
- 基于 httpx 的异步客户端，兼容 OpenAI 接口。
- 单次调用超时 300s。遇到 429、5xx、超时、连接错误时按指数退避重试，最多 5 次。
- 解析 usage（包括 reasoning tokens）和响应头里的 `x-litellm-response-cost`。
- 每次调用写一条 JSONL 记录，带上用途标签。
- 上下文超限等错误映射成具体的异常类型。

**模型白名单**：harness 运行时的客户端只能调用配置里的执行模型（`qwen3.6-35b-a3b`），调别的模型直接报错。

**模型特性适配**
- Qwen：
  - 用 `chat_template_kwargs.enable_thinking` 开关思考，可选开 `preserve_thinking`。
  - 网关不接受 `reasoning_effort` 和顶层的 `enable_thinking`。
  - 已知问题一：tool call 偶尔会写进 `reasoning_content`。解析器要兜底解析 Qwen3-Coder 的 XML 格式（`<tool_call><function=…><parameter=…>`）。
  - 已知问题二：参数 JSON 可能被截断或格式不对。能修就修，修不了就把错误回传给模型。
- GLM（只在 RSI 里用）：
  - 思考模式一直开着；网关会拒绝 `thinking` 参数。
  - 历史消息里要原样回传 `reasoning_content`。

**工具执行器**
- 用 pydantic 校验参数，带超时。
- 输出太长时保留头尾各 4K 字符。
- 工具抛异常时，转成结构化的错误消息返回给模型。

**Agent 循环**
- 预算：步数、墙钟时间、token、成本都有上限。
- 连续出错：3 次后注入工具 schema 提醒，5 次后终止。
- 重复调用：同一个工具、同样的参数连续调用 3 次，就注入提示。
- 上下文管理：Qwen 的上下文预算是 128K。超过 96K 时，把较早的工具输出折叠成摘要；系统提示、任务描述和最近 8 轮保持原样。
- 预算用完时，由 policy 的兜底逻辑提交结果。
- 任何异常都不会让整批任务崩掉。

**轨迹**：写成带版本号（schema_version）的 JSONL，每一步记录消息、工具调用及结果、用量和耗时，最后记录最终预测和评分结果。

### 5.2 能力包（RSI 唯一可以改的地方）

| 文件 | 作用 |
|---|---|
| `prompt.md` | system prompt 模板 |
| `skills/*.md` | 精选的技能手册，建议不超过 3 份，内容聚焦，可以删 |
| `tools.py` | 这个 benchmark 专用的工具，基于 core 的 Tool 协议 |
| `policy.py` | 求解编排：分哪些阶段、怎么自检、何时注入引导消息、best-of-n、兜底提交 |
| `config.yaml` | 采样参数、思考开关、预算、上下文阈值 |

### 5.3 设计原则（来自已有工作）

- 自检只看确定性的执行结果，比如跑测试、计时、执行 SQL。不让执行模型自己给自己打分。
- 引导信息尽量放在工具返回和运行中的提示里，而不是一股脑塞进 system prompt。
- 技能手册少而精。没起作用的工具和规则可以删掉。

### 5.4 baseline v0（PDF 要求的"简单工具"版本）

- BIRD：两个工具，`execute_sql`（在本题沙箱里执行，返回截断后的结果或错误）和 `submit`；prompt 尽量简短。
- SWE：两个工具，`bash`（在本题容器里执行）和 `submit`；prompt 尽量简短，但写明官方的约束。
- 两边共用的设置：
  - 开启思考；采样参数 T=0.6、top_p=0.95、top_k=20；单次 max_tokens 16K。
  - 求解预算：BIRD 30 步 / 10 分钟，SWE 60 步 / 45 分钟。
- v0 冻结后打 tag `baseline-v0`，后面所有对比都以它为起点。

## 6. 评测框架（goaly.bench）

**runner**
- 基于 asyncio，按资源池限流：LLM 并发数、数据库槽位、SWE 求解用的 CPU 槽位、SWE 评测 worker（不超过 4 个）。
- 单题出异常不影响其他题；每题结果落盘，中断后可以续跑。
- 评测缓存的 key 是 (harness 内容哈希, 题目, seed)。

**run manifest**：记录 git commit、harness 内容哈希、配置哈希、数据集 revision、模型名和采样参数、开始和结束时间、机器信息。

**结果记录**：每题记录分数、是否正确、耗时、tokens（输入、输出、思考分开）、成本、步数、错误类型。

**指标**
- 主指标：BIRD 用成功率，SWE 用 SR 的调和平均。
- 多 seed 时给出均值 ± 标准差，以及 bootstrap 95% CI。
- 逐题配对比较：数赢和输的题数，做单侧符号检验；SWE 用配对 bootstrap。
- "稳定做对集合"：每个 seed 都做对的题。
- 按类别、按仓库分解。

**CLI**
- `goaly run --bench {birdcritic,swefficiency} --split {train,val,heldout} --harness <path|tag> --seeds N --out runs/<id>`
- 也可以用 `--instances <官方格式 jsonl>` 直接传任务文件，供主办方跑他们的 held-out。这时只需要执行模型的 API 和对应环境（BIRD 要数据库，SWE 要 Docker）。

## 7. 数据切分

| | train | val | held-out |
|---|---|---|---|
| BIRD（PG-530，按类别分层） | 150 | 100 | 280 |
| SWE（从 Lite 的 100 题中选，按仓库分层） | 12 | 12 | 12 |

- 随机种子固定为 `20261014`，切分出的 ID 列表提交到 `src/goaly/bench/splits/`。
- 三份数据的用法：
  - **train**：分析器可以看完整轨迹、grader 反馈和标准答案。
  - **val**：结果只交给决策模块，分析器看不到。
  - **held-out**：开发期间不用任何改动过的版本去跑。唯一的例外是 v0，它的 held-out 结果用于最终对比。
- SWE 的题如果单题评测超过 25 分钟，就换成同一仓库的另一题，并记录下来。

## 8. RSI 系统（goaly.rsi）

### 8.1 车道

- BIRD 和 SWE 两条车道并行运行。
- 每条车道维护自己的 best（git tag `best/bird`、`best/swe`），只能改自己的能力包。
- RSI 运行期间，core 和 bench 都冻结。

### 8.2 一轮的流程

1. **分析**（GLM，用途标签 `rsi-analyst`）
   - 输入：
     - 账本摘要，包括被拒绝过的尝试；
     - 当前 best 在 train 上的逐题结果；
     - 失败轨迹的摘要，带 grader 反馈和标准答案；
     - 该 benchmark 的计分规则说明。
   - 输出结构化 JSON，内容是失败模式的归类，以及 3 个假设。每个假设写清：
     - 针对哪种失败模式；
     - 改动类型：prompt / skill / tool / policy / config / 删除；
     - 具体怎么改；
     - 预期效果和风险。
2. **实现**（GLM 编码 agent，用途标签 `rsi-impl`）
   - 取排名前 k 的假设（BIRD k=3，SWE k=2），各开一个 git worktree 并行实现。
   - 权限：
     - 整个仓库只读；
     - 只能写 `src/goaly/harness/packs/<lane>/` 和 `tests/harness/packs/<lane>/`；
     - 可以跑能力包的单测；
     - 可以拿 1–2 道题做冒烟测试。
   - 每个假设最终产出一个 commit。
3. **守卫**：任何一项不过就直接拒绝。
   - diff 只能落在允许的路径里；
   - `bench/` 和 `core/` 的 checksum 没变；
   - 防硬编码扫描通过；
   - 能力包单测通过。
4. **筛选**：拿 best 在相同题目上的缓存结果做对比，没有净改善就拒绝。
   - BIRD：在"该假设针对的 train 失败题（最多 15 道）+ 5 道 train 已做对的题"上跑 1 个 seed。
   - SWE：在该假设针对的 3 道 train 题上跑。
5. **验证**：BIRD 在 val 的 100 题上跑 2 个 seed；SWE 在 val 的 12 题上跑 1 个 seed。
6. **决策**：规则见 §8.3。如果有多个候选过线，接受 Δ 最大的那个；其他过线的假设放回队列，下一轮基于新的 best 重新实现。
7. **复跑确认**（只对 BIRD）：被接受的候选在 val 上再跑 1 个 seed。3 个 seed 合并后 Δ 仍然 ≥ 2pp，才更新 best。
8. **记录**：所有候选都写进账本。GLM 为每个候选写一条 lesson，说明做了什么、结果怎样、为什么，并引用实验 ID。

### 8.3 决策规则

**BIRD**：每题的解题率取各 seed 的平均。以下四条都满足才接受：
- 净收益：Δ = 候选平均成功率 − best 平均成功率 ≥ 3pp。
- 显著性：赢的题数（候选解题率高于 best）多于输的题数，且单侧符号检验 p ≤ 0.1。
- 回归：best 在所有 seed 都做对、候选在所有 seed 都做错的题，最多 1 道。
- 成本护栏：如果候选平均每题的 token 或耗时超过 best 的 1.5 倍，Δ 的门槛翻倍。

**SWE**：以下三条都满足才接受：
- 收益：val 上 SR 调和平均的相对提升 ≥ 10%，且配对 bootstrap P(Δ>0) ≥ 0.8。
- 回归：best 上有真实加速（SR > 1/gold）的题，在候选上跌到 SR ≤ 1/gold 的，最多 1 道。
- 成本护栏：同 BIRD。

所有阈值都写在配置里，每次决策都记录当时用的阈值。

### 8.4 停止条件

连续 4 轮没有接受任何改动，或者该车道的时间预算用完。

### 8.5 防作弊与安全边界

| 风险 | 防线 |
|---|---|
| 改评分代码或底座 | 实现者只能写本车道的目录；每次评测前校验 `bench/` 和 `core/` 的 checksum |
| 硬编码答案 | 扫描 diff，命中以下任一项就拒绝：任何 split 的 instance_id；任何 split 的标准 SQL 或 gold patch 片段（规范化后长度 ≥ 40 字符）；db_id 作为完整标识符出现。如果 db_id 本身是常见英文单词，只记警告不拒绝 |
| 依赖改进模型 | harness 运行时有模型白名单（在 core 里，RSI 改不了） |
| 针对 workload 写特判、跨次缓存结果 | 官方评分每次计时都在新进程里跑，并检测调用栈；prompt 和技能手册明令禁止；held-out 会暴露这类过拟合 |
| 修改测试文件 | 官方评分前会把测试文件重置 |
| 在 val 上过拟合 | 分析器看不到 val 的轨迹；held-out 只在最后用一次 |

### 8.6 账本与 lessons

- `experiments.jsonl`：每条记录包含实验 ID、车道、父版本、假设、改动类型、diff 统计、筛选和验证的指标、决策及理由、成本。
- `lessons.md`：按条目增量追加（ACE 风格），每条带"有用 / 有害"计数，由 GLM 维护，可以删。不整篇重写，避免丢信息。

## 9. 成本与效率核算

- 每次 LLM 调用记录：用途（exec / rsi-analyst / rsi-impl / rsi-lesson）、模型、tokens、成本（取自响应头）、耗时。
- 报告里分开列两部分：
  - 最终 harness 在 held-out 上的执行成本和时间；
  - RSI 实验的总成本，按车道、按用途拆开。
- 机器侧：每题的墙钟时间，以及 SWE 评测消耗的 CPU 时间。

## 10. 最终评测与复现

- **冻结**：把 core、`best/bird`、`best/swe` 合在一起，打 tag `final`。
- **运行**：
  - final 在两个 benchmark 的 held-out 上各跑 3 次，在 val 上也跑 3 次。
  - v0 的 held-out 3 次在 baseline 阶段已经跑完。
- **报告内容**：
  - 起点、终点和提升幅度（均值 ± 标准差，以及置信区间）；
  - 多次运行的分数波动；
  - 按类别、按仓库的分解；
  - 效率和实验成本；
  - 分数随轮次变化的曲线；
  - lessons，每条附实验 ID。
- **复现**：`make final` 依次完成：
  1. 准备环境（镜像和数据库）；
  2. 运行 final；
  3. 评分；
  4. 生成报告。

  依赖用 `uv.lock` 锁定，所有版本信息都记在 manifest 里。

## 11. 工程规范与开发流程

**技术栈与测试**
- Python 3.12 + uv；ruff；mypy --strict；import-linter；pre-commit；GitHub Actions。
- pytest，按 TDD 写测试。单测不访问网络；需要 docker、数据库或真实 API 的测试打 integration 标记，CI 里不跑。

**配置与日志**
- 配置用 pydantic 校验。
- 密钥只从环境变量或 `~/.config/goaly/.env` 读取。
- 日志统一用结构化 JSON。

**语言约定**：代码、注释、README 和最终报告用英文；设计文档和实施计划用中文。

**协作流程**
- 一个 Linear ticket 对应一个分支、一个小 PR，非测试代码大约不超过 400 行。
- 有依赖关系的 PR 用 stacked 方式提交。
- PR 由项目负责人审核并合并。review 意见用新 commit 修复，不 force-push。
- RSI 自动产生的改动只提交到 `rsi/*` 分支，定期汇总成"RSI 结果 PR"。

## 12. 风险与对策

| 风险 | 对策 |
|---|---|
| SWE 评测太慢 | T15 先实测耗时；必要时缩小子集；空补丁跳过评测；结果缓存；v0 的 held-out 提前在后台跑 |
| BIRD 的 GT 迟迟不到 | 先做其他环节；必要时 train 暂时改用 Six-Gym-PG 的公开数据 |
| 时间紧，关键路径约 20 小时 | BIRD 和 SWE 两条线并行开发；stacked PR，审核不阻塞开发；RSI 放在夜间跑 |
| 上游接口在 review 时被要求修改 | 接口在 T05、T06 先定下来；下游 PR 跟着 rebase |
| Qwen 的 tool call 输出异常 | core 里做兜底解析和容错（T04） |
| 评测有噪声 | 配对比较、多 seed、复跑确认 |
| SWE 镜像太占磁盘 | 只拉子集用到的镜像，按批次清理 |

## 13. 排期（关键路径）

| 时段（从开始算） | 内容 |
|---|---|
| 0–1h | T01、T02 |
| 1–3h | T03、T04、T08、T09、T13、T14 并行开发；后台下载镜像和数据库 dump |
| 3–5h | T05、T06、T10、T15 |
| 5–7h | T07、T11、T16、T18；开始跑 baseline 实验（T12、T17） |
| 7–9h | T19、T20、T21 |
| 9–17h | T22：两条车道运行 |
| 17–20h | T23、T24 |

## 附录 A：Ticket 对照

T01–T24 依次对应 CIR-1633 到 CIR-1656，即 T_n = CIR-(1632+n)。依赖关系与 Linear 中的 blocked-by 保持一致。
