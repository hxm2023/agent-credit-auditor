# Agent-RL Credit Auditor —— 项目详细介绍

> **GitHub**: https://github.com/hxm2023/agent-credit-auditor
>
> **定位**：CPU-first 的信用估计器审计与 exact benchmark 工具（engineering
> project，非论文项目）。它不是新的 credit-assignment 算法，而是一套逼每个
> 候选方法回答四个问题的审计系统。
>
> **一句话**：在 GRPO/agent-RL 生态里，"更细粒度 credit"很容易画出漂亮的
> 学习曲线，但最常见的成果是混淆——target 没说清、预算不公平、机制没发生。
> Credit Auditor 把这些混淆变成显式 contract 和 fail-closed Gate。

---

## 1. 为什么做这件事（动机与背景）

我最初做 GRPO Agent credit assignment 时，线上 rollout policy 没有随
trainer 更新：trainer 的权重在变，rollout 用的还是初始化时创建的静态
client；token、old-logprob 与 mask 的身份也对不上。当时拿到了 36.5%→63.5%
的 success curve，但审计后发现那根本不是训练后 policy 的表现。

我没有继续调大模型，而是把这件事拆成两个工具：

| 工具 | 职责 |
|---|---|
| **GRPO-Guard**（在线） | 保证轨迹链路可信：policy/token/mask 身份、canonical hashing、envelope 生命周期 |
| **Credit Auditor**（离线，本工具） | 保证 estimator 可信：estimand、target、成本、机制——即使轨迹正确，也要审计估计器本身 |

两个项目独立建仓。旧 `grpo-credit-assignment` 仓库只作为 legacy evidence
museum（事故博物馆），不直接改名发布。

---

## 2. 核心贡献：四个审计问题

Credit Auditor 逼每个候选 credit estimator 回答：

1. **Estimand（目标）**：你在估什么？完整 policy gradient、某一步的局部
   梯度、root-marginal、还是 continuation-specific effect？很多方法用局部
   sibling contrast 更新共享 prefix，却从未证明它仍估计完整梯度。
2. **Bias（偏差）**：在有限可枚举世界里，相对独立 oracle，是无偏还是偏差
   可解释？oracle 与主实现**零共享代码**、不同算法、独立进程。
3. **Cost（成本）**：在相同 transition/token/intervention budget 下，
   fixed-budget MSE 是否优于强基线？基线用完整 rollout、候选却额外调用
   branch oracle——成本不匹配是最常见的隐性作弊。
4. **Mechanism（机制）**：正结果真的来自声称的 adaptive/local/causal
   机制，还是退化成固定超参数？（D002 的 adaptive claim 被自己的校准结果
   否决——widths 全部退化为 global control）

每个实验只允许一个 primary estimand；secondary 必须另列，不能看到结果后
切换 headline。

---

## 3. 技术架构

```
configs/           冻结协议 + seed manifests（预注册，内容哈希）
src/credit_auditor/
  schema.py        EstimandSpec/SamplingSpec/CostSpec/EstimatorSpec/ClaimDecision
  canonical.py     确定性 JSON + SHA-256 内容哈希
  runner.py        protocol-first 流水线（14 步固定顺序）、no-overwrite、原子发布
  worlds/          精确有限 MDP/SCM 枚举器（Bernoulli、shared-logit、continuation、telemetry）
  estimands/       形式化 target（独立于 estimator）
  estimators/      dense / sibling / HH-HT / BPO-like / PC-RSG / branching
  oracles/         自包含 Bellman + enumeration oracle（独立子进程、stdlib-only）
  audit/           T/S/C/U/M/D/E/N 八类 Gate + A1-A14 故障注入矩阵
  adapters/        GRPO-Guard envelope 校验（fail-closed）+ legacy bundle 校验器
  experiments/     m0 / v001 / d002 / continuation / census / self-audit 驱动
tests/             数学单元 / oracle 独立性 / 协议证据 / 故障注入 / 冻结回归
scripts/           一键复现（reproduce_all.sh）+ 各实验包 + smoke + 服务器脚本
artifacts/         规范运行输出（result/manifest/report，no-overwrite + SHA256SUMS）
```

### 关键工程特性

- **协议优先**：protocol JSON + seed manifest 在运行前冻结并内容哈希；任何
  修改都是 decision-logged 的版本升级，不是原地编辑。`validate-protocol`
  拒绝未知 gate 名、reason code、claim gate——冻结配置里没有静默笔误。
- **证据链**：每个数字追溯到 artifact 目录 + git commit + SHA256SUMS；
  manifest 记录 source hash、dirty 标志、parent selection hash、raw rows
  引用。
- **双 oracle 独立性**：enumeration（朴素路径枚举）与 Bellman（值函数 DP）
  两个自包含子进程，stdlib-only、AST 级 import 隔离、monkeypatch 破坏测试；
  12 个冻结问题上三路对齐**精确到 ==0**（Fraction 精确算术，无浮点舍入）。
- **no-overwrite**：canonical 输出拒绝覆盖；篡改的 frozen selection（自哈希
  校验）被拒绝；A8 阻止 test-time reselection。
- **工程门禁（CI）**：ruff（E/F/W/I + format）、pyright（src 严格，0 错误）、
  pytest-cov ≥85%（当前 94.5%）、smoke/full 测试分层、fresh-clone 复现步骤。

---

## 4. 关键成果（全部数字可追溯到 artifacts + commit + SHA256SUMS）

### 4.1 三个主实验包

| 包 | 结论 | 意义 |
|---|---|---|
| **M0 target audit** | dense/uniform-HH 无偏（容差内）；local sibling 对局部 estimand 无偏、对完整梯度有偏；**propagated sibling 与 BPO-like 被拒绝**（T003）；paired-replay 在预注册 matched-cost 正例上赢 57×（MSE ratio 0.017），uncoupled 对照组输 7× | 审计器既批准真实正例，又拒绝错误 target 与传播 |
| **V001 utility failure** | PC-RSG 式 residual correction 校准误差 ~1e-16（"校准准确"）但 fixed-budget MSE 比 dense 差 **26.5×** | "校准准确"从不等同于效用成立；成本会计是关键 |
| **D002 dual verdict** | calibrated mapping 以 median ratio **0.205**（bootstrap CI [0.177, 0.229]）击败 dense optimal-constant/RLOO envelope——指标通过；但校准选出的 widths 全部等于 global control **（[2,2,2,2]）**，adaptive variable-width claim 失败（MECH001） | 总指标很好不能覆盖机制失败；只保留窄 claim |

### 4.2 深化与自审计

| 项目 | 数字 | 意义 |
|---|---|---|
| **Fraction 精确交叉验证** | 主实现 vs 双 oracle **mismatch == 0**（12 问题） | "exact benchmark"字面成立（H≤6 无浮点舍入） |
| **Predefined Fault Mutation Regression Suite** | 13 类冻结故障模板全部触发预期 reason code（回归 TPR=1.0，9 类 N=200、4 类 runner 型 N=30），无对照误报（FPR=0.0，Wilson CI） | 预定义模板的软件回归检测；模板与预期 code 同源构造，故不称通用 TPR/FPR（对外部未知故障不声明）；开发中真实抓出 3 个对照组构造 bug |
| **机制理论** | paired-replay 方差闭式、K-sample prefix floor、HH 1/q 放大——全部数值 ==0 diff 验证 | 不仅测出结论，还知道为什么是结构性的 |
| **CTRI 大规模普查** | Fraction 精确 sign/rank 稳定性：N=5k → 100k → **10M（autodl2 服务器，48 CPU worker，88s）**，reversal 率收敛于 **3.2744% ± 0.006pp** | 三档规模速率稳定，服务器算力利用 |
| **collapse 统计检验** | D002 [2,2,2,2] 落在候选宽度零分布的 0 分位（p ≤ 0.05） | MECH001 从硬编码谓词升级为假设检验 |

### 4.3 真实场景使用（"这个项目真的用起来过吗"——是）

- **Qwen3-4B GRPO manifest-level 集成检查**（autodl2，GRPO-Guard 监督的
  smoke run，1 个 optimizer step）：检查到 398 个 TRL 同步调用均带 ack
  （static_rollout 离线信号 CLEAR），manifest 含 5 个权重分片的 hash
  metadata（17.6GB）。**边界**：尚未独立重哈希权重 bytes，也未完成
  estimator-level 真实轨迹审计——这是 manifest-level integration smoke，
  不是真实 credit assignment audit。`artifacts/v0.1.6/real_training_audit/`
- **GRPO-Guard envelope 集成**（§25）：真实 Guard 轨迹 envelope 通过
  CreditAuditBundle 校验（ALLOW，hash-only 引用，fail-closed 钉在
  `grpo-guard-envelope-1.0`，无写回）
- **在线/离线故障映射**：Guard 在线故障（static_rollout、mask_shift、
  misbound_logprob、retokenization、f5-f8）→ 离线可检测信号 → Auditor gate
  的完整对应表；真实场景故障注入演示（f5/f7/f8 全部检出）

---

## 5. 工程实践与交付状态

- **190 个 CPU 测试**（smoke ~85s 快速层；完整套件为发布门禁），覆盖率 94.5%
- **GitHub CI 双 job 全绿**：quality（ruff/format/pyright/依赖审计）+
  test（协议校验 → smoke → fresh-clone M0 复现 → 完整套件 + 覆盖率门禁）
- **GitHub Release v0.1.6 + tag**（v0.1.6 为外部评审 P0 修复版：LF SHA256SUMS
  证据链、严格 provenance、CI release gate、干净树重新生成全部九包）；`uv build`
  sdist/wheel 已验证
- **Dockerfile**（CPU-only 发布镜像）、`.python-version`、`SECURITY.md`
- **一键复现**：`bash scripts/reproduce_all.sh artifacts/v0.1.x` 从干净克隆
  重建全部实验包与发布报告（fresh-clone 验证通过）
- **版本历史**：v0.1.0-v0.1.6，每版干净工作树发布，manifest dirty=false

---

## 6. 诚信与边界（项目的立身之本）

- **模式**：`docs_only_semantic`——旧仓库代码不在本机，所有历史数字
  （144/202、24.81×、0.694、192/192、ρ=0.735）是事故背景，**从未作为复现
  结果出现**。升级到 `legacy_exact` 需要带外锚定的签名迁移 bundle
  （`validate-legacy-bundle` 已就绪）。
- **designed vs discovered**：M0 的 paired-replay 正例与 D002 的 metric PASS
  是在**预注册的 designed world** 上演示的（decision log D8/D9）；发现的
  正例（dense/HH 无偏、V001 失败、propagated 有偏）是冻结世界上的真实发现。
  这个区分写进了面试叙事（docs/tech_narrative.md）。
- **Claims 政策**（§23）：允许——finite-MDP 无偏性、matched-budget 比较、
  dual verdict 演示、窄域 synthetic 效率 claim；禁止——"提出新算法"、
  "真实 LLM 收益"、"K=8 证明 adaptive 有效"、任何旧数字。
- **已知非目标**：不证明真实 LLM 下游收益；不外推 prevalence；v0.1 不连
  接 GPU trainer。v0.2 的真实轨迹 estimator 级审计受 GRPO-Guard schema
  发布进度约束（§20.2 门）。

---

## 7. 快速上手

```bash
git clone https://github.com/hxm2023/agent-credit-auditor.git
cd agent-credit-auditor
uv sync --extra dev --frozen

# 一键复现全部九个实验包 + 发布报告
bash scripts/reproduce_all.sh artifacts/v0.1.6

# 或单独跑
bash scripts/run_m0.sh        # target audit
bash scripts/run_v001.sh      # 预期效用失败
bash scripts/run_d002.sh      # 校准 + 冻结 test（双裁决）
uv run credit-auditor audit --artifact-dir artifacts/local/M0
```

---

## 8. 面试/求职定位

- **30 秒**：GRPO credit assignment 的线上审计失败 → 停止调模型 → 建两个
  工具（Guard 在线 / Auditor 离线）→ Auditor 把 estimator 的 target、成本、
  机制变成可复现的 CPU 检查。
- **10 分钟**：见 `docs/tech_narrative.md`（含 designed-vs-discovered 边界
  与 §24.4 高频追问速答）。
- **深度**：`docs/mechanism_theory.md`（闭式推导）、
  `docs/real_training_audit_report.md`（真实训练审计）、
  `docs/online_offline_fault_map.md`（系统级故障映射）。
- **工程信号**：CI 双 job 全绿、94.5% 覆盖率、ruff/pyright 门禁、
  Dockerfile、GitHub Release、no-overwrite 证据纪律。

---

*文档版本：2026-08-23 · 项目 v0.1.6 · 所有数字可追溯至
https://github.com/hxm2023/agent-credit-auditor 的 artifacts + commit +
SHA256SUMS。*
