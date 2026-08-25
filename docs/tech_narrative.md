# Technical narrative — 10-minute interview story

This document turns the 30-second story (docs/resume_claims.md) into a
10-minute narrative with the honest designed-vs-discovered distinction baked
in. It is the script for the "core project deep-dive" section of an
interview. Every number here traces to artifacts/v0.1.2+ + commit + SHA256SUMS.

---

## Act 1 — the failure that started it (2 min)

> 我做 GRPO Agent credit assignment 时，线上 rollout policy 没有随 trainer 更新：
> trainer 的权重在变，rollout 用的还是初始化时创建的静态 client；old-logprob、
> token 与 mask 的身份也对不上。我们当时拿到 36.5%→63.5% 的 success curve，
> 但审计后发现那根本不是训练后 policy 的表现。我没有继续调大模型，而是把
> 这件事拆成两个工具：GRPO-Guard（在线保证轨迹链路可信）和 Credit Auditor
> （离线保证 estimator 的 target、成本和机制可信）。

**面试要点**：讲清"为什么先做审计而不是继续训练"——算力诚实、负结果管理、
工程判断力。这是整个故事的根基。

## Act 2 — 四个审计问题（2 min）

> Credit Auditor 不是新算法，它逼每个候选方法回答四个问题：
> 1. **Estimand**：你在估什么？完整 policy gradient、某一步的局部梯度、
>    root-marginal、还是 continuation-specific effect？——很多方法用局部
>    sibling contrast 更新共享 prefix，却从未证明它仍估计完整梯度。
> 2. **Bias**：在有限可枚举世界里，相对独立 oracle，无偏还是偏差可解释？
> 3. **Cost**：相同 transition budget 下，fixed-budget MSE 是否优于强基线？
>    基线用完整 rollout、候选却额外调用 branch oracle——成本不匹配。
> 4. **Mechanism**：正结果真的来自声称的 adaptive/local/causal 机制，还是
>    退化成固定超参数？

**面试要点**：这四问是"领域判断力"的证据——不是只会推公式，而是知道
GRPO 生态里 credit assignment 常见的混淆点（target 混淆、预算不公平、
机制退化、split 污染）。

## Act 3 — 工程实现：exact 与独立性（2 min）

> 实现上三件事保证可信：
> - **Exact worlds**：Bernoulli MDP / shared-logit MDP / continuation 家族，
>   全部精确枚举；M0 的 designed cases 用 fractions.Fraction，主实现与两个
>   oracle 的偏差**精确为 0**（不是 1e-16 容差）。
> - **独立 oracle**：enumeration 与 Bellman DP 两个自包含子进程，stdlib-only、
>   import-graph 隔离测试、monkeypatch 破坏测试——共用 bug 无法静默通过。
> - **协议优先**：protocol JSON + seed manifest 内容哈希冻结；run 之前先
>   校验 gate 名/reason code；canonical 输出 no-overwrite；每个数字追溯到
>   artifact + commit + SHA256SUMS。

**面试要点**：面试官会问"如何防止 oracle 和主实现共用 bug"——答案：
不同算法 + 独立进程 + import 隔离 + 双 oracle 交叉验证 + Fraction 精确对齐。

## Act 4 — 三个实验包的发现（3 min）

### M0 — target audit
> dense 与 uniform-HH 在冻结问题上无偏（容差内）；local sibling 对局部
> estimand 无偏、对完整梯度有偏；**propagated sibling 与 BPO-like 被拒绝**
> （T003：局部对比传播到共享 prefix 后期望为 0，不是完整梯度）。一个
> paired-replay（coupled sibling）估计器在预注册的 matched-cost 正例上赢
> 57 倍（MSE ratio 0.017），而 uncoupled 对照组输 7 倍——**正例来自配对的
> 机制，不是巧合**。

### V001 — calibration accurate ≠ utility
> PC-RSG 风格的 residual correction 校准误差 ~1e-16（"校准准确"），但
> fixed-budget MSE 比 dense 差 26.5 倍——residual noise amplification +
> branch continuation cost。**"校准准确"从来不代表效用成立**。

### D002 — 双裁决（旗舰）
> 在冻结的 48 个 held-out 问题上，calibrated mapping 以 median ratio 0.205
> （CI [0.177, 0.229]）击败 dense optimal-constant/RLOO envelope——指标通过；
> 但校准选出的 widths 全部等于 2（= global control），**adaptive
> variable-width 机制 claim 失败**（MECH001）。结论：总指标很好不能覆盖
> 机制失败；只保留窄 claim（fixed mapping 效率），adaptive claim 关闭。

**面试要点**：D002 是"审计工具的核心价值演示"——工具必须能同时输出
metric PASS 和 mechanism FAIL，而不是把失败藏起来。

## Act 5 — 诚实边界：designed vs discovered（1 min，必须主动讲）

> 我要主动区分两类正例：
> - **发现的正例**：dense/HH 无偏、propagated sibling 有偏、V001 效用失败
>   ——这些是冻结世界上的发现，不是我设计的。
> - **设计出来的正例**：M0 的 paired-replay 57 倍胜出和 D002 的 metric PASS
>   是在**预注册的 designed world** 上演示的——世界构造、种子、协议在跑之前
>   就冻结并写进 decision log。D002 的 mechanism-fail 部分是结构性的：
>   预注册的 raw-MSE 校准目标函数本来就偏好最大宽度，所以 collapse 是
>   预期的演示，不是事后发现的退化。
> 为什么这样设计：Auditor 的职责不是发现新方法，而是**演示它能否区分
> 真伪机制**。designed positive 是可控的探针；probe 有意义是因为
> uncoupled control 输了 7 倍——机制对照把"设计出来的赢"变成"机制确实
> 在工作"的证据。

**面试要点**：这一分钟是诚信分。面试官一定会问"你的正例是不是 tuned
出来的"——主动承认 designed 部分 + 展示 decision log D8/D9 的预注册证据，
把弱点变成纪律的证据。

## Act 6 — 真实链路与下一步（1 min）

> v0.1.2 之后做了真实链路连接：GRPO-Guard 签发的真实轨迹 envelope（frozen
> fixtures）现在能流过 Auditor 的 CreditAuditBundle 校验——schema 版本
> 钉在 grpo-guard-envelope-1.0（Guard 仓库自己的版本），fail-closed，
> 只按 hash 引用、不写回。下一步 v0.2 是：Guard 发布正式 schema 包后，
> 对真实训练轨迹做 estimator 级审计（不再只是 envelope 校验）。

**面试要点**：展示 roadmap 意识——工具链闭环（Guard 在线 + Auditor 离线）
是长期愿景，不是一次性 demo。

---

## Act 7 — 从 exact 到真实训练：证据桥与闭环（2 min，新）

> exact 世界再漂亮，面试官会问"这能预测真实训练吗"。我把这条路打通了：
> - **证据桥（Stage 2）**：在可控 tool-agent 世界（观察依赖的工具调用 MDP）上，
>   exact 层的预测公式 var·cost/B + bias² 定量复现了采样预算下的固定预算 MSE
>   （全部估计器-任务对比率 0.87-1.07）——exact 结论不是孤立的玩具。同时发现
>   一个 transfer finding：paired-replay 在独立坐标设计世界里的无偏性，在观察
>   依赖世界里不成立（配对对比漏掉了决策通过未来观察/动作的间接效应）——exact
>   层当场抓到，这正是"审计判断能预测失败"的证据。
> - **真实闭环（Stage 3）**：在共享 8×A800 服务器上跑了 18 次 Guard 监督的真实
>   GRPO（2 个 tool-use 任务 × 3 个 credit estimator × 3 seeds，Qwen3-4B LoRA，
>   一次一张卡、让位他人）。结果：dense/local 每 epoch 都有真实 Guard 验证的
>   更新；**paired-branch 的可靠性门在全部 9 次里选择弃权**（零 credit → 零更新）
>   ——我把这个保守门行为如实写进报告，而不是包装成成功。tau2 任务则是一个
>   诚实的负结果：base 模型产生不了有效工具调用，奖励全零，所有估计器零信号。

**面试要点**：这一段回答"exact 数学审计和真实训练有什么关系"——桥接的不是
玩具迁移论，而是同一套判断（target/cost/mechanism）在真实数据上仍然工作，
而且敢于报告门弃权与零信号这样的负结果。

## 高频追问速答（对应设计 §24.4）

1. **local vs full gradient**：local sibling 估计单步局部效应
   E[Δ·s_t] = p(1-p)(Q(1)-Q(0))；传播到 prefix 后期望变 0（Δ 条件均值 0），
   不再是完整梯度。
2. **为什么 local credit 不能传播**：Δ 在 prefix 上条件均值为 0，传播项的
   期望恒为 0 ≠ 目标分量（T003）。
3. **HH vs HT**：WR 用选择概率（Hansen-Hurwitz），WOR 用包含概率
   （Horvitz-Thompson）；混用 = S003。
4. **有偏为何能赢 MSE**：MSE = bias² + var/n；稀疏估计器方差低 + 周期便宜
   → fixed-budget MSE 可能更好（BPO-like 在 242/3000 cells 赢过——历史背景，
   新协议不复现该数）。
5. **transition budget 代表真实 GPU 成本吗**：不直接；报告同时给
   transition / model-forward / token 三口径，primary unit 预先冻结。
6. **oracle 如何避免共用 bug**：不同算法（枚举 vs DP）+ 独立进程 +
   import 隔离测试 + monkeypatch 破坏测试 + Fraction 精确对齐（==0）。
7. **[2,2,2,2] 为什么否定 adaptive**：校准选择的 widths 无 diversity 且
   等于 global control——没有 adaptive 机制存在的证据；MECH001。
8. **calibration/test 如何不可事后重选**：seed 冻结 + 内容哈希 + test 阶段
   校验 selection 自哈希（篡改即拒绝）+ no-overwrite。
9. **exact MDP 结果如何迁移到 LLM**：不迁移；v0.1 明确不外推 prevalence；
   真实 Agent 效用需要 Guard 可信 envelope + 冻结模型/token/seed 协议
   （v0.2）。
10. **为什么 10/10 gates 通过的 minimal logging 仍被 kill**：数学与实现
    正确 ≠ 新颖——它等价于经典 decision-reduct / FD / hitting-set；
    只能作为 teaching asset。

---

## 底线声明（在任何场合都主动讲）

- 所有数字来自 docs_only_semantic 模式的新冻结协议；历史数字
  （144/202、24.81×、0.694、ρ=0.735）是事故背景，不是复现。
- 没有声称真实 LLM Agent 下游收益；没有声称新 credit assignment 算法。
- 每个数字可追溯到 artifact 目录 + commit + SHA256SUMS。
