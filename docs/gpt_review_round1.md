 # 结论

  可以作为“核心项目之一”，但以当前状态，不建议把它放成大模型后训练算法岗的第一核心项目。

  更合适的排序是：

  1. 最强的真实后训练算法项目：GRPO reward hacking、Agent-TTRL，或有真实模型效果的 credit assignment 工作
  2. Agent-RL Credit Auditor：证明你的实验判断、算法审计和工程能力
  3. 其他项目

  如果投的是后训练评测、训练可靠性、RL environment/evals、研究工程岗位，修完下面的 P0 问题后，它甚至可以放第一。若投华为昇腾/MindSpeed-RL、大模型后训练算法与训练系统岗位，当前还缺真实训练、NPU、分布式
  和性能优化证据。

  我审查的是 hire/project_introduction.md:1 和公开仓库 main@2d1b1ce、v0.1.5 发布物、代码、CI、artifacts。

   用法                       当前适合度
  ━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━
   后训练算法岗第一核心           5.5/10
  ─────────────────────────  ────────────
   后训练评测/可靠性核心            7/10
  ─────────────────────────  ────────────
   补工程能力的第二核心             8/10
  ─────────────────────────  ────────────
   华为昇腾后训练/计算方向        4.5/10
  ─────────────────────────  ────────────
   完成真实轨迹闭环后             8–9/10

  ## 它真正有价值的地方

  第一，故事非常好，而且有真实性。

  你不是“为了做开源项目而造框架”，而是从 36.5%→63.5% 的虚假成功曲线出发，发现 rollout policy 没更新，然后主动停掉实验，把问题拆成：

  - GRPO-Guard：轨迹和训练身份是否可信
  - Credit Auditor：estimand、偏差、成本和机制是否可信

  “发现漂亮结果不可信，于是停止调参并重建证据系统”是很强的研究判断力信号。

  第二，数学深度比一般学生工程项目强。

  项目涉及：

  - full-gradient 与 local estimand 的区分
  - score-function gradient
  - HH/HT correction
  - bias、variance、fixed-budget MSE
  - paired replay 的协方差作用
  - independent enumeration/Bellman oracle
  - mechanism claim 与 metric claim 分离

  这些内容可以支撑一次很深的技术面，而不是停留在“我调用 TRL 跑了 GRPO”。

  第三，它确实补充了你的工程画像。

  我在干净克隆上验证了：

  - uv sync --frozen 成功
  - Ruff lint/format 通过
  - uv run pyright src/ 为 0 error
  - 当前 main 的 GitHub quality job 已通过

  所以它不是纯设计文档。typed schema、protocol-first runner、独立 oracle、CI、Docker、release packaging 都有实际代码。

  第四，它很契合评测和可靠性方向。当前 OpenAI 的 Agent Post-Training Evals 岗强调 RL environments、测量可靠性和方差、持续评测系统；Anthropic Code RL 也强调
  reward/verifier、诊断模型为何没有提升和训练管线可靠性。前半部分与你的项目很匹配。OpenAI Frontier Evals & Environments
  (https://openai.com/careers/research-engineer-frontier-evals-and-environments-san-francisco/)、Anthropic Code RL (https://job-boards.greenhouse.io/anthropic/jobs/5254364008)

  ## 当前不能作为第一核心的原因

  最大的缺口不是代码量，而是“真实后训练闭环”仍然没有建立：

  当前证据
  有限合成 MDP
      ↓
  exact oracle
      ↓
  估计器审计

  岗位希望看到
  真实 LLM trajectory
      ↓
  credit estimator
      ↓
  GRPO update
      ↓
  训练效果 / 稳定性 / 成本

  当前主实验仍是 H≤6 的有限合成世界。它证明的是审计逻辑在这些世界中成立，不证明：

  - Auditor 能处理真实 agent trajectory
  - 它能判断哪个 estimator 在真实 LLM 上更好
  - 它能降低 GPU/NPU 成本
  - 它能提升任务成功率
  - 它参与过多卡训练、rollout 调度或性能优化

  Anthropic 的算法工程岗位还要求真实训练实验、端到端系统、PyTorch/分布式和性能能力；MindSpeed-RL 当前重点包括
  GRPO、训推共卡、权重重切分、动态批处理、profiler、精度分析和确定性计算。这正是当前项目缺少的后半段。MindSpeed-RL (https://gitee.com/ascend/MindSpeed-RL/blob/master/README.md?skip_mobile=true)

  ## 必须先修的 P0 问题

  这些不只是文档瑕疵，其中一些会直接反噬“审计可信”的核心叙事。

  ### 1. 发布证据链目前是坏的

  文档多处声称“所有数字可由 SHA256SUMS 追溯”，但在 main@2d1b1ce：

  sha256sum --quiet -c artifacts/v0.1.5/SHA256SUMS

  98 个条目中有 73 个 checksum mismatch。进一步执行：

  credit-auditor audit \
    --artifact-dir artifacts/v0.1.5/M0

  返回 integrity=fail，包括 REPORT.md、protocol.json、result.json、gate_decision.json、oracle_result.json 和 run_manifest.json 六项不匹配。

  这不等于结果造假，更像发布后文件被重写但 checksum 没有重新生成；然而从审计器接口看，发布物就是失败的。

  更严重的是，provenance validator (https://github.com/hxm2023/agent-credit-auditor/blob/2d1b1ce580e262b472baf4153bd15ac07addc317/src/credit_auditor/audit/provenance.py#L31-L66) 会接受一个空的
  SHA256SUMS。自审计甚至把“空 checksum 文件”当作正常对照组。

  必须修成：

  - SHA256SUMS 为空直接失败
  - 必须覆盖预期文件集合，拒绝缺项、重复项和路径逃逸
  - CI 对每个 checked-in artifact pack 执行真实校验
  - 顶层 release checksum 最后生成
  - v0.1.6 重新生成全套 artifacts，不能复制旧包
  - 添加“发布物本身必须通过 Auditor”的 CI gate

  修完前，不要在简历写“完整不可篡改证据链”。

  ### 2. “真实 Qwen3-4B 审计”被表述得过强

  真实训练审计代码 (https://github.com/hxm2023/agent-credit-auditor/blob/2d1b1ce580e262b472baf4153bd15ac07addc317/src/credit_auditor/audit/real_training.py#L25-L95) 实际做的是：

  - 统计日志中的 ack=true
  - 检查 manifest 必需字段
  - 检查 weight hash 是否为长度 64 的字符串
  - 汇总 manifest 声称的字节数

  它没有：

  - 读取并重新哈希 17.6GB 权重文件
  - 独立证明 runtime rollout 已实际加载新权重
  - 审计真实 trajectory 的 credit estimator
  - 比较真实 old-logprob、action mask、reward 与 optimizer 输入
  - 运行超过 1 个 optimizer step 的训练验证

  因此 hire/project_introduction.md:113 应改为：

  > 在一次 Qwen3-4B、1 optimizer-step 的 GRPO smoke run 上完成 manifest-level integration；检查到 398 个同步调用均带 ack，且 manifest 包含 5 个权重分片的 hash metadata。尚未独立重哈希权重 bytes，也未
  > 完成 estimator-level 真实轨迹审计。

  这仍然有价值，但属于集成 smoke，不是“真实 credit assignment audit”。

  ### 3. 自审计的 TPR/FPR 不适合作为简历 headline

  文档写“N=200/类”，但 代码 (https://github.com/hxm2023/agent-credit-auditor/blob/2d1b1ce580e262b472baf4153bd15ac07addc317/src/credit_auditor/experiments/self_audit.py#L286-L310) 中 A7/A10/A12/A13
  只有 N=30。

  更重要的是，这些 fault 和 expected reason code 由同一个测试函数构造。因此它测量的是：

  > 对 13 种预定义 mutation 的软件回归检测率

  而不是：

  > 对未知真实故障的通用 TPR/FPR

  建议改名为：

  > Predefined Fault Mutation Regression Suite：13 类冻结故障模板全部触发预期 reason code，无对照误报。

  不要把 TPR=1/FPR=0 放在简历主 bullet。

  ### 4. V001 的置信区间门有逻辑错误

  V001 定义 ratio 为 candidate/baseline。若要求“改进的置信区间下界大于 0”，等价条件应是：

  ratio_ci_hi < 1

  当前 实现 (https://github.com/hxm2023/agent-credit-auditor/blob/2d1b1ce580e262b472baf4153bd15ac07addc317/src/credit_auditor/experiments/v001.py#L157-L166) 使用 ci_lo < 1，可能让跨过 1
  的不确定结果通过。

  当前 26.5× 的失败结论不会因此改变，但作为通用 gate 必须修复并加入边界测试。

  ### 5. 版本与 provenance 不一致

  当前：

  - pyproject.toml 是 0.1.6
  - GitHub 只有 v0.1.5 release/tag
  - README 仍写“six packs”和 artifacts/v0.1.1
  - 文档却声称 v0.1.0–v0.1.6 都是干净发布
  - dependency_lock_sha 实际写的是字符串 "uv.lock"，不是文件 SHA
  - runner 的 dirty 检测忽略 untracked source 文件

  应发布一个真正的 v0.1.6，重新生成 artifacts、checksum、coverage 和 release notes；否则把 main 标成 0.1.6-dev。

  ## 升级成第一核心项目的最短路线

  不要继续加更多 CPU pack。现在需要纵向打通。

  ### 阶段一：真实 trajectory bridge，建议 2–3 周

  接入真实 GRPO/Agent-RL trajectory：

  prompt
  input_ids
  generated_tokens
  action_mask
  old_logprobs
  policy_version
  reward components
  tool calls / observations
  termination reason
  optimizer-consumed tokens/mask

  至少支持：

  - trajectory-level GRPO/RLOO
  - local/step-level credit estimator
  - paired replay或你的 credit estimator
  - Guard envelope 到 Auditor bundle 的真实转换

  关键是 Auditor 必须审计实际 optimizer 消费的数据，而不只是 manifest。

  ### 阶段二：建立 exact-to-real 证据桥

  做三层验证：

  1. 有限 MDP：exact oracle，可精确测 bias/MSE
  2. 小型可控 agent environment：高预算 Monte Carlo reference
  3. 真实 LLM tool-use：实际训练与 held-out evaluation

  最强的最终 claim 不是“我提出了新算法”，而是：

  > Auditor 在训练前判定 estimator 的 target/cost/mechanism 风险；这些判定能够预测等预算真实训练中的失败、方差或收益排序。

  这会让有限 MDP 不再是孤立玩具。

  ### 阶段三：真实闭环实验

  最低实验配置建议：

  - 2 个 agent/tool-use task
  - 3 个 estimator
  - 至少 3 seeds
  - 相同 rollout token、tool call、verifier call 和 GPU-hour 预算
  - final checkpoint，而不是挑 peak
  - 同时报 success、reward hacking、KL、长度、invalid tool call、估计器方差和吞吐

  最好真实发现并修复一个问题，例如：

  - estimator target 与共享 prefix 不一致
  - action mask 错一位
  - old-logprob 来自错误 policy version
  - audit 预判某方法虽低方差但成本后不占优

  这一个真实案例，比继续增加 CTRI 10M census 更有求职价值。

  ### 阶段四：华为/昇腾定向升级

  如果目标包括华为计算产品线，建议只做一条完整 NPU 纵向路径：

  - 接入 MindSpeed-RL 的 GRPO trajectory/logging
  - 在昇腾 NPU 上完成一次小模型训练
  - CostSpec 增加 generated tokens、update tokens、tool calls、NPU time、HBM、通信量
  - 使用 profiler/msprobe 定位一次训推或动态 batch 问题
  - 最好向 MindSpeed-RL 提交一个真实 adapter、trace schema 或诊断 PR

  这会把项目从“CPU 数学审计工具”升级为“昇腾后训练可靠性组件”。



  ## 最终建议

  对你这种已有两个 GRPO 研究项目的人，这个项目方向是对的：它补的是实验判断、可靠性和工程纪律，不需要硬改成第三篇算法论文。

  但当前仓库公开可验证的 checksum、版本和真实训练表述存在明显问题。顺序应当是：

  1. 先修发布证据链与文档过度表述
  2. 再把它作为第二核心项目
  3. 接入真实 trajectory 和等预算训练闭环
  4. 完成后再考虑升为第一核心，或用于后训练 eval/reliability 岗主打

  一句话判断：

  > 这是一个很有潜力、面试深度很强的“研究工程补位项目”，但当前还是 exact synthetic auditor，不是完整的大模型后训练算法项目；把 exact audit 的判断真正闭环到 LLM/NPU 训练结果，才会成为你的王牌项目。