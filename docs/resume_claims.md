# Resume claims (design §23.1 — docs_only_semantic variant)

Only these claims may appear in public materials. Everything traces to
artifacts/v0.1.1 + commit + SHA256SUMS.

## Allowed bullets

> **Agent-RL Credit Auditor：GRPO 信用估计器审计与 Exact Benchmark｜Python、NumPy**

- Built a finite-MDP/SCM exact benchmark with import-isolated Bellman and
  enumeration oracles (separate subprocesses, stdlib-only code, three-way
  agreement verified to 1e-16) that audits branching credit estimators under
  explicit estimand, sampling-law, and matched-transition-budget contracts,
  with target / support / cost / split / mechanism reason-coded gates
  (T001-T005, S001-S004, C001-C003, U001-U002, MECH001-MECH002, D001-D003,
  E001-E003, N001, P001).
- In docs-only semantic mode with new frozen protocols/seeds (decision log
  D1-D9): reproduced the failure TYPES of the legacy routes —
  (a) M0: local-to-prefix propagation and BPO-like selection bias are rejected
  (T003) while dense/uniform-HH stay unbiased within 1e-9; a paired-replay
  branching estimator wins the pre-registered matched-budget positive case
  (bias < 1e-16, MSE ratio 0.017 vs dense on the frozen focal world; the
  uncoupled control loses 7.0x vs dense — the win is genuinely
  mechanism-driven; both ratios trace to artifacts/v0.1.1/M0);
  (b) V001: the PC-RSG-style residual correction is calibration-accurate
  (expectation error ~1e-16) yet fixed-budget MSE fails ~26.5x vs dense —
  "calibration accurate" does not imply utility;
  (c) D002: the calibrated global-K mapping beats the strong dense envelope on
  48 frozen held-out problems (median MSE ratio 0.205, bootstrap CI
  [0.177, 0.229]) WHILE the adaptive variable-width mechanism claim fails
  (selected widths collapse to [2,2,2,2] = the global control, MECH001) — a
  metric-pass/mechanism-fail dual verdict.
- Turned the old project's failures (static rollout, zero-support sampling,
  unmatched cost, test-time reselection, width collapse) into a frozen
  regression package: >=100 CPU tests, no-overwrite evidence bundles with
  run_manifest + SHA256SUMS, fresh-clone one-command reproduction.
- Not claimed: real LLM-agent downstream utility, new credit-assignment
  algorithms, or reproduction of the legacy numbers (144/202, 24.81x, 0.694,
  192/192, rho=0.735) — all explicitly excluded.

## Forbidden (never write, §23.2)

- "proposed a new Agent credit assignment algorithm"
- "improved performance on real LLM agents"
- "global K=8 proves variable-width adaptive works"
- "the 56 legacy tests constitute an open-source release"
- "CTRI is new partial-identification theory"
- "minimal logging is a new minimal-sensing theorem"
- "exact finite-MDP results represent real task distributions"

## 30-second story (面试 30 秒版, design §24.1)

> 我最初做 GRPO Agent credit assignment，但审计发现线上 rollout policy 没有随
> trainer 更新，token、old-logprob 和 mask 身份也不闭合，所以我停止使用旧成功率
> 结论。随后我没有继续调大模型，而是先做 Credit Auditor：对每个 estimator 显式
> 定义 estimand、sampling 和成本，用独立 exact oracle 检查 bias/MSE，再用机制
> 对照验证正结果来自声称机制。最终一个方法虽然 MSE 很漂亮，但 mapping 退化成
> global K=8，我主动把 adaptive claim 关掉，只保留窄结果。
