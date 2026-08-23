# Evidence bridge (Stage 2 of the real-trajectory bridge)

External-review direction (docs/gpt_review_round1.md, Stage 2): make the
exact-MDP results a predictor, not an isolated toy — "Auditor 在训练前判定
estimator 的 target/cost/mechanism 风险；这些判定能够预测等预算真实训练中的
失败、方差或收益排序". This page documents the CPU-executable first half of
that bridge: exact verdicts predicting fixed-budget MSE on controllable
tool-agent tasks.

## Three layers

1. **Exact layer** (H=4, full enumeration): for each estimator the cycle
   distribution is enumerated exactly (path x sibling draws), giving the
   bias vs the exact target, the intrinsic cycle variance, and the
   matched-budget predictor
       predicted MSE = var_cycle * cost / B + bias^2
   where B is the transition budget. This is a QUANTITATIVE prediction from
   the exact layer alone — no sampling.
2. **MC agreement gate**: an independent high-budget Monte Carlo of the same
   target (own RNG stream, own code path, never imports estimator code) must
   agree with the exact target (6-sigma gate) — validating both paths.
3. **Sampled layer**: the same estimators consume sampled TRAJECTORY RECORDS
   (aca-trajectory-record-1.0) under matched transition budgets (dense H,
   local sibling H + (H-t+1), paired-replay H + sum(H-t+1), pc-rsg
   H + E_q[H-t+1]); fixed-budget MSE measured over seeded replicates.

The estimators consume records ONLY — no environment access — so real LLM
trajectories drop into the same code path unchanged (Stage 3 harness, gated
on the Guard trajectory schema package, design §20.2).

## Frozen tasks

`worlds/tool_agent.py`: observation-dependent tool-use MDP. Behavior policy
P(a_t=1) depends on the previous tool observation (good 0.55 / bad 0.85);
tool success probabilities per task; terminal reward = phase-correct-tool
bonus + success count - length penalty. Two H=4 tasks for the exact layer +
two H=12 twins for the sampling-only scale layer.

## Estimators (same definitions as the M0 exact worlds)

- dense: g = R sum_t s_t, s_t = a_t - pi_t (unbiased, cost H)
- local_sibling: contrast at t on coordinate t only (unbiased for the local
  effect; biased for the full gradient — T003)
- paired_replay: per-coordinate coupled contrast (unbiased in the M0
  designed world)
- pc_rsg: backbone + 1/q residual correction (V001 story)

Sibling semantics: fresh draw a'_t ~ policy, shared suffix (the paired-replay
coupling). The exact and sampled layers agree on this definition; the bridge
self-validates (cycle MC == exact mean, 6-sigma) — the M0 oracle-alignment
discipline at the cycle level.

## Results (artifacts/evidence_bridge, seeded, reproducible)

- **Prediction ratios 0.87-1.07** — the exact-layer predictor reproduces the
  sampled fixed-budget MSE on every estimator-task pair at H=4, not just the
  ordering.
- **Transfer finding**: in observation-dependent worlds the coupled contrast
  misses the indirect effect of a_t through future observations/actions, so
  the designed-world unbiasedness of paired_replay/pc_rsg does NOT transfer;
  the exact layer flags both as biased (0.08/0.02 and 0.62/0.32 on the two
  tasks), while dense stays unbiased (bias ~ 1e-16). The local sibling's
  full-gradient bias (0.40/0.23) reproduces T003.
- **Scale**: at H=12 (MC target only) paired_replay wins fixed-budget MSE on
  both tasks — the cycle-variance advantage visible in the exact layer
  (var 1.49 vs dense 14.19) compounds while the fixed bias penalty does not.

## Boundaries

- This is a demonstration on controllable environments (tool class, like the
  real-training audit), not a protocol pack and not a real-training result.
- The third layer (real LLM trajectories + matched-budget training loop) is
  the Stage 3 harness; it needs the Guard trajectory schema package and GPU
  training, both outside the CPU-only Auditor scope.
- No claim about real LLM agent performance; no new credit-assignment
  algorithm.

Run: `bash scripts/run_evidence_bridge.sh` (writes artifacts/evidence_bridge/
result.json + REPORT.md + exported trajectory records).
