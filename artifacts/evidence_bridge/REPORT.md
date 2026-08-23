# Evidence bridge (Stage 2): exact verdicts -> sampled fixed-budget MSE

- controllable tool-agent tasks (frozen specs, observation-dependent policy)
- exact layer (H=4 enumeration): bias + intrinsic cycle variance + matched-budget
  predictor p = var_cycle * cost + bias^2
- sampled layer: fixed-budget MSE (B=20000 transitions,
  R=40 replicates) of the SAME estimators over trajectory records
- MC agreement gate: independent high-budget MC vs the exact target

## tool_selection (H=4)

- exact target: 0.664510 | MC target: 0.674524 (se 8.41e-03) | agreement gate: PASS (rel diff 1.51e-02)
- Spearman(exact predictor -> sampled MSE): 0.200

| estimator | exact bias | exact var | cost | unbiased | predicted MSE | sampled MSE | ratio |
|---|---:|---:|---:|---|---:|---:|---:|
| dense | 1.11e-16 | 14.191 | 4.0 | True | 0.0028 | 0.0030 | 1.07 |
| local_sibling | 4.03e-01 | 0.549 | 7.0 | False | 0.1626 | 0.1631 | 1.00 |
| paired_replay | 8.12e-02 | 1.491 | 18.0 | False | 0.0079 | 0.0069 | 0.87 |
| pc_rsg | 6.20e-01 | 44.933 | 7.5 | False | 0.4015 | 0.4072 | 1.01 |

## evidence_chain (H=4)

- exact target: 0.326796 | MC target: 0.339282 (se 7.05e-03) | agreement gate: PASS (rel diff 3.82e-02)
- Spearman(exact predictor -> sampled MSE): 0.400

| estimator | exact bias | exact var | cost | unbiased | predicted MSE | sampled MSE | ratio |
|---|---:|---:|---:|---|---:|---:|---:|
| dense | 2.78e-16 | 9.964 | 4.0 | True | 0.0020 | 0.0020 | 0.98 |
| local_sibling | 2.35e-01 | 0.033 | 7.0 | False | 0.0550 | 0.0552 | 1.00 |
| paired_replay | 1.87e-02 | 0.250 | 18.0 | False | 0.0006 | 0.0005 | 0.88 |
| pc_rsg | 3.17e-01 | 24.749 | 7.5 | False | 0.1096 | 0.1011 | 0.92 |

## tool_selection_large (H=12)

- MC target (H=12, exact infeasible): 1.020124 (se 1.37e-02)

| estimator | exact bias | exact var | cost | unbiased | predicted MSE | sampled MSE | ratio |
|---|---:|---:|---:|---|---:|---:|---:|
| dense | - | - | 12.0 | - | - | 0.0459 | - |
| local_sibling | - | - | 23.0 | - | - | 0.8756 | - |
| paired_replay | - | - | 102.0 | - | - | 0.0114 | - |
| pc_rsg | - | - | 19.5 | - | - | 1.4611 | - |

## evidence_chain_large (H=12)

- MC target (H=12, exact infeasible): 0.818582 (se 1.22e-02)

| estimator | exact bias | exact var | cost | unbiased | predicted MSE | sampled MSE | ratio |
|---|---:|---:|---:|---|---:|---:|---:|
| dense | - | - | 12.0 | - | - | 0.0255 | - |
| local_sibling | - | - | 23.0 | - | - | 0.5416 | - |
| paired_replay | - | - | 102.0 | - | - | 0.0062 | - |
| pc_rsg | - | - | 19.5 | - | - | 1.0403 | - |

## Findings

- Exact-layer predictor vs measured fixed-budget MSE: prediction ratios 0.87-1.07 across all estimator-task pairs at H=4 (the exact layer QUANTITATIVELY
  reproduces the sampled MSE, not just the ordering).
- Transfer finding: in observation-dependent tool-agent worlds the coupled
  contrast misses the INDIRECT effect of a_t (through future observations and
  actions), so the designed-world unbiasedness of paired-replay/pc_rsg (M0,
  independent coordinates) does not hold here — the exact layer flags both
  as biased; the local sibling remains biased for the full gradient (T003).
- Scale: at H=12 (MC target only) paired-replay wins fixed-budget MSE on both
  tasks — its cycle-variance advantage (exact layer: var 1.49 vs dense 14.19)
  compounds, while the fixed bias penalty does not.

## Honesty notes
- This is the evidence-bridge DEMONSTRATION on controllable tool-agent tasks:
  exact verdicts predicting fixed-budget MSE ordering. The third layer (real LLM
  trajectories) is the Stage 3 harness — estimators consume records only, so real
  records drop in unchanged, gated on the Guard trajectory schema package
  (design 20.2). No claim about real LLM agent performance.
- The MC reference is an independent code path with its own RNG stream; the
  estimators never import it (oracle independence).
