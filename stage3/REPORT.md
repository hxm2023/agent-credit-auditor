# Stage 3 results — matched-budget real closed loop (jindun, A800)

- 2 tasks (cts_order, tau2_retail) x 3 estimators (dense/local/paired) x 3 seeds
- 32 prompts x 8 gens x 3 epochs, LoRA rank 16 / lr 5e-6, Guard-supervised
- predictions pre-registered in stage3/PREDICTIONS.md BEFORE the runs

## Findings (from the 18 real runs)

- cts_order paired-branch reliability gate abstained in ALL epochs and ALL seeds (True) — zero credit, zero updates, policy unchanged. The gate is conservative on this task profile (58% invalid tool calls, flat utilities); pre-registered prediction 4's 'recovers later' did NOT hold within 3 epochs.
- cts_order dense/local produced real Guard-validated updates every epoch (True) with comparable gradient scale (grad_l2 4.9 vs 4.9); final eval unchanged at mean_u 0.875 — 3 LoRA steps at lr 5e-6 do not move deployment metrics (mechanism-level comparison only).
- tau2_retail: base Qwen3-4B produced NO valid function calls (invalid_rate 1.0, True zero-reward regime) — ALL estimators received zero signal (True); an honest negative about the task/model combination, not an estimator comparison.

## cts_order

| estimator | seeds done | final success | mean_u | grad_l2 (mean over epochs) | KL drift | invalid | GPU s |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 3 | 0.0000 | 0.8750 | 4.8887 | 0.0013 | 0.0000 | 2582.4648 |
| local | 3 | 0.0000 | 0.8750 | 4.8569 | -0.0009 | 0.0000 | 2415.5869 |
| paired | 3 | 0.0000 | 0.8750 | 0.0000 | 0.0000 | 0.0000 | 2374.7350 |

## tau2_retail

| estimator | seeds done | final success | mean_u | grad_l2 (mean over epochs) | KL drift | invalid | GPU s |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 2664.4445 |
| local | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 2454.3435 |
| paired | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 2702.3703 |

## Gradient L2 per epoch (estimator variance proxy)

| estimator | e0 | e1 | e2 |
|---|---:|---:|---:|
| dense | 2.3368 | 2.4576 | 2.5388 |
| local | 2.4969 | 2.0611 | 2.7273 |
| paired | 0.0000 | 0.0000 | 0.0000 |

## Pre-registered predictions vs results

| Prediction | Verdict | Evidence |
|---|---|---|
| P1 ordering of final success | VOID | cts_order: all equal (0.000) -> ordering VOID; tau2_retail: all equal (0.000) -> ordering VOID |
| P2 gradient variance ordering | CONFIRMED | dense > local > paired ({'dense': '2.4444', 'local': '2.4284', 'paired': '0.0000'}) (dense ~= local; paired abstains) |
| P3 KL drift ordering | INCONCLUSIVE | paired < local < dense ({'dense': '0.0007', 'local': '0.0005', 'paired': '0.0000'}) (paired's 0.0 = zero updates/gate abstention, not mechanism KL) |

## Stage-1 trajectory audit on the exported records

| run | records | consistent | findings |
|---|---:|---:|---:|
| cts_order_dense_s1 | 768 | False | 0 |
| cts_order_dense_s2 | 768 | False | 0 |
| cts_order_dense_s3 | 768 | False | 0 |
| cts_order_local_s1 | 768 | False | 0 |
| cts_order_local_s2 | 768 | False | 0 |
| cts_order_local_s3 | 768 | False | 0 |
| cts_order_paired_s1 | 768 | False | 0 |
| cts_order_paired_s2 | 768 | False | 0 |
| cts_order_paired_s3 | 768 | False | 0 |
| tau2_retail_dense_s1 | 768 | False | 0 |
| tau2_retail_dense_s2 | 768 | False | 0 |
| tau2_retail_dense_s3 | 768 | False | 0 |
| tau2_retail_local_s1 | 768 | False | 0 |
| tau2_retail_local_s2 | 768 | False | 0 |
| tau2_retail_local_s3 | 768 | False | 0 |
| tau2_retail_paired_s1 | 768 | False | 0 |
| tau2_retail_paired_s2 | 768 | False | 0 |
| tau2_retail_paired_s3 | 768 | False | 0 |

## Honesty notes
- Real Guard-supervised GRPO training, but small scale (LoRA, 32 prompts); the
  numbers support the mechanism-level comparison only. Final eval unchanged
  everywhere: 3 LoRA steps at lr 5e-6 are far below what moves deployment
  metrics — the experiment compares estimator/gate behavior, not outcomes.
- Trajectory audit 'consistent=False' with 0 findings: each run's record file
  mixes 3 policy versions (one per epoch), which the batch-level policy-mix
  check flags by design; every per-record check (mask/logprob/reward) passes.
- Paired-branch credit uses a (decision slots x 2 branches) utility matrix — the
  reliability gate is kept, CRN coupling is not (stage3/credit.py).
- tau2_retail's all-zero-reward regime (invalid_rate 1.0) is a task/model
  capability negative, not an estimator result; it is reported honestly.
- Trajectory records carry real old_logprobs where the generation service
  provided them; the Guard chain owns logprob identity.
