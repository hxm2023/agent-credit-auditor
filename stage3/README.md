# Stage 3 — matched-budget real closed loop (jindun)

The external review's Stage 3: 2 tool-use tasks x 3 credit estimators x
3 seeds with matched rollout/token/optimizer budgets, final checkpoints,
and honest reporting (success, reward, KL, length, invalid calls, estimator
variance, throughput). Written 2026-08-24; predictions pre-registered in
`PREDICTIONS.md` BEFORE the runs.

## What runs

- **Tasks**: `cts_order` (CTS evidence-utility world, the r002 environment)
  and `tau2_retail` (TAU2 base split via the tau2 server: /init /exec /eval,
  reward = pass_pct of the evaluation criteria).
- **Estimators** (mapping to the Stage-2 bridge set):
  - `dense` — group-relative GRPO advantage over all masked tokens (the
    Guard's `grpo_loss`)
  - `local` — the SAME advantage applied ONLY at tool-call decision token
    positions (the local-sibling analog; custom `stage3_loss`)
  - `paired` — per-decision signed credit from the paired-branch reliability
    gate (`agent_ttrl.credit.paired_credit` over a (slots x 2 branches)
    utility matrix), applied at decision positions
- **Loop**: 3 epochs; ALL epochs roll out with the trainer's own current
  policy and score exactly (Guard `behavior_logprob_source=
  "exact_behavior_scorer"`, schema 7.5 — the trainer emits GenerationEvent +
  ScoringEvent per sequence). The vLLM generation-service mode is NOT
  exercised: vLLM 0.26 cannot initialize on this server's Blackwell GPU
  (SM 12.x requires a newer vLLM); documented in each run's metrics.
  Guard chain per epoch: identity validation -> ScoringEvent -> reward
  event -> pre-update ALLOW -> materialize -> guarded update -> commit
  adapter + manifest -> canary. 32 prompts x 8 gens x 3 epochs, LoRA
  rank 16 / alpha 32 / lr 5e-6.
- **Output per run**: `metrics.json` (per-epoch success/mean_u/calls/
  invalid/grad_l2/loss + final eval on held-out prompts + KL drift vs base +
  GPU seconds), the Guard event store (auditable), and
  `trajectory_records.jsonl` (aca-trajectory-record-1.0) for the Auditor's
  Stage-1 trajectory audit.

## How to run (on the jindun server; the final 18/18 matrix was executed by
scripts/run_jindun_seq.sh + run_jindun_sweep.sh, one GPU at a time, free-card-only)

```bash
# On jindun (js3.blockelite.cn, working folder /data_3/repo/agood/Agent-RL-Credit-Auditor):
# 1. tau2 server (needed only for the tau2_retail task)
python3 stage3/tau2_server_jindun.py   # NotRequired shim for py3.10 included

# 2. dry-run first (8 prompts, 1 epoch, cts_order x dense)
GRPO_GUARD_REPO=/data_3/repo/agood/grpo-guard-src ATTRL_DIR=/data_3/repo/agood/agent-ttrl GRPO_GUARD_MODEL_PATH=/data_3/repo/agood/models_cache/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c STAGE3_CUDA_DEVICE=cuda:6 .venv_jindun/bin/python stage3/train.py   --task cts_order --estimator dense --seed 1 --out stage3/out_jindun/dryrun   --prompts 8 --gens 4 --epochs 1

# 3. full matrix (18 runs, one GPU at a time, free-card-only, yields to
#    other users; 5 attempts per run)
bash stage3/run_jindun_seq.sh
bash stage3/run_jindun_sweep.sh   # fills any gaps -> 18/18
```
```

Result collection: `uv run python scripts/stage3_report.py <results_dir>` in
this repo writes the comparison against the pre-registered predictions.

## Honest boundaries

- The estimators are the real-rollout mapping of the Stage-2 definitions
  (documented in credit.py); the paired-branch credit uses a
  (slots x branches) utility matrix, not CRN-coupled branches — the
  reliability gate's paired structure is kept, the coupling is not.
- Real data, real training, Guard-supervised; but the runs are SMALL
  (LoRA, 32 prompts) — the numbers support the mechanism-level comparison,
  not deployment claims.
- The Auditor's Stage-1 audit runs on the exported trajectory records; the
  Stage-2 verdicts predict the ordering (PREDICTIONS.md); both are compared
  honestly in the results report.
