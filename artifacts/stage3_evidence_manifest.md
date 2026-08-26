# Stage 3 evidence manifest (where the full evidence lives)

The GitHub repo carries the reproducible evidence per run — `metrics.json` +
`trajectory_records.jsonl` (aca-trajectory-record-1.0) + the reports — for
all 18 matrix runs (`artifacts/stage3_jindun/`) and the 3 autodl2 partial
runs (`artifacts/stage3/`). The FULL Guard-supervised evidence chain (the
append-only Guard event stores: generation/scoring/reward events, identity
and pre-update validation decisions, materialized update handles) lives
outside the repo because of size:

## 18-run matrix (jindun, 2026-08-24/25)

- Guard event stores per run: `<run>/events/` + `<run>/store/` under
  `/data_3/repo/agood/Agent-RL-Credit-Auditor/stage3/out_jindun/` on
  js3.blockelite.cn (NOT cleared — jindun is a permanent shared server).
- Each run's store contains: sealed GenerationEvent + ScoringEvent +
  RewardEvent + ValidationDecisionEvent for every epoch, the materialized
  `ValidatedBatchHandle` inputs (sequence/loss-mask/logprob artifacts), and
  the committed policy manifests with adapter sha256s (also recorded in
  `metrics.json` per epoch).

## autodl2 partial runs (3 dense, superseded; server cleared 2026-08-26)

- The full out tree (Guard stores + run logs) was archived before the
  rented server was released:
  `C:\Users\w1828\backups\stage3_out_autodl2_20260826.tgz` (1.1 GB, local).
- `metrics.json` + `trajectory_records.jsonl` for the 3 runs remain in the
  repo at `artifacts/stage3/` — the numbers they back (dense cts_order
  grad_l2 4.6-5.1, final eval mean_u 0.875) match the jindun runs.

## Related backups (user's other projects, same shutdown)

- `C:\Users\w1828\backups\attrl_artifacts_20260826.tgz` (72 MB): agent-ttrl
  r002/r003 run outputs from autodl2 (agent-ttrl itself also lives on
  jindun at /data_3/repo/agood/agent-ttrl).
- The GRPO-Guard source is a git repo at
  https://github.com/hxm2023/GRPO-Guard (no server-only code).
