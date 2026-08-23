# Real full-model post-training audit: lessons and upgrades (v0.1.5)

On 2026-08-23 the Auditor was run against a REAL full-model post-training
scenario: a Qwen3-4B GRPO training loop supervised by GRPO-Guard on the
autodl2 server (208 cores / 1 TB RAM; 2×RTX 6000D reserved for GPU projects —
the audit itself is CPU-only). Artifacts were pulled with the project
owner's explicit authorization (see `artifacts/real_training/` and the
decision trail).

## What was audited

| Artifact | Source | What the Auditor checked |
|---|---|---|
| `smoke_out/smoke_result.json` | real Qwen3-4B GRPO smoke run | TRL parameter sync calls — every rollout-side param sync must be acked |
| `loop_out/ckpt_v1/policy_manifest.json` | real training loop, policy v1 | traceability fields, per-shard weights sha256 (17.6 GB across 5 shards) |
| `loop_out/events/lease.json` | Guard trainer lease | run bookkeeping |
| `loop_out/run_manifest.json` | Guard run manifest | cross-schema observation |

## Result

- **Rollout policy sync: CLEAR** — 398/398 TRL sync calls acked, 1 optimizer
  step committed. The offline static_rollout signal (the legacy failure: a
  rollout policy that never tracks the trainer) is negative on real data.
- **Policy traceability: OK** — Qwen/Qwen3-4B, policy v1 (parent v0), all 5
  weight shards hashed, tokenizer/template/config/code hashes present.

## Lessons learned (real-scenario driven upgrades)

1. **Real data needs a directory-audit interface, not a frozen protocol.**
   Real training artifacts are not frozen reproducible worlds; the Auditor
   gained `audit/real_training.py` + `scripts/run_real_training_audit.sh`
   that audit a data directory and emit a report. This is a new tool class
   (real-scenario usage) beside the protocol packs (frozen evidence).

2. **Manifest schemas differ by owner.** Guard's `run_manifest.json` is
   minimal (`run_id`, `closed_loop`) — its schema is Guard's, not the
   Auditor's §18. The Auditor must NOT apply its own manifest-field check to
   another project's artifacts; the §18 check stays scoped to Auditor
   packages. The real-training audit reads fields by source schema.

3. **The strongest offline static-rollout signal is the sync-ack ledger.**
   `trl_observed_sync_calls` with `ack: false` is the direct traceable
   evidence of the legacy failure; the audit converts it into a verdict.

4. **Guard's blob store is binary and stays Guard-side.** The Auditor does
   not fork a decoder (§25); it audits what the manifests expose.

5. **Real-data fixtures are schema tests.** The fetched real artifacts'
   structure is mirrored into unit-test fixtures so the audit functions are
   regression-covered without depending on the remote host.

## Honesty notes

- This is a HEALTH audit of a real run (no fault found in the sync ledger);
  the value is the tool's demonstrated use on real full-model training
  artifacts, not a fault discovery.
- The audit covers offline-detectable signals from manifests; a full
  estimator-level audit of real trajectories (logprob/mask/reward on real
  rollout events) is the v0.2 milestone, gated on Guard's published schema
  package (§20.2).
