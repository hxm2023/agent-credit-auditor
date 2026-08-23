# Online/offline fault map: GRPO-Guard → Credit Auditor

The two projects are one trust chain: GRPO-Guard validates the ONLINE
trajectory chain (policy/token/mask identity, canonical hashing, envelope
lifecycle); Credit Auditor validates the OFFLINE estimator claims (estimand,
target, cost, mechanism). The map below connects every Guard online fault to
the offline-detectable signal and the Auditor gate/reason code that catches
it — answering "has this project really been used?" with a concrete
fault-to-detector correspondence.

| Guard online fault | Online detection (Guard) | Offline-detectable signal | Auditor gate / reason |
|---|---|---|---|
| static_rollout (rollout policy not synced with trainer) | F1 boundary family: behavior logprob source / policy version mismatch | envelope `training_contract.max_policy_lag_versions > 0` or `behavior_logprob_source` inconsistent with the scoring event | `sampling_support` (S002_Q_NOT_LOGGED), `target_identity` (T004_CONTINUATION_TARGET_MISMATCH) |
| mask_shift (completion/loss mask misaligned) | F1 family: mask-shape / span checks | decision spans inconsistent with the completion mask (span out of range or misaligned) | `target_identity` (T005_CLIPPING_SCOPE_MISMATCH) |
| misbound_logprob (old-logprob bound to wrong policy) | F1 family: authoritative logprob event checks | logged q / old-logprob without the authoritative behavior event | `sampling_support` (S002_Q_NOT_LOGGED) |
| retokenization (trainer/rollout tokenize differently) | F1 family: token identity checks | token identity not closed across the envelope chain | `target_identity` (T005_CLIPPING_SCOPE_MISMATCH) |
| f5_split_leakage (held-out used in training) | f5_f8_v02: D003_SPLIT_OVERLAP | calibration/test seed overlap in the audit protocol | `heldout_split` (D001_SPLIT_OVERLAP / D003_SPLIT_OVERLAP) |
| f6_evaluator_alias (same evaluator under two names) | f5_f8_v02: R006_EVALUATOR_ALIAS | split/evaluator identifiers not bijective | `heldout_split` (D001_SPLIT_OVERLAP) |
| f7_event_reorder (scoring event after the update) | f5_f8_v02: L005_SCORING_AFTER_UPDATE | manifest event ordering inconsistent | `provenance` (P001_EVIDENCE_INCOMPLETE) |
| f8_artifact_mutation (artifact edited after publish) | f5_f8_v02: T001_ARTIFACT_HASH_MISMATCH | package SHA256SUMS mismatch on disk | `provenance` (P001_EVIDENCE_INCOMPLETE, hash lineage) |

The Auditor's own release pipeline implements the offline half: `check_split_disjoint` (f5), the provenance audit's SHA256SUMS verification (f8), the no-overwrite discipline (f8-adjacent), the selection self-hash check (f8/lineage), and the envelope bundle validation (§25, f1-adjacent). The real-scenario demo (`scripts/run_real_scenario_demo.sh`) injects these Guard fault patterns into the Auditor's own artifacts and shows the detection firing.

## Trajectory-level signals (v0.2-prep, Stage 1 of the real-trajectory bridge)

The four f1-family faults are ALSO offline-detectable at the TRAJECTORY level —
the rollout records an optimizer step actually consumes (tokens + action mask +
old logprobs + rewards). `src/credit_auditor/audit/trajectory_audit.py` checks
per-record consistency and the estimator-consumption agreement; the demo
(`scripts/run_trajectory_demo.sh`) injects each fault into the frozen fixture
records and shows the detector firing.

| Trajectory signal | Offline check | Auditor reason code |
|---|---|---|
| mask_shift | `action_mask` length != generated tokens; mask values not in {0,1} | `T005_CLIPPING_SCOPE_MISMATCH` |
| misbound_logprob | `old_logprobs` missing / length mismatch / NaN / Inf / positive | `S002_Q_NOT_LOGGED` |
| retokenization | token span inconsistent with the record (length/identity breaks) | `T005_CLIPPING_SCOPE_MISMATCH` |
| stale_policy | `policy_version` missing, or two versions mixed inside one batch | `T004_CONTINUATION_TARGET_MISMATCH` |
| silent_mask_drift | estimator-applied mask != `optimizer_consumed_mask` | `T005_CLIPPING_SCOPE_MISMATCH` |
| unparseable / missing fields | record JSON/fields broken | `P001_EVIDENCE_INCOMPLETE` |

The trajectory records are anchored hash-only via
`src/credit_auditor/adapters/trajectory_bundle.py` (pinned schema
`aca-trajectory-bundle-1.0`, fail-closed on unknown schema/missing hash/mutation).

Honesty notes: the mapping is a design correspondence (both projects' reason
codes were written against the same failure taxonomy); the demos run the
offline detectors on Auditor-owned fixtures with injected Guard fault patterns,
not on live Guard production data. The trajectory record format is the
Auditor's own spec for frozen fixtures; real Guard trajectories flow through
the envelope adapter (pinned to the Guard schema, §25) until Guard publishes
its trajectory schema package (§20.2 gate).
