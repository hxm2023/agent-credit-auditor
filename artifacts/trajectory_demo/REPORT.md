# Trajectory-level fault demo (v0.2-prep, Stage 1)

- record schema: aca-trajectory-record-1.0 | bundle schema: aca-trajectory-bundle-1.0
- fixtures: tests/fixtures/trajectories/clean_trajectories.jsonl (frozen, committed)
- bundle: hash-only refs, fail-closed on mutation

| injected fault | detected | reason code(s) |
|---|---|---|
| clean copy (no fault) | True | n,o,n,e |
| f1 mask_shift | True | T005_CLIPPING_SCOPE_MISMATCH |
| f2 misbound_logprob | True | S002_Q_NOT_LOGGED |
| f3 retokenization | True | S002_Q_NOT_LOGGED,T005_CLIPPING_SCOPE_MISMATCH |
| f4 stale_policy | True | T004_CONTINUATION_TARGET_MISMATCH |
| f5 mixed_policy | True | T004_CONTINUATION_TARGET_MISMATCH |
| f6 silent_mask_drift | True | T005_CLIPPING_SCOPE_MISMATCH |

## Verdict: ALL DETECTORS FIRE (clean baseline consistent=True)

## Honesty notes
- The detectors check offline consistency of optimizer-consumed trajectory records;
  they do not evaluate training outcomes. Real Guard trajectories still flow through
  the envelope adapter (pinned to the Guard schema, design 25); this record format is
  the Auditor's own fixture spec for the trajectory-level audit.
