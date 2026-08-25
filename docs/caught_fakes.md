# 我抓过的假货（Caught fakes）

Legacy-incident narrative, in the same style as GRPO-Guard's incident
storytelling. Every "fake" below is a FAILURE TYPE that the Auditor now
catches as a frozen regression fixture (§3, §22). The historical numbers are
incident background only — in `docs_only_semantic` mode they are never
reproduced as results (§13.6); each case below maps to the gate that would
have caught it.

## Case 1 — CPC 的"假自适应"（fake adaptivity）

**The claim that looked real**: a variable-width, per-decision adaptive
credit method with beautiful learning curves and a headline correlation.

**What was actually there**: the "adaptive" mapping was structurally a fixed
global width. When the mechanism check ran (the D002-style calibration →
frozen-test dual verdict), the calibrated widths **collapsed to [2,2,2,2] —
exactly the global control**. A metric pass with no mechanism behind it:
the method's advantage came from the fixed mapping, not from adaptation.

**How the Auditor catches it today**: `MECH001` (adaptive mapping
degeneration) — the pre-registered mechanism gate compares the calibrated
hyperparameters against the fixed control and fails the claim when there is
no diversity (§13.3: a metric pass does not license a mechanism claim).
Frozen regression: the D002 dual verdict (`artifacts/v0.1.6/D002_test`).

## Case 2 — ρ=0.735 的"假方差"（fake variance）

**The claim that looked real**: a correlation `ρ=0.735` between the method's
variance and something meaningful — strong evidence the estimator was
working.

**What was actually there**: the baseline it was computed against was
broken. In the ALFWorld fork-oracle runs (`53/60`), the alternatives were
**no-ops**: the environment's per-rollout draws only sampled success or
failure, so the same goal was all-success or all-fail — the reward had **no
variance to measure**. `ρ=0.735` is a correlation computed on a variance
that was undefined by construction. (CPC-iter2 `43/50` showed the same
pattern.)

**How the Auditor catches it today**: the **non-degeneracy pre-gate**
(§10.2): alternatives must change state, group variance must be non-zero —
else the instance is INVALID before any estimator runs. The fault-injection
matrix pins it as `A11` (alternative action is a no-op → non-degeneracy
fail). Frozen regression: the A11 template in the self-audit pack
(`artifacts/v0.1.6/SELFAUDIT`).

## Case 3 — 86% no-op 的"假成功曲线"（fake success curve）

**The claim that looked real**: the training curve `36.5% → 63.5%` — the
policy was clearly improving.

**What was actually there**: the rollout policy was **static** — the online
rollout never tracked the trainer's weights (the GRPO-Guard `static_rollout`
incident). The curve measured the OLD policy's behavior under changing
prompts, not training. A large fraction of the "updates" were no-ops on the
real policy; per-iteration success was the success of the stale rollout.
(The same pattern in the pilot's `4.7% → 10.9%`.)

**How the Auditor catches it today**: this one is split across the two
projects — GRPO-Guard's online envelope contract (policy identity, sync
ledger, `max_policy_lag_versions = 0`) catches it ONLINE; the Auditor's
trajectory-level audit catches it OFFLINE on the records the optimizer
consumed: `T004` (stale/mixed policy_version) and `S002` (misbound
old-logprobs). Frozen regression: the trajectory demo
(`scripts/run_trajectory_demo.sh`, f4/f5 cases) + the M0 target audit
(`T003` propagated-sibling rejection).

## The pattern

All three fakes share one structure: **a metric that looked like evidence
but was not measuring the claimed mechanism** — a fixed width dressed as
adaptive, a variance computed on no variance, a success curve from a policy
that never trained. The Auditor's four questions exist to make each of
these failures explicit before the number reaches a resume:

1. What exactly are you estimating? (target identity — catches Case 1's
   hidden fixed mapping and Case 3's stale policy)
2. Unbiased vs an independent oracle? (bias — the oracle that Case 2 never
   had)
3. What does it cost at a matched budget? (cost — the "wins" that bought
   their advantage with more compute)
4. Is the mechanism real, or did it degenerate? (mechanism — Case 1's
   collapse and Case 2's no-op alternatives)
