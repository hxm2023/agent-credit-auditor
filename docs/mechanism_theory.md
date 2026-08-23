# Mechanism theory: why the audit verdicts hold

This document derives the analytic structure behind the Auditor's key verdicts
(the paired-replay win, the K-sample loss, the HH amplification, the collapse
statistical test). Every formula is verified numerically against the exact
enumerations (see `tests/unit/test_mechanism_theory.py`); the derivations are
the interview-grade version of "why".

## Notation

Bernoulli decision world: a_t ~ Bernoulli(p), score s_t = a_t - p, terminal
reward R. All expectations are exact over the finite path space.

## 1. Paired-replay variance: why the win is structural

The paired-replay estimator at decision t is

    ĝ = Δ·s_t,   Δ = R(a, ξ) - R(a', ξ)   (coupled continuation, deterministic)

For the focal world R = w(2a_t - 1) + (noise independent of a_t), the contrast
is deterministic: Δ = 2w(a - a'). Then:

    E[ĝ] = 2w·p(1-p)                                    (the local effect)
    E[ĝ²] = (2w)²·E[(a-a')²(a-p)²]
         = (2w)²·p(1-p)·(p² + (1-p)²)
    Var(ĝ) = 4w²·p(1-p)·(p² + (1-p)²) - 4w²·p²(1-p)²
           = 4w²·p(1-p)·(1 - 3p + 3p² - 2p³ + p²)        (closed form)

Compare with dense REINFORCE at the same coordinate:

    Var(R·s_t) = E[R²]·p(1-p) - (2wp(1-p))²,   E[R²] ≈ noise² + w²

**Why paired-replay wins**: the continuation noise (which dominates E[R²])
cancels exactly in the contrast, so the variance is driven by the SMALL focal
effect w instead of the LARGE reward variance. The uncoupled control (fresh
continuations) loses because the noise does not cancel: Var(Δ̄) ≈ 2σ²/Kb,
which at the D002 cost convention cannot beat the envelope.

## 2. K-sample estimator: the prefix floor

The K-sample continuation-averaging estimator (shared prefix through depth d)
has exact second moments

    E[ĝ_K ĝ_Kᵀ] = (1/K)·M_dense + (1 - 1/K)·E[μ(pre)μ(pre)ᵀ]

where μ(pre) = E[ĝ | prefix] = Q(s_d)·s_pre + g_from(s_d). The first term
shrinks with K, but the second does NOT:

    Var(ĝ_K) → Var(E[ĝ | prefix])   as K → ∞        (the prefix floor)

The floor is Var over prefixes of the conditional mean — it is the variance
the K samples share and can never average out. Because the D002 cost
convention charges K continuation transitions, the K-sample estimator's
variance/cost product is bounded below by the floor·(K(h-d)+…)/h, which loses
to the dense envelope (dense variance/cost = 1, RLOO = 0.5). This is why the
historical 0.694 win is not reconstructable from the summary semantics: any
shared-prefix structure carries the floor.

## 3. HH sparse sampling: the 1/q amplification

The Hansen-Hurwitz estimator at sampled time T ~ q is ĝ_T = R·s_T/q_T. The
exact coordinate variance (deconditioning over the sampling probability) is

    Var(ĝ_t) = E[(R s_t)²]/q_t - target_t²
             = Var(R s_t)/q_t + target_t²·(1/q_t - 1)

so sparse sampling multiplies the noise variance by 1/q_t (uniform q over H:
×H per coordinate) plus a target-correction term. This is the mechanism
behind the V001 utility failure: the residual correction's noise is amplified
while its cost is paid in full.

## 4. The collapse statistical test

The mechanism gate's MECH001 decision is formalized as a hypothesis test
(`collapse_statistical_evidence` in `audit/mechanism.py`):

    H0: the selected widths are drawn uniformly from the candidate width
        multiset (the 2401-mapping space).
    Test statistic: Shannon diversity H(selected widths).
    p-value: P(H(random draw of size k) ≤ H(observed)).

Under the frozen RNG seed the null distribution is reproducible; the D002
selection [2,2,2,2] has H = 0, which is at the 0-quantile of the null →
statistically collapsed at any alpha. A genuinely adaptive selection (e.g.
[2,4,2,8], H > 0) passes the gate and clears the test.

## 5. Numerical verification summary

| Formula | Verified against | Result |
|---|---|---|
| §1 Var(ĝ_paired) closed form | exact enumeration over 2^H paths | == 0 diff |
| §2 E[ĝ_K ĝ_Kᵀ] split | brute-force joint enumeration (K=2) | == 0 diff |
| §3 1/q amplification (+target term) | exact HH moments | == 0 diff |
| §4 collapse p-value | frozen null resampling | deterministic |

Honesty notes: these are exact-world identities, not LLM claims; they explain
the designed-world verdicts (M0 positive, D002 mechanism-fail) and the
historical non-reproducibility (prefix floor), but they do not transfer to
real-task prevalence.
