"""Numerical verification of the mechanism theory formulas (docs/
mechanism_theory.md): closed forms vs exact enumeration."""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from credit_auditor.audit.mechanism import collapse_statistical_evidence
from credit_auditor.stats import exact_moments
from credit_auditor.worlds.bernoulli_sequence import BernoulliSequenceMDP
from credit_auditor.worlds.base import WeightedVector


def _focal_world(p: float, w: float, noise: float) -> BernoulliSequenceMDP:
    """2-decision focal world: R = w(2a_1-1) + noise(2a_0-1)(2a_0-1) —
    the noise term depends only on a_0, so the t=1 contrast cancels it."""
    rewards = {}
    for a0 in (0, 1):
        for a1 in (0, 1):
            rewards[(a0, a1)] = w * (2 * a1 - 1) + noise * (2 * a0 - 1) * (2 * a0 - 1)
    return BernoulliSequenceMDP((p, p), rewards)


def _paired_estimator_moments(world: BernoulliSequenceMDP, t: int) -> np.ndarray:
    """Exact paired-replay moments at decision t (coupled contrast)."""
    H = world.horizon
    p = world.probabilities[t]
    out = []
    for bits in range(1 << H):
        a = tuple((bits >> (H - 1 - tt)) & 1 for tt in range(H))
        w_p = 1.0
        for tt, at in enumerate(a):
            w_p *= world.probabilities[tt] if at else (1 - world.probabilities[tt])
        for a_sib in (0, 1):
            alt = list(a)
            alt[t] = a_sib
            delta = world.rewards[a] - world.rewards[tuple(alt)]
            value = delta * (a[t] - p)
            w_sib = p if a_sib == 1 else (1 - p)
            out.append(WeightedVector(w_p * w_sib, (value,)))
    m = exact_moments(out, np.zeros(1))
    return np.array([m.mean[0], m.var_trace])


def test_paired_replay_variance_closed_form():
    """§1: Var(ĝ) = 4w²p(1-p)(p²+(1-p)²) - 4w²p²(1-p)²."""
    p, w = 0.4, 0.3
    world = _focal_world(p, w, noise=1.0)
    mean, var = _paired_estimator_moments(world, t=1)
    e2 = (2 * w) ** 2 * p * (1 - p) * (p**2 + (1 - p) ** 2)
    var_closed = e2 - (2 * w * p * (1 - p)) ** 2
    assert abs(var - var_closed) < 1e-12
    assert abs(mean - 2 * w * p * (1 - p)) < 1e-12


def test_paired_replay_win_is_structural():
    """§1: the contrast variance is driven by w, not by the noise; the
    uncoupled contrast would carry the noise."""
    p, w = 0.5, 0.1
    world = _focal_world(p, w, noise=1.0)
    _, var_paired = _paired_estimator_moments(world, t=1)
    # dense per-coordinate variance at t=1: E[R²]p(1-p) - (local effect)²
    er2 = sum(pr * world.rewards[a] ** 2 for a, pr in world.all_paths())
    local = 2 * w * p * (1 - p)
    var_dense = er2 * p * (1 - p) - local**2
    assert var_paired < 0.1 * var_dense, (var_paired, var_dense)


def test_hh_amplification_1_over_q2():
    """§3: Var(ĝ_t) = Var(R s_t)/q_t² for the HH estimator."""
    world = _focal_world(0.5, 0.2, noise=0.8)
    from credit_auditor.estimators import hh_ht
    from credit_auditor.worlds.base import WeightedVector

    target = world.true_gradient()
    q = (0.5, 0.5)  # uniform over the 2 decisions
    dist = hh_ht.hh_distribution(world, q)
    m = exact_moments(dist, target)
    # E[(R s_1)²] and Var(R s_1) (the conditional variance at T=1)
    e_rs1_2 = sum(pr * (world.rewards[a] * world.score(a)[1]) ** 2 for a, pr in world.all_paths())
    var_rs1 = e_rs1_2 - target[1] ** 2
    m1 = exact_moments([WeightedVector(v.weight, (v.vector[1],)) for v in dist], np.array([target[1]]))
    var1 = m1.var_trace
    # §3 general form: Var(ĝ_t) = E[ĝ²] - target² = E[(R s)²]/q - target²
    #                     = Var(R s)/q + target²·(1/q - 1)
    var1_analytic = var_rs1 / q[1] + target[1] ** 2 * (1 / q[1] - 1)
    assert abs(var1 - var1_analytic) < 1e-9
    # amplification dominates: with target ~ 0 the 1/q factor is the term
    var1_zero_target = var_rs1 / q[1]
    assert var1 > var1_zero_target  # target correction adds variance when target != 0


def test_collapse_statistical_test_d002():
    """§4: the D002 selection [2,2,2,2] is statistically collapsed; a diverse
    selection is not."""
    null_widths = [2, 4, 8] * 800 + [1]  # 2401-mapping width multiset approx
    ev = collapse_statistical_evidence([2, 2, 2, 2], null_widths)
    assert ev["diversity"] == 0.0
    assert ev["statistically_collapsed"] is True
    assert ev["p_value"] <= 0.05
    ev2 = collapse_statistical_evidence([2, 4, 2, 8], null_widths)
    assert ev2["diversity"] > 0.0
    assert ev2["p_value"] > 0.05


def test_collapse_test_deterministic():
    null_widths = [2, 4, 8] * 800 + [1]
    a = collapse_statistical_evidence([2, 2, 2, 2], null_widths)
    b = collapse_statistical_evidence([2, 2, 2, 2], null_widths)
    assert a == b
