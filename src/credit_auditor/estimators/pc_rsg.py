"""PC-RSG-style estimator (design §9.2, V001 expected-utility-failure).

Cycle = dense backbone rollout (cost H) + at a q-sampled decision t, a
residual correction built from a coupled paired contrast:
    g = R(tau) s(tau) + (contrast_t - R(tau) s_t) / q_t   (coordinate t only)
The correction is unbiased (E[contrast_t] = local effect = the backbone's
target coordinate), so calibration can be accurate while the FIXED-BUDGET
utility still fails: the 1/q_t amplification of the contrast noise and the
branch continuation cost dominate. This is the semantic reproduction of the
historical V001 failure TYPE (residual noise amplification + branch cost),
not a claim of the historical 24.81x number.
"""
from __future__ import annotations

from credit_auditor.worlds.base import WeightedVector
from credit_auditor.worlds.bernoulli_sequence import BernoulliSequenceMDP


def pc_rsg_distribution(
    world: BernoulliSequenceMDP, q: tuple[float, ...], branch_width: int = 2
) -> list[WeightedVector]:
    H = world.horizon
    if len(q) != H or abs(sum(q) - 1.0) > 1e-12:
        raise ValueError(f"q must be a length-{H} probability vector")
    out: list[WeightedVector] = []
    # Marginalized exact enumeration: only the sibling action at the sampled
    # coordinate t enters the value; all other sibling coordinates integrate
    # to weight 1, so each outcome is (t, tau, a'_t) with weight P(tau) q_t P(a'_t).
    for t, qt in enumerate(q):
        for a, p in world.all_paths():
            s = world.score(a)
            r = world.rewards[a]
            backbone = [r * st for st in s]
            for apt in (0, 1):
                alt = a[:t] + (apt,) + a[t + 1 :]
                # score-weighted coupled contrast: E[Delta * s_t] = local effect
                contrast = (world.rewards[a] - world.rewards[alt]) * s[t]
                w_sib = world.probabilities[t] if apt else (1.0 - world.probabilities[t])
                vec = list(backbone)
                if qt > 0:
                    vec[t] += (contrast - r * s[t]) / qt
                out.append(WeightedVector(p * qt * w_sib, tuple(vec)))
    return out


def cycle_cost_formula(horizon: int, q: tuple[float, ...]) -> float:
    """Expected cycle cost: dense backbone H + branch at the q-sampled
    decision (t + 2(H-t) + 1 under d002_branching_v1 with width 2)."""
    return float(horizon) + sum(q[t] * (t + 2 * (horizon - t) + 1) for t in range(horizon))


def mechanism_signature() -> dict:
    return {
        "estimator_family": "pc_rsg",
        "contrast_source": "sparse_residual_correction",
        "updated_coordinates": "backbone_all_plus_sampled_residual",
        "calibration": "frozen_q_and_branch_width",
    }
