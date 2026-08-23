"""Dense REINFORCE estimator and stronger dense envelope (§9.2).

- dense_distribution: g = R(tau) * score(tau)  (unbiased, cost H)
- dense_optimal_constant_distribution: per-coordinate optimal constant
  c_t = E[R s_t^2]/E[s_t^2]; g_t = (R - c_t) s_t (unbiased, variance-minimized)
- root_rloo_distribution(n=2): leave-one-out over 2-sample sets, exact joint
  distribution (ordered pairs with replacement); g = 0.5[(R1-R2)s1 + (R2-R1)s2]
"""

from __future__ import annotations

from credit_auditor.worlds.base import WeightedVector
from credit_auditor.worlds.bernoulli_sequence import BernoulliSequenceMDP


def dense_distribution(world: BernoulliSequenceMDP) -> list[WeightedVector]:
    out = []
    for a, p in world.all_paths():
        s = world.score(a)
        out.append(WeightedVector(p, tuple(world.rewards[a] * st for st in s)))
    return out


def dense_optimal_constant_distribution(world: BernoulliSequenceMDP) -> list[WeightedVector]:
    H = world.horizon
    num = [0.0] * H
    den = [0.0] * H
    for a, p in world.all_paths():
        s = world.score(a)
        r = world.rewards[a]
        for t in range(H):
            num[t] += p * r * s[t] ** 2
            den[t] += p * s[t] ** 2
    c = [num[t] / den[t] for t in range(H)]
    out = []
    for a, p in world.all_paths():
        s = world.score(a)
        r = world.rewards[a]
        out.append(WeightedVector(p, tuple((r - c[t]) * s[t] for t in range(H))))
    return out


def root_rloo_distribution(world: BernoulliSequenceMDP, n: int = 2) -> list[WeightedVector]:
    """Exact joint distribution over n-sample sets (ordered, with replacement).
    For n=2 the set enumeration is exact and cheap. Expectation equals the
    full gradient because E[s_t] = 0 coordinate-wise."""
    if n != 2:
        raise NotImplementedError("exact root-RLOO currently implemented for n=2 only")
    paths = world.all_paths()
    out: list[WeightedVector] = []
    for a1, p1 in paths:
        s1 = world.score(a1)
        r1 = world.rewards[a1]
        for a2, p2 in paths:
            s2 = world.score(a2)
            r2 = world.rewards[a2]
            g = tuple(0.5 * ((r1 - r2) * s1[t] + (r2 - r1) * s2[t]) for t in range(world.horizon))
            out.append(WeightedVector(p1 * p2, g))
    return out


def mechanism_signature() -> dict:
    return {"estimator_family": "dense", "contrast_source": "none", "updated_coordinates": "all"}
