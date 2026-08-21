"""Hansen-Hurwitz / Horvitz-Thompson estimators (§5.4, §9.2).

hh_distribution(world, q): sample decision time T ~ q, full rollout, estimate
    g_T = R(tau) * s_T / q_T   (coordinate T only; others zero).
Unbiased for the full gradient iff q_t > 0 for every coordinate with nonzero
target (S001 otherwise). With-replacement sampling requires the Hansen-Hurwitz
(selection-probability) correction; without-replacement requires Horvitz-
Thompson (inclusion-probability) correction (S003).
"""
from __future__ import annotations

from credit_auditor.worlds.base import WeightedVector
from credit_auditor.worlds.bernoulli_sequence import BernoulliSequenceMDP


def hh_distribution(world: BernoulliSequenceMDP, q: tuple[float, ...]) -> list[WeightedVector]:
    H = world.horizon
    if len(q) != H:
        raise ValueError(f"q must have length {H}")
    if any(x < 0 for x in q) or abs(sum(q) - 1.0) > 1e-12:
        raise ValueError("q must be a probability vector")
    out: list[WeightedVector] = []
    for t, qt in enumerate(q):
        for a, p in world.all_paths():
            s = world.score(a)
            vec = [0.0] * H
            vec[t] = world.rewards[a] * s[t] / qt if qt > 0 else 0.0
            out.append(WeightedVector(p * qt, tuple(vec)))
    return out


def uniform_hh_distribution(world: BernoulliSequenceMDP) -> list[WeightedVector]:
    return hh_distribution(world, tuple([1.0 / world.horizon] * world.horizon))


def mechanism_signature() -> dict:
    return {"estimator_family": "hh_ht", "contrast_source": "none", "updated_coordinates": "single_sampled"}
