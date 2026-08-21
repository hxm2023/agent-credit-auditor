"""BPO-like estimator (§9.2, §3.1 lesson).

This is a LITERAL-BPO-STYLE estimator, not a faithful implementation of any
specific published algorithm: select a decision time t* by (1-eps) top-entropy
+ eps uniform floor, branch both actions at t* with a paired contrast, and
PROPAGATE the contrast to every coordinate with no selection correction.

Properties (exact): the selection is not HH-corrected and the contrast is
propagated to prefix coordinates, so the expectation is NOT the full gradient
in general. The old project found such literal BPO biased in 144/202 cases;
this semantic port must show the same failure TYPE (T002/T003) on designed
cases. Never report as a claim about official BPO (§9.2).
"""
from __future__ import annotations

import math

from credit_auditor.worlds.base import WeightedVector
from credit_auditor.worlds.bernoulli_sequence import BernoulliSequenceMDP


def _entropies(world: BernoulliSequenceMDP) -> list[float]:
    h: list[float] = []
    for p in world.probabilities:
        e = 0.0
        for q in (p, 1.0 - p):
            if q > 0:
                e -= q * math.log(q)
        h.append(e)
    return h


def bpo_like_distribution(
    world: BernoulliSequenceMDP, epsilon: float = 0.1
) -> list[WeightedVector]:
    H = world.horizon
    ents = _entropies(world)
    t_star = max(range(H), key=lambda t: ents[t])
    q_sel = [epsilon / H + (1.0 - epsilon) if t == t_star else epsilon / H for t in range(H)]
    out: list[WeightedVector] = []
    for t in range(H):
        for a, p in world.all_paths():
            for sb in range(1 << H):
                ap = tuple((sb >> (H - 1 - tt)) & 1 for tt in range(H))
                w_sib = 1.0
                for tt, apt in enumerate(ap):
                    w_sib *= world.probabilities[tt] if apt else (1.0 - world.probabilities[tt])
                alt = a[:t] + (ap[t],) + a[t + 1 :]
                delta = world.rewards[a] - world.rewards[alt]
                s = world.score(a)
                vec = tuple(delta * st for st in s)
                out.append(WeightedVector(p * w_sib * q_sel[t], vec))
    return out


def mechanism_signature() -> dict:
    return {
        "estimator_family": "bpo_like",
        "contrast_source": "selected_contrast",
        "updated_coordinates": "all_including_prefix",
        "selection_correction": "none",
        "disclaimer": "BPO-like literal port; not a claim about official BPO",
    }
