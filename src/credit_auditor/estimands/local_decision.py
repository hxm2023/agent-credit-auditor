"""LOCAL_DECISION_GRADIENT(t) estimand (§5.2, §5.3).

For BernoulliSequenceMDP the local action effect at time t given prefix h is
Q(h,1) - Q(h,0) and the local target coordinate is
    sum_h P(h) * p_t * (1-p_t) * (Q(h,1) - Q(h,0))
which equals dJ/dtheta_t (terminal-reward case). The estimator under test may
claim either the full vector (sparse at other coordinates -> bias) or the
single coordinate (unbiased). The audit decides per claimed estimand.
"""

from __future__ import annotations

import numpy as np

from credit_auditor.worlds.bernoulli_sequence import BernoulliSequenceMDP

ESTIMAND_ID = "local_decision_gradient"


def target(world: BernoulliSequenceMDP, t: int) -> np.ndarray:
    g = np.zeros(world.horizon, dtype=np.float64)
    q = world.q_values()
    probs = world.probabilities
    for bits in range(1 << t):
        h = tuple((bits >> (t - 1 - tt)) & 1 for tt in range(t))
        p_h = 1.0
        for tt, ht in enumerate(h):
            p_h *= probs[tt] if ht else (1.0 - probs[tt])
        g[t] += p_h * probs[t] * (1.0 - probs[t]) * (q[h + (1,)] - q[h + (0,)])
    return g
