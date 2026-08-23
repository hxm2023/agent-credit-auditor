"""ROOT_MARGINAL_GRADIENT estimand (§5.2, §8.2 case 1).

Shared-logit world: several decision times share one root logit theta so
p_t = sigma(theta) for t in S (shared coordinate set). The root marginal is
    dJ/dtheta = sum_{t in S} E[ R * (a_t - p_t) ] * (dp_t/dtheta * 1)  -- logit form
For logit parameterization dp_t/dtheta = p_t(1-p_t) and the score is
(a_t - p_t), giving dJ/dtheta = sum_{t in S} E[R (a_t - p_t)] exactly
(chain rule: dJ/dtheta = sum_t dJ/dp_t * p_t(1-p_t), and dJ/dp_t = E[R (a_t-p_t)/(p_t(1-p_t))]).
A "flat leaf average" that averages per-decision terms without the chain-rule
sum is a different object; the audit must catch it.
"""

from __future__ import annotations

import numpy as np

from credit_auditor.worlds.bernoulli_sequence import BernoulliSequenceMDP

ESTIMAND_ID = "root_marginal_gradient"


def shared_logit_world(seed: int, shared_times: tuple[int, ...], horizon: int) -> BernoulliSequenceMDP:
    """World where decisions at `shared_times` share one logit (same p)."""
    import hashlib

    def draw(i: int) -> float:
        h = hashlib.sha256(f"ACA-ROOT::{seed}::{i}".encode()).digest()
        return int.from_bytes(h[:8], "big") / 2**64

    p_root = 0.2 + 0.6 * draw(0)
    probs: list[float] = []
    for t in range(horizon):
        probs.append(p_root if t in shared_times else 0.15 + 0.7 * draw(1 + t))
    rewards: dict[tuple[int, ...], float] = {}
    for bits in range(1 << horizon):
        a = tuple((bits >> (horizon - 1 - t)) & 1 for t in range(horizon))
        rewards[a] = 2 * draw(100 + bits) - 1
    return BernoulliSequenceMDP(tuple(probs), rewards)


def root_marginal_gradient(world: BernoulliSequenceMDP, shared_times: tuple[int, ...]) -> np.ndarray:
    """dJ/dtheta_root = sum_{t in S} E[R (a_t - p_t)] (logit chain rule)."""
    g_full = world.true_gradient()
    return np.asarray([g_full[t] for t in shared_times])


def flat_leaf_average(world: BernoulliSequenceMDP, shared_times: tuple[int, ...]) -> np.ndarray:
    """(1/|S|) sum_{t in S} E[R (a_t - p_t)] — the WRONG flat aggregation that
    a naive estimator may produce; equals the root marginal only if |S| == 1."""
    g_full = world.true_gradient()
    return np.asarray([g_full[t] / len(shared_times) for t in shared_times])
