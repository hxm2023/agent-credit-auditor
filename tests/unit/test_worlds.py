"""BernoulliSequenceMDP math tests (§17.1)."""
from __future__ import annotations

import math

import numpy as np

from credit_auditor.worlds.bernoulli_sequence import BernoulliSequenceMDP, deterministic_world


def test_path_probabilities_sum_to_one():
    world = deterministic_world(seed=1, horizon=4)
    total = sum(p for _, p in world.all_paths())
    assert total == 1.0


def test_all_paths_count():
    world = deterministic_world(seed=2, horizon=5)
    assert len(world.all_paths()) == 32


def test_score_vector():
    world = BernoulliSequenceMDP(probabilities=(0.25, 0.5, 0.75), rewards={(0, 0, 0): 1.0, (0, 0, 1): 1.0, (0, 1, 0): 1.0, (0, 1, 1): 1.0, (1, 0, 0): 1.0, (1, 0, 1): 1.0, (1, 1, 0): 1.0, (1, 1, 1): 1.0})
    s = world.score((1, 0, 1))
    assert s == (0.75, -0.5, 0.25)


def test_true_gradient_matches_finite_difference():
    world = deterministic_world(seed=3, horizon=4)

    def logit(p: float) -> float:
        return math.log(p / (1 - p))

    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    eps = 1e-6
    g_num = []
    for t in range(world.horizon):
        p0 = world.probabilities[t]
        def j_with(delta: float) -> float:
            probs = list(world.probabilities)
            probs[t] = sigmoid(logit(p0) + delta)
            w = BernoulliSequenceMDP(tuple(probs), world.rewards)
            return w.expected_reward()
        g_num.append((j_with(eps) - j_with(-eps)) / (2 * eps))
    g_exact = world.true_gradient()
    np.testing.assert_allclose(g_num, g_exact, rtol=1e-5, atol=1e-7)


def test_expected_reward_matches_enumeration():
    world = deterministic_world(seed=4, horizon=3)
    manual = sum(p * world.rewards[a] for a, p in world.all_paths())
    assert abs(manual - world.expected_reward()) < 1e-12


def test_q_values_match_direct_conditional_expectation():
    world = deterministic_world(seed=5, horizon=4)
    q = world.q_values()
    # Q(empty prefix) = expected reward
    assert abs(q[()] - world.expected_reward()) < 1e-12
    # Q(prefix) = E[reward | prefix], checked by enumeration
    for t in range(1, world.horizon):
        for bits in range(1 << t):
            h = tuple((bits >> (t - 1 - tt)) & 1 for tt in range(t))
            # conditional expectation by enumeration of suffixes
            p_cond = 0.0
            r_cond = 0.0
            for suff_bits in range(1 << (world.horizon - t)):
                suff = tuple((suff_bits >> (world.horizon - t - 1 - s)) & 1 for s in range(world.horizon - t))
                full = h + suff
                p = 1.0
                for tt, at in enumerate(full):
                    p *= world.probabilities[tt] if at else (1.0 - world.probabilities[tt])
                p_cond += p
                r_cond += p * world.rewards[full]
            assert abs(q[h] - r_cond / p_cond) < 1e-9


def test_world_rejects_incomplete_rewards():
    import pytest
    with pytest.raises(ValueError):
        BernoulliSequenceMDP(probabilities=(0.5, 0.5), rewards={(0, 0): 1.0})


def test_world_serialization_roundtrip():
    world = deterministic_world(seed=6, horizon=3)
    spec = world.to_spec()
    assert spec["world"] == "bernoulli_sequence_mdp"
    w2 = BernoulliSequenceMDP.from_spec(spec)
    assert w2.probabilities == world.probabilities
    assert w2.rewards == world.rewards
    np.testing.assert_allclose(w2.true_gradient(), world.true_gradient())


def test_noop_alternative_detection():
    """E001: an alternative that never changes reward is a no-op."""
    world = BernoulliSequenceMDP(
        probabilities=(0.5, 0.5),
        rewards={(0, 0): 1.0, (0, 1): 1.0, (1, 0): 2.0, (1, 1): 2.0},
    )
    # action at t=1 is irrelevant: no effect on reward -> group variance zero
    from credit_auditor.audit.environment import noop_alternative_detection
    flags = noop_alternative_detection(world)
    assert flags == [False, True]
