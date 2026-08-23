"""Coverage + correctness for the non-focal D002 paths (ksample moments,
branching bucket moments, RLOO envelope) that the D002 pipeline tests do not
exercise (they use the focal world only)."""
from __future__ import annotations

import numpy as np

from credit_auditor.estimators import branching
from credit_auditor.estimators.dense import root_rloo_distribution
from credit_auditor.stats import exact_moments
from credit_auditor.worlds.base import WeightedVector
from credit_auditor.worlds.bernoulli_sequence import deterministic_world
from credit_auditor.worlds.d002_shared_logits import generate_problem, true_gradient


def test_ksample_moments_unbiased_on_stochastic_world():
    p = generate_problem("cov_k", 424242)
    target = true_gradient(p)
    mapping = {b.bucket_id: (2, 4) for b in p.buckets}
    mb = branching.estimator_moments(p, mapping)
    assert np.max(np.abs(mb.mean - target)) < 1e-9
    stats = branching.moments_to_stats(mb, target)
    assert stats["var_trace"] > 0
    assert branching.mapping_cycle_cost(p, mapping) > 0


def test_branching_bucket_moments_deep_width():
    p = generate_problem("cov_b", 424243)
    target = true_gradient(p)
    b = p.buckets[0]
    for width in (2, 4, 8):
        m = branching.branching_bucket_moments(p, b, d=2, width=width)
        assert m.second_moment.shape == (3, 3)
        assert np.isfinite(m.second_moment).all()
    # deep branch stays within the bucket horizon
    m = branching.branching_bucket_moments(p, b, d=b.horizon - 1, width=4)
    assert np.isfinite(m.second_moment).all()


def test_dense_optimal_constant_and_rloo_on_d002():
    p = generate_problem("cov_o", 424244)
    target = true_gradient(p)
    parts_opt = [branching.dense_optimal_constant_moments(p, b) for b in p.buckets]
    parts_rloo = [branching.root_rloo_moments(p, b) for b in p.buckets]
    for parts in (parts_opt, parts_rloo):
        mean = sum((x.mean for x in parts), np.zeros(3))
        assert np.max(np.abs(mean - target)) < 1e-9


def test_bernoulli_root_rloo_n2_exact():
    """dense.root_rloo_distribution (Bernoulli world, n=2) exact moments."""
    world = deterministic_world(seed=9, horizon=4)
    target = world.true_gradient()
    dist = root_rloo_distribution(world, n=2)
    m = exact_moments(dist, target)
    assert m.max_abs_bias() < 1e-12
    assert m.var_trace > 0
    assert m.var_trace < 1.5 * exact_moments(
        [WeightedVector(p, tuple(world.rewards[a] * s for s in world.score(a))) for a, p in world.all_paths()],
        target,
    ).var_trace
