"""Exact moments tests (§5.5, §17.1)."""

from __future__ import annotations

import numpy as np
import pytest

from credit_auditor.stats import average_mse_at_budget, exact_moments, fixed_budget_mse
from credit_auditor.worlds.base import WeightedVector


def test_mse_identity_bias2_plus_vartrace():
    dist = [WeightedVector(0.25, (1.0, 0.0)), WeightedVector(0.25, (0.0, 1.0)), WeightedVector(0.5, (-1.0, -1.0))]
    m = exact_moments(dist, target=np.array([0.0, 0.0]))
    assert abs(m.mse - (m.bias_sq + m.var_trace)) < 1e-15


def test_exact_moments_known_values():
    # deterministic estimator = target -> zero bias, zero variance
    dist = [WeightedVector(1.0, (3.0, 4.0))]
    m = exact_moments(dist, target=np.array([3.0, 4.0]))
    assert m.bias_sq == 0.0
    assert m.var_trace == 0.0
    assert m.mse == 0.0


def test_exact_moments_bias():
    dist = [WeightedVector(1.0, (5.0, 0.0))]
    m = exact_moments(dist, target=np.array([3.0, 0.0]))
    assert abs(m.bias_sq - 4.0) < 1e-12


def test_constant_baseline_does_not_change_expectation():
    """§17.1: subtracting a constant from the reward changes variance only.
    Estimator g = R*s with E[s] = 0 per coordinate; E[(R-c)s] = E[Rs]."""
    # two paths: (R=2, s=(+.5,+.5)) w.p. .5, (R=4, s=(-.5,-.5)) w.p. .5
    dist = [WeightedVector(0.5, (1.0, 1.0)), WeightedVector(0.5, (-2.0, -2.0))]
    dist_c = [WeightedVector(0.5, (0.5, 0.5)), WeightedVector(0.5, (-1.5, -1.5))]  # baseline c=1
    m = exact_moments(dist, target=np.array([-0.5, -0.5]))
    m_c = exact_moments(dist_c, target=np.array([-0.5, -0.5]))
    np.testing.assert_allclose(m.mean, m_c.mean, rtol=1e-12)
    assert m.var_trace != m_c.var_trace


def test_fixed_budget_floor_and_mse():
    dist = [WeightedVector(1.0, (3.0, 4.0))]
    m = exact_moments(dist, target=np.array([3.0, 4.0]))
    m = fixed_budget_mse(m, budget=100, cycle_cost=6)
    assert m.n_cycles == 16
    assert m.unused_budget == 4.0
    assert m.mse_at_budget == 0.0


def test_fixed_budget_infeasible():
    m = exact_moments([WeightedVector(1.0, (1.0,))], target=np.array([1.0]))
    m = fixed_budget_mse(m, budget=5, cycle_cost=6)
    assert m.n_cycles == 0
    assert m.mse_at_budget is None


def test_fixed_budget_mse_formula():
    # var_trace = 2, bias = 0, cost 4, budget 40 -> n=10 -> MSE = 2/10 = 0.2
    dist = [WeightedVector(0.5, (1.0, 0.0)), WeightedVector(0.5, (-1.0, 0.0))]
    m = exact_moments(dist, target=np.array([0.0, 0.0]))
    assert abs(m.var_trace - 1.0) < 1e-12
    m = fixed_budget_mse(m, budget=40, cycle_cost=4)
    assert m.n_cycles == 10
    assert abs(m.mse_at_budget - 0.1) < 1e-12


def test_average_mse_excludes_infeasible():
    m1 = exact_moments([WeightedVector(1.0, (1.0,))], target=np.array([1.0]))
    m2 = exact_moments([WeightedVector(1.0, (2.0,))], target=np.array([1.0]))
    # m2 infeasible (cost 12 > budget 10); m1 (cost 6) feasible with n=1
    avg = average_mse_at_budget([m1, m2], budget=10, cycle_cost=[6.0, 12.0])
    m1b = fixed_budget_mse(m1, budget=10, cycle_cost=6.0)
    assert avg == m1b.mse_at_budget
    assert m1b.n_cycles == 1


def test_negative_total_weight_rejected():
    with pytest.raises(ValueError):
        exact_moments([WeightedVector(0.0, (1.0,))], target=np.array([1.0]))
