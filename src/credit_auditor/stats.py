"""Exact bias / variance / MSE and fixed-budget MSE (design §5.5).

For a discrete estimator distribution P(g_hat = x):
    mu   = E[g_hat]
    b    = mu - g
    Bias^2 = ||b||_2^2
    VarTrace = E||g_hat - mu||_2^2
    MSE  = Bias^2 + VarTrace

Fixed budget B with single-cycle cost c:
    n = floor(B/c)
    MSE_B = Bias^2 + VarTrace/n
This requires n iid estimator cycles under a frozen world. Cycles sharing
dense backbone / common random numbers / adaptive state must compute the JOINT
estimator distribution instead (§5.5). Shared calibration/backbone costs must
be deducted per protocol, not divided out.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from credit_auditor.worlds.base import WeightedVector


@dataclass
class ExactMoments:
    mean: np.ndarray
    target: np.ndarray
    bias: np.ndarray
    bias_sq: float
    var_trace: float
    mse: float
    n_cycles: int | None = None
    mse_at_budget: float | None = None
    unused_budget: float | None = None
    n_outcomes: int = 0

    def max_abs_bias(self) -> float:
        return float(np.max(np.abs(self.bias))) if self.bias.size else 0.0


def exact_moments(distribution: list[WeightedVector], target: np.ndarray) -> ExactMoments:
    """Compute exact moments from a probability-weighted estimator distribution."""
    total = sum(v.weight for v in distribution)
    if total <= 0:
        raise ValueError("distribution total weight must be > 0")
    n = len(distribution)
    dim = len(distribution[0].vector) if n else 0
    mean = np.zeros(dim)
    var = np.zeros(dim)
    for v in distribution:
        vec = np.asarray(v.vector, dtype=np.float64)
        mean += (v.weight / total) * vec
    for v in distribution:
        diff = np.asarray(v.vector, dtype=np.float64) - mean
        var += (v.weight / total) * (diff * diff)
    target_arr = np.asarray(target, dtype=np.float64)
    bias = mean - target_arr
    bias_sq = float(bias @ bias)
    var_trace = float(var.sum())
    return ExactMoments(
        mean=mean,
        target=target_arr,
        bias=bias,
        bias_sq=bias_sq,
        var_trace=var_trace,
        mse=bias_sq + var_trace,
        n_outcomes=n,
    )


def fixed_budget_mse(moments: ExactMoments, budget: int, cycle_cost: float) -> ExactMoments:
    """MSE under a fixed total budget: n = floor(B/c) iid cycles (§5.5)."""
    if cycle_cost <= 0:
        raise ValueError("cycle cost must be positive")
    n = int(budget // cycle_cost)
    if n < 1:
        # INFEASIBLE_BUDGET: budget below one cycle cost; no gradient work
        # possible. Reported as None (not averaged into results).
        moments.n_cycles = 0
        moments.unused_budget = float(budget)
        return moments
    moments.n_cycles = n
    moments.unused_budget = float(budget) - n * cycle_cost
    moments.mse_at_budget = moments.bias_sq + moments.var_trace / n
    return moments


def average_mse_at_budget(moments_list: list[ExactMoments], budget: int, cycle_cost: float | list[float]) -> float:
    """Average of per-problem fixed-budget MSE; infeasible cells are excluded.
    Cycle cost may differ per problem (e.g. different horizons); pass a list
    aligned with moments_list in that case."""
    costs = [cycle_cost] * len(moments_list) if isinstance(cycle_cost, (int, float)) else list(cycle_cost)
    if len(costs) != len(moments_list):
        raise ValueError("cycle costs must align with moments_list")
    vals = []
    for m, c in zip(moments_list, costs):
        m2 = fixed_budget_mse(m, budget, c)
        if m2.mse_at_budget is not None:
            vals.append(m2.mse_at_budget)
    if not vals:
        raise ValueError("no feasible cells at this budget")
    return float(np.mean(vals))
