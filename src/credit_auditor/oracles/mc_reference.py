"""Independent high-budget Monte Carlo reference for the evidence bridge.

The MC reference computes the full-policy-gradient target
    g* = E[ R(tau) * sum_t (a_t - pi_t) ]
directly from raw rollouts: it samples with its OWN RNG stream and computes
the target from first principles (no estimator code, no exact enumeration).
It never imports estimator modules; estimators never import this module
(oracle independence, same rule as the exact oracles).

Two uses:
- agreement gate: at small horizon, MC(g*) must agree with the exact target
  from ToolAgentTask.exact_target() (validates both paths);
- sampled-layer reference: at large horizon (exact infeasible), the MC
  estimate is the target the fixed-budget MSE is measured against.
"""

from __future__ import annotations

import random

from credit_auditor.worlds.tool_agent import ToolAgentTask, action_prob, observation_prob, reward


def mc_target(world: ToolAgentTask, n: int = 1_000_000, seed: int = 20260824) -> dict:
    """MC estimate of g* with per-coordinate decomposition and SE."""
    rng = random.Random(seed)
    H = world.horizon
    acc = 0.0
    acc2 = 0.0
    for _ in range(n):
        actions: list[int] = []
        obs: list[int] = []
        for t in range(H):
            pa = action_prob(world.spec, t, tuple(obs))
            a = 1 if rng.random() < pa else 0
            po = observation_prob(world.spec, t, a)
            o = 1 if rng.random() < po else 0
            actions.append(a)
            obs.append(o)
        s = [a - action_prob(world.spec, t, tuple(obs[:t])) for t, a in enumerate(actions)]
        v = reward(world.spec, tuple(actions)) * sum(s)
        acc += v
        acc2 += v * v
    mean = acc / n
    var = acc2 / n - mean * mean
    return {
        "target": mean,
        "n": n,
        "seed": seed,
        "se": (max(var, 0.0) / n) ** 0.5,
    }
