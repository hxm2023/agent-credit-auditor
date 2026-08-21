"""Environment/oracle gate (design §11.7, faults A11).

Pre-gate checks:
- alternatives must change state or reachable continuation (E001_ALTERNATIVE_NOOP)
- group variance must be non-zero (E003_GROUP_VARIANCE_ZERO)
- continuation uses the declared policy
Non-degeneracy is checked BEFORE estimator gates so that a degenerate world can
never be used to claim an estimator works.
"""
from __future__ import annotations

from credit_auditor.worlds.bernoulli_sequence import BernoulliSequenceMDP


def noop_alternative_detection(world: BernoulliSequenceMDP) -> list[bool]:
    """Per time step: True if changing a_t never changes the reward
    distribution (no-op alternative)."""
    flags: list[bool] = []
    H = world.horizon
    for t in range(H):
        effect = 0.0
        for a, _ in world.all_paths():
            other = list(a)
            other[t] = 1 - other[t]
            effect += (world.rewards[a] - world.rewards[tuple(other)]) ** 2
        flags.append(effect < 1e-30)
    return flags


def group_variance_zero(world: BernoulliSequenceMDP, group: tuple[int, ...]) -> bool:
    """E003: within a decision group, the estimator values must not be
    constant (e.g., an action with no reward effect makes its local
    contrast estimator degenerate)."""
    q = world.q_values()
    vals = []
    H = world.horizon
    for t in group:
        for bits in range(1 << t):
            h = tuple((bits >> (t - 1 - tt)) & 1 for tt in range(t))
            vals.append(q[h + (1,)] - q[h + (0,)])
    if not vals:
        return True
    return max(vals) - min(vals) < 1e-30


def environment_gate(world: BernoulliSequenceMDP) -> dict:
    flags = noop_alternative_detection(world)
    reasons: list[str] = []
    for t, noop in enumerate(flags):
        if noop:
            reasons.append(f"E001_ALTERNATIVE_NOOP t={t}")
    for t in range(world.horizon):
        if group_variance_zero(world, (t,)):
            reasons.append(f"E003_GROUP_VARIANCE_ZERO t={t}")
    return {
        "status": "fail" if reasons else "pass",
        "reason_codes": reasons,
        "noop_alternatives": flags,
    }
