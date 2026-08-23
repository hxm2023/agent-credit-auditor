"""Continuation / partial-restore finite worlds (design §8.4, §13.4) —
support_only diagnostic pack.

Semantics (docs_only_semantic, new frozen universe):
- response functions map full outcomes to Fraction values;
- two observation regimes: MARGINAL (only the logged coordinates of the
  actual outcome) and PAIRED-REPLAY (the replay summary includes the
  sibling/continuation information);
- a target sign is identifiable from a fiber only if ALL outcomes in the
  fiber share that sign; mixed-sign fibers force abstention (zero false-safe:
  the auditor never claims identifiability on a mixed fiber);
- a continuation family is a set of continuation policies; action values and
  sign/rank stability are computed EXACTLY over the family;
- the coordinate-box relaxation (each coordinate allowed its own range) is
  compared with the coupled family (joint feasibility).

NOVELTY STATUS: CLASSICAL COUPLED/NONRECTANGULAR ROBUST ADVANTAGE AND
CROSS-WORLD PARTIAL IDENTIFICATION EQUIVALENCE — not a new theory (§3.4).
CLAIM STATUS: SUPPORT_ONLY DIAGNOSTIC.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction

# ---------------------------------------------------------------------------
# Response functions and observation regimes
# ---------------------------------------------------------------------------

Outcome = tuple[int, ...]


def marginal_key(outcome: Outcome, logged: tuple[int, ...]) -> tuple[int, ...]:
    """Marginal observation: the logged coordinates of the outcome."""
    return tuple(outcome[i] for i in logged)


def paired_replay_key(
    outcome: Outcome,
    sibling: Outcome,
    logged_outcome: tuple[int, ...],
    logged_sibling: tuple[int, ...],
) -> tuple[int, ...]:
    """Paired-replay observation: the logged coordinates of the outcome AND
    of the sibling replay (design §8.4: replica sees the replay summary). The
    two logged sets may differ (e.g. the replay summary logs the latent)."""
    return tuple(outcome[i] for i in logged_outcome) + tuple(sibling[i] for i in logged_sibling)


def fiber_signs(
    outcomes: list[Outcome],
    values: Mapping[Outcome, Fraction],
    sign: Callable[[Fraction], str],
    key_fn: Callable[[Outcome], tuple[int, ...]],
) -> dict[tuple[int, ...], set[str]]:
    """Sign set per observation fiber. A fiber with BOTH signs is NOT
    identifiable (abstention)."""
    out: dict[tuple[int, ...], set[str]] = {}
    for o in outcomes:
        k = key_fn(o)
        out.setdefault(k, set()).add(sign(values[o]))
    return out


def identifiable_fibers(fiber_signs_map: dict[tuple[int, ...], set[str]]) -> list[tuple[int, ...]]:
    return [k for k, signs in fiber_signs_map.items() if len(signs) == 1]


def mixed_fibers(fiber_signs_map: dict[tuple[int, ...], set[str]]) -> list[tuple[int, ...]]:
    return [k for k, signs in fiber_signs_map.items() if len(signs) > 1]


# ---------------------------------------------------------------------------
# Continuation family: exact action values, sign/rank stability
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContinuationPolicy:
    """A continuation policy: per-state action probabilities (Fractions)."""

    probs: Mapping[tuple[int, ...], Fraction]  # state -> P(a=1)


def action_value(
    reward: Mapping[tuple[int, int, int], Fraction],
    transitions: Mapping[tuple[int, int, int], Fraction],
    policy: ContinuationPolicy,
    s: int,
    a: int,
    horizon: int,
) -> Fraction:
    """Expected accumulated reward from (s, a) under the continuation policy,
    exact Fraction arithmetic. State space: binary chain."""
    if horizon == 1:
        return Fraction(reward[(s, a, 1)]) * transitions[(s, a, 1)] + Fraction(reward[(s, a, 0)]) * (
            1 - transitions[(s, a, 1)]
        )
    v = Fraction(0)
    for sp in (0, 1):
        p_tr = transitions[(s, a, sp)] if sp == 1 else 1 - transitions[(s, a, 1)]
        if p_tr == 0:
            continue
        p_a1 = policy.probs.get((horizon - 1, sp), Fraction(1, 2))
        sub = p_a1 * action_value(reward, transitions, policy, sp, 1, horizon - 1) + (1 - p_a1) * action_value(
            reward, transitions, policy, sp, 0, horizon - 1
        )
        v += p_tr * (Fraction(reward[(s, a, sp)]) + sub)
    return v


def family_action_values(
    reward: Mapping[tuple[int, int, int], Fraction],
    transitions: Mapping[tuple[int, int, int], Fraction],
    policies: list[ContinuationPolicy],
    s: int,
    horizon: int,
) -> dict[tuple[int, int], list[Fraction]]:
    """Q(s, a) under every family member: {(s, a): [values over policies]}."""
    out: dict[tuple[int, int], list[Fraction]] = {}
    for a in (0, 1):
        out[(s, a)] = [action_value(reward, transitions, p, s, a, horizon) for p in policies]
    return out


def sign_stability(values: list[Fraction]) -> tuple[str, int]:
    """Dominant sign over the family and the count of agreeing members."""
    pos = sum(1 for v in values if v > 0)
    neg = sum(1 for v in values if v < 0)
    zero = len(values) - pos - neg
    if pos == 0 and neg == 0:
        return "zero", zero
    return ("positive" if pos > neg else "negative"), max(pos, neg)


def rank_reversals(values_by_action: dict[tuple[int, int], list[Fraction]], s: int) -> int:
    """Number of family members where the Q(s,1) vs Q(s,0) RANK differs from
    the majority rank."""
    diffs = [v1 - v0 for v1, v0 in zip(values_by_action[(s, 1)], values_by_action[(s, 0)])]
    if not diffs:
        return 0
    majority = 1 if sum(1 for d in diffs if d > 0) >= sum(1 for d in diffs if d < 0) else -1
    return sum(1 for d in diffs if (d > 0) != (majority > 0))


def coordinate_box_vs_coupled(values_by_action: dict[tuple[int, int], list[Fraction]], s: int) -> dict:
    """Coordinate-box relaxation (per-action ranges independently) vs the
    coupled family (joint realizations): the box is strictly larger when the
    joint realizations are nonrectangular."""
    q1 = values_by_action[(s, 1)]
    q0 = values_by_action[(s, 0)]
    box_points = set(itertools.product(q1, q0))
    coupled_points = set(zip(q1, q0))
    return {
        "box_size": len(box_points),
        "coupled_size": len(coupled_points),
        "nonrectangular": coupled_points != box_points,
    }
