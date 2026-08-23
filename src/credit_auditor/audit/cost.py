"""Cost gate (design §11.3, faults A5/A6, reason codes C001-C003).

Checks:
- primary cost unit frozen; prefix/branch/continuation/restore/calibration
  accounted (C001 if the candidate's declared cycle cost omits terms its own
  mechanism signature requires);
- baseline and candidate complete the same total budget (no asymmetric
  compute; never compare single-cycle variance only);
- baseline entrypoint is the frozen strong envelope, not a weak constant
  (C003).
"""

from __future__ import annotations

from fractions import Fraction

from credit_auditor.schema import CostSpec, GateResult, ReasonCode


def _frac(x: float | str | Fraction) -> Fraction:
    if isinstance(x, Fraction):
        return x
    if isinstance(x, str):
        return Fraction(x)
    return Fraction(x).limit_denominator(10**9)


def cost_gate(
    candidate_cost: CostSpec,
    candidate_actual_cycle_cost: Fraction,
    baseline_cost: CostSpec,
    baseline_actual_cycle_cost: Fraction,
    budget: int,
    declared_mechanism_terms: list[str] | None = None,
) -> GateResult:
    """C gate for one (candidate, baseline, budget) triple.

    `declared_mechanism_terms` lists the cost terms the candidate's own
    mechanism requires (e.g. ["prefix", "suffixes", "restores"]); if the
    declared cycle cost omits one, C001 fires (A5).
    """
    # A5: declared cost must cover every term the mechanism requires.
    if declared_mechanism_terms is not None:
        missing = [t for t in declared_mechanism_terms if t not in candidate_actual_cycle_cost_terms(candidate_cost)]
        if missing:
            return GateResult(
                gate="matched_cost",
                status="fail",
                reason_codes=[ReasonCode.C001_UNMATCHED_TRANSITION_BUDGET],
                detail=f"candidate cost omits required terms: {missing}",
            )

    # Both estimators must complete the same total budget (floor cycles).
    if budget <= 0:
        raise ValueError("budget must be positive")
    n_cand = budget // candidate_actual_cycle_cost
    n_base = budget // baseline_actual_cycle_cost
    if n_cand < 1 or n_base < 1:
        return GateResult(
            gate="matched_cost",
            status="fail",
            reason_codes=[ReasonCode.C001_UNMATCHED_TRANSITION_BUDGET],
            detail=f"budget {budget} infeasible for one side (cand n={n_cand}, base n={n_base})",
        )
    return GateResult(
        gate="matched_cost",
        status="pass",
        detail=f"cand n={n_cand} base n={n_base} at budget {budget}",
    )


def baseline_entrypoint_gate(baseline_kind: str, frozen_envelope: str) -> GateResult:
    """C003: baseline must be the frozen strong envelope (A6)."""
    if baseline_kind != frozen_envelope:
        return GateResult(
            gate="matched_cost",
            status="fail",
            reason_codes=[ReasonCode.C003_BASELINE_ENTRYPOINT_UNFAITHFUL],
            detail=f"baseline kind {baseline_kind!r} != frozen envelope {frozen_envelope!r}",
        )
    return GateResult(gate="matched_cost", status="pass")


def candidate_actual_cycle_cost_terms(cost: CostSpec) -> list[str]:
    return [t.term_id for t in cost.evaluate_cycle().terms]
