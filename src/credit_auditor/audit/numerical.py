"""Numerical-margin gate (fault A14, reason code N001).

Near-zero float values must never be counted as sign reversals (§10.3):
sign claims require a pre-registered margin, and values below it are marked
near-zero rather than positive/negative.
"""
from __future__ import annotations

from credit_auditor.schema import GateResult, ReasonCode


def sign_of(value: float, margin: float) -> str:
    """'positive' / 'negative' / 'near_zero' with an explicit margin."""
    if value > margin:
        return "positive"
    if value < -margin:
        return "negative"
    return "near_zero"


def sign_reversal_gate(
    a: float, b: float, margin: float, claim: str = "sign reversal"
) -> GateResult:
    """A14: a 'sign reversal' claim requires both values beyond the margin,
    otherwise N001 fires."""
    sa, sb = sign_of(a, margin), sign_of(b, margin)
    if sa == "near_zero" or sb == "near_zero":
        return GateResult(
            gate="numerical_margin",
            status="fail",
            reason_codes=[ReasonCode.N001_NEAR_ZERO_SIGN],
            detail=f"claim '{claim}': values {a:.3e} ({sa}), {b:.3e} ({sb}) below margin {margin:.1e}",
        )
    if sa != sb:
        return GateResult(gate="numerical_margin", status="pass", detail=f"reversal {sa} -> {sb}")
    return GateResult(gate="numerical_margin", status="fail", reason_codes=[ReasonCode.N001_NEAR_ZERO_SIGN], detail=f"no reversal: both {sa}")
