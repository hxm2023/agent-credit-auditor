"""Sampling/support gate (design §11.2, faults A3/A4, reason codes S001-S004)."""
from __future__ import annotations

import numpy as np

from credit_auditor.schema import GateResult, ReasonCode, SamplingSpec


def support_gate(q: tuple[float, ...], target: np.ndarray, min_support: float = 1e-6) -> GateResult:
    """S001: every target-relevant coordinate must have positive sampling
    support."""
    zeroed = [t for t, qt in enumerate(q) if qt < min_support and abs(target[t]) > min_support]
    if zeroed:
        return GateResult(
            gate="sampling_support",
            status="fail",
            reason_codes=[ReasonCode.S001_ZERO_SUPPORT],
            detail=f"zero support at coordinates {zeroed}",
        )
    return GateResult(gate="sampling_support", status="pass")


def correction_gate(spec: SamplingSpec, q_logged: bool = True) -> GateResult:
    """S003/S002: replacement law must match the correction; q must be logged."""
    if not q_logged:
        return GateResult(
            gate="sampling_support",
            status="fail",
            reason_codes=[ReasonCode.S002_Q_NOT_LOGGED],
            detail="selection probabilities not recorded",
        )
    law = spec.decision_sampling.replacement
    correction = spec.correction
    if law == "with_replacement" and correction.name != "hansen_hurwitz":
        return GateResult(
            gate="sampling_support",
            status="fail",
            reason_codes=[ReasonCode.S003_WRONG_HH_HT_CORRECTION],
            detail=f"WR sampling requires Hansen-Hurwitz; got {correction.name}",
        )
    if law == "without_replacement" and correction.name != "horvitz_thompson":
        return GateResult(
            gate="sampling_support",
            status="fail",
            reason_codes=[ReasonCode.S003_WRONG_HH_HT_CORRECTION],
            detail=f"WOR sampling requires Horvitz-Thompson; got {correction.name}",
        )
    return GateResult(gate="sampling_support", status="pass")
