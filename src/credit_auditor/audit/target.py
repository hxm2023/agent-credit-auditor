"""Target gate (design §11.1, faults A1/A2, reason codes T001-T005).

Checks: claimed estimand defined; estimator expectation == target within
tolerance; local credit NOT propagated to shared prefix; continuation policy
consistent; clipped/unclipped scope not conflated.
"""

from __future__ import annotations

import numpy as np

from credit_auditor.schema import GateResult, ReasonCode
from credit_auditor.stats import ExactMoments


def compare_oracle(primary: np.ndarray, oracle: np.ndarray, tol_rel: float, tol_abs: float) -> tuple[bool, float]:
    """Primary-vs-oracle mismatch check (E002 upstream of the target gate)."""
    max_mismatch = float(np.max(np.abs(primary - oracle)))
    if max_mismatch > tol_abs:
        return False, max_mismatch
    return True, max_mismatch


def target_gate(
    moments: ExactMoments,
    tolerance: dict,
    mechanism_signature: dict,
    claimed_estimand: str,
    estimand_id: str,
) -> GateResult:
    """Return the T-gate result for one (estimator, estimand) pair."""
    rel = tolerance.get("bias_rel", 1e-9)
    abs_tol = tolerance.get("bias_abs", 1e-12)
    max_abs_target = float(np.max(np.abs(moments.target))) if moments.target.size else 0.0
    scale = max(max_abs_target, 1.0)
    bias_ok = moments.max_abs_bias() <= max(abs_tol, rel * scale)

    if claimed_estimand != estimand_id:
        return GateResult(
            gate="target_identity",
            status="fail",
            reason_codes=[ReasonCode.T001_ESTIMAND_UNSPECIFIED],
            detail=f"claimed={claimed_estimand} actual={estimand_id}",
        )

    if bias_ok:
        return GateResult(gate="target_identity", status="pass", detail=f"max_abs_bias={moments.max_abs_bias():.3e}")

    # Bias found: choose the reason code by the declared mechanism structure.
    updated = mechanism_signature.get("updated_coordinates", "")
    contrast = mechanism_signature.get("contrast_source", "")
    propagated = updated == "all_including_prefix"
    if propagated and contrast in ("local_sibling", "selected_contrast"):
        code = ReasonCode.T003_LOCAL_TO_PREFIX_PROPAGATION
    elif contrast == "truncated_continuation":
        code = ReasonCode.T004_CONTINUATION_TARGET_MISMATCH
    elif contrast == "partial_retention":
        code = ReasonCode.T005_CLIPPING_SCOPE_MISMATCH
    else:
        code = ReasonCode.T002_BIAS_EXCEEDS_TOLERANCE
    return GateResult(
        gate="target_identity",
        status="fail",
        reason_codes=[code],
        detail=f"max_abs_bias={moments.max_abs_bias():.3e} bias_sq={moments.bias_sq:.3e}",
    )


def retention_gate(moments: ExactMoments, tolerance: dict) -> GateResult:
    """T005: partial outcome retention changes the target."""
    rel = tolerance.get("bias_rel", 1e-9)
    abs_tol = tolerance.get("bias_abs", 1e-12)
    if moments.max_abs_bias() <= max(abs_tol, rel * max(float(np.max(np.abs(moments.target))), 1.0)):
        return GateResult(gate="target_identity", status="pass")
    return GateResult(
        gate="target_identity",
        status="fail",
        reason_codes=[ReasonCode.T005_CLIPPING_SCOPE_MISMATCH],
        detail=f"retained-outcome estimator bias={moments.max_abs_bias():.3e}",
    )
