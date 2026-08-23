"""Mechanism gate (design §11.5, fault A9, reason codes MECH001/MECH002).

Every claimed adaptive/local/coupled mechanism must have an observable
signature and control:
- variable-width: selected widths must have pre-registered diversity AND
  differ from the global control; all-equal widths => MECH001 collapse.
- root aggregation: root-vs-flat results must be material (MECH002).
"""

from __future__ import annotations

from credit_auditor.schema import GateResult, ReasonCode


def width_diversity_gate(selected_widths: list[int], global_control_width: int | None = None) -> GateResult:
    """MECH001: adaptive variable-width claim requires >= 2 distinct selected
    widths and (when a global control exists) at least one width != control."""
    distinct = len(set(selected_widths))
    if distinct < 2:
        return GateResult(
            gate="mechanism",
            status="fail",
            reason_codes=[ReasonCode.MECH001_ADAPTIVE_MAPPING_COLLAPSED_TO_GLOBAL],
            detail=f"selected widths {selected_widths} have {distinct} distinct value(s); "
            f"adaptive claim collapses to a fixed width",
        )
    if global_control_width is not None and all(w == global_control_width for w in selected_widths):
        return GateResult(
            gate="mechanism",
            status="fail",
            reason_codes=[ReasonCode.MECH001_ADAPTIVE_MAPPING_COLLAPSED_TO_GLOBAL],
            detail=f"selected widths {selected_widths} all equal the global control {global_control_width}",
        )
    return GateResult(gate="mechanism", status="pass")


def collapse_statistical_evidence(
    selected_widths: list[int],
    null_widths: list[int],
    alpha: float = 0.05,
) -> dict:
    """Statistical formalization of the mechanism collapse (design §11.5).

    Null hypothesis: the selected widths are drawn at random from the
    candidate width distribution (null_widths, e.g. the 2401-mapping space's
    width multiset). Under the null, the Shannon diversity H of the selected
    widths is compared to the null distribution of H over random draws of the
    same size: if the observed H sits at or below the alpha quantile, the
    collapse is statistically significant (the selection is indistinguishable
    from a fixed-width choice, or worse).

    Returns {"diversity": H, "null_mean": ..., "null_alpha_quantile": ...,
             "p_value": ..., "statistically_collapsed": bool}.
    """
    import random

    import numpy as np

    def diversity(ws: list[int]) -> float:
        counts = np.bincount(ws)
        probs = counts[counts > 0] / len(ws)
        return float(-np.sum(probs * np.log(probs))) if len(probs) > 1 else 0.0

    k = len(selected_widths)
    rng = random.Random(20260823)  # frozen: the null distribution is reproducible
    null_diversities = []
    for _ in range(10000):
        draw = [rng.choice(null_widths) for _ in range(k)]
        null_diversities.append(diversity(draw))
    null_diversities = np.asarray(null_diversities)
    observed = diversity(selected_widths)
    p_value = float(np.mean(null_diversities <= observed))
    return {
        "diversity": observed,
        "null_mean": float(np.mean(null_diversities)),
        "null_alpha_quantile": float(np.quantile(null_diversities, alpha)),
        "p_value": p_value,
        "statistically_collapsed": p_value <= alpha,
    }


def root_vs_leaf_materiality_gate(root_effect: float, leaf_effect: float, tolerance: float = 1e-9) -> GateResult:
    """MECH002: a claimed root-aggregation mechanism must differ materially
    from the flat leaf result."""
    if abs(root_effect - leaf_effect) < tolerance:
        return GateResult(
            gate="mechanism",
            status="fail",
            reason_codes=[ReasonCode.MECH002_ROOT_VS_LEAF_NOT_MATERIAL],
            detail=f"root-vs-leaf difference {abs(root_effect - leaf_effect):.3e} below tolerance",
        )
    return GateResult(gate="mechanism", status="pass")
