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
