"""Faults A5/A6 (§14): unmatched budget and unfaithful baseline."""

from __future__ import annotations

from fractions import Fraction

from credit_auditor.audit.cost import baseline_entrypoint_gate, cost_gate
from credit_auditor.schema import CostSpec, ReasonCode


def _branch_cost_spec() -> CostSpec:
    return CostSpec(calculator_id="d002_branching_v1", parameters={"horizon": 6, "depth": 4, "width": 8})


def _dense_cost_spec() -> CostSpec:
    return CostSpec(calculator_id="dense_horizon_v1", parameters={"horizon": 6})


def test_A5_branch_cost_omitted():
    """A5: candidate does not count branch continuation cost -> C001."""
    spec = _branch_cost_spec()
    gate = cost_gate(
        candidate_cost=spec,
        candidate_actual_cycle_cost=Fraction(27, 1),
        baseline_cost=_dense_cost_spec(),
        baseline_actual_cycle_cost=Fraction(6, 1),
        budget=512,
        declared_mechanism_terms=["prefix", "suffixes", "restores"],
    )
    assert gate.status == "pass"  # honest spec passes

    # A5 injection: candidate's declared cost is dense (H) although its
    # mechanism needs branching terms -> the term-level check fires.
    gate2 = cost_gate(
        candidate_cost=_dense_cost_spec(),  # wrongly declares dense cost
        candidate_actual_cycle_cost=Fraction(6, 1),
        baseline_cost=_dense_cost_spec(),
        baseline_actual_cycle_cost=Fraction(6, 1),
        budget=512,
        declared_mechanism_terms=["prefix", "suffixes", "restores"],
    )
    assert gate2.status == "fail"
    assert ReasonCode.C001_UNMATCHED_TRANSITION_BUDGET in gate2.reason_codes


def test_A6_weak_baseline():
    """A6: baseline uses a weak constant instead of the frozen strong
    envelope -> C003."""
    gate = baseline_entrypoint_gate(baseline_kind="plain_reinforce", frozen_envelope="dense_optimal_constant_root_rloo")
    assert gate.status == "fail"
    assert ReasonCode.C003_BASELINE_ENTRYPOINT_UNFAITHFUL in gate.reason_codes


def test_correct_baseline_entrypoint_passes():
    gate = baseline_entrypoint_gate(
        baseline_kind="dense_optimal_constant_root_rloo", frozen_envelope="dense_optimal_constant_root_rloo"
    )
    assert gate.status == "pass"


def test_cost_gate_matched_budget_cycle_counts():
    gate = cost_gate(
        candidate_cost=_branch_cost_spec(),
        candidate_actual_cycle_cost=Fraction(27, 1),
        baseline_cost=_dense_cost_spec(),
        baseline_actual_cycle_cost=Fraction(6, 1),
        budget=512,
    )
    assert gate.status == "pass"
    assert "n=18" in gate.detail  # 512//27
    assert "n=85" in gate.detail  # 512//6


def test_cost_gate_infeasible_budget():
    gate = cost_gate(
        candidate_cost=_branch_cost_spec(),
        candidate_actual_cycle_cost=Fraction(27, 1),
        baseline_cost=_dense_cost_spec(),
        baseline_actual_cycle_cost=Fraction(6, 1),
        budget=10,
    )
    assert gate.status == "fail"
    assert ReasonCode.C001_UNMATCHED_TRANSITION_BUDGET in gate.reason_codes
