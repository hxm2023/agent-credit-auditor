"""Schema unit tests (§7)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from credit_auditor.schema import (
    AuditDecision,
    ClaimCeiling,
    ClaimDecision,
    ClaimStatus,
    CostBreakdown,
    CostTerm,
    EstimandSpec,
    EstimatorSpec,
    GateResult,
    HeadlineDecision,
    ReasonCode,
    SamplingSpec,
)


def test_estimand_spec_roundtrip():
    s = EstimandSpec(
        estimand_id="full_score_gradient",
        world_family="bernoulli_sequence_mdp",
        policy_parameterization="independent_logits",
        reward_semantics="terminal",
    )
    assert s.discount == 1.0
    assert s.clipping == "none"


def test_sampling_spec_defaults():
    s = SamplingSpec(decision_sampling={"replacement": "with_replacement"})
    assert s.restore.state_identity == "exact"
    assert s.correction.name == "none"
    assert s.continuation.samples_per_branch == 1


def test_sampling_spec_rejects_unknown_replacement():
    with pytest.raises(ValidationError):
        SamplingSpec(decision_sampling={"replacement": "sometimes"})


def test_estimator_spec_roundtrip():
    s = EstimatorSpec(estimator_id="hh_local_sibling", claimed_estimand="full_score_gradient")
    assert s.version == "v1"
    assert s.required_assumptions == []


def test_claim_decision_roundtrip():
    c = ClaimDecision(
        claim_id="global_k8_efficiency",
        claim_text="global K=8 improves finite-MDP fixed-budget MSE under protocol X",
        status=ClaimStatus.PASS,
        required_gates=["integrity", "matched_cost", "utility"],
        gate_results=[GateResult(gate="utility", status="pass")],
        reason_codes=[ReasonCode.U001_PRIMARY_THRESHOLD_MET],
        claim_ceiling=ClaimCeiling(
            allowed=["fixed-width synthetic efficiency"],
            forbidden=["adaptive variable-width credit assignment"],
        ),
    )
    assert c.status == ClaimStatus.PASS
    assert c.claim_ceiling.forbidden == ["adaptive variable-width credit assignment"]


def test_integrity_invalid_dominates_all_claims():
    d = AuditDecision(
        experiment_integrity=ClaimStatus.INVALID,
        claims=[
            ClaimDecision(claim_id="a", claim_text="x", status=ClaimStatus.PASS),
            ClaimDecision(claim_id="b", claim_text="y", status=ClaimStatus.FAIL),
        ],
        headline_decision=HeadlineDecision(proposed_new_method_claim=ClaimStatus.INVALID),
    )
    assert all(c.status == ClaimStatus.INVALID for c in d.claims)
    assert d.headline_decision.proposed_new_method_claim == ClaimStatus.INVALID


def test_strong_claim_failure_keeps_narrow_claim():
    """§7.5: stronger claim failing does not erase a pre-defined narrow claim."""
    d = AuditDecision(
        experiment_integrity=ClaimStatus.PASS,
        claims=[
            ClaimDecision(claim_id="global_k8_efficiency", claim_text="x", status=ClaimStatus.PASS),
            ClaimDecision(claim_id="variable_width_adaptivity", claim_text="y", status=ClaimStatus.FAIL),
        ],
        headline_decision=HeadlineDecision(
            proposed_new_method_claim=ClaimStatus.FAIL, retained_narrow_claim="global_k8_efficiency"
        ),
    )
    statuses = {c.claim_id: c.status for c in d.claims}
    assert statuses == {"global_k8_efficiency": ClaimStatus.PASS, "variable_width_adaptivity": ClaimStatus.FAIL}


def test_cost_breakdown_rejects_total_not_equal_sum():
    with pytest.raises(ValidationError):
        CostBreakdown(
            primary_unit="environment_transition",
            terms=[CostTerm(term_id="a", quantity="1/1", unit_cost="1/1", subtotal="1/1")],
            total="2/1",
        )


def test_cost_breakdown_accepts_decomposable_total():
    b = CostBreakdown(
        primary_unit="environment_transition",
        terms=[CostTerm(term_id="a", quantity="1/1", unit_cost="1/1", subtotal="1/1")],
        total="1/1",
    )
    assert b.total_fraction().numerator == 1
