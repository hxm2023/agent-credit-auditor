"""Auditor's own fault-injection matrix, A1-A4 (§14) — the Auditor must
detect each injected error with the fixed reason code."""
from __future__ import annotations

import numpy as np

from credit_auditor.audit.sampling import correction_gate, support_gate
from credit_auditor.audit.target import target_gate
from credit_auditor.estimators import bpo_like, hh_ht, sibling
from credit_auditor.schema import Correction, ReasonCode, SamplingSpec
from credit_auditor.stats import exact_moments
from credit_auditor.worlds.bernoulli_sequence import deterministic_world

TOL = {"bias_rel": 1e-9, "bias_abs": 1e-12, "near_zero_target_abs": 1e-8}


def test_A1_local_target_labeled_as_full_gradient():
    """A1: estimator whose target is local is labeled full gradient -> T002."""
    world = deterministic_world(seed=201, horizon=4)
    t = 2
    target_full = world.true_gradient()
    dist = sibling.local_sibling_distribution(world, t)
    m = exact_moments(dist, target_full)
    gate = target_gate(
        m, TOL, sibling.mechanism_signature_local(),
        claimed_estimand="full_score_gradient", estimand_id="full_score_gradient",
    )
    assert gate.status == "fail"
    assert ReasonCode.T002_BIAS_EXCEEDS_TOLERANCE in gate.reason_codes


def test_A2_sibling_credit_propagated_to_shared_prefix():
    """A2: sibling contrast propagated to prefix -> T003."""
    world = deterministic_world(seed=202, horizon=4)
    t = 1
    m = exact_moments(sibling.propagated_sibling_distribution(world, t), world.true_gradient())
    gate = target_gate(
        m, TOL, sibling.mechanism_signature_propagated(),
        claimed_estimand="full_score_gradient", estimand_id="full_score_gradient",
    )
    assert gate.status == "fail"
    assert ReasonCode.T003_LOCAL_TO_PREFIX_PROPAGATION in gate.reason_codes


def test_A3_zero_support_q():
    """A3: adaptive q with a zeroed time step -> S001."""
    world = deterministic_world(seed=203, horizon=4)
    target = world.true_gradient()
    q = (0.0, 1.0 / 3, 1.0 / 3, 1.0 / 3)
    gate = support_gate(q, target, min_support=1e-6)
    assert gate.status == "fail"
    assert ReasonCode.S001_ZERO_SUPPORT in gate.reason_codes


def test_A4_wr_sampling_with_ht_correction():
    """A4: with-replacement sampling using HT inclusion correction -> S003."""
    spec = SamplingSpec(
        decision_sampling={"replacement": "with_replacement"},
        correction=Correction(name="horvitz_thompson", version="v1"),
    )
    gate = correction_gate(spec)
    assert gate.status == "fail"
    assert ReasonCode.S003_WRONG_HH_HT_CORRECTION in gate.reason_codes


def test_A4b_wor_sampling_with_hh_correction():
    spec = SamplingSpec(
        decision_sampling={"replacement": "without_replacement"},
        correction=Correction(name="hansen_hurwitz", version="v1"),
    )
    gate = correction_gate(spec)
    assert gate.status == "fail"
    assert ReasonCode.S003_WRONG_HH_HT_CORRECTION in gate.reason_codes


def test_A4c_correct_pairs_pass():
    for law, corr in (("with_replacement", "hansen_hurwitz"), ("without_replacement", "horvitz_thompson")):
        spec = SamplingSpec(decision_sampling={"replacement": law}, correction=Correction(name=corr))
        assert correction_gate(spec).status == "pass"


def test_q_not_logged():
    gate = correction_gate(SamplingSpec(decision_sampling={"replacement": "with_replacement"}), q_logged=False)
    assert gate.status == "fail"
    assert ReasonCode.S002_Q_NOT_LOGGED in gate.reason_codes


def test_bpo_like_is_biased_and_detected():
    """BPO-like literal port must be detected as biased (T003 propagation)."""
    world = deterministic_world(seed=204, horizon=4)
    m = exact_moments(bpo_like.bpo_like_distribution(world), world.true_gradient())
    gate = target_gate(
        m, TOL, bpo_like.mechanism_signature(),
        claimed_estimand="full_score_gradient", estimand_id="full_score_gradient",
    )
    assert gate.status == "fail"
    assert gate.reason_codes == [ReasonCode.T003_LOCAL_TO_PREFIX_PROPAGATION]
    assert m.bias_sq > 1e-6


def test_hh_unbiased_with_full_support():
    world = deterministic_world(seed=205, horizon=4)
    m = exact_moments(hh_ht.uniform_hh_distribution(world), world.true_gradient())
    assert m.max_abs_bias() < 1e-9
    assert m.bias_sq < 1e-12
