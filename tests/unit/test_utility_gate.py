"""Utility-gate boundary tests (P0-4, GPT review): with ratio =
candidate/baseline, the correct condition for "improvement CI lower bound >
0" is ratio_ci_hi < 1 — a CI that crosses 1 must FAIL."""

from __future__ import annotations

import numpy as np

from credit_auditor.experiments.v001 import _bootstrap_median_improvement


def _gate_verdict(ratios: list[float], seed_key: str) -> dict:
    """Replicate the V001 gate decision on a synthetic ratio vector."""
    boot = _bootstrap_median_improvement(np.asarray(ratios), seed_key, replicates=2000)
    passed = boot["median"] <= 0.8 and boot["ci_hi"] < 1.0
    return {"passed": passed, **boot}


def test_ci_crossing_one_fails():
    """A ratio distribution whose CI crosses 1 is an uncertain result: the
    gate must fail. (The median alone would mislead — the bimodal case has a
    middle-median of exactly 1, and the crossing-CI case below fails.)"""
    # bimodal: half clearly better (0.5), half clearly worse (1.5)
    ratios = [0.5] * 100 + [1.5] * 100
    v = _gate_verdict(ratios, "ACA-SEM-TEST-CROSS")
    assert v["ci_lo"] < 1.0 < v["ci_hi"]  # the CI provably crosses 1
    assert v["passed"] is False  # crossing CI must fail the gate


def test_ci_strictly_below_one_passes():
    ratios = [0.6] * 200
    v = _gate_verdict(ratios, "ACA-SEM-TEST-OK")
    assert v["passed"] is True
    assert v["ci_hi"] < 1.0


def test_median_above_threshold_fails():
    ratios = [0.9] * 200
    v = _gate_verdict(ratios, "ACA-SEM-TEST-ABOVE")
    assert v["passed"] is False


def test_gate_is_deterministic():
    a = _gate_verdict([0.6] * 200, "ACA-SEM-TEST-DET")
    b = _gate_verdict([0.6] * 200, "ACA-SEM-TEST-DET")
    assert a["passed"] == b["passed"]
    assert a["ci_hi"] == b["ci_hi"]
