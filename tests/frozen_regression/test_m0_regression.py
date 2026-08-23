"""M0 frozen regression: the formal run must keep its expected outcome
structure. Runs the full driver (fresh artifacts each time, no-overwrite
respected via unique dirs)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from credit_auditor import runner
from credit_auditor.experiments import m0 as m0_exp

pytestmark = pytest.mark.full

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def m0_run(tmp_path_factory):
    m0_exp.register()
    out = runner.run(
        protocol_path=ROOT / "configs/protocols/m0_regression_v1.json",
        output_dir=tmp_path_factory.mktemp("m0") / "M0",
        seed_manifests=[ROOT / "configs/seeds/m0_problems.json"],
    )
    return out


def test_m0_integrity_pass(m0_run):
    gd = json.loads((m0_run / "gate_decision.json").read_text(encoding="utf-8"))
    assert gd["experiment_integrity"] == "pass"


def test_m0_claims(m0_run):
    gd = json.loads((m0_run / "gate_decision.json").read_text(encoding="utf-8"))
    statuses = {c["claim_id"]: c["status"] for c in gd["claims"]}
    assert statuses["dense_unbiased_full_gradient"] == "pass"
    assert statuses["propagated_sibling_rejected"] == "pass"
    assert statuses["paired_replay_matched_cost_positive"] == "pass"


def test_m0_all_problems_pass_env_and_oracle(m0_run):
    man = json.loads((m0_run / "run_manifest.json").read_text(encoding="utf-8"))
    assert man["raw_rows_count"] == 12
    from credit_auditor.canonical import read_jsonl_zst

    raw = read_jsonl_zst(m0_run / "raw_rows.jsonl.zst")
    assert len(raw) == 12
    for p in raw:
        assert p["environment_gate"]["status"] == "pass", p["problem_id"]
        assert p["oracle"]["oracle_enumeration_match"] and p["oracle"]["oracle_bellman_match"], p["problem_id"]
        statuses = {e["estimator"]: e["gate_status"] for e in p["estimators"]}
        assert statuses["dense"] == "pass"
        assert statuses["dense_optimal_constant"] == "pass"
        assert statuses["uniform_hh"] == "pass"
        assert statuses["local_sibling_local_estimand"] == "pass"
        assert statuses["local_sibling_as_full"] == "fail"
        assert statuses["propagated_sibling"] == "fail"
        assert statuses["bpo_like"] == "fail"


def test_m0_designed_case_failures_detected(m0_run):
    man = json.loads((m0_run / "run_manifest.json").read_text(encoding="utf-8"))
    by_case = {c["case"]: c for c in man["designed_cases"]}
    assert by_case["outcome_retention"]["gate_status"] == "fail"
    assert by_case["outcome_retention"]["reason_codes"] == ["T005_CLIPPING_SCOPE_MISMATCH"]
    assert by_case["completion_deadline"]["gate_status"] == "fail"
    assert by_case["completion_deadline"]["reason_codes"] == ["T004_CONTINUATION_TARGET_MISMATCH"]
    shared = by_case["shared_logit_predictable_width"]
    assert shared["gate"]["status"] == "fail"


def test_m0_matched_cost_positive(m0_run):
    man = json.loads((m0_run / "run_manifest.json").read_text(encoding="utf-8"))
    case = next(c for c in man["designed_cases"] if c["case"] == "matched_cost_positive")
    assert case["narrow_positive"] is True
    assert case["mechanism_control_passes"] is True
    assert case["paired_sibling"]["bias_sq"] < 1e-20
    assert case["paired_sibling"]["mse_at_budget"] < case["dense"]["mse_at_budget"]
    assert case["uncoupled_control"]["mse_at_budget"] > case["dense"]["mse_at_budget"]


def test_m0_bpo_case_reason_codes(m0_run):
    man = json.loads((m0_run / "run_manifest.json").read_text(encoding="utf-8"))
    case = next(c for c in man["designed_cases"] if c["case"] == "bpo_prefix_propagation")
    statuses = {e["estimator"]: e for e in case["estimators"]}
    assert statuses["propagated_sibling"]["reason_codes"] == ["T003_LOCAL_TO_PREFIX_PROPAGATION"]
    assert statuses["bpo_like"]["reason_codes"] == ["T003_LOCAL_TO_PREFIX_PROPAGATION"]
    assert statuses["dense"]["gate_status"] == "pass"
