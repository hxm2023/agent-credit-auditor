"""D002 frozen regression: the dual verdict (metric PASS + mechanism FAIL)
must be stable; calibration/test splits are frozen and disjoint."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from credit_auditor import runner
from credit_auditor.experiments import d002 as d002_exp

pytestmark = pytest.mark.full

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def d002_pipeline(tmp_path_factory):
    d002_exp.register()
    tmp = tmp_path_factory.mktemp("d002")
    cal = runner.run(
        protocol_path=ROOT / "configs/protocols/d002_regression_v1.json",
        output_dir=tmp / "D002_cal",
        phase="calibration",
        seed_manifests=[ROOT / "configs/seeds/d002_calibration.json"],
    )
    test = runner.run(
        protocol_path=ROOT / "configs/protocols/d002_regression_v1.json",
        output_dir=tmp / "D002_test",
        phase="test",
        frozen_selection=cal / "selection.json",
        seed_manifests=[ROOT / "configs/seeds/d002_test.json"],
    )
    return cal, test


def test_d002_dual_verdict(d002_pipeline):
    _, test = d002_pipeline
    gd = json.loads((test / "gate_decision.json").read_text(encoding="utf-8"))
    statuses = {c["claim_id"]: c["status"] for c in gd["claims"]}
    assert statuses["global_k8_efficiency"] == "pass"
    assert statuses["variable_width_adaptivity"] == "fail"
    assert gd["experiment_integrity"] == "pass"
    assert gd["headline_decision"]["proposed_new_method_claim"] == "fail"
    assert gd["headline_decision"]["retained_narrow_claim"] == "global_k8_efficiency"


def test_d002_widths_collapse(d002_pipeline):
    cal, test = d002_pipeline
    selection = json.loads((cal / "selection.json").read_text(encoding="utf-8"))
    widths = [v[1] for v in selection["selected_mapping"].values()]
    assert len(set(widths)) == 1, "widths must collapse (mechanism-fail demonstration)"


def test_d002_metric_strong(d002_pipeline):
    _, test = d002_pipeline
    res = json.loads((test / "result.json").read_text(encoding="utf-8"))
    boot = res["bootstrap"]
    assert boot["median"] < 0.8
    assert boot["ci_hi"] < 1.0
    assert boot["ci_lo"] > 0.0


def test_d002_oracle_alignment(d002_pipeline):
    _, test = d002_pipeline
    res = json.loads((test / "result.json").read_text(encoding="utf-8"))
    for row in res["problems"]:
        assert row["oracle_max_mismatch"] < 1e-9
        assert row["mapping_bias_sq"] < 1e-9


def test_d002_test_refuses_unfrozen_selection(tmp_path):
    """A8: the test phase without a frozen selection fails closed with an
    evidence package (exit_status driver_failed), never a silent run."""
    d002_exp.register()
    out = runner.run(
        protocol_path=ROOT / "configs/protocols/d002_regression_v1.json",
        output_dir=tmp_path / "D002_test_noselection",
        phase="test",
        seed_manifests=[ROOT / "configs/seeds/d002_test.json"],
    )
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["exit_status"] == "driver_failed"


def test_d002_tampered_selection_rejected(tmp_path):
    """A8 lineage: an edited selection (tampered widths) must be rejected by
    the self-hash check, not silently used."""
    import copy

    d002_exp.register()
    cal = runner.run(
        protocol_path=ROOT / "configs/protocols/d002_regression_v1.json",
        output_dir=tmp_path / "D002_cal",
        phase="calibration",
        seed_manifests=[ROOT / "configs/seeds/d002_calibration.json"],
    )
    sel = json.loads((cal / "selection.json").read_text(encoding="utf-8"))
    tampered = copy.deepcopy(sel)
    first_bucket = next(iter(tampered["selected_mapping"]))
    tampered["selected_mapping"][first_bucket][1] = 8  # edit a width
    tampered_path = tmp_path / "tampered_selection.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    out = runner.run(
        protocol_path=ROOT / "configs/protocols/d002_regression_v1.json",
        output_dir=tmp_path / "D002_test_tampered",
        phase="test",
        frozen_selection=tampered_path,
        seed_manifests=[ROOT / "configs/seeds/d002_test.json"],
    )
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["exit_status"] == "driver_failed"
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert "content hash mismatch" in result.get("error", "")


def test_d002_test_manifest_records_parent_selection_hash(d002_pipeline):
    _, test = d002_pipeline
    manifest = json.loads((test / "run_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest.get("parent_calibration_selection_sha256", "")) == 64


def test_d002_selection_hash_stable(d002_pipeline):
    """The frozen selection must be content-hashed and versioned (A8 lineage)."""
    cal, _ = d002_pipeline
    sel = json.loads((cal / "selection.json").read_text(encoding="utf-8"))
    assert len(sel["selection_sha256"]) == 64
    # re-hash the published selection (canonical serialization) and compare
    from credit_auditor.canonical import sha256_json

    body = {k: v for k, v in sel.items() if k != "selection_sha256"}
    assert sha256_json(body) == sel["selection_sha256"]
