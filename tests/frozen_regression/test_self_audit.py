"""Self-audit frozen regression: all fault types must keep TPR=1.0, FPR=0.0."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from credit_auditor import runner
from credit_auditor.experiments import self_audit as sa

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.full


@pytest.fixture(scope="module")
def self_audit_run(tmp_path_factory):
    sa.register()
    return runner.run(
        protocol_path=ROOT / "configs/protocols/self_audit_v1.json",
        output_dir=tmp_path_factory.mktemp("selfaudit") / "SELFAUDIT",
    )


def test_self_audit_all_tpr_one(self_audit_run):
    res = json.loads((self_audit_run / "result.json").read_text(encoding="utf-8"))
    assert len(res["rows"]) == 13
    for r in res["rows"]:
        assert r["tpr"] == 1.0, r["fault"]
        assert r["fpr"] == 0.0, r["fault"]


def test_self_audit_claim_pass(self_audit_run):
    gd = json.loads((self_audit_run / "gate_decision.json").read_text(encoding="utf-8"))
    assert gd["claims"][0]["status"] == "pass"
    assert gd["experiment_integrity"] == "pass"


def test_self_audit_sample_sizes(self_audit_run):
    res = json.loads((self_audit_run / "result.json").read_text(encoding="utf-8"))
    heavy = {r["fault"]: r["n_fault"] for r in res["rows"] if r["n_fault"] < 200}
    assert set(heavy) == {"A7_SPLIT_OVERLAP", "A10_ORACLE_IMPORT", "A12_EVIDENCE_MISSING", "A13_OVERWRITE"}
    assert all(n == 30 for n in heavy.values())


def test_self_audit_wilson_ci_sane(self_audit_run):
    res = json.loads((self_audit_run / "result.json").read_text(encoding="utf-8"))
    for r in res["rows"]:
        lo, hi = r["tpr_ci"]
        assert 0.0 <= lo <= hi <= 1.0
