"""Optional support-only packs (§13.4 continuation, §13.5 minimal logging):
stable SUPPORT_ONLY verdicts, banners, and honest abstention."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from credit_auditor import runner
from credit_auditor.experiments import continuation as cont_exp
from credit_auditor.experiments import minimal_logging as ml_exp

pytestmark = pytest.mark.full

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def cont_run(tmp_path_factory):
    cont_exp.register()
    return runner.run(
        protocol_path=ROOT / "configs/protocols/continuation_support_only_v1.json",
        output_dir=tmp_path_factory.mktemp("cont") / "CONT",
    )


@pytest.fixture(scope="module")
def ml_run(tmp_path_factory):
    ml_exp.register()
    return runner.run(
        protocol_path=ROOT / "configs/protocols/minimal_logging_teaching_v1.json",
        output_dir=tmp_path_factory.mktemp("ml") / "ML",
    )


def test_continuation_support_only_verdicts(cont_run):
    gd = json.loads((cont_run / "gate_decision.json").read_text(encoding="utf-8"))
    statuses = {c["claim_id"]: c["status"] for c in gd["claims"]}
    assert statuses == {"u1_zero_false_safe_abstention": "support_only", "u2u3_stability_reported": "support_only"}
    assert gd["experiment_integrity"] == "pass"


def test_continuation_abstention_honest(cont_run):
    """Zero false-safe: mixed marginal fibers force abstention; paired-replay
    identifies the replay summaries."""
    res = json.loads((cont_run / "result.json").read_text(encoding="utf-8"))
    assert res["u1"]["marginal"]["mixed_fibers"], "marginal regime must mix signs"
    assert res["u1"]["paired_replay"]["mixed_fibers"] == []
    assert res["u1"]["marginal"]["abstention_correct"] is True


def test_continuation_stability_reported(cont_run):
    res = json.loads((cont_run / "result.json").read_text(encoding="utf-8"))
    u = res["u2u3"]
    assert len(u["q0_values"]) == 3
    assert u["sign_stability"]["Q0"][0] in ("positive", "negative", "zero")
    assert isinstance(u["rank_reversals"], int)
    assert "nonrectangular" in u["box_vs_coupled"]


def test_continuation_banner_present(cont_run):
    text = (cont_run / "REPORT.md").read_text(encoding="utf-8")
    assert "SUPPORT_ONLY" in text
    assert "PARTIAL IDENTIFICATION" in text
    assert "not a new theory" in text


def test_minimal_logging_support_only(ml_run):
    gd = json.loads((ml_run / "gate_decision.json").read_text(encoding="utf-8"))
    assert gd["claims"][0]["status"] == "support_only"
    text = (ml_run / "REPORT.md").read_text(encoding="utf-8")
    assert "NOVELTY STATUS: CLASSICAL DECISION-REDUCT / FD / HITTING-SET EQUIVALENCE" in text
    assert "TEACHING OR TELEMETRY-SCHEMA DIAGNOSTIC ONLY" in text


def test_minimal_logging_counts(ml_run):
    res = json.loads((ml_run / "result.json").read_text(encoding="utf-8"))
    assert res["point_total"] == 65536
    assert res["point_eligible"] > 0
    assert res["sign_eligible"] >= res["point_eligible"]
    assert sum(res["point_min_sizes"].values()) == res["point_eligible"]
    assert sum(res["sign_min_sizes"].values()) == res["sign_eligible"]
    assert res["runtime_cpu_seconds"] > 0


def test_optional_pack_evidence_complete(cont_run, ml_run):
    from credit_auditor.audit.provenance import audit_artifact_dir

    for d in (cont_run, ml_run):
        audit = audit_artifact_dir(d)
        assert audit["integrity"] == "pass", audit["errors"]
