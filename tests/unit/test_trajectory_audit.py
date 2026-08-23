"""Trajectory-level audit tests (v0.2-prep real-trajectory bridge): the
offline detectors for mask_shift / misbound_logprob / retokenization /
stale_policy / silent_mask_drift must fire on the frozen fixtures, and the
hash-anchored bundle must fail closed on mutation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from credit_auditor.adapters.trajectory_bundle import (
    BUNDLE_SCHEMA,
    trajectory_to_bundle,
    validate_trajectory_bundle,
)
from credit_auditor.audit.trajectory_audit import audit_trajectory_dir

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/trajectories/clean_trajectories.jsonl"


def _load() -> list[dict]:
    return [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write(tmp_path: Path, name: str, recs: list[dict]) -> Path:
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8", newline="\n")
    return p


def _codes(audit: dict) -> list[str]:
    return sorted({f["code"] for f in audit["findings"]})


def test_clean_fixture_is_consistent():
    a = audit_trajectory_dir(FIXTURE)
    assert a["records"] == 4
    assert a["consistent"] is True


def test_mask_shift_detected(tmp_path):
    recs = _load()
    recs[0]["action_mask"] = recs[0]["action_mask"][:-1]
    p = _write(tmp_path, "f1.jsonl", recs)
    a = audit_trajectory_dir(p)
    assert a["consistent"] is False
    assert "T005_CLIPPING_SCOPE_MISMATCH" in _codes(a)


def test_misbound_logprob_detected(tmp_path):
    recs = _load()
    recs[1]["old_logprobs"] = recs[1]["old_logprobs"] + [-0.9]
    p = _write(tmp_path, "f2.jsonl", recs)
    a = audit_trajectory_dir(p)
    assert "S002_Q_NOT_LOGGED" in _codes(a)


def test_logprob_nan_detected(tmp_path):
    recs = _load()
    recs[0]["old_logprobs"] = [float("nan"), -0.5, -0.4]
    p = _write(tmp_path, "f2b.jsonl", recs)
    a = audit_trajectory_dir(p)
    assert "S002_Q_NOT_LOGGED" in _codes(a)


def test_retokenization_detected(tmp_path):
    recs = _load()
    recs[0]["generated_tokens"] = recs[0]["generated_tokens"][1:]
    p = _write(tmp_path, "f3.jsonl", recs)
    a = audit_trajectory_dir(p)
    assert "T005_CLIPPING_SCOPE_MISMATCH" in _codes(a)


def test_stale_policy_detected(tmp_path):
    recs = _load()
    recs[2]["policy_version"] = ""
    p = _write(tmp_path, "f4.jsonl", recs)
    a = audit_trajectory_dir(p)
    assert "T004_CONTINUATION_TARGET_MISMATCH" in _codes(a)


def test_mixed_policy_batch_detected(tmp_path):
    recs = _load()
    recs[3]["policy_version"] = "v2"
    p = _write(tmp_path, "f5.jsonl", recs)
    a = audit_trajectory_dir(p)
    assert a["batch_finding"] is not None
    assert a["batch_finding"]["code"] == "T004_CONTINUATION_TARGET_MISMATCH"


def test_silent_mask_drift_detected(tmp_path):
    recs = _load()
    recs[0]["optimizer_consumed_mask"] = [0, 1, 1]
    p = _write(tmp_path, "f6.jsonl", recs)
    a = audit_trajectory_dir(p)
    assert "T005_CLIPPING_SCOPE_MISMATCH" in _codes(a)


def test_unparseable_line_detected(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"trajectory_id": "ok"}\nnot-json\n', encoding="utf-8", newline="\n")
    a = audit_trajectory_dir(p)
    assert a["consistent"] is False
    assert "P001_EVIDENCE_INCOMPLETE" in _codes(a)


def test_bundle_anchors_and_rejects_mutation(tmp_path):
    recs = _load()
    for i, r in enumerate(recs):
        _write(tmp_path, f"r{i}.jsonl", [r])
    bundle = trajectory_to_bundle(tmp_path)
    assert bundle.schema_version == BUNDLE_SCHEMA
    v = validate_trajectory_bundle(bundle.to_dict(), tmp_path)
    assert v["status"] == "ALLOW"
    # mutation must fail closed
    (tmp_path / "r0.jsonl").write_text(json.dumps({"trajectory_id": "tampered"}), encoding="utf-8")
    v2 = validate_trajectory_bundle(bundle.to_dict(), tmp_path)
    assert v2["status"] == "REJECT"
    assert any("mismatch" in r for r in v2["reasons"])


def test_bundle_rejects_unknown_schema():
    v = validate_trajectory_bundle({"schema_version": "other-9.9"}, Path("."))
    assert v["status"] == "REJECT"


def test_bundle_rejects_missing_file(tmp_path):
    recs = _load()
    _write(tmp_path, "r0.jsonl", recs)
    bundle = trajectory_to_bundle(tmp_path).to_dict()
    bundle["records"].append({"uri": "trajectory://missing.jsonl", "sha256": "a" * 64})
    v = validate_trajectory_bundle(bundle, tmp_path)
    assert v["status"] == "REJECT"


@pytest.mark.parametrize("line", [1, 2, 3])
def test_cli_audit_trajectories_reports(tmp_path, line):
    """The CLI subcommand surfaces consistency and exits nonzero on faults."""
    from credit_auditor.cli import main

    _write(tmp_path, "x.jsonl", _load())
    rc = main(["audit-trajectories", "--data-dir", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "trajectory_audit_report.md").is_file()
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{}", encoding="utf-8")
    rc2 = main(["audit-trajectories", "--data-dir", str(tmp_path)])
    assert rc2 == 1
