"""Faults A10, A12, A13, A14 (§14): oracle independence, evidence
completeness, no-overwrite, numerical margin."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from credit_auditor.audit.numerical import sign_of, sign_reversal_gate
from credit_auditor.audit.provenance import audit_artifact_dir
from credit_auditor.canonical import NoOverwriteError, atomic_write_json
from credit_auditor.oracles.isolation import check_import_isolation, check_process_import_isolation
from credit_auditor.schema import ReasonCode

ROOT = Path(__file__).resolve().parents[2]
ORACLE_DIR = ROOT / "src" / "credit_auditor" / "oracles"


def test_A10_oracle_importing_estimator_detected(tmp_path):
    """A10: an oracle that imports estimator helpers violates independence."""
    evil = tmp_path / "evil_oracle.py"
    evil.write_text(
        "import json, sys\n"
        "from credit_auditor.estimators import dense  # forbidden\n"
        "def main():\n"
        "    spec = json.load(sys.stdin)\n"
        "    sys.stdout.write(json.dumps({'gradient': [0.0], 'oracle': 'evil', 'input_sha256': 'x', 'precision': 'float64'}))\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )
    bad = check_import_isolation(evil)
    assert bad == ["credit_auditor.estimators"]
    # process-level check also fails
    res = check_process_import_isolation(evil)
    assert res["isolated"] is False


def test_A12_missing_raw_results_detected(tmp_path):
    """A12: only a report without raw result/manifest -> P001."""
    d = tmp_path / "exp"
    d.mkdir()
    (d / "REPORT.md").write_text("# only a report\n", encoding="utf-8")
    audit = audit_artifact_dir(d)
    assert audit["integrity"] == "fail"
    assert any("missing" in e for e in audit["errors"])


def test_A13_canonical_output_overwrite_refused(tmp_path):
    """A13: a second write to a canonical path must raise, never overwrite.
    The runner calls refuse_existing BEFORE any write; atomic_write_json is
    only used after that gate passes."""
    p = tmp_path / "canonical.json"
    atomic_write_json({"v": 1}, p)
    from credit_auditor.canonical import refuse_existing

    with pytest.raises(NoOverwriteError):
        refuse_existing(p)
    # the original content is intact (no silent overwrite)
    assert json.loads(p.read_text(encoding="utf-8")) == {"v": 1}


def test_A14_near_zero_sign_not_counted():
    """A14: a 1e-16 float must not be claimed as a sign reversal."""
    assert sign_of(1e-16, margin=1e-8) == "near_zero"
    assert sign_of(-1e-16, margin=1e-8) == "near_zero"
    assert sign_of(0.05, margin=1e-8) == "positive"
    gate = sign_reversal_gate(1e-16, -1e-16, margin=1e-8)
    assert gate.status == "fail"
    assert ReasonCode.N001_NEAR_ZERO_SIGN in gate.reason_codes


def test_A14b_real_reversal_with_margin_passes():
    gate = sign_reversal_gate(0.123, -0.045, margin=1e-8)
    assert gate.status == "pass"


def test_A14c_no_reversal_fails():
    gate = sign_reversal_gate(0.123, 0.045, margin=1e-8)
    assert gate.status == "fail"


def test_legacy_oracle_imports_clean():
    """The shipped oracles stay import-clean (regression guard for A10)."""
    for script in ("enumeration_oracle.py", "bellman_oracle.py"):
        assert check_import_isolation(ORACLE_DIR / script) == []
