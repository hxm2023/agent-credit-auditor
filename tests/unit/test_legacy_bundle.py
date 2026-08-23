"""Legacy bundle validator tests (design §13.6): structure, hashes, anchor,
mode gating."""

from __future__ import annotations

import json
from pathlib import Path

from credit_auditor.adapters.legacy_bundle import (
    REQUIRED_FILES,
    bundle_root_sha256,
    legacy_mode_for,
    sha256_file,
    validate_legacy_bundle,
)


def _make_bundle(tmp_path: Path) -> Path:
    """Synthetic structurally-complete bundle with matching SHA256SUMS."""
    bundle = tmp_path / "bundle"
    for rel in REQUIRED_FILES:
        p = bundle / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"content:{rel}", encoding="utf-8")
    manifest = {
        "schema_version": "legacy-bundle-manifest-1.0",
        "protocol_hashes": {
            "d002_preimplementation": "a" * 64,
            "d002_calibration_seeds": "b" * 64,
            "d002_test_seeds": "c" * 64,
            "d002_superseding_protocol": "d" * 64,
            "credit_transport_protocol": "e" * 64,
            "minimal_logging_protocol": "f" * 64,
        },
        "files": REQUIRED_FILES,
    }
    (bundle / "legacy_bundle_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    listed = sorted((set(REQUIRED_FILES) | {"legacy_bundle_manifest.json"}) - {"SHA256SUMS"})
    sums = "\n".join(f"{sha256_file(bundle / rel)}  {rel}" for rel in listed)
    (bundle / "SHA256SUMS").write_text(sums + "\n", encoding="utf-8")
    return bundle


def test_structural_bundle_unanchored(tmp_path):
    """Without the out-of-band anchor the bundle is UNANCHORED, never VALID."""
    bundle = _make_bundle(tmp_path)
    out = validate_legacy_bundle(bundle)
    assert out["status"] == "UNANCHORED"
    assert out["structural_ok"] is True


def test_anchored_valid_bundle(tmp_path):
    bundle = _make_bundle(tmp_path)
    anchor = bundle_root_sha256(bundle)
    out = validate_legacy_bundle(bundle, anchor_root_sha256=anchor)
    assert out["status"] == "VALID"


def test_tampered_file_rejected(tmp_path):
    bundle = _make_bundle(tmp_path)
    (bundle / "src/credit_v2/finite_mdp.py").write_text("tampered", encoding="utf-8")
    anchor = bundle_root_sha256(bundle)
    out = validate_legacy_bundle(bundle, anchor_root_sha256=anchor)
    assert out["status"] == "REJECT"
    assert any("SHA256SUMS" in r for r in out["reasons"])


def test_wrong_anchor_rejected(tmp_path):
    bundle = _make_bundle(tmp_path)
    out = validate_legacy_bundle(bundle, anchor_root_sha256="f" * 64)
    assert out["status"] == "REJECT"
    assert any("anchor" in r for r in out["reasons"])


def test_missing_files_rejected(tmp_path):
    bundle = _make_bundle(tmp_path)
    (bundle / "configs/d002_protocol_20260821_061930.json").unlink()
    out = validate_legacy_bundle(bundle)
    assert out["status"] == "UNANCHORED"
    assert out["structural_ok"] is False
    assert any("missing" in r for r in out["structural_reasons"])


def test_manifest_missing_hash_keys(tmp_path):
    bundle = _make_bundle(tmp_path)
    manifest = json.loads((bundle / "legacy_bundle_manifest.json").read_text(encoding="utf-8"))
    del manifest["protocol_hashes"]["d002_superseding_protocol"]
    (bundle / "legacy_bundle_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    out = validate_legacy_bundle(bundle)
    assert out["structural_ok"] is False
    assert any("protocol hash keys" in r for r in out["structural_reasons"])


def test_mode_gating_fails_closed():
    assert legacy_mode_for("docs_only_semantic", bundle_valid=True) == "docs_only_semantic"
    assert legacy_mode_for("legacy_exact", bundle_valid=False) == "docs_only_semantic"
    assert legacy_mode_for("legacy_exact", bundle_valid=True) == "legacy_exact"


def test_root_hash_deterministic(tmp_path):
    bundle = _make_bundle(tmp_path)
    assert bundle_root_sha256(bundle) == bundle_root_sha256(bundle)
