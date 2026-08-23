"""Legacy migration bundle validator (design §13.6, §22) — legacy_exact
readiness.

docs_only_semantic is the only authorized mode until a signed migration
bundle arrives. This validator makes the upgrade path executable: it checks
the bundle's required structure (configs, src/credit_v2, scripts, formal
logs, manifest, SHA256SUMS), verifies content hashes, and requires the
bundle's root digest to equal a trust anchor delivered out-of-band (design
doc update / signed release / trusted channel). A bundle carrying only its
own SHA256SUMS is NOT self-anchored (§13.6).

The validator itself is fully testable with synthetic bundles; it does not
depend on the real legacy code existing anywhere.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

# §13.6 required bundle layout (relative paths)
REQUIRED_FILES = [
    "configs/v001_phase_diagram_20260821_042226.json",
    "configs/d002_protocol_20260821_061930.json",
    "configs/d002_sanity_seeds_20260821_061130.json",
    "configs/d002_calibration_seeds_20260821_061130.json",
    "configs/d002_test_seeds_20260821_061130.json",
    "src/credit_v2/finite_mdp.py",
    "src/credit_v2/root_marginal.py",
    "src/credit_v2/phase_diagram.py",
    "src/credit_v2/d002_mdp.py",
    "src/credit_v2/d002_experiment.py",
    "src/credit_v2/d002_oracle.py",
    "src/credit_v2/independent_oracle.py",
    "src/credit_v2/independent_root_oracle.py",
    "scripts/run_d002.py",
    "deep-experiment-logs/M0_FORMAL_VALIDATION/G001_G002_attempt_04/exact_audit.json",
    "deep-experiment-logs/M0_FORMAL_VALIDATION/V001_attempt_01/REPORT.md",
    "deep-experiment-logs/M0_FORMAL_VALIDATION/V001_attempt_01/phase_diagram.csv",
    "deep-experiment-logs/M0_FORMAL_VALIDATION/V001_attempt_01/phase_diagram.json",
    "deep-experiment-logs/M0_FORMAL_VALIDATION/RMTPG_D002_calibration_attempt_01/calibration.json",
    "deep-experiment-logs/M0_FORMAL_VALIDATION/RMTPG_D002_calibration_attempt_01/frozen_mapping.json",
    "deep-experiment-logs/M0_FORMAL_VALIDATION/RMTPG_D002_test_attempt_01/REPORT.md",
    "deep-experiment-logs/M0_FORMAL_VALIDATION/RMTPG_D002_test_attempt_01/test_results.json",
    "deep-experiment-logs/M0_FORMAL_VALIDATION/RMTPG_D002_test_attempt_01/test_rows.csv",
    "legacy_bundle_manifest.json",
    "SHA256SUMS",
]

# §22.6 historical protocol hashes the manifest should declare (informational:
# the validator checks the manifest declares them, not the values themselves —
# a real bundle's manifest is the authoritative source).
EXPECTED_PROTOCOL_HASH_KEYS = [
    "d002_preimplementation",
    "d002_calibration_seeds",
    "d002_test_seeds",
    "d002_superseding_protocol",
    "credit_transport_protocol",
    "minimal_logging_protocol",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bundle_root_sha256(bundle_dir: Path) -> str:
    """Root digest of the bundle tree (all files under it, sorted paths)."""
    h = hashlib.sha256()
    for rel in sorted(p.relative_to(bundle_dir).as_posix() for p in bundle_dir.rglob("*") if p.is_file()):
        h.update(rel.encode("utf-8"))
        h.update(b"\x00")
        h.update(sha256_file(bundle_dir / rel).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def validate_legacy_bundle(bundle_dir: Path, anchor_root_sha256: str | None = None) -> dict:
    """Validate a legacy migration bundle (§13.6).

    Returns {"status": "VALID", ...} or {"status": "REJECT", "reasons": [...]}.
    The anchor is the out-of-band root digest; without it the bundle is NOT
    self-anchored and can only be reported as structurally sound, never as
    trusted for legacy_exact mode.
    """
    reasons: list[str] = []
    bundle_dir = Path(bundle_dir)
    if not bundle_dir.is_dir():
        return {"status": "REJECT", "reasons": [f"bundle dir not found: {bundle_dir}"]}

    # 1. required files present
    missing = [r for r in REQUIRED_FILES if not (bundle_dir / r).is_file()]
    if missing:
        reasons.append(f"missing required files: {missing[:5]} ... ({len(missing)} total)")

    # 2. SHA256SUMS matches content (the sums file itself is never listed)
    sums_path = bundle_dir / "SHA256SUMS"
    if sums_path.is_file():
        declared = {}
        for line in sums_path.read_text(encoding="utf-8").splitlines():
            h, _, rel = line.partition("  ")
            declared[rel] = h
        mismatches = [
            rel
            for rel, h in declared.items()
            if rel != "SHA256SUMS" and (not (bundle_dir / rel).is_file() or sha256_file(bundle_dir / rel) != h)
        ]
        if mismatches:
            reasons.append(f"SHA256SUMS mismatches: {mismatches[:5]} ... ({len(mismatches)} total)")
    else:
        reasons.append("missing SHA256SUMS")

    # 3. manifest structure
    manifest_path = bundle_dir / "legacy_bundle_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            missing_keys = [k for k in EXPECTED_PROTOCOL_HASH_KEYS if k not in manifest.get("protocol_hashes", {})]
            if missing_keys:
                reasons.append(f"manifest missing protocol hash keys: {missing_keys}")
            if not isinstance(manifest.get("files"), list):
                reasons.append("manifest missing 'files' list")
        except json.JSONDecodeError:
            reasons.append("legacy_bundle_manifest.json is not valid JSON")
    else:
        reasons.append("missing legacy_bundle_manifest.json")

    # 4. trust anchor
    root = bundle_root_sha256(bundle_dir)
    if anchor_root_sha256 is None:
        return {
            "status": "UNANCHORED",
            "reasons": ["no out-of-band trust anchor provided; the bundle is NOT self-anchored (§13.6)"],
            "root_sha256": root,
            "structural_ok": not reasons,
            "structural_reasons": reasons,
        }
    if root != anchor_root_sha256:
        reasons.append(f"root sha256 {root[:16]}... != anchor {anchor_root_sha256[:16]}...")
    if reasons:
        return {"status": "REJECT", "reasons": reasons, "root_sha256": root}
    return {
        "status": "VALID",
        "root_sha256": root,
        "protocol_hashes": json.loads(manifest_path.read_text(encoding="utf-8")).get("protocol_hashes", {}),
    }


def legacy_mode_for(protocol_reconstruction_mode: str, bundle_valid: bool) -> str:
    """Mode gating (§13.6): legacy_exact only when the protocol asks for it AND
    a valid anchored bundle is present."""
    if protocol_reconstruction_mode == "docs_only_semantic":
        return "docs_only_semantic"
    if protocol_reconstruction_mode == "legacy_exact" and bundle_valid:
        return "legacy_exact"
    return "docs_only_semantic"  # fail closed: no bundle, no legacy_exact
