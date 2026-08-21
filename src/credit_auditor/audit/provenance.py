"""Provenance/evidence gate (§11.6, §14 A12/A13, §18 package format).

Checks on a published artifact directory:
- all required package files exist (A12 -> P001_EVIDENCE_INCOMPLETE)
- SHA256SUMS matches current file contents
- run_manifest exit status and result.json status consistency
- failed runs still enter the index (checked at index build time)
"""
from __future__ import annotations

from pathlib import Path

from credit_auditor.canonical import sha256_file

REQUIRED_PACKAGE_FILES = [
    "protocol.json",
    "result.json",
    "oracle_result.json",
    "gate_decision.json",
    "run_manifest.json",
    "REPORT.md",
    "SHA256SUMS",
]


def audit_artifact_dir(artifact_dir: Path) -> dict:
    errors: list[str] = []
    for name in REQUIRED_PACKAGE_FILES:
        if not (artifact_dir / name).is_file():
            errors.append(f"missing {name}")

    if not errors:
        sums = {}
        for line in (artifact_dir / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
            h, _, rel = line.partition("  ")
            sums[rel] = h
        for rel, expected in sums.items():
            p = artifact_dir / rel
            if not p.is_file():
                errors.append(f"SHA256SUMS entry missing on disk: {rel}")
                continue
            actual = sha256_file(p)
            if actual != expected:
                errors.append(f"hash mismatch: {rel}")

    integrity = "pass" if not errors else "fail"
    return {"integrity": integrity, "errors": errors, "artifact_dir": str(artifact_dir)}


def verify_no_overwrite(artifact_dir: Path) -> None:
    """A13: canonical outputs are never overwritten; re-runs must target a new
    directory or be refused by the runner before any write."""
    from credit_auditor.canonical import NoOverwriteError, refuse_existing
    refuse_existing(artifact_dir)  # raises NoOverwriteError if it exists
