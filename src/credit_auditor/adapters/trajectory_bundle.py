"""Trajectory bundle adapter (v0.2-prep, Stage 1 of the real-trajectory bridge).

Converts a directory of trajectory records into a hash-anchored bundle that
follows the Auditor's evidence discipline: every record file is referenced by
sha256 only, the bundle schema is pinned (`aca-trajectory-bundle-1.0`), and
validation fails closed on unknown schema majors or missing/unmatched hashes.

Real Guard trajectories keep flowing through the envelope adapter
(guard_integration.py, schema pinned to the Guard repo, design §25). This
adapter is the Auditor's own record format for frozen trajectory fixtures and
for the trajectory-level audit's output side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from credit_auditor.canonical import sha256_file

BUNDLE_SCHEMA = "aca-trajectory-bundle-1.0"


@dataclass
class TrajectoryRecordRef:
    uri: str
    sha256: str


@dataclass
class TrajectoryBundle:
    schema_version: str
    record_refs: list[TrajectoryRecordRef] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "records": [{"uri": r.uri, "sha256": r.sha256} for r in self.record_refs],
        }


def trajectory_to_bundle(data_dir: Path, schema: str = BUNDLE_SCHEMA) -> TrajectoryBundle:
    """Hash every *.jsonl record file under data_dir into a bundle."""
    data_dir = Path(data_dir)
    refs = []
    for path in sorted(data_dir.rglob("*.jsonl")):
        refs.append(
            TrajectoryRecordRef(uri=f"trajectory://{path.relative_to(data_dir).as_posix()}", sha256=sha256_file(path))
        )
    return TrajectoryBundle(schema_version=schema, record_refs=refs)


def validate_trajectory_bundle(bundle: dict, data_dir: Path) -> dict:
    """Fail-closed validation: pinned schema + every ref present and hashed."""
    if bundle.get("schema_version") != BUNDLE_SCHEMA:
        return {"status": "REJECT", "reason": f"unknown schema {bundle.get('schema_version')!r}"}
    records = bundle.get("records", [])
    missing: list[str] = []
    mismatched: list[str] = []
    for rec in records:
        uri = rec.get("uri", "")
        sha = rec.get("sha256")
        if not isinstance(sha, str) or len(sha) != 64:
            missing.append(uri)
            continue
        rel = uri.removeprefix("trajectory://")
        path = data_dir / rel
        if not path.is_file():
            missing.append(uri)
            continue
        if sha256_file(path) != sha:
            mismatched.append(uri)
    reasons = []
    if missing:
        reasons.append(f"missing or unhashable refs: {missing[:5]} ({len(missing)} total)")
    if mismatched:
        reasons.append(f"hash mismatch: {mismatched[:5]} ({len(mismatched)} total)")
    if not records:
        reasons.append("bundle carries no record refs")
    return {
        "status": "ALLOW" if not reasons else "REJECT",
        "records": len(records),
        "reasons": reasons,
    }
