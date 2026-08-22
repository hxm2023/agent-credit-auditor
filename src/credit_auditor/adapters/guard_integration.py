"""GRPO-Guard real-trajectory integration (design §25) — the Auditor consumes
real Guard-issued trajectory envelopes by hash-only references.

The Guard repo (C:/Users/w1828/repos/GRPO-Guard) owns the envelope schema
`grpo-guard-envelope-1.0` (its design doc §schema). The Auditor never imports
the grpo_guard package and never forks its schema: it reads the SERIALIZED
envelope JSON, extracts the hash-bearing references, and builds its own
versioned CreditAuditBundle. Fail-closed rules (§25):
- unknown schema major / unknown required extension -> REJECT
- every referenced artifact must carry a sha256
- the bundle never writes back to Guard artifacts

This module is the "real-trajectory connection": a real GRPO rollout envelope
(Guard-issued, from its frozen test fixtures or a live run) flows through the
Auditor's bundle validation, demonstrating the exact-toy -> real-toolchain
bridge. The online envelope validation itself remains Guard's job; the
Auditor only interprets what the envelope permits (design §25).
"""
from __future__ import annotations

import json
from pathlib import Path

from credit_auditor.adapters.grpo_guard_envelope import (
    GUARD_SCHEMA,
    CreditAuditBundle,
    GuardEnvelopeRef,
    validate_guard_envelope,
)


def _ref(event: dict | None, default_uri: str = "") -> GuardEnvelopeRef | None:
    if not event:
        return None
    sha = event.get("event_sha256") or event.get("sha256")
    if not sha:
        return None
    return GuardEnvelopeRef(uri=event.get("uri") or default_uri, sha256=sha)


def envelope_to_bundle(envelope: dict, pinned_schema: str = GUARD_SCHEMA) -> tuple[CreditAuditBundle, dict]:
    """Build the Auditor's CreditAuditBundle from a real Guard envelope.

    Returns (bundle, validation) where validation is the fail-closed check
    result of validate_guard_envelope. The envelope itself carries no
    schema_version field (Guard's schema is declared externally); the pin is
    applied here, and unknown required_extensions reject the bundle.
    """
    env_sha = envelope.get("envelope_sha256")
    if not isinstance(env_sha, str) or len(env_sha) != 64:
        raise ValueError(f"envelope without a valid envelope_sha256: {envelope.get('envelope_id')}")

    validation = validate_guard_envelope(
        {
            "schema_version": pinned_schema,
            "required_extensions": envelope.get("required_extensions", []) or [],
            "content_sha256": env_sha,
        },
        pinned_schema=pinned_schema,
    )
    if validation["status"] != "ALLOW":
        return None, validation

    bundle = CreditAuditBundle(
        guard_schema_version=pinned_schema,
        guard_envelope_refs=[
            GuardEnvelopeRef(uri=f"guard-envelope:{envelope.get('envelope_id')}", sha256=env_sha)
        ],
        decision_token_spans=[],
        restore_protocol_ref=None,
        branch_event_refs=[],
        continuation_policy_manifest_refs=[_ref(envelope.get("policy_manifest"), "manifest://policy")] if envelope.get("policy_manifest") else [],
        selection_probabilities=None,
        cost_observations={},
        target_policy_scoring_event=_ref(envelope.get("generation_event"), "event://generation"),
    )
    return bundle, validation


def load_real_envelope(path: Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "envelope_sha256" not in data:
        raise ValueError(f"{path} is not a Guard trajectory envelope (missing envelope_sha256)")
    return data


def summarize_bundle(bundle: CreditAuditBundle, envelope_id: str) -> dict:
    return {
        "envelope_id": envelope_id,
        "bundle_schema": bundle.schema_version,
        "guard_schema": bundle.guard_schema_version,
        "guard_envelope_refs": [r.sha256[:12] for r in bundle.guard_envelope_refs],
        "policy_manifest_ref": [r.sha256[:12] for r in bundle.continuation_policy_manifest_refs],
        "target_scoring_event": bundle.target_policy_scoring_event.sha256[:12] if bundle.target_policy_scoring_event else None,
        "selection_law": "strict_on_policy" if bundle.selection_probabilities is None else bundle.selection_probabilities.law,
    }
