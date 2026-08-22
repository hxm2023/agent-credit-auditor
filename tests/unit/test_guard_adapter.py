"""GRPO-Guard envelope adapter tests (design §25): fail-closed on everything
unknown, hash-only references, no write-back."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from credit_auditor.adapters.grpo_guard_envelope import (
    CreditAuditBundle,
    GuardEnvelopeRef,
    validate_guard_envelope,
)

VALID_ENVELOPE = {
    "schema_version": "grpo-guard-envelope-1.0",
    "required_extensions": [],
    "content_sha256": "a" * 64,
}


def test_valid_envelope_allowed():
    out = validate_guard_envelope(VALID_ENVELOPE)
    assert out["status"] == "ALLOW"


def test_unknown_major_fails_closed():
    env = dict(VALID_ENVELOPE, schema_version="grpo-guard-envelope-2.0")
    out = validate_guard_envelope(env)
    assert out["status"] == "REJECT"
    assert "major" in out["reason"]


def test_unknown_required_extension_fails_closed():
    env = dict(VALID_ENVELOPE, required_extensions=["decision_spans_v9"])
    out = validate_guard_envelope(env)
    assert out["status"] == "REJECT"
    assert "extension" in out["reason"]


def test_missing_content_hash_fails_closed():
    env = {k: v for k, v in VALID_ENVELOPE.items() if k != "content_sha256"}
    out = validate_guard_envelope(env)
    assert out["status"] == "REJECT"


def test_pinned_version_mismatch_fails_closed():
    out = validate_guard_envelope(VALID_ENVELOPE, pinned_schema="grpo-guard-envelope-0.9")
    assert out["status"] == "REJECT"


def test_bundle_requires_hashes_on_all_refs():
    with pytest.raises(ValidationError):
        CreditAuditBundle(
            guard_schema_version="grpo-guard-envelope-1.0",
            guard_envelope_refs=[GuardEnvelopeRef(uri="x", sha256="short")],
        )


def test_bundle_roundtrip():
    b = CreditAuditBundle(
        guard_schema_version="grpo-guard-envelope-1.0",
        guard_envelope_refs=[GuardEnvelopeRef(uri="s3://guard/evt/1", sha256="b" * 64)],
        decision_token_spans=[[83, 91], [104, 112]],
        selection_probabilities={"values": [0.25, 0.75], "law": "with_replacement"},
        cost_observations={"environment_transitions": 37, "generated_tokens": 412, "model_forwards": 39},
    )
    assert b.schema_version == "credit-audit-bundle-1.0"
    assert b.cost_observations.environment_transitions == 37


def test_bundle_canonical_hash_stable():
    from credit_auditor.canonical import sha256_json
    b = CreditAuditBundle(
        guard_schema_version="grpo-guard-envelope-1.0",
        guard_envelope_refs=[GuardEnvelopeRef(uri="u", sha256="c" * 64)],
    )
    assert sha256_json(b.model_dump(mode="json")) == sha256_json(b.model_dump(mode="json"))


def test_no_write_back_api():
    """§25: the adapter exposes validation only — there is no API that writes
    to Guard artifacts."""
    import inspect
    from credit_auditor.adapters import grpo_guard_envelope as g
    public = [n for n in dir(g) if not n.startswith("_")]
    assert not any("write" in n.lower() or "publish" in n.lower() for n in public)
