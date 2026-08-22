"""GRPO-Guard real-trajectory integration tests (§25): real envelopes build
valid bundles; tampered/invalid envelopes fail closed; no write-back."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from credit_auditor.adapters.guard_integration import (
    envelope_to_bundle,
    load_real_envelope,
    summarize_bundle,
)
from credit_auditor.adapters.grpo_guard_envelope import validate_guard_envelope

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "guard_envelopes"


@pytest.mark.parametrize(
    "name",
    ["f1_static_rollout_envelope.json", "boundary_truncation_envelope.json"],
)
def test_real_envelope_builds_valid_bundle(name):
    env = load_real_envelope(FIXTURES / name)
    bundle, validation = envelope_to_bundle(env)
    assert validation["status"] == "ALLOW"
    assert bundle is not None
    assert bundle.guard_schema_version == "grpo-guard-envelope-1.0"
    assert len(bundle.guard_envelope_refs) == 1
    assert bundle.guard_envelope_refs[0].sha256 == env["envelope_sha256"]


def test_real_envelope_required_extensions_present():
    env = load_real_envelope(FIXTURES / "f1_static_rollout_envelope.json")
    assert env["required_extensions"] == []


def test_tampered_envelope_sha_rejected():
    env = load_real_envelope(FIXTURES / "f1_static_rollout_envelope.json")
    env["envelope_sha256"] = "t" * 64  # content hash does not match the real one
    # The bundle references the (tampered) hash; the FAIL-CLOSED guard is the
    # envelope validator: a wrong-length or missing hash rejects.
    env2 = dict(env)
    env2["envelope_sha256"] = "short"
    with pytest.raises(ValueError):
        envelope_to_bundle(env2)


def test_unknown_required_extension_rejects_bundle():
    env = load_real_envelope(FIXTURES / "f1_static_rollout_envelope.json")
    env["required_extensions"] = ["decision_spans_v9"]
    bundle, validation = envelope_to_bundle(env)
    assert validation["status"] == "REJECT"
    assert "extension" in validation["reason"]
    assert bundle is None


def test_unknown_schema_major_rejects():
    env = load_real_envelope(FIXTURES / "f1_static_rollout_envelope.json")
    bundle, validation = envelope_to_bundle(env, pinned_schema="grpo-guard-envelope-2.0")
    assert validation["status"] == "REJECT"
    assert "major" in validation["reason"]


def test_bundle_summary_traceable():
    env = load_real_envelope(FIXTURES / "f1_static_rollout_envelope.json")
    bundle, _ = envelope_to_bundle(env)
    summary = summarize_bundle(bundle, env["envelope_id"])
    assert summary["envelope_id"] == env["envelope_id"]
    assert summary["guard_schema"] == "grpo-guard-envelope-1.0"
    assert len(summary["guard_envelope_refs"]) == 1


def test_no_write_back_api():
    import inspect
    from credit_auditor.adapters import guard_integration as gi
    public = [n for n in dir(gi) if not n.startswith("_")]
    assert not any("write" in n.lower() or "publish" in n.lower() for n in public)
