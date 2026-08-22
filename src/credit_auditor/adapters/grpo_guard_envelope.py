"""GRPO-Guard envelope adapter (design §25) — v0.2 preparation, CPU-only.

Guard is the online owner of trajectory envelopes and canonical hashing. The
Auditor does NOT fork, modify, or re-publish a copy of the Guard core schema:
it pins a declared Guard schema version and FAILS CLOSED on anything unknown
(unknown major, unknown required extension, incompatible hash rules). The
Auditor's own `CreditAuditBundle` references Guard envelopes ONLY by hash and
never writes back to Guard artifacts.

v0.1.1 scope: schema + fail-closed validation + tests. No Guard server or GPU
is required; a real v0.2 smoke additionally needs GRPO-Guard's published
schema package and an allowed envelope (CLAUDE.md/§20.2 gate).
"""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from credit_auditor.schema import CanonicalModel

GUARD_SCHEMA = "grpo-guard-envelope-1.0"
CREDIT_AUDIT_BUNDLE_SCHEMA = "credit-audit-bundle-1.0"

KNOWN_GUARD_MAJORS = {"1"}
KNOWN_REQUIRED_EXTENSIONS: set[str] = set()  # none in the pinned 1.x core


class GuardEnvelopeRef(CanonicalModel):
    uri: str
    sha256: str


class SelectionProbabilities(CanonicalModel):
    values: list[float]
    law: str  # e.g. with_replacement


class CostObservations(CanonicalModel):
    environment_transitions: int | None = None
    generated_tokens: int | None = None
    model_forwards: int | None = None


class CreditAuditBundle(CanonicalModel):
    """The Auditor's own versioned bundle (§25). References Guard envelopes by
    hash only; cannot be written back to Guard artifacts."""

    schema_version: str = CREDIT_AUDIT_BUNDLE_SCHEMA
    guard_schema_version: str
    guard_envelope_refs: list[GuardEnvelopeRef] = Field(default_factory=list)
    decision_token_spans: list[list[int]] = Field(default_factory=list)
    restore_protocol_ref: GuardEnvelopeRef | None = None
    branch_event_refs: list[GuardEnvelopeRef] = Field(default_factory=list)
    continuation_policy_manifest_refs: list[GuardEnvelopeRef] = Field(default_factory=list)
    selection_probabilities: SelectionProbabilities | None = None
    cost_observations: CostObservations = Field(default_factory=CostObservations)
    target_policy_scoring_event: GuardEnvelopeRef | None = None

    @model_validator(mode="after")
    def _refs_have_hashes(self) -> "CreditAuditBundle":
        """§25: every Guard reference must carry a sha256; a bundle without
        one is invalid."""
        for ref in (
            list(self.guard_envelope_refs)
            + list(self.branch_event_refs)
            + list(self.continuation_policy_manifest_refs)
            + ([self.restore_protocol_ref] if self.restore_protocol_ref else [])
            + ([self.target_policy_scoring_event] if self.target_policy_scoring_event else [])
        ):
            if not isinstance(ref.sha256, str) or len(ref.sha256) != 64:
                raise ValueError(f"guard reference without a valid sha256: {ref.uri}")
        return self


def validate_guard_envelope(
    envelope: dict,
    pinned_schema: str = GUARD_SCHEMA,
    known_majors: set[str] | None = None,
    known_required_extensions: set[str] | None = None,
) -> dict:
    """Fail-closed validation of a Guard envelope (§25).

    Returns {"status": "ALLOW"} or {"status": "REJECT", "reason": ...}.
    Unknown schema major, unknown required extensions, or missing canonical
    hashes all REJECT — the Auditor never downgrades an unknown envelope.
    """
    majors = known_majors or KNOWN_GUARD_MAJORS
    extensions = known_required_extensions or KNOWN_REQUIRED_EXTENSIONS

    schema = envelope.get("schema_version")
    if not isinstance(schema, str):
        return {"status": "REJECT", "reason": "missing or non-string schema_version"}
    version = schema.rsplit("-", 1)[-1] if "-" in schema else schema
    major = version.split(".")[0] if "." in version else version
    if major not in majors:
        return {
            "status": "REJECT",
            "reason": f"unknown Guard schema major {major!r} (known: {sorted(majors)}); fail closed",
        }
    if schema != pinned_schema:
        return {
            "status": "REJECT",
            "reason": f"schema {schema!r} != pinned {pinned_schema!r}; pin a declared version",
        }
    for ext in envelope.get("required_extensions", []) or []:
        if ext not in extensions:
            return {
                "status": "REJECT",
                "reason": f"unknown required extension {ext!r} (known: {sorted(extensions)}); fail closed",
            }
    if not isinstance(envelope.get("content_sha256"), str) or len(envelope["content_sha256"]) != 64:
        return {"status": "REJECT", "reason": "missing/invalid content_sha256"}
    return {"status": "ALLOW", "schema_version": schema}
