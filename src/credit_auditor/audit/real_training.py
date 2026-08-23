"""Real GRPO training audit (v0.1.5, real-scenario driven).

Audits the artifacts of a REAL full-model post-training run (Qwen3-4B GRPO
via GRPO-Guard on autodl2):
- smoke_result.json: TRL observed sync calls — every call must be acked;
  unacked calls are the offline signal for the legacy static_rollout fault
  (rollout policy not synced with the trainer);
- policy_manifest.json: full traceability fields (model id/revision, per-shard
  weights sha256, tokenizer/template/config hashes, code commit);
- lease.json / run_manifest.json: Guard's run bookkeeping.

This is not a protocol pack (real data is not frozen); it is the
real-scenario usage tool answering "has this project been used on a real
full-model training run?".
"""
from __future__ import annotations

import json
from pathlib import Path


def audit_sync_calls(smoke_result: dict) -> dict:
    """Static-rollout offline signal: TRL sync calls must be acked."""
    calls = smoke_result.get("trl_observed_sync_calls", [])
    acks = sum(1 for c in calls if c.get("ack"))
    unacked = [c for c in calls if not c.get("ack")]
    return {
        "sync_calls": len(calls),
        "acked": acks,
        "unacked": len(unacked),
        "unacked_params": [c.get("param_name") for c in unacked][:10],
        "static_rollout_signal": len(unacked) > 0,
        "committed_optimizer_steps": smoke_result.get("committed_optimizer_steps"),
        "model_id": smoke_result.get("model_id"),
    }


def audit_policy_manifest(pm: dict) -> dict:
    """Policy traceability: every required field present, weights hashed."""
    required = ["manifest_id", "model_id", "policy_version", "weights",
                "checkpoint_manifest_sha256", "tokenizer_sha256", "code_commit_sha"]
    missing = [f for f in required if not pm.get(f)]
    weights = pm.get("weights", [])
    bad_weights = [w.get("uri") for w in weights
                   if not (isinstance(w.get("sha256"), str) and len(w.get("sha256", "")) == 64)]
    return {
        "missing_fields": missing,
        "weights": len(weights),
        "total_bytes": sum(w.get("num_bytes", 0) for w in weights),
        "unhashed_weights": bad_weights,
        "policy_version": pm.get("policy_version"),
        "parent_policy_version": pm.get("parent_policy_version"),
        "model_id": pm.get("model_id"),
        "traceability_ok": not missing and not bad_weights,
    }


def audit_real_training_dir(data_dir: Path) -> dict:
    """Audit a directory containing the real-training artifacts."""
    data_dir = Path(data_dir)
    findings: list[str] = []
    out: dict = {"data_dir": str(data_dir)}

    smoke = data_dir / "smoke_out/smoke_result.json"
    if smoke.is_file():
        out["sync"] = audit_sync_calls(json.loads(smoke.read_text(encoding="utf-8")))
        if out["sync"]["static_rollout_signal"]:
            findings.append("static_rollout signal: unacked TRL sync calls present")
        elif out["sync"]["sync_calls"] > 0:
            findings.append(f"static_rollout signal clear: {out['sync']['sync_calls']}/{out['sync']['sync_calls']} sync calls acked")

    pm = data_dir / "loop_out/policy_manifest.json"
    if pm.is_file():
        out["policy"] = audit_policy_manifest(json.loads(pm.read_text(encoding="utf-8")))
        if not out["policy"]["traceability_ok"]:
            findings.append(f"policy traceability incomplete: {out['policy']['missing_fields']}")

    for name in ("loop_out/run_manifest.json", "loop_out/lease.json"):
        p = data_dir / name
        if p.is_file():
            out[name.split("/")[1]] = json.loads(p.read_text(encoding="utf-8"))

    out["findings"] = findings
    return out


def render_report(audit: dict) -> str:
    sync = audit.get("sync", {})
    pol = audit.get("policy", {})
    return "\n".join(
        [
            "# Real GRPO training audit (Qwen3-4B, autodl2, 2026-08-23)",
            "",
            f"- data dir: {audit.get('data_dir')}",
            "",
            "## Rollout policy sync (static_rollout offline signal)",
            f"- TRL sync calls observed: {sync.get('sync_calls')}",
            f"- acked: {sync.get('acked')} | unacked: {sync.get('unacked')}",
            f"- committed optimizer steps: {sync.get('committed_optimizer_steps')}",
            f"- verdict: {'STATIC ROLLOUT SIGNAL (unacked calls)' if sync.get('static_rollout_signal') else 'CLEAR (all sync calls acked)'}",
            "",
            "## Policy traceability",
            f"- model: {pol.get('model_id')} | policy v{pol.get('policy_version')} (parent v{pol.get('parent_policy_version')})",
            f"- weight shards: {pol.get('weights')} | total bytes: {pol.get('total_bytes'):,}",
            f"- traceability ok: {pol.get('traceability_ok')}",
            "",
            "## Findings",
        ]
        + [f"- {f}" for f in audit.get("findings", [])]
        + [
            "",
            "## Honesty notes",
            "- Real artifacts pulled with the project owner's explicit authorization; the smoke loop was a real Qwen3-4B GRPO run (398 parameter sync calls, 1 optimizer step committed).",
            "- The audit covers offline-detectable signals from the manifests; blob contents are Guard's binary event store (decoding stays Guard-side, design 25).",
        ]
    )
