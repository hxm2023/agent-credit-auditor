"""Real GRPO training audit tests: sync-ack signal, policy traceability,
schema-shaped fixtures mirroring the fetched real artifacts."""
from __future__ import annotations

import json
from pathlib import Path

from credit_auditor.audit.real_training import (
    audit_policy_manifest,
    audit_real_training_dir,
    audit_sync_calls,
    render_report,
)


def _smoke_result(acked: int, unacked: int) -> dict:
    calls = [{"param_name": f"p{i}", "param_shape": [1], "timestamp": 0.0, "ack": True} for i in range(acked)]
    calls += [{"param_name": f"u{i}", "param_shape": [1], "timestamp": 0.0, "ack": False} for i in range(unacked)]
    return {"model_id": "Qwen/Qwen3-4B", "trl_observed_sync_calls": calls, "committed_optimizer_steps": 1}


def _policy_manifest() -> dict:
    return {
        "manifest_id": "pm-1",
        "model_id": "Qwen/Qwen3-4B",
        "policy_version": 1,
        "parent_policy_version": 0,
        "weights": [{"uri": "artifact://m.safetensors", "num_bytes": 100, "sha256": "a" * 64}],
        "checkpoint_manifest_sha256": "b" * 64,
        "tokenizer_sha256": "c" * 64,
        "chat_template_sha256": "d" * 64,
        "code_commit_sha": "e" * 40,
        "config_sha256": "f" * 64,
    }


def test_sync_all_acked_clear():
    out = audit_sync_calls(_smoke_result(acked=398, unacked=0))
    assert out["static_rollout_signal"] is False
    assert out["sync_calls"] == 398 and out["acked"] == 398


def test_sync_unacked_signals_static_rollout():
    out = audit_sync_calls(_smoke_result(acked=390, unacked=8))
    assert out["static_rollout_signal"] is True
    assert out["unacked"] == 8
    assert out["unacked_params"][0] == "u0"


def test_policy_traceability_ok():
    out = audit_policy_manifest(_policy_manifest())
    assert out["traceability_ok"] is True
    assert out["weights"] == 1


def test_policy_missing_hash_detected():
    pm = _policy_manifest()
    pm["weights"][0]["sha256"] = "short"
    out = audit_policy_manifest(pm)
    assert out["traceability_ok"] is False
    assert out["unhashed_weights"] == ["artifact://m.safetensors"]


def test_audit_dir_with_real_fixtures(tmp_path):
    """The fetched real artifacts' structure must audit cleanly (schema
    fixtures mirroring artifacts/real_training)."""
    smoke = tmp_path / "smoke_out"
    loop = tmp_path / "loop_out"
    smoke.mkdir(parents=True)
    loop.mkdir(parents=True)
    (smoke / "smoke_result.json").write_text(json.dumps(_smoke_result(398, 0)), encoding="utf-8")
    (loop / "policy_manifest.json").write_text(json.dumps(_policy_manifest()), encoding="utf-8")
    (loop / "run_manifest.json").write_text(json.dumps({"run_id": "loop-x", "closed_loop": True}), encoding="utf-8")
    (loop / "lease.json").write_text(json.dumps({"lease_id": "guard-trainer", "epoch": 1}), encoding="utf-8")
    audit = audit_real_training_dir(tmp_path)
    assert audit["sync"]["static_rollout_signal"] is False
    assert audit["policy"]["traceability_ok"] is True
    assert "static_rollout signal clear" in audit["findings"][0]
    report = render_report(audit)
    assert "398/398" in report
    assert "CLEAR" in report
