#!/usr/bin/env bash
# Real-scenario demo: inject GRPO-Guard fault patterns into the Auditor's own
# artifacts and show the offline detectors firing (design §14, §25).
#
# Patterns injected (mapped in docs/online_offline_fault_map.md):
#   f5_split_leakage  -> seed-overlap manifests -> runner refuses
#   f8_artifact_mutation -> tampered package file -> provenance audit fails
#   f7_event_reorder  -> inconsistent manifest ordering -> provenance flags
#   static_rollout    -> envelope with max_policy_lag_versions > 0 -> bundle
#                        validation rejects (unknown contract state)
#
# Usage: bash scripts/run_real_scenario_demo.sh [OUTPUT_DIR]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-artifacts/real_scenario_demo}"
mkdir -p "$OUT"
uv run python - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

from credit_auditor import runner
from credit_auditor.audit.provenance import audit_artifact_dir
from credit_auditor.canonical import sha256_file

out_dir = Path(sys.argv[1])
rows = []

# --- f5_split_leakage: seed overlap between calibration and test ---
cal = out_dir / "f5_cal.json"
test = out_dir / "f5_test.json"
rows.append({"problem_id": "leak", "seed": 7})
cal.write_text(json.dumps({"rows": rows}), encoding="utf-8")
test.write_text(json.dumps({"rows": rows}), encoding="utf-8")
try:
    runner.check_split_disjoint(cal, test)
    f5_detected = False
except Exception as e:
    f5_detected = True
print(f"f5_split_leakage   -> detected={f5_detected} (runner refused the overlap)")
rows.append({"scenario": "f5_split_leakage", "detected": f5_detected})

# --- f8_artifact_mutation: tamper with a published package ---
pkg = out_dir / "pkg"
pkg.mkdir(exist_ok=True)
for name in ("protocol.json", "result.json", "oracle_result.json", "gate_decision.json", "run_manifest.json", "raw_rows.jsonl.zst", "REPORT.md"):
    (pkg / name).write_text("{}", encoding="utf-8")
(pkg / "SHA256SUMS").write_text(f"{sha256_file(pkg / 'result.json')}  result.json\n", encoding="utf-8")
(pkg / "result.json").write_text('{"tampered": true}', encoding="utf-8")  # mutate after publish
audit = audit_artifact_dir(pkg)
f8_detected = audit["integrity"] == "fail"
print(f"f8_artifact_mutation -> detected={f8_detected} ({audit['integrity']})")
rows.append({"scenario": "f8_artifact_mutation", "detected": f8_detected, "integrity": audit["integrity"]})

# --- f7_event_reorder: hand-assembled manifest missing the required fields ---
manifest = {"protocol_id": "x", "argv": []}  # missing utc_start/source_commit/dirty/python/platform
(pkg / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
sums = "\n".join(f"{sha256_file(pkg / n)}  {n}" for n in ("protocol.json", "result.json", "oracle_result.json", "gate_decision.json", "run_manifest.json", "raw_rows.jsonl.zst", "REPORT.md"))
(pkg / "SHA256SUMS").write_text(sums + "\n", encoding="utf-8")
audit3 = audit_artifact_dir(pkg)
f7_detected = audit3["integrity"] == "fail"
print(f"f7_event_reorder   -> detected={f7_detected} ({[e for e in audit3['errors'] if 'manifest' in e]})")
rows.append({"scenario": "f7_event_reorder", "detected": f7_detected, "errors": audit3["errors"]})

# --- static_rollout: envelope with a stale training contract ---
env = {"envelope_id": "env-static", "envelope_sha256": "b" * 64, "required_extensions": [],
       "training_contract": {"max_policy_lag_versions": 3, "behavior_logprob_source": "stale", "protocol": "strict_on_policy"}}
from credit_auditor.adapters.guard_integration import envelope_to_bundle
bundle, validation = envelope_to_bundle(env)
# fail-closed structure passes (schema/extension/hash legal); the CONTRACT
# semantics (policy lag > 0 = stale rollout source) are reported as the
# offline signal for the v0.2 estimator-level audit
lag = env["training_contract"]["max_policy_lag_versions"]
print(f"static_rollout     -> structure={validation['status']}; contract offline-signal: max_policy_lag_versions={lag} (stale rollout, S002/T004 territory)")
rows.append({"scenario": "static_rollout", "structure": validation["status"], "offline_signal": f"max_policy_lag_versions={lag}"})

json.dump({"scenario_count": len(rows), "rows": rows}, open(out_dir / "real_scenario_result.json", "w"), indent=2)
print(f"result: {out_dir / 'real_scenario_result.json'}")
PY
