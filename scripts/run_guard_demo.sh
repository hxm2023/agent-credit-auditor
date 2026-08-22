#!/usr/bin/env bash
# Real-trajectory Guard integration demo (design §25).
#
# Reads REAL Guard-issued trajectory envelopes (from the Auditor's frozen
# fixtures, or any GRPO-Guard checkout), builds the Auditor's
# CreditAuditBundle by hash-only references, and runs the fail-closed
# validation. This is the exact-toy -> real-toolchain bridge: a real GRPO
# rollout envelope flows through the Auditor's bundle validation.
#
# Usage: bash scripts/run_guard_demo.sh [ENVELOPE_DIR] [OUTPUT_DIR]
set -euo pipefail
cd "$(dirname "$0")/.."
ENV_DIR="${1:-tests/fixtures/guard_envelopes}"
OUT="${2:-artifacts/guard_demo}"

mkdir -p "$OUT"
uv run python - "$ENV_DIR" "$OUT" <<'PY'
import json
import sys
from pathlib import Path

from credit_auditor.adapters.guard_integration import envelope_to_bundle, load_real_envelope, summarize_bundle

env_dir = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
rows = []
for env_path in sorted(env_dir.glob("*.json")):
    env = load_real_envelope(env_path)
    bundle, validation = envelope_to_bundle(env)
    row = {
        "envelope": env_path.name,
        "envelope_id": env.get("envelope_id"),
        "envelope_stage": env.get("envelope_stage"),
        "required_extensions": env.get("required_extensions"),
        "validation": validation["status"],
        "bundle": summarize_bundle(bundle, env.get("envelope_id")) if bundle else None,
    }
    rows.append(row)
    status = validation["status"]
    print(f"{env_path.name:45s} {status:6s} envelope_id={env.get('envelope_id')}")
json.dump({"guard_schema": "grpo-guard-envelope-1.0", "envelopes": rows}, open(out_dir / "guard_demo_result.json", "w"), indent=2)
print(f"result: {out_dir / 'guard_demo_result.json'}")
PY
