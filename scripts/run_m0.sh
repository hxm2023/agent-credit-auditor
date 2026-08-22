#!/usr/bin/env bash
# M0 target audit (design §15). Usage: scripts/run_m0.sh [OUTPUT_DIR]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-artifacts/local/M0}"
uv run credit-auditor run \
  --protocol configs/protocols/m0_regression_v1.json \
  --output "$OUT" \
  --seed configs/seeds/m0_problems.json
uv run credit-auditor audit --artifact-dir "$OUT"
