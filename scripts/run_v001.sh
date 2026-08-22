#!/usr/bin/env bash
# V001 expected utility failure (design §15). Usage: scripts/run_v001.sh [OUTPUT_DIR]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-artifacts/local/V001}"
uv run credit-auditor run \
  --protocol configs/protocols/v001_failure_v1.json \
  --output "$OUT" \
  --seed configs/seeds/v001_problems.json \
  --seed configs/seeds/v001_calibration.json
uv run credit-auditor audit --artifact-dir "$OUT"
