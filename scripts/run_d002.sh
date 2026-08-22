#!/usr/bin/env bash
# D002 calibration + test (design §15). Usage: scripts/run_d002.sh [ROOT_DIR]
# The test phase refuses any selection other than the frozen calibration output.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="${1:-artifacts/local/D002}"
uv run credit-auditor run \
  --protocol configs/protocols/d002_regression_v1.json \
  --phase calibration --output "$ROOT/D002_cal" \
  --seed configs/seeds/d002_calibration.json
uv run credit-auditor run \
  --protocol configs/protocols/d002_regression_v1.json \
  --phase test --output "$ROOT/D002_test" \
  --frozen-selection "$ROOT/D002_cal/selection.json" \
  --seed configs/seeds/d002_test.json
uv run credit-auditor audit --artifact-dir "$ROOT/D002_test"
