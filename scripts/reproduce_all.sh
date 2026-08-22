#!/usr/bin/env bash
# One-command reproduction of the full v0.1.1 release (design §15, §28
# fresh-clone reproduction). Usage: scripts/reproduce_all.sh [ARTIFACT_ROOT]
# Artifact root defaults to artifacts/v0.1.1; the run refuses existing outputs
# (no-overwrite), so delete or move the root to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="${1:-artifacts/v0.1.1}"

echo "== validate protocols =="
for p in configs/protocols/*.json; do
  uv run credit-auditor validate-protocol "$p"
done

echo "== M0 target audit =="
uv run credit-auditor run \
  --protocol configs/protocols/m0_regression_v1.json \
  --output "$ROOT/M0" --seed configs/seeds/m0_problems.json

echo "== V001 expected utility failure =="
uv run credit-auditor run \
  --protocol configs/protocols/v001_failure_v1.json \
  --output "$ROOT/V001" \
  --seed configs/seeds/v001_problems.json \
  --seed configs/seeds/v001_calibration.json

echo "== D002 calibration + test (frozen selection) =="
uv run credit-auditor run \
  --protocol configs/protocols/d002_regression_v1.json \
  --phase calibration --output "$ROOT/D002_cal" \
  --seed configs/seeds/d002_calibration.json
uv run credit-auditor run \
  --protocol configs/protocols/d002_regression_v1.json \
  --phase test --output "$ROOT/D002_test" \
  --frozen-selection "$ROOT/D002_cal/selection.json" \
  --seed configs/seeds/d002_test.json

echo "== optional support-only packs =="
uv run credit-auditor run \
  --protocol configs/protocols/continuation_support_only_v1.json \
  --output "$ROOT/CONT"
uv run credit-auditor run \
  --protocol configs/protocols/minimal_logging_teaching_v1.json \
  --output "$ROOT/ML"

echo "== release report (runs the full test suite for TEST_LOG) =="
uv run python -c "
from pathlib import Path
from credit_auditor.report import build_release_report
print('report:', build_release_report(Path('$ROOT')))
"

echo "== done: $ROOT =="
