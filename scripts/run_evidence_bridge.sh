#!/usr/bin/env bash
# Evidence bridge (Stage 2 of the real-trajectory bridge): exact verdicts
# predict sampled fixed-budget MSE on controllable tool-agent tasks.
#
# Three layers (docs/evidence_bridge.md):
#   1. exact: bias + intrinsic cycle variance + matched-budget predictor
#      p = var_cycle * cost / B + bias^2  (H=4 enumeration)
#   2. MC agreement gate: independent high-budget MC vs the exact target
#   3. sampled: fixed-budget MSE of the SAME estimators over trajectory
#      records (matched transitions, seeded replicates); H=12 uses MC target
#
# Usage: bash scripts/run_evidence_bridge.sh [OUTPUT_DIR]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-artifacts/evidence_bridge}"
mkdir -p "$OUT"
uv run python -c "
from pathlib import Path
from credit_auditor.experiments.evidence_bridge import run_bridge
summary = run_bridge(Path('$OUT'))
for t in summary['tasks']:
    print(f\"{t['task']}: gate={t['mc_agreement_gate']} rho={t['spearman_exact_to_sampled']}\")
print('report:', '$OUT/REPORT.md')
"