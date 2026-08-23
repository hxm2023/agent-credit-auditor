#!/usr/bin/env bash
# Real GRPO training audit (v0.1.5): audits the artifacts of a real
# full-model post-training run (Qwen3-4B GRPO via GRPO-Guard on autodl2).
#
# Usage: bash scripts/run_real_training_audit.sh [DATA_DIR] [OUTPUT_DIR]
#   DATA_DIR  default: artifacts/real_training (fetched from autodl2 with
#             the owner's authorization)
set -euo pipefail
cd "$(dirname "$0")/.."
DATA="${1:-artifacts/real_training}"
OUT="${2:-artifacts/real_training_audit}"
mkdir -p "$OUT"
uv run python - "$DATA" "$OUT" <<'PY'
import json
import sys
from pathlib import Path

from credit_auditor.audit.real_training import audit_real_training_dir, render_report

data_dir = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
audit = audit_real_training_dir(data_dir)
(out_dir / "audit_result.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
(out_dir / "REPORT.md").write_text(render_report(audit), encoding="utf-8")
print(render_report(audit))
PY
