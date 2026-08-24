#!/usr/bin/env bash
# Stage 3 full matrix launcher (runs on autodl2): 2 tasks x 3 estimators x
# 3 seeds = 18 Guard-supervised GRPO runs, sequential (GPU0 trainer + GPU1
# vLLM each), matched budgets (32 prompts x 8 gens x 3 epochs, LoRA rank 16,
# lr 5e-6). Per run: metrics.json + Guard event/store + trajectory records.
#
# Usage (on autodl2): bash stage3/run_matrix.sh [OUT]  (nohup-friendly)
set -euo pipefail
STAGE3=/root/autodl-tmp/agent-ttrl/stage3
OUT=${1:-$STAGE3/out}
PY=/root/autodl-tmp/grpo-guard/.venv/bin/python
mkdir -p "$OUT"

# tau2 retail server (needed by the tau2_retail task; single instance)
TAU2_LOG="$OUT/tau2_server.log"
start_tau2() {
  if ! curl -s -m 2 -X POST http://127.0.0.1:8800/reset > /dev/null 2>&1; then
    echo "== starting tau2 server =="
    TAU2_SERVER_PORT=8800 nohup /root/autodl-tmp/appworld-venv/bin/python \
      /root/autodl-tmp/agent-ttrl/scripts/tau2_server.py > "$TAU2_LOG" 2>&1 &
    sleep 8
  fi
  curl -s -m 5 -X POST http://127.0.0.1:8800/reset > /dev/null && echo "tau2 server ok"
}

start_tau2

echo "== stage3 matrix: 2 tasks x 3 estimators x 3 seeds =="
for task in cts_order tau2_retail; do
  for estimator in dense local paired; do
    for seed in 1 2 3; do
      run="$OUT/${task}_${estimator}_s${seed}"
      if [ -f "$run/metrics.json" ]; then
        echo "skip (exists): $run"
        continue
      fi
      echo "== run: $task $estimator seed=$seed =="
      STAGE3_VLLM_PORT=$((8007 + seed)) STAGE3_GROUP_PORT=$((51227 + seed)) \
        "$PY" "$STAGE3/train.py" --task "$task" --estimator "$estimator" \
        --seed "$seed" --out "$run" --prompts 32 --gens 8 --epochs 3 \
        > "$OUT/${task}_${estimator}_s${seed}.log" 2>&1
      echo "== done: $run (rc=$?) =="
    done
  done
done
echo "== matrix complete: $OUT =="
