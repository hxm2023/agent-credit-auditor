#!/usr/bin/env bash
# Stage 3 matrix launcher for jindun (js3.blockelite.cn). GPU policy:
#   - NEVER use cards with other users' experiments (memory.used > 1 GiB)
#   - use at most 3 cards, prefer 5/6/7 (6 and 7 first), fall back to 0/1/4
#   - if the chosen card becomes busy mid-run, the run's retry logic handles
#     it (run_matrix retries up to 3 times; this wrapper re-checks each run)
# Usage: bash stage3/run_jindun_matrix.sh [OUT]
set -uo pipefail
REPO=/data_3/repo/agood/Agent-RL-Credit-Auditor
STAGE3=$REPO/stage3
OUT=${1:-$STAGE3/out_jindun}
PY=$REPO/.venv_jindun/bin/python
mkdir -p "$OUT"

pick_gpu() {
  # free GPUs first (memory.used <= 1024 MiB), priority 6 7 5 4 0 1 2 3
  for i in 6 7 5 4 0 1 2 3; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $i 2>/dev/null | tr -d ' ')
    if [ -n "$used" ] && [ "$used" -le 1024 ]; then
      echo $i
      return 0
    fi
  done
  return 1
}

# tau2 server (needed by the tau2_retail task; single instance, CPU)
start_tau2() {
  if ! curl -s -m 2 -X POST http://127.0.0.1:8800/reset > /dev/null 2>&1; then
    echo "== starting tau2 server =="
    TAU2_SERVER_PORT=8800 nohup /usr/bin/python3 \
      /data_3/repo/agood/agent-ttrl/scripts/tau2_server.py > "$OUT/tau2_server.log" 2>&1 < /dev/null &
    sleep 8
  fi
  curl -s -m 5 -X POST http://127.0.0.1:8800/reset > /dev/null && echo "tau2 server ok"
}
start_tau2

echo "== jindun stage3 matrix: 2 tasks x 3 estimators x 3 seeds =="
for task in cts_order tau2_retail; do
  for estimator in dense local paired; do
    for seed in 1 2 3; do
      run="$OUT/${task}_${estimator}_s${seed}"
      if [ -f "$run/metrics.json" ]; then
        echo "skip (exists): $run"
        continue
      fi
      for attempt in 1 2 3; do
        gpu=$(pick_gpu) || { echo "[$(date +%H:%M:%S)] no free GPU; sleeping 300"; sleep 300; attempt=$((attempt - 1)); continue; }
        echo "== run: $task $estimator seed=$seed (attempt $attempt, GPU $gpu) =="
        GRPO_GUARD_REPO=/data_3/repo/agood/grpo-guard-src \
        ATTRL_DIR=/data_3/repo/agood/agent-ttrl \
        GRPO_GUARD_MODEL_PATH=/data_3/repo/agood/models_cache/models--Qwen--Qwen3-4B/snapshots \
        STAGE3_CUDA_DEVICE="cuda:$gpu" \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
          "$PY" "$STAGE3/train.py" --task "$task" --estimator "$estimator" \
          --seed "$seed" --out "$run" --prompts 32 --gens 8 --epochs 3 \
          > "$OUT/${task}_${estimator}_s${seed}.log" 2>&1
        rc=$?
        if [ -f "$run/metrics.json" ]; then
          echo "== done: $run (rc=$rc) =="
          break
        fi
        echo "== failed (rc=$rc), attempt $attempt of 3 =="
        sleep 10
      done
    done
  done
done
echo "== jindun matrix complete: $OUT =="
