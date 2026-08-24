#!/usr/bin/env bash
# jindun cts_order matrix (9 runs: 3 estimators x 3 seeds).
# GPU policy: free cards only (<=1 GiB used), priority 6 7 4 0 1 2,
# never >1 card per run, sleep 300s when no card is free.
set -uo pipefail
REPO=/data_3/repo/agood/Agent-RL-Credit-Auditor
STAGE3=$REPO/stage3
OUT=$STAGE3/out_jindun
PY=$REPO/.venv_jindun/bin/python
MODEL=/data_3/repo/agood/models_cache/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c
mkdir -p "$OUT"

pick_gpu() {
  for i in 6 7 4 0 1 2; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $i 2>/dev/null | tr -d ' ')
    [ -n "$used" ] && [ "$used" -le 1024 ] || continue
    # skip GPUs that already host one of OUR trainers (multi-launcher safety)
    own=0
    uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $i 2>/dev/null)
    for line in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null); do
      pid=${line%,*}; puuid=${line#*,}
      if [ "$puuid" = "$uuid" ] && strings /proc/$pid/cmdline 2>/dev/null | grep -q 'stage3/train.py'; then
        own=1
        break
      fi
    done
    [ "$own" -eq 0 ] && { echo $i; return 0; }
  done
  return 1
}

echo "== jindun cts_order matrix (3 estimators x 3 seeds) =="
for estimator in dense local paired; do
  for seed in 1 2 3; do
    run="$OUT/cts_order_${estimator}_s${seed}"
    if [ -f "$run/metrics.json" ]; then
      echo "skip (exists): $run"
      continue
    fi
    for attempt in 1 2 3; do
      gpu=$(pick_gpu) || { echo "[$(date +%H:%M:%S)] no free GPU; sleeping 300"; sleep 300; attempt=$((attempt - 1)); continue; }
      echo "== run: $estimator s$seed (attempt $attempt, GPU $gpu) =="
      GRPO_GUARD_REPO=/data_3/repo/agood/grpo-guard-src \
      ATTRL_DIR=/data_3/repo/agood/agent-ttrl \
      GRPO_GUARD_MODEL_PATH=$MODEL \
      STAGE3_CUDA_DEVICE="cuda:$gpu" \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        "$PY" "$STAGE3/train.py" --task cts_order --estimator "$estimator" \
          --seed "$seed" --out "$run" --prompts 32 --gens 8 --epochs 3 \
          > "$OUT/cts_order_${estimator}_s${seed}.log" 2>&1
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
echo "== jindun cts_order matrix complete =="
