#!/usr/bin/env bash
# jindun stage3 matrix, sequential (one run at a time, one GPU at a time):
# cts_order x 3 estimators x 3 seeds, then tau2_retail x 3 x 3.
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
    if [ -n "$used" ] && [ "$used" -le 1024 ]; then
      echo $i
      return 0
    fi
  done
  return 1
}

echo "== jindun sequential matrix =="
for task in cts_order tau2_retail; do
  for estimator in dense local paired; do
    for seed in 1 2 3; do
      run="$OUT/${task}_${estimator}_s${seed}"
      if [ -f "$run/metrics.json" ]; then
        echo "skip (exists): $run"
        continue
      fi
      for attempt in 1 2 3 4 5; do
        gpu=$(pick_gpu) || { echo "[$(date +%H:%M:%S)] no free GPU; sleeping 300"; sleep 300; attempt=$((attempt - 1)); continue; }
        echo "== run: $task $estimator s$seed (attempt $attempt, GPU $gpu) =="
        GRPO_GUARD_REPO=/data_3/repo/agood/grpo-guard-src         ATTRL_DIR=/data_3/repo/agood/agent-ttrl         GRPO_GUARD_MODEL_PATH=$MODEL         STAGE3_CUDA_DEVICE="cuda:$gpu"         PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True           "$PY" "$STAGE3/train.py" --task "$task" --estimator "$estimator"             --seed "$seed" --out "$run" --prompts 32 --gens 8 --epochs 3             > "$OUT/${task}_${estimator}_s${seed}.log" 2>&1
        rc=$?
        if [ -f "$run/metrics.json" ]; then
          echo "== done: $run (rc=$rc) =="
          break
        fi
        echo "== failed (rc=$rc), attempt $attempt of 5 =="
        sleep 10
      done
    done
  done
done
echo "== jindun sequential matrix complete =="
