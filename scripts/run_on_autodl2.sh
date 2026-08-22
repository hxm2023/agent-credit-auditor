#!/usr/bin/env bash
# OPTIONAL server-scale census (user-executed; CPU-only, 0 GPU).
#
# Runs the CTRI sign/rank stability census at N=10^7 on autodl2's CPU cores
# (shared with GRPO-Guard / agent-ttrl — Auditor never contends for GPU and
# uses bounded parallelism), then rsyncs the result back.
#
# Usage: bash scripts/run_on_autodl2.sh [N] [PARALLEL]
#   N        families (default 10000000)
#   PARALLEL worker processes (default 32; the server has ~200 cores)
set -euo pipefail
cd "$(dirname "$0")/.."
N="${1:-10000000}"
P="${2:-32}"
REMOTE="/tmp/aca_auditor_census"
SRV=autodl2

echo "== sync repo (exclude venv/artifacts/.git) =="
rsync -az --delete \
  --exclude .venv --exclude artifacts --exclude .git --exclude __pycache__ \
  ./ "$SRV:$REMOTE/"

echo "== run census on server (N=$N, P=$P workers) =="
ssh "$SRV" "cd $REMOTE && export PATH=\$HOME/miniconda3/bin:\$PATH && \
  python - <<'PY'
import hashlib, json, multiprocessing as mp, time, sys
sys.path.insert(0, '.')
from credit_auditor.experiments.continuation_scale import generate_family, census_family

N = $N
def work(seed):
    return census_family(generate_family(seed))

if __name__ == '__main__':
    t0 = time.perf_counter()
    with mp.Pool($P) as pool:
        results = pool.map(work, range(N))
    counts = {'sign_reversal': 0, 'rank_reversal': 0, 'mixed_q1': 0, 'mixed_q0': 0, 'abstain': 0}
    for r in results:
        for k in counts:
            counts[k] += int(r[k])
    elapsed = time.perf_counter() - t0
    out = {'families': N, 'counts': counts, 'rates': {k: v / N for k, v in counts.items()},
           'elapsed_s': elapsed, 'host': 'autodl2', 'workers': $P}
    json.dump(out, open('census_result.json', 'w'), indent=2)
    print('N=', N, 'rates=', {k: round(v, 6) for k, v in out['rates'].items()}, 'time=', round(elapsed, 1), 's')
PY"

echo "== rsync result back =="
mkdir -p artifacts/scale_supplement
rsync -az "$SRV:$REMOTE/census_result.json" artifacts/scale_supplement/
echo "result: artifacts/scale_supplement/census_result.json"
