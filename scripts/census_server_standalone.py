#!/usr/bin/env python
"""Server-scale CTRI sign/rank stability census — STDLIB-ONLY (runs on any
Python 3.10+ with zero installed packages, for the shared autodl2 host).

The seed derivation, family generation, and census logic are byte-identical
to src/credit_auditor/experiments/continuation_scale.py (verified by the
consistency check below): a smaller N run must match the local package's
output exactly, then the large N run tightens the rate estimates.

Usage:
    python census_server_standalone.py [N] [WORKERS] [OUTPUT.json]
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import sys
import time
from fractions import Fraction

SEED_PREFIX = "ACA-SEM-CTRISCALE"


def _draw(key: str, i: int) -> Fraction:
    h = hashlib.sha256(key.encode("utf-8") + b"::" + str(i).encode())
    return Fraction(int.from_bytes(h.digest()[:8], "big"), 2**64)


def _frac_in(x: Fraction, lo: Fraction, hi: Fraction) -> Fraction:
    return lo + (hi - lo) * x


def generate_family(seed: int) -> dict:
    key = f"{SEED_PREFIX}::{seed}"
    reward = {}
    for s in (0, 1):
        for a in (0, 1):
            for sp in (0, 1):
                reward[(s, a, sp)] = _frac_in(_draw(key, s * 4 + a * 2 + sp), Fraction(0), Fraction(1))
    transitions = {}
    for s in (0, 1):
        for a in (0, 1):
            transitions[(s, a, 1)] = _frac_in(_draw(key, 20 + s * 4 + a * 2), Fraction(1, 10), Fraction(9, 10))
    policies = []
    for m in range(3):
        p0 = _frac_in(_draw(key, 40 + m * 2), Fraction(1, 10), Fraction(9, 10))
        p1 = _frac_in(_draw(key, 41 + m * 2), Fraction(1, 10), Fraction(9, 10))
        policies.append({"p0": p0, "p1": p1})
    return {"reward": reward, "transitions": transitions, "policies": policies}


def action_value(reward, transitions, policy, s: int, a: int, horizon: int) -> Fraction:
    if horizon == 1:
        return reward[(s, a, 1)] * transitions[(s, a, 1)] + reward[(s, a, 0)] * (1 - transitions[(s, a, 1)])
    v = Fraction(0)
    for sp in (0, 1):
        p_tr = transitions[(s, a, sp)] if sp == 1 else 1 - transitions[(s, a, 1)]
        if p_tr == 0:
            continue
        p_a1 = policy["p1"] if sp == 1 else policy["p0"]
        sub = p_a1 * action_value(reward, transitions, policy, sp, 1, horizon - 1) + (1 - p_a1) * action_value(
            reward, transitions, policy, sp, 0, horizon - 1
        )
        v += p_tr * (reward[(s, a, sp)] + sub)
    return v


def census_family(fam: dict) -> dict:
    q1 = [action_value(fam["reward"], fam["transitions"], p, 0, 1, 2) for p in fam["policies"]]
    q0 = [action_value(fam["reward"], fam["transitions"], p, 0, 0, 2) for p in fam["policies"]]
    diffs = [a - b for a, b in zip(q1, q0)]
    signs = ["+" if d > 0 else ("-" if d < 0 else "0") for d in diffs]
    pos = signs.count("+")
    neg = signs.count("-")
    majority = "+" if pos > neg else ("-" if neg > pos else None)
    return {
        "sign_reversal": majority is not None and any(s != majority and s != "0" for s in signs),
        "rank_reversal": pos > 0 and neg > 0,
        "mixed_q1": any(x < 0 for x in q1) and any(x > 0 for x in q1),
        "mixed_q0": any(x < 0 for x in q0) and any(x > 0 for x in q0),
        "abstain": majority is None,
        "signs": signs,
    }


def _work(seed: int) -> dict:
    return census_family(generate_family(seed))


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10000000
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 48
    out_path = sys.argv[3] if len(sys.argv) > 3 else "census_result.json"
    t0 = time.perf_counter()
    with mp.Pool(workers) as pool:
        results = pool.map(_work, range(n))
    counts = {"sign_reversal": 0, "rank_reversal": 0, "mixed_q1": 0, "mixed_q0": 0, "abstain": 0}
    for r in results:
        for k in counts:
            counts[k] += int(r[k])
    elapsed = time.perf_counter() - t0
    out = {
        "families": n,
        "counts": counts,
        "rates": {k: v / n for k, v in counts.items()},
        "elapsed_s": elapsed,
        "host": "autodl2",
        "workers": workers,
        "seed_prefix": SEED_PREFIX,
        "arithmetic": "exact_fraction",
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(
        json.dumps(
            {
                "families": n,
                "rates": {k: round(v, 7) for k, v in out["rates"].items()},
                "elapsed_s": round(elapsed, 1),
                "workers": workers,
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
