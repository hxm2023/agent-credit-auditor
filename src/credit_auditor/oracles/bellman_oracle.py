#!/usr/bin/env python
"""INDEPENDENT Bellman gradient oracle (design §10.2).

SELF-CONTAINED: imports nothing from `credit_auditor` (verified by
oracles.isolation tests). Same JSON contract as enumeration_oracle.py.

Algorithm (different from path enumeration):
    Q(h) = expected terminal reward from prefix h (computed backwards),
    dJ/dtheta_t = sum_{h : |h|=t} P(h) p_t(1-p_t) (Q(h,1) - Q(h,0)).
This is value-function dynamic programming; the primary side may use
path enumeration, so a shared bug in one enumeration routine cannot
silently pass both.
"""
from __future__ import annotations

import hashlib
import json
import sys


def main() -> int:
    spec = json.load(sys.stdin)
    probs = [float(x) for x in spec["probabilities"]]
    rewards = spec["rewards"]
    h = len(probs)

    # Q for full sequences = terminal reward
    q: dict[str, float] = {}
    for key, r in rewards.items():
        q[key] = float(r)
    # Backwards recursion over prefixes
    for t in range(h - 1, -1, -1):
        p_t = probs[t]
        for bits in range(1 << t):
            h_pref = ",".join(str((bits >> (t - 1 - tt)) & 1) for tt in range(t)) if t > 0 else ""
            prefix = h_pref if h_pref else ""
            q1 = q[(prefix + ",1") if prefix else "1"]
            q0 = q[(prefix + ",0") if prefix else "0"]
            q[prefix] = p_t * q1 + (1 - p_t) * q0

    g = [0.0] * h
    for t in range(h):
        p_t = probs[t]
        for bits in range(1 << t):
            h_pref = ",".join(str((bits >> (t - 1 - tt)) & 1) for tt in range(t)) if t > 0 else ""
            prefix = h_pref if h_pref else ""
            p_h = 1.0
            for tt in range(t):
                ht = (bits >> (t - 1 - tt)) & 1
                p_h *= probs[tt] if ht else (1.0 - probs[tt])
            k1 = (prefix + ",1") if prefix else "1"
            k0 = (prefix + ",0") if prefix else "0"
            g[t] += p_h * p_t * (1.0 - p_t) * (q[k1] - q[k0])

    input_sha = hashlib.sha256(json.dumps(spec, sort_keys=True).encode("utf-8")).hexdigest()
    out = {
        "gradient": [float(x) for x in g],
        "oracle": "bellman",
        "input_sha256": input_sha,
        "precision": "float64",
        "n_states": sum(1 << t for t in range(h + 1)),
    }
    sys.stdout.write(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
