#!/usr/bin/env python
"""INDEPENDENT enumeration oracle (design §10.2).

SELF-CONTAINED: imports nothing from `credit_auditor` (verified by
oracles.isolation tests). Reads one JSON world spec on stdin:

    {"world": "bernoulli_sequence_mdp", "probabilities": [...],
     "rewards": {"0,0": r00, "0,1": r01, ...}}

writes one JSON result on stdout:

    {"gradient": [...], "oracle": "enumeration", "input_sha256": ...,
     "precision": "float64", "n_paths": N}

Algorithm: direct 2^H path enumeration of E[R(tau) * score(tau)] —
deliberately the most naive route so any shared-bug risk with the primary
path enumeration is visible. Independent process per §10.1.
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
    g = [0.0] * h
    for bits in range(1 << h):
        a = tuple((bits >> (h - 1 - t)) & 1 for t in range(h))
        p = 1.0
        for t, at in enumerate(a):
            p *= probs[t] if at else (1.0 - probs[t])
        r = rewards[",".join(str(x) for x in a)]
        for t in range(h):
            g[t] += p * r * (a[t] - probs[t])
    input_sha = hashlib.sha256(json.dumps(spec, sort_keys=True).encode("utf-8")).hexdigest()
    out = {
        "gradient": [float(x) for x in g],
        "oracle": "enumeration",
        "input_sha256": input_sha,
        "precision": "float64",
        "n_paths": 1 << h,
    }
    sys.stdout.write(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
