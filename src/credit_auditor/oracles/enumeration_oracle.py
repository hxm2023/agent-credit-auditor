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


def _bernoulli_sequence(spec):
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
    return g


def _d002(spec):
    """D002 shared-logit world: enumerate all (states, actions) paths per
    bucket; gradient wrt the 3 shared logits (score a - p)."""
    logits = [float(x) for x in spec["logits"]]
    g = [0.0] * 3
    for b in spec["buckets"]:
        H = b["horizon"]
        init = b["initial_state"]
        pmap = b["param_map"]
        tbase = b["transition_base"]
        toff = b["transition_offset"]
        teff = b["transition_effect"]
        trew = b["terminal_rewards"]

        def tp(t, s, a):
            if focal is not None:
                return float(a)  # semantic focal world: deterministic s'=a
            v = tbase[t * 2 + s] + toff[t * 2 + s] + (2 * a - 1) * teff[t * 2 + s]
            return max(0.05, min(0.95, v))

        focal = b.get("focal_reward")
        for abits in range(1 << H):
            acts = tuple((abits >> (H - 1 - t)) & 1 for t in range(H))
            for sbits in range(1 << (H + 1)):
                states = tuple((sbits >> (H - t)) & 1 for t in range(H + 1))
                if states[0] != init:
                    continue
                p = 1.0
                for t in range(H):
                    s, a = states[t], acts[t]
                    pj = logits[pmap[t * 2 + s]]
                    p *= pj if a == 1 else (1.0 - pj)
                    p_tr = tp(t, s, a)
                    p *= p_tr if states[t + 1] == 1 else (1.0 - p_tr)
                if focal is not None:
                    nt = focal["noise_times"]
                    if len(nt) == 2:
                        p1 = logits[pmap[nt[0] * 2 + states[nt[0]]]]
                        p2 = logits[pmap[nt[1] * 2 + states[nt[1]]]]
                        r = focal["noise"] * (acts[nt[0]] - p1) * (acts[nt[1]] - p2)
                    else:
                        p1 = logits[pmap[nt[0] * 2 + states[nt[0]]]]
                        r = focal["noise"] * (acts[nt[0]] - p1)
                    r += sum(focal["w"] * (2 * acts[t] - 1) for t in range(H) if t not in nt)
                else:
                    r = trew[f"{states[H-1]},{acts[H-1]},{states[H]}"]
                for t in range(H):
                    j = pmap[t * 2 + states[t]]
                    g[j] += p * r * (acts[t] - logits[j])
    return g


def main() -> int:
    spec = json.load(sys.stdin)
    world = spec.get("world", "bernoulli_sequence_mdp")
    if world == "bernoulli_sequence_mdp":
        g = _bernoulli_sequence(spec)
        n = 1 << len(spec["probabilities"])
    elif world == "d002_shared_logits_mdp":
        g = _d002(spec)
        n = sum(1 << (4 * b["horizon"]) for b in spec["buckets"])
    else:
        raise ValueError(f"unknown world {world!r}")
    input_sha = hashlib.sha256(json.dumps(spec, sort_keys=True).encode("utf-8")).hexdigest()
    out = {
        "gradient": [float(x) for x in g],
        "oracle": "enumeration",
        "input_sha256": input_sha,
        "precision": "float64",
        "n_paths": n,
    }
    sys.stdout.write(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
