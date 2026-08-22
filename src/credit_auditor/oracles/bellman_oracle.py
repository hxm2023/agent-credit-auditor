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


def _d002_bellman(spec):
    """D002 shared-logit world via DP (different algorithm from enumeration):
    dJ/dtheta_j = sum_{(t,s): map=j} reach(t,s) * p_j(1-p_j) * (Q(t,s,1)-Q(t,s,0))."""
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

        def e_r(t, s, a):
            p1 = tp(t, s, a)
            return p1 * trew[f"{s},{a},1"] + (1 - p1) * trew[f"{s},{a},0"]

        focal = b.get("focal_reward")
        q = {}
        qpol = {}
        for t in range(H - 1, -1, -1):
            for s in (0, 1):
                for a in (0, 1):
                    if focal is not None:
                        # action-based reward: expected future focal reward is
                        # zero-mean; only the current focal term contributes
                        nt = focal["noise_times"]
                        q[(t, s, a)] = focal["w"] * (2 * a - 1) if t not in nt else 0.0
                    elif t == H - 1:
                        # terminal reward arrives at the last transition only
                        q[(t, s, a)] = e_r(t, s, a)
                    else:
                        q[(t, s, a)] = sum(
                            (tp(t, s, a) if sp == 1 else 1 - tp(t, s, a)) * qpol.get((t + 1, sp), 0.0)
                            for sp in (0, 1)
                        )
                pa = logits[pmap[t * 2 + s]]
                qpol[(t, s)] = pa * q[(t, s, 1)] + (1 - pa) * q[(t, s, 0)]
        reach = {(0, init): 1.0}
        for t in range(H - 1):
            for s in (0, 1):
                pr = reach.get((t, s), 0.0)
                if pr <= 0:
                    continue
                pa = logits[pmap[t * 2 + s]]
                for a in (0, 1):
                    p_tr = tp(t, s, a)
                    for sp in (0, 1):
                        reach[(t + 1, sp)] = reach.get((t + 1, sp), 0.0) + pr * (pa if a == 1 else 1 - pa) * (p_tr if sp == 1 else 1 - p_tr)
        for t in range(H):
            for s in (0, 1):
                j = pmap[t * 2 + s]
                pj = logits[j]
                g[j] += reach.get((t, s), 0.0) * pj * (1.0 - pj) * (q[(t, s, 1)] - q[(t, s, 0)])
    return g


def _bernoulli_fraction_bellman(spec):
    """EXACT Bellman DP with fractions.Fraction (design §10.3): output as
    "a/b" strings, enabling mismatch == 0 against the primary."""
    from fractions import Fraction

    probs = [Fraction(x) for x in spec["probabilities"]]
    rewards = spec["rewards"]
    h = len(probs)
    q: dict[str, Fraction] = {}
    for key, r in rewards.items():
        q[key] = Fraction(r)
    for t in range(h - 1, -1, -1):
        p_t = probs[t]
        for bits in range(1 << t):
            h_pref = ",".join(str((bits >> (t - 1 - tt)) & 1) for tt in range(t)) if t > 0 else ""
            prefix = h_pref if h_pref else ""
            q1 = q[(prefix + ",1") if prefix else "1"]
            q0 = q[(prefix + ",0") if prefix else "0"]
            q[prefix] = p_t * q1 + (1 - p_t) * q0
    g = [Fraction(0)] * h
    for t in range(h):
        p_t = probs[t]
        for bits in range(1 << t):
            h_pref = ",".join(str((bits >> (t - 1 - tt)) & 1) for tt in range(t)) if t > 0 else ""
            prefix = h_pref if h_pref else ""
            p_h = Fraction(1)
            for tt in range(t):
                ht = (bits >> (t - 1 - tt)) & 1
                p_h *= probs[tt] if ht else 1 - probs[tt]
            k1 = (prefix + ",1") if prefix else "1"
            k0 = (prefix + ",0") if prefix else "0"
            g[t] += p_h * p_t * (1 - p_t) * (q[k1] - q[k0])
    return [f"{x.numerator}/{x.denominator}" for x in g]


def _bernoulli_bellman(spec):
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
    return g, h


def main() -> int:
    spec = json.load(sys.stdin)
    world = spec.get("world", "bernoulli_sequence_mdp")
    if world == "d002_shared_logits_mdp":
        g = _d002_bellman(spec)
        oracle_name = "bellman_d002"
        n_states = sum(2 * (b["horizon"] + 1) for b in spec["buckets"])
    elif world == "bernoulli_fraction_mdp":
        g = _bernoulli_fraction_bellman(spec)
        oracle_name = "bellman_fraction"
        n_states = sum(1 << t for t in range(len(spec["probabilities"]) + 1))
    else:
        g, h = _bernoulli_bellman(spec)
        oracle_name = "bellman"
        n_states = sum(1 << t for t in range(h + 1))

    input_sha = hashlib.sha256(json.dumps(spec, sort_keys=True).encode("utf-8")).hexdigest()
    # Fraction worlds return "a/b" strings for EXACT comparison (== 0).
    is_fraction = world == "bernoulli_fraction_mdp"
    out = {
        "gradient": [x for x in g] if is_fraction else [float(x) for x in g],
        "oracle": oracle_name,
        "input_sha256": input_sha,
        "precision": "exact_fraction" if is_fraction else "float64",
        "n_states": n_states,
    }
    sys.stdout.write(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
