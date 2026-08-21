"""D002 shared-logit world (design §8.3, docs_only_semantic reconstruction).

Semantics per §8.3.1 (new frozen seeds; no historical parity claims):
- 3 shared action probabilities p_j ~ Uniform(0.15, 0.85); policy at decision
  (t, s) uses p_{map(t,s)}, logit parameterization with score (a - p_j).
- 4 buckets: late_reusable H=6, early_sensitive H=6, short_mixed H=3,
  medium_mixed H=5; depth candidates {2,4}/{2,4}/{1,2}/{1,3}.
- transitions: p(s'=1|t,s,a) = clip(base + offset + (2a-1)*effect, 0.05, 0.95)
  with per-(t,s) base/offset/effect draws and per-bucket scale vectors.
- parameter map: [0,1,2] placed first, remaining positions drawn from
  {0,1,2}, Fisher-Yates shuffled, paired per time (state-0, state-1), each
  shared logit used at least once.
- initial state Bernoulli(0.5); terminal reward table r(s,a,s') Bernoulli(0.5)
  with non-constant guard.
- reward = terminal r(s_{H-1}, a_{H-1}, s_H).

Gradient target (3-dim, logit space):
    dJ/dtheta_j = sum over decisions (t,s) with map(t,s)=j of E[R (a_t - p_j)].
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from credit_auditor.worlds.base import ExactWorld

BUCKET_SPECS = {
    "late_reusable": {"horizon": 6, "depths": (2, 4), "scales": (0.05, 0.05, 0.05, 0.05, 0.30, 0.35)},
    "early_sensitive": {"horizon": 6, "depths": (2, 4), "scales": (0.35, 0.30, 0.08, 0.08, 0.05, 0.05)},
    "short_mixed": {"horizon": 3, "depths": (1, 2), "scales": (0.20, 0.20, 0.20)},
    "medium_mixed": {"horizon": 5, "depths": (1, 3), "scales": (0.10, 0.30, 0.15, 0.30, 0.20)},
}
BUCKET_ORDER = ("late_reusable", "early_sensitive", "short_mixed", "medium_mixed")


@dataclass(frozen=True)
class D002Bucket:
    bucket_id: str
    horizon: int
    initial_state: int
    param_map: tuple[int, ...]  # index t*2 + s -> logit index
    transition_base: tuple[float, ...]  # length H
    transition_offset: tuple[float, ...]
    transition_effect: tuple[float, ...]  # per (t, s) — flattened t*2+s
    terminal_rewards: dict[tuple[int, int, int], float]  # (s, a, s') -> r
    focal_reward: dict | None = None  # semantic focal world config (D9)


@dataclass(frozen=True)
class D002Problem(ExactWorld):
    problem_id: str
    seed: int
    logits: tuple[float, ...]  # 3 probabilities p_j
    buckets: tuple[D002Bucket, ...]

    def horizon(self) -> int:
        return sum(b.horizon for b in self.buckets)

    def to_spec(self) -> dict:
        spec = {
            "world": "d002_shared_logits_mdp",
            "problem_id": self.problem_id,
            "seed": self.seed,
            "logits": list(self.logits),
            "buckets": [
                {
                    "bucket_id": b.bucket_id,
                    "horizon": b.horizon,
                    "initial_state": b.initial_state,
                    "param_map": list(b.param_map),
                    "transition_base": list(b.transition_base),
                    "transition_offset": list(b.transition_offset),
                    "transition_effect": list(b.transition_effect),
                    "terminal_rewards": {f"{s},{a},{sp}": r for (s, a, sp), r in b.terminal_rewards.items()},
                "focal_reward": b.focal_reward,
            }
            for b in self.buckets
        ],
        }
        return spec

    @classmethod
    def from_spec(cls, spec: dict) -> "D002Problem":
        buckets = []
        for b in spec["buckets"]:
            buckets.append(
                D002Bucket(
                    bucket_id=b["bucket_id"],
                    horizon=b["horizon"],
                    initial_state=b["initial_state"],
                    param_map=tuple(b["param_map"]),
                    transition_base=tuple(b["transition_base"]),
                    transition_offset=tuple(b["transition_offset"]),
                    transition_effect=tuple(b["transition_effect"]),
                    terminal_rewards={tuple(int(x) for x in k.split(",")): v for k, v in b["terminal_rewards"].items()},
                    focal_reward=b.get("focal_reward"),
                )
            )
        return cls(problem_id=spec["problem_id"], seed=spec["seed"], logits=tuple(spec["logits"]), buckets=tuple(buckets))


def _draw(key: str, i: int) -> float:
    h = hashlib.sha256(key.encode("utf-8") + b"::" + str(i).encode())
    return int.from_bytes(h.digest()[:8], "big") / 2**64


def _randbelow(key: str, i: int, n: int) -> int:
    return min(n - 1, int(_draw(key, i) * n))


def _clip(x: float) -> float:
    return max(0.05, min(0.95, x))


def generate_problem_focal(problem_id: str, seed: int, w: float = 0.05, noise: float = 1.0) -> D002Problem:
    """Semantic D002 world v2 (decision log D9): shared-logit Bernoulli world
    with deterministic states (s_{t+1} = a_t) and the m0-focal terminal reward
    R = noise*(2a_{n1}-1)(2a_{n2}-1) + sum_{t not in noise} w*(2a_t-1), where
    the noise times are the two middle times of each bucket (zero-target
    coordinates). The paired-replay contrast at a focal decision is exact and
    deterministic; the width parameter is therefore irrelevant, which is what
    the mechanism gate detects (width collapse)."""
    key = f"ACA-SEM-D002F::{seed}"
    logits = tuple(0.15 + 0.7 * _draw(key, i) for i in range(3))
    buckets: list[D002Bucket] = []
    for bi, bucket_id in enumerate(BUCKET_ORDER):
        spec = BUCKET_SPECS[bucket_id]
        H = spec["horizon"]
        # STATE-INDEPENDENT policy (decision log D9): both states at time t
        # share one logit, so decisions are independent and the centered-noise
        # zero-target property holds exactly. Each logit appears at least once.
        j_t = [_randbelow(key, 100 + bi * 100 + t, 3) for t in range(H)]
        used = set(j_t)
        for j in range(3):
            if j not in used:
                j_t[j % H] = j
        param_map = tuple(v for t in range(H) for v in (j_t[t], j_t[t]))
        # deterministic transitions: s_{t+1} = a_t (P=1)
        base = tuple(0.5 for _ in range(2 * H))
        offset = tuple(0.0 for _ in range(2 * H))
        effect = tuple(0.5 for _ in range(2 * H))  # p(s'=1|s,a) = 0.5 + (2a-1)*0.5 = a
        initial_state = 0
        # focal reward: NON-ADJACENT noise times (the state s_{n2} = a_{n2-1}
        # would correlate with a_{n1} if adjacent, breaking the centered-noise
        # zero-target property); per horizon the pair is fixed and frozen.
        _NOISE_PAIRS = {6: (2, 5), 5: (1, 4), 4: (0, 3), 3: (0, 2)}
        noise_times = _NOISE_PAIRS[H]
        rewards: dict[tuple[int, int, int], float] = {}
        for s in (0, 1):
            for a in (0, 1):
                for sp in (0, 1):
                    rewards[(s, a, sp)] = 0.0  # reward is action-based (focal structure)
        buckets.append(
            D002Bucket(
                bucket_id=bucket_id,
                horizon=H,
                initial_state=initial_state,
                param_map=tuple(param_map),
                transition_base=tuple(base),
                transition_offset=tuple(offset),
                transition_effect=tuple(effect),
                terminal_rewards=rewards,
                focal_reward={  # type: ignore[attr-defined]
                    "w": w,
                    "noise": noise,
                    "noise_times": noise_times,
                },
            )
        )
    return D002Problem(problem_id=problem_id, seed=seed, logits=logits, buckets=tuple(buckets))


def focal_reward(problem: D002Problem, b: D002Bucket, actions: tuple[int, ...]) -> float:
    """Terminal reward of the semantic focal world: CENTERED noise interaction
    (a_n1 - p(s_n1))(a_n2 - p(s_n2)) at the noise times (zero target regardless
    of p) plus small focal effects w(2a_t - 1) elsewhere. States are
    reconstructed deterministically: s_0 = initial, s_{t+1} = a_t."""
    cfg = b.focal_reward  # type: ignore[attr-defined]
    H = b.horizon
    states = [b.initial_state]
    for t in range(H):
        states.append(actions[t])
    nt = cfg["noise_times"]
    r = 0.0
    if len(nt) == 2:
        n1, n2 = nt
        p1 = action_prob(problem, b, n1, states[n1])
        p2 = action_prob(problem, b, n2, states[n2])
        r = cfg["noise"] * (actions[n1] - p1) * (actions[n2] - p2)
    else:
        n1 = nt[0]
        p1 = action_prob(problem, b, n1, states[n1])
        r = cfg["noise"] * (actions[n1] - p1)
    r += sum(cfg["w"] * (2 * actions[t] - 1) for t in range(H) if t not in nt)
    return r


def generate_problem(problem_id: str, seed: int) -> D002Problem:
    key = f"ACA-SEM-D002::world::{seed}"
    logits = tuple(0.15 + 0.7 * _draw(key, i) for i in range(3))
    buckets: list[D002Bucket] = []
    for bi, bucket_id in enumerate(BUCKET_ORDER):
        spec = BUCKET_SPECS[bucket_id]
        H = spec["horizon"]
        # parameter map: [0,1,2] then remaining, shuffled, paired per time
        positions: list[int] = [0, 1, 2] + [_randbelow(key, 100 + bi * 100 + i, 3) for i in range(2 * H - 3)]
        for i in range(len(positions) - 1, 0, -1):
            j = _randbelow(key, 200 + bi * 100 + i, i + 1)
            positions[i], positions[j] = positions[j], positions[i]
        param_map = tuple(positions)
        # transitions per (t, s)
        base: list[float] = []
        offset: list[float] = []
        effect: list[float] = []
        for t in range(H):
            scale_t = spec["scales"][t]
            for s in range(2):
                base.append(0.2 + 0.6 * _draw(key, 300 + bi * 100 + t * 2 + s))
                offset.append(-0.1 + 0.2 * _draw(key, 400 + bi * 100 + t * 2 + s))
                effect.append(-scale_t + 2 * scale_t * _draw(key, 500 + bi * 100 + t * 2 + s))
        initial_state = 1 if _draw(key, 600 + bi) < 0.5 else 0
        # terminal reward table with non-constant guard
        rewards: dict[tuple[int, int, int], float] = {}
        vals = [1.0 if _draw(key, 700 + bi * 10 + (s * 4 + a * 2 + sp)) < 0.5 else 0.0 for s in (0, 1) for a in (0, 1) for sp in (0, 1)]
        if len(set(vals)) == 1:
            vals[-1] = 1.0 - vals[-1]
        for s in (0, 1):
            for a in (0, 1):
                for sp in (0, 1):
                    rewards[(s, a, sp)] = vals[s * 4 + a * 2 + sp]
        buckets.append(
            D002Bucket(
                bucket_id=bucket_id,
                horizon=H,
                initial_state=initial_state,
                param_map=tuple(param_map),
                transition_base=tuple(base),
                transition_offset=tuple(offset),
                transition_effect=tuple(effect),
                terminal_rewards=rewards,
            )
        )
    return D002Problem(problem_id=problem_id, seed=seed, logits=logits, buckets=tuple(buckets))


def transition_prob(b: D002Bucket, t: int, s: int, a: int) -> float:
    if b.focal_reward is not None:
        # semantic focal world: exactly deterministic transitions s_{t+1} = a_t
        return float(a)
    idx = t * 2 + s
    return _clip(b.transition_base[idx] + b.transition_offset[idx] + (2 * a - 1) * b.transition_effect[idx])


def action_prob(problem: D002Problem, b: D002Bucket, t: int, s: int) -> float:
    return problem.logits[b.param_map[t * 2 + s]]


def score(problem: D002Problem, b: D002Bucket, t: int, s: int, a: int) -> float:
    """Score wrt the shared logit at (t, s): a - p (logit parameterization)."""
    return float(a) - action_prob(problem, b, t, s)


def enumerate_bucket(problem: D002Problem, b: D002Bucket) -> list[tuple[tuple[int, ...], tuple[int, ...], float, float]]:
    """Exact path enumeration: (states, actions, prob, reward).

    Path probability includes the policy draw of each action:
    P(tau) = P(s_0) * prod_t pi(a_t|s_t) * P(s_{t+1}|s_t,a_t)."""
    H = b.horizon
    out: list[tuple[tuple[int, ...], tuple[int, ...], float, float]] = []
    for abits in range(1 << H):
        actions = tuple((abits >> (H - 1 - t)) & 1 for t in range(H))
        for sbits in range(1 << (H + 1)):
            states = tuple((sbits >> (H - t)) & 1 for t in range(H + 1))
            if states[0] != b.initial_state:
                continue
            p = 1.0
            for t in range(H):
                s = states[t]
                a = actions[t]
                p_a = action_prob(problem, b, t, s)
                p *= (p_a if a == 1 else 1.0 - p_a)
                p_tr = transition_prob(b, t, s, a)
                p *= p_tr if states[t + 1] == 1 else 1.0 - p_tr
            if b.focal_reward is not None:
                r = focal_reward(problem, b, actions)
            else:
                r = b.terminal_rewards[(states[H - 1], actions[H - 1], states[H])]
            out.append((states, actions, p, r))
    return out


def true_gradient(problem: D002Problem) -> np.ndarray:
    g = np.zeros(3)
    for b in problem.buckets:
        for states, actions, p, r in enumerate_bucket(problem, b):
            for t in range(b.horizon):
                j = b.param_map[t * 2 + states[t]]
                g[j] += p * r * (actions[t] - problem.logits[j])
    return g
