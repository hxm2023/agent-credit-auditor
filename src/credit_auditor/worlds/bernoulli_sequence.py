"""BernoulliSequenceMDP (design §8.1) with exact enumeration.

Actions are independent Bernoulli per time step, a_t ~ Bernoulli(p_t),
p_t = sigma(theta_t). Terminal reward R(a_0..a_{H-1}).

Score function gradient wrt logit theta_t: (a_t - p_t).
    J(theta) = sum_tau P(tau) R(tau)
    dJ/dtheta_t = E[ R(tau) * (a_t - p_t) ]          (§5.1, §8.1)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from credit_auditor.worlds.base import ExactWorld


@dataclass(frozen=True)
class BernoulliSequenceMDP(ExactWorld):
    probabilities: tuple[float, ...]
    rewards: Mapping[tuple[int, ...], float]

    def __post_init__(self) -> None:
        H = len(self.probabilities)
        for a in self.rewards:
            if not (isinstance(a, tuple) and len(a) == H and all(x in (0, 1) for x in a)):
                raise ValueError(f"reward key must be a {H}-bit action tuple: {a}")
        if len(self.rewards) != (1 << H):
            raise ValueError(f"rewards must cover all {1 << H} action sequences")

    @property
    def horizon(self) -> int:
        return len(self.probabilities)

    def all_paths(self) -> list[tuple[tuple[int, ...], float]]:
        """All 2^H action sequences with exact path probabilities."""
        H = self.horizon
        probs = self.probabilities
        paths: list[tuple[tuple[int, ...], float]] = []
        for bits in range(1 << H):
            a = tuple((bits >> (H - 1 - t)) & 1 for t in range(H))
            p = 1.0
            for t, at in enumerate(a):
                p *= probs[t] if at else (1.0 - probs[t])
            paths.append((a, p))
        return paths

    def score(self, a: tuple[int, ...]) -> tuple[float, ...]:
        """Score function (a_t - p_t) for logit parameters."""
        return tuple(int(at) - p for at, p in zip(a, self.probabilities))

    def reward(self, a: tuple[int, ...]) -> float:
        return self.rewards[a]

    def true_gradient(self) -> np.ndarray:
        g = np.zeros(self.horizon, dtype=np.float64)
        for a, p in self.all_paths():
            g += p * self.rewards[a] * np.asarray(self.score(a))
        return g

    def expected_reward(self) -> float:
        return sum(p * self.rewards[a] for a, p in self.all_paths())

    # ------------------------------------------------------------------
    # Q-function (used by the primary side; the oracle has its own copy)
    # ------------------------------------------------------------------
    def q_values(self) -> dict[tuple[int, ...], float]:
        """Expected reward from every prefix (Bellman recursion from the end)."""
        q: dict[tuple[int, ...], float] = {}
        for a, _ in self.all_paths():
            q[a] = self.rewards[a]
        for t in range(self.horizon - 1, -1, -1):
            p_t = self.probabilities[t]
            for bits in range(1 << t):
                h = tuple((bits >> (t - 1 - tt)) & 1 for tt in range(t))
                q[h] = p_t * q[h + (1,)] + (1 - p_t) * q[h + (0,)]
        return q

    def to_spec(self) -> dict:
        from credit_auditor.worlds.base import reward_map_to_spec

        return {
            "world": "bernoulli_sequence_mdp",
            "probabilities": list(self.probabilities),
            "rewards": reward_map_to_spec(dict(self.rewards)),
        }

    @classmethod
    def from_spec(cls, spec: dict) -> BernoulliSequenceMDP:
        from credit_auditor.worlds.base import reward_map_from_spec

        return cls(
            probabilities=tuple(float(x) for x in spec["probabilities"]),
            rewards=reward_map_from_spec(spec["rewards"]),
        )


def make_world(probabilities: Sequence[float], rewards: Mapping[tuple[int, ...], float]) -> BernoulliSequenceMDP:
    return BernoulliSequenceMDP(tuple(float(p) for p in probabilities), rewards)


def deterministic_world(seed: int, horizon: int, scale: float = 1.0) -> BernoulliSequenceMDP:
    """Frozen deterministic world from a seed (used by tests and protocols)."""
    import hashlib

    def draw(i: int) -> float:
        h = hashlib.sha256(f"ACA-WORLD::{seed}::{i}".encode()).digest()
        return int.from_bytes(h[:8], "big") / 2**64

    probs = [0.15 + 0.7 * draw(i) for i in range(horizon)]
    rewards: dict[tuple[int, ...], float] = {}
    for bits in range(1 << horizon):
        a = tuple((bits >> (horizon - 1 - t)) & 1 for t in range(horizon))
        rewards[a] = scale * (2 * draw(100 + bits) - 1)
    return BernoulliSequenceMDP(tuple(probs), rewards)
