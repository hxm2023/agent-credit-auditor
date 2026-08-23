"""Fraction-exact Bernoulli world (design §10.3).

Small worlds (H <= 6) can be enumerated with exact fractions.Fraction
arithmetic, pushing oracle alignment from float64 tolerance to EXACT equality
(mismatch == 0). The serialized spec carries Fraction values as "a/b" strings
so the self-contained oracle processes can also compute exactly.

Float64 enumeration remains the documented path for larger worlds; the
Fraction layer is the exact cross-validation for the designed cases.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from credit_auditor.worlds.base import ExactWorld


def frac(s: str | Fraction | int) -> Fraction:
    return s if isinstance(s, Fraction) else Fraction(s)


@dataclass(frozen=True)
class BernoulliFractionMDP(ExactWorld):
    probabilities: tuple[Fraction, ...]
    rewards: Mapping[tuple[int, ...], Fraction]

    def __post_init__(self) -> None:
        H = len(self.probabilities)
        for a in self.rewards:
            if not (isinstance(a, tuple) and len(a) == H and all(x in (0, 1) for x in a)):
                raise ValueError(f"reward key must be a {H}-bit action tuple")
        if len(self.rewards) != (1 << H):
            raise ValueError(f"rewards must cover all {1 << H} action sequences")
        for p in self.probabilities:
            if not (0 < p < 1):
                raise ValueError(f"probabilities must be in (0, 1): {p}")

    @property
    def horizon(self) -> int:
        return len(self.probabilities)

    def all_paths(self) -> list[tuple[tuple[int, ...], Fraction]]:
        H = self.horizon
        out: list[tuple[tuple[int, ...], Fraction]] = []
        for bits in range(1 << H):
            a = tuple((bits >> (H - 1 - t)) & 1 for t in range(H))
            p = Fraction(1)
            for t, at in enumerate(a):
                p *= self.probabilities[t] if at else 1 - self.probabilities[t]
            out.append((a, p))
        return out

    def score(self, a: tuple[int, ...]) -> tuple[Fraction, ...]:
        return tuple(Fraction(int(at)) - p for at, p in zip(a, self.probabilities))

    def true_gradient(self) -> tuple[Fraction, ...]:
        g = [Fraction(0)] * self.horizon
        for a, p in self.all_paths():
            r = self.rewards[a]
            for t in range(self.horizon):
                g[t] += p * r * (int(a[t]) - self.probabilities[t])
        return tuple(g)

    def expected_reward(self) -> Fraction:
        return sum((p * self.rewards[a] for a, p in self.all_paths()), Fraction(0))

    def to_spec(self) -> dict:
        return {
            "world": "bernoulli_fraction_mdp",
            "probabilities": [f"{p.numerator}/{p.denominator}" for p in self.probabilities],
            "rewards": {",".join(map(str, a)): f"{r.numerator}/{r.denominator}" for a, r in self.rewards.items()},
        }

    @classmethod
    def from_float_world(cls, world) -> BernoulliFractionMDP:
        """Exact conversion of a float BernoulliSequenceMDP via its rational
        origin (seeded worlds use dyadic rationals, exactly representable)."""
        probs = tuple(Fraction(world.probabilities[t]).limit_denominator(2**60) for t in range(world.horizon))
        rewards = {a: Fraction(r).limit_denominator(2**60) for a, r in world.rewards.items()}
        return cls(probabilities=probs, rewards=rewards)
