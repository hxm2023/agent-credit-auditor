"""World base types: serializable exact worlds + weighted-vector distributions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ExactWorld(Protocol):
    """An exactly enumerable world. `to_spec()` must produce a JSON-serializable
    spec that both the primary path and the independent oracle process accept."""

    def to_spec(self) -> dict[str, Any]: ...

    def horizon(self) -> int: ...


@dataclass
class WeightedVector:
    """A weighted vector (estimator outcome). Weights are probabilities;
    they must sum to 1 for a distribution, but can be arbitrary for joint
    probability-of-cycles computations."""

    weight: float
    vector: tuple[float, ...]

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"WeightedVector(w={self.weight:.4g}, v={self.vector})"


def normalize(vectors: list[WeightedVector]) -> list[WeightedVector]:
    total = sum(v.weight for v in vectors)
    if total <= 0:
        raise ValueError("weighted vectors have non-positive total weight")
    return [WeightedVector(v.weight / total, v.vector) for v in vectors]


def serialize_world_spec(world: ExactWorld) -> dict[str, Any]:
    return world.to_spec()


def reward_map_to_spec(rewards: dict[tuple[int, ...], float]) -> dict[str, float]:
    return {",".join(map(str, a)): r for a, r in rewards.items()}


def reward_map_from_spec(spec_rewards: dict[str, float]) -> dict[tuple[int, ...], float]:
    return {tuple(int(x) for x in k.split(",")): float(v) for k, v in spec_rewards.items()}
