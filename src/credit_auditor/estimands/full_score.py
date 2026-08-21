"""FULL_SCORE_GRADIENT estimand (§5.1, §5.2): E[ R(tau) * score(tau) ]."""
from __future__ import annotations

import numpy as np

from credit_auditor.worlds.bernoulli_sequence import BernoulliSequenceMDP

ESTIMAND_ID = "full_score_gradient"


def target(world: BernoulliSequenceMDP) -> np.ndarray:
    return world.true_gradient()


def expected_reward(world: BernoulliSequenceMDP) -> float:
    return world.expected_reward()
