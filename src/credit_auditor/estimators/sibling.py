"""Sibling contrast estimators (§5.3, §9.2).

Cycle = (trajectory tau, sibling action vector a' drawn with its own policy
probability). Contrast at depth t uses the same suffix (PAIRED REPLAY /
latent-noise coupling), which cancels continuation noise exactly.

- local_sibling_distribution(world, t): updates ONLY coordinate t.
  Unbiased for LOCAL_DECISION_GRADIENT(t); biased for FULL_SCORE_GRADIENT.
- propagated_sibling_distribution(world, t): applies the t-contrast to ALL
  coordinates (the classic local-to-prefix misuse). Expectation at t' < t is
  ZERO (contrast is mean-zero conditional on the prefix), so it is biased for
  the full gradient whenever the target is nonzero there (T003).
- paired_sibling_distribution(world, skip=()): updates every focal coordinate
  with its own paired contrast; zero contribution at skipped coordinates.
  Unbiased for the full gradient; the paired-replay mechanism control is the
  uncoupled variant (paired_sibling_uncoupled).
"""

from __future__ import annotations

from credit_auditor.worlds.base import WeightedVector
from credit_auditor.worlds.bernoulli_sequence import BernoulliSequenceMDP


def _sibling_weight(a: tuple[int, ...], world: BernoulliSequenceMDP) -> float:
    w = 1.0
    for t, at in enumerate(a):
        w *= world.probabilities[t] if at else (1.0 - world.probabilities[t])
    return w


def local_sibling_distribution(world: BernoulliSequenceMDP, t: int) -> list[WeightedVector]:
    H = world.horizon
    out: list[WeightedVector] = []
    for a, p in world.all_paths():
        for sb in range(1 << H):
            ap = tuple((sb >> (H - 1 - tt)) & 1 for tt in range(H))
            w_sib = _sibling_weight(ap, world)
            alt = a[:t] + (ap[t],) + a[t + 1 :]
            delta = world.rewards[a] - world.rewards[alt]
            vec = [0.0] * H
            vec[t] = delta * (a[t] - world.probabilities[t])
            out.append(WeightedVector(p * w_sib, tuple(vec)))
    return out


def propagated_sibling_distribution(world: BernoulliSequenceMDP, t: int) -> list[WeightedVector]:
    H = world.horizon
    out: list[WeightedVector] = []
    for a, p in world.all_paths():
        for sb in range(1 << H):
            ap = tuple((sb >> (H - 1 - tt)) & 1 for tt in range(H))
            w_sib = _sibling_weight(ap, world)
            alt = a[:t] + (ap[t],) + a[t + 1 :]
            delta = world.rewards[a] - world.rewards[alt]
            s = world.score(a)
            vec = tuple(delta * st for st in s)
            out.append(WeightedVector(p * w_sib, vec))
    return out


def paired_sibling_distribution(world: BernoulliSequenceMDP, skip: tuple[int, ...] = ()) -> list[WeightedVector]:
    """Paired-replay full-gradient estimator (design §8.2 case 5 / matched
    cost positive). Coordinates in `skip` contribute exactly zero (their
    target is zero in the designed world); all other coordinates use their
    own paired contrast with the SAME suffix (coupled)."""
    H = world.horizon
    out: list[WeightedVector] = []
    for a, p in world.all_paths():
        for sb in range(1 << H):
            ap = tuple((sb >> (H - 1 - tt)) & 1 for tt in range(H))
            w_sib = _sibling_weight(ap, world)
            vec = [0.0] * H
            for t in range(H):
                if t in skip:
                    continue
                alt = a[:t] + (ap[t],) + a[t + 1 :]
                delta = world.rewards[a] - world.rewards[alt]
                vec[t] = delta * (a[t] - world.probabilities[t])
            out.append(WeightedVector(p * w_sib, tuple(vec)))
    return out


def paired_sibling_uncoupled_distribution(
    world: BernoulliSequenceMDP, skip: tuple[int, ...] = ()
) -> list[WeightedVector]:
    """Mechanism control: sibling's suffix is a FRESH independent draw
    (uncoupled). Expectation is unchanged (still unbiased) but continuation
    noise no longer cancels, so fixed-budget MSE collapses on noise-heavy
    worlds — proving the positive result requires the PAIRED mechanism."""
    H = world.horizon
    out: list[WeightedVector] = []
    for a, p in world.all_paths():
        for sb in range(1 << H):
            ap = tuple((sb >> (H - 1 - tt)) & 1 for tt in range(H))
            w_sib = _sibling_weight(ap, world)
            for fbits in range(1 << (H - 1)):
                fsuff = tuple((fbits >> (H - 2 - s)) & 1 for s in range(H - 1))
                w_fs = 1.0
                for tt, ft in enumerate(fsuff):
                    w_fs *= world.probabilities[tt + 1] if ft else (1.0 - world.probabilities[tt + 1])
                vec = [0.0] * H
                for t in range(H):
                    if t in skip:
                        continue
                    fresh = a[:t] + (ap[t],) + fsuff[: H - t - 1]
                    delta = world.rewards[a] - world.rewards[fresh]
                    vec[t] = delta * (a[t] - world.probabilities[t])
                out.append(WeightedVector(p * w_sib * w_fs, tuple(vec)))
    return out


def mechanism_signature_local() -> dict:
    return {"estimator_family": "sibling", "contrast_source": "local_sibling", "updated_coordinates": "single_branch"}


def mechanism_signature_propagated() -> dict:
    return {
        "estimator_family": "sibling",
        "contrast_source": "local_sibling",
        "updated_coordinates": "all_including_prefix",
    }


def mechanism_signature_paired() -> dict:
    return {"estimator_family": "sibling", "contrast_source": "paired_replay", "updated_coordinates": "all_focal"}
