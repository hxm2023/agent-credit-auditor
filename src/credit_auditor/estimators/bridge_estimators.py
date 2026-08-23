"""Trajectory-view estimators for the evidence bridge (Stage 2).

The SAME estimator definitions run on both bridge layers:
- exact layer: `exact_distribution` enumerates the full (path, sibling) cycle
  distribution of each estimator over the tool-agent task — its mean and
  variance are the Auditor's verdicts (bias vs the exact target, intrinsic
  cycle variance, mechanism shape);
- sampled layer: estimators consume trajectory RECORDS only
  (aca-trajectory-record-1.0: generated_tokens = actions, behavior_probs =
  pi_t, rewards.final). They never touch the environment, so the same code
  path runs when real LLM trajectories arrive (Stage 3, gated on the Guard
  schema package). Cycle sampling (backbone + coupled sibling replays) lives
  in `sample_cycle_records` here; the MC reference never imports this module.

Estimators (formulas over the task's score s_t = a_t - pi_t):
- dense:           g = R(tau) * sum_t s_t                (unbiased, cost H)
- local_sibling:   g = (R(tau) - R(tau'^t)) * s_t         (coordinate t only;
                   unbiased for the local effect at t, biased for the full
                   gradient unless the target is zero elsewhere — T003)
- paired_replay:   g = sum_t (R(tau) - R(tau'^t)) * s_t   (per-coordinate
                   coupled contrast; unbiased for the full gradient)
- pc_rsg:          g = R(tau)*sum s_t + (contrast_t - R*s_t)/q_t at a
                   q-sampled coordinate t (unbiased per-coordinate correction;
                   V001 shows calibration accurate != fixed-budget utility)
"""

from __future__ import annotations

from credit_auditor.worlds.tool_agent import ToolAgentTask, action_prob, reward

# ---------------- exact layer ----------------


def _path_score(world: ToolAgentTask, actions: tuple[int, ...], obs: tuple[int, ...]) -> list[float]:
    s: list[float] = []
    prefix: list[int] = []
    for t, a in enumerate(actions):
        s.append(a - action_prob(world.spec, t, tuple(prefix)))
        prefix.append(obs[t])
    return s


def exact_distribution(world: ToolAgentTask, estimator: str, **kw) -> list[tuple[float, float]]:
    """Exact (weight, value) cycle distribution of one estimator."""
    H = world.horizon
    paths = world.all_paths()  # (actions, obs, p, r)
    out: list[tuple[float, float]] = []

    if estimator == "dense":
        for actions, obs, p, r in paths:
            out.append((p, r * sum(_path_score(world, actions, obs))))
        return out

    if estimator == "local_sibling":
        t = kw["t"]
        for actions, obs, p, r in paths:
            s_t = _path_score(world, actions, obs)[t]
            for flip in (0, 1):
                pa = action_prob(world.spec, t, tuple(obs[:t]))
                w_sib = pa if flip == 1 else 1.0 - pa
                alt = actions[:t] + (flip,) + actions[t + 1 :]
                delta = r - reward(world.spec, alt)
                vec = [0.0] * H
                vec[t] = delta * s_t
                out.append((p * w_sib, sum(vec)))
        return out

    if estimator == "paired_replay":
        skip = set(kw.get("skip", ()))
        for actions, obs, p, r in paths:
            s = _path_score(world, actions, obs)
            for mask in range(1 << H):
                w_sib = 1.0
                value = 0.0
                for t in range(H):
                    if t in skip:
                        continue
                    pa = action_prob(world.spec, t, tuple(obs[:t]))
                    flip = (mask >> t) & 1
                    w_sib *= pa if flip else 1.0 - pa
                    alt = actions[:t] + (flip,) + actions[t + 1 :]
                    value += (r - reward(world.spec, alt)) * s[t]
                out.append((p * w_sib, value))
        return out

    if estimator == "pc_rsg":
        q = kw["q"]
        t = kw.get("t")
        for actions, obs, p, r in paths:
            s = _path_score(world, actions, obs)
            backbone = r * sum(s)
            for tt in [t] if t is not None else range(H):
                for flip in (0, 1):
                    pa = action_prob(world.spec, tt, tuple(obs[:tt]))
                    w_sib = pa if flip == 1 else 1.0 - pa
                    alt = actions[:tt] + (flip,) + actions[tt + 1 :]
                    contrast = r - reward(world.spec, alt)
                    correction = (contrast - r * s[tt]) / q[tt]
                    out.append((p * q[tt] * w_sib, backbone + correction))
        return out

    raise ValueError(f"unknown bridge estimator: {estimator}")


# ---------------- record consumers (sampled layer) ----------------


def _score_from_record(record: dict) -> list[float]:
    return [a - p for a, p in zip(record["generated_tokens"], record["behavior_probs"])]


def dense_estimate(record: dict) -> float:
    """One cycle = one trajectory record: g = R * sum_t (a_t - pi_t)."""
    return record["rewards"]["final"] * sum(_score_from_record(record))


def local_sibling_estimate(record: dict, sibling_record: dict, t: int) -> float:
    """g = (R(tau) - R(tau'^t)) * s_t — coordinate t only."""
    s_t = _score_from_record(record)[t]
    delta = record["rewards"]["final"] - sibling_record["rewards"]["final"]
    return delta * s_t


def paired_replay_estimate(record: dict, sibling_records: list[dict], skip: tuple[int, ...] = ()) -> float:
    """g = sum_t (R(tau) - R(tau'^t)) * s_t over focal coordinates."""
    s = _score_from_record(record)
    value = 0.0
    for t, sibling in enumerate(sibling_records):
        if t in skip:
            continue
        value += (record["rewards"]["final"] - sibling["rewards"]["final"]) * s[t]
    return value


def pc_rsg_estimate(record: dict, sibling_record: dict, q: tuple[float, ...]) -> float:
    """Backbone + residual correction at the q-sampled coordinate t (carried
    as sibling metadata)."""
    t = sibling_record["sibling_at"]
    s = _score_from_record(record)
    r = record["rewards"]["final"]
    backbone = r * sum(s)
    contrast = r - sibling_record["rewards"]["final"]
    return backbone + (contrast - r * s[t]) / q[t]


# ---------------- cycle sampling (env-side; records only downstream) ----------------


def sample_cycle_records(world: ToolAgentTask, rng, estimator: str, **kw) -> tuple[list[dict], int]:
    """Sample one cycle and return its trajectory RECORDS (backbone first)
    plus the transition cost. Sibling semantics: the sibling action at t is a
    FRESH DRAW from the policy; the suffix is SHARED (paired-replay coupling,
    same as the exact layer)."""
    actions, obs, r = world.sample_rollout(rng)
    backbone = world.to_records([(actions, obs, r)])[0]
    H = world.horizon

    def _sibling_record(t: int, a_prime: int, r_alt: float) -> dict:
        alt_actions = actions[:t] + (a_prime,) + actions[t + 1 :]
        rec = world.to_records([(alt_actions, obs, r_alt)])[0]
        rec["sibling_at"] = t  # q-sampled/residual coordinate (estimator metadata)
        return rec

    def _fresh_draw(t: int) -> int:
        pa = action_prob(world.spec, t, tuple(obs[:t]))
        return 1 if rng.random() < pa else 0

    if estimator == "dense":
        return [backbone], H

    if estimator == "local_sibling":
        t = kw["t"]
        a_prime = _fresh_draw(t)
        r_alt = ToolAgentTask.sibling_reward(world.spec, actions, t, a_prime)
        return [backbone, _sibling_record(t, a_prime, r_alt)], H + (H - t + 1)

    if estimator == "paired_replay":
        siblings: list[dict] = []
        cost = H
        for t in range(H):
            if t in kw.get("skip", ()):
                continue
            a_prime = _fresh_draw(t)
            r_alt = ToolAgentTask.sibling_reward(world.spec, actions, t, a_prime)
            siblings.append(_sibling_record(t, a_prime, r_alt))
            cost += H - t + 1
        return [backbone] + siblings, cost

    if estimator == "pc_rsg":
        q = kw["q"]
        t = rng.choices(range(H), weights=list(q), k=1)[0]
        a_prime = _fresh_draw(t)
        r_alt = ToolAgentTask.sibling_reward(world.spec, actions, t, a_prime)
        return [backbone, _sibling_record(t, a_prime, r_alt)], H + (H - t + 1)

    raise ValueError(f"unknown bridge estimator: {estimator}")


def estimator_cost(world: ToolAgentTask, estimator: str, **kw) -> float:
    """Expected transitions per cycle (matched-budget accounting)."""
    H = world.horizon
    if estimator == "dense":
        return float(H)
    if estimator == "local_sibling":
        return float(H + (H - kw["t"] + 1))
    if estimator == "paired_replay":
        skip = set(kw.get("skip", ()))
        return float(H + sum(H - t + 1 for t in range(H) if t not in skip))
    if estimator == "pc_rsg":
        q = kw["q"]
        return float(H + sum(q[t] * (H - t + 1) for t in range(H)))
    raise ValueError(f"unknown bridge estimator: {estimator}")
