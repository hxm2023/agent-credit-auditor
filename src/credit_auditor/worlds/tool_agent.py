"""Controllable tool-agent tasks (Stage 2 of the real-trajectory bridge).

A tool-use-structured MDP with OBSERVATION-DEPENDENT decisions: at each step
the agent calls a tool (a_t = 1) or not (a_t = 0); the tool emits an
observation o_t (success/failure) that shifts the next call probability; the
terminal reward is a structured task score over the whole outcome path.

The task is exactly enumerable at small horizon (path probabilities =
product over steps of the prefix-dependent action prob x observation prob)
AND sampleable at large horizon — the two layers of the evidence bridge:
- exact layer: Auditor verdicts (bias/MSE/mechanism) from enumeration;
- sampled layer: fixed-budget MSE of the same estimators over sampled
  trajectory records, measured against an independent high-budget MC
  reference.

Score (score-function feature for the gradient estimand):
    s_t = a_t - pi_t(1 | obs prefix)   (centered action indicator)
so the full-policy-gradient target is g* = E[ R(tau) * sum_t s_t ].
The sibling/replay samplers support LATENT-NOISE COUPLING (paired replay):
the observation noise stream is drawn once and both arms consume the same
stream, which cancels continuation noise in contrasts exactly.

Task semantics (tool-use flavor):
    R(tau) = w_correct * I{phase-correct tool used} + w_ok * (#successful
    calls) - w_cost * length; two frozen task configs (tool_selection,
    evidence_chain) are registered below, each with a large sampling twin.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

_CALL_P_GOOD = 0.55  # P(call next) after a successful observation
_CALL_P_BAD = 0.85  # P(call next) after a failed observation
_SKIP_OBS_OK = 0.2  # P(observation ok | skipped the tool)


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    horizon: int
    tool_success: tuple[float, ...]  # p_t: success prob of tool t
    phase: tuple[int, int]  # (start, end) of the "correct tool" phase
    w_correct: float  # reward weight for using the phase-correct tool
    w_ok: float  # reward per successful call
    w_cost: float  # penalty per step


TOOL_SELECTION = TaskSpec(
    task_id="tool_selection",
    horizon=4,
    tool_success=(0.7, 0.4, 0.7, 0.4),
    phase=(1, 3),
    w_correct=4.0,
    w_ok=0.5,
    w_cost=0.2,
)
EVIDENCE_CHAIN = TaskSpec(
    task_id="evidence_chain",
    horizon=4,
    tool_success=(0.3, 0.8, 0.3, 0.8),
    phase=(0, 4),
    w_correct=3.0,
    w_ok=0.4,
    w_cost=0.1,
)

# scaled-up sampling twins (exact enumeration infeasible at H=12)
TOOL_SELECTION_LARGE = TaskSpec(
    task_id="tool_selection_large",
    horizon=12,
    tool_success=(0.7, 0.4, 0.7, 0.4, 0.7, 0.4, 0.7, 0.4, 0.7, 0.4, 0.7, 0.4),
    phase=(2, 10),
    w_correct=4.0,
    w_ok=0.5,
    w_cost=0.2,
)
EVIDENCE_CHAIN_LARGE = TaskSpec(
    task_id="evidence_chain_large",
    horizon=12,
    tool_success=(0.3, 0.8, 0.3, 0.8, 0.3, 0.8, 0.3, 0.8, 0.3, 0.8, 0.3, 0.8),
    phase=(0, 12),
    w_correct=3.0,
    w_ok=0.4,
    w_cost=0.1,
)


def action_prob(spec: TaskSpec, t: int, obs_prefix: tuple[int, ...]) -> float:
    """Behavior policy: P(a_t = 1 | observations so far)."""
    base = _CALL_P_GOOD if (not obs_prefix or obs_prefix[-1] == 1) else _CALL_P_BAD
    return base


def observation_prob(spec: TaskSpec, t: int, a: int) -> float:
    """P(o_t = 1 | a_t): a call succeeds with tool_success[t], a skip rarely."""
    return spec.tool_success[t] if a == 1 else _SKIP_OBS_OK


def reward(spec: TaskSpec, actions: tuple[int, ...]) -> float:
    """Structured task score over the outcome path."""
    total = -spec.w_cost * len(actions)
    for a in actions:
        if a == 1:
            total += spec.w_ok
    start, end = spec.phase
    if any(a == 1 for t, a in enumerate(actions) if start <= t < end):
        total += spec.w_correct
    return total


class ToolAgentTask:
    """Exact-enumerable + sampleable tool-agent task (one frozen TaskSpec)."""

    def __init__(self, spec: TaskSpec) -> None:
        self.spec = spec
        self.horizon = spec.horizon

    # ---- exact layer ----
    def all_paths(self) -> list[tuple[tuple[int, ...], tuple[int, ...], float, float]]:
        """Every (actions, observations, probability, reward) path."""
        out: list[tuple[tuple[int, ...], tuple[int, ...], float, float]] = []

        def rec(t: int, actions: list[int], obs: list[int], p: float) -> None:
            if t == self.horizon:
                out.append((tuple(actions), tuple(obs), p, reward(self.spec, tuple(actions))))
                return
            pa = action_prob(self.spec, t, tuple(obs))
            for a, p_a in ((0, 1.0 - pa), (1, pa)):
                po = observation_prob(self.spec, t, a)
                for o, p_o in ((0, 1.0 - po), (1, po)):
                    rec(t + 1, actions + [a], obs + [o], p * p_a * p_o)

        rec(0, [], [], 1.0)
        return out

    def _score(self, actions: tuple[int, ...], obs: tuple[int, ...]) -> list[float]:
        s: list[float] = []
        prefix: list[int] = []
        for t, a in enumerate(actions):
            s.append(a - action_prob(self.spec, t, tuple(prefix)))
            prefix.append(obs[t])
        return s

    def exact_target(self) -> float:
        """g* = E[ R(tau) * sum_t s_t ], s_t = a_t - pi_t(obs prefix)."""
        target = 0.0
        for actions, obs, p, r in self.all_paths():
            target += p * r * sum(self._score(actions, obs))
        return target

    def exact_distribution(self, estimator: str, **kw) -> list[tuple[float, float]]:
        """Exact (weight, value) distribution of a trajectory-view estimator.

        Estimators in estimators/bridge_estimators.py: 'dense' (R*s),
        'local_sibling' (contrast at t, coordinate t only),
        'paired_replay' (per-coordinate coupled contrast, zero at skipped
        coordinates), 'pc_rsg' (backbone + residual correction at q-sampled
        t). Sibling weights integrate over the sibling policy exactly.
        """
        from credit_auditor.estimators.bridge_estimators import exact_distribution

        return exact_distribution(self, estimator, **kw)

    def exact_estimator_moments(self, estimator: str, **kw) -> dict:
        dist = self.exact_distribution(estimator, **kw)
        mean = sum(w * v for w, v in dist)
        var = sum(w * (v - mean) ** 2 for w, v in dist)
        return {"mean": mean, "var": var, "n_outcomes": len(dist)}

    # ---- sampled layer ----
    def sample_rollout(self, rng) -> tuple[tuple[int, ...], tuple[int, ...], float]:
        """Sample (actions, observations, reward) from the behavior policy."""
        actions: list[int] = []
        obs: list[int] = []
        for t in range(self.horizon):
            pa = action_prob(self.spec, t, tuple(obs))
            a = 1 if rng.random() < pa else 0
            po = observation_prob(self.spec, t, a)
            o = 1 if rng.random() < po else 0
            actions.append(a)
            obs.append(o)
        return tuple(actions), tuple(obs), reward(self.spec, tuple(actions))

    @staticmethod
    def sibling_reward(spec: TaskSpec, actions: tuple[int, ...], t: int, a_prime: int) -> float:
        """Reward of the sibling trajectory: the backbone action vector with
        coordinate t REPLACED by a fresh draw a_prime (shared suffix, the
        paired-replay coupling — continuation noise cancels in the contrast)."""
        return reward(spec, actions[:t] + (a_prime,) + actions[t + 1 :])

    def to_records(
        self,
        rollouts: list[tuple[tuple[int, ...], tuple[int, ...], float]],
        policy_version: str = "v1",
    ) -> list[dict]:
        """Export rollouts as aca-trajectory-record-1.0 records. For the
        bridge, `generated_tokens` ARE the action values (0/1 decisions) —
        the score feature is s_t = a_t - pi_t; the real-trajectory adapter
        will map real token ids to decisions with its own feature map."""
        records = []
        for i, (actions, obs, r) in enumerate(rollouts):
            logprobs: list[float] = []
            probs: list[float] = []
            prefix: list[int] = []
            for t, a in enumerate(actions):
                pa = action_prob(self.spec, t, tuple(prefix))
                logprobs.append(math.log(pa) if a == 1 else math.log(1.0 - pa))
                probs.append(pa)
                prefix.append(obs[t])
            records.append(
                {
                    "trajectory_id": f"{self.spec.task_id}_{i}",
                    "policy_version": policy_version,
                    "generated_tokens": list(actions),
                    "action_mask": [1] * len(actions),
                    "old_logprobs": logprobs,
                    "behavior_probs": probs,
                    "rewards": {"final": r},
                    "termination_reason": "done",
                }
            )
        return records


def save_records(records: list[dict], path: Path) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
        newline="\n",
    )
