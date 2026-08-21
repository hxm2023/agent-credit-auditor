"""D002 branching estimator (design §8.3, §9.2 global-K / variable-width).

Cycle for one bucket = (sampled trajectory tau, sibling action a'_d at the
mapped depth d, K/2 independent continuation pairs). Value at the 3 shared
logits:
    g_j = sum over decisions (t,s_t) != (d,s_d), map=j of R(tau)(a_t - p_j)
        + DeltaBar (a_d - p_j)   [at the branched decision]
with DeltaBar = (2/K) sum_k [ R(s_d, a_d, xi_k) - R(s_d, a'_d, xi'_k) ] over
K/2 independent continuation pairs (fresh draws; latent-noise-independent
restore per §7.2).

Exact moments are computed WITHOUT enumerating continuation pairs, via
conditional value functions Q(a)=E[R|s_d,a], sigma^2(a)=Var(R|s_d,a):
    E[DeltaBar | a, a'] = Q(a) - Q(a')
    Var(DeltaBar | a, a') = (sigma^2(a) + sigma^2(a')) / (K/2)
Cross terms with the dense part factor through Q(s_d, a_d): the trajectory's
future reward is independent of the fresh pairs given (s_d, a_d).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from credit_auditor.worlds.d002_shared_logits import (
    D002Problem,
    action_prob,
    enumerate_bucket,
    transition_prob,
)


@dataclass
class BucketMoments:
    mean: np.ndarray  # 3 shared-logit coordinates
    second_moment: np.ndarray  # 3x3


@dataclass
class _Dp:
    q: dict  # (t, s, a) -> E[remaining reward | t,s,a]
    qpol: dict  # (t, s) -> E[remaining reward | t,s] following policy
    q2: dict  # (t, s, a) -> E[remaining reward^2 | t,s,a]
    q2pol: dict  # (t, s) -> E[remaining reward^2 | t,s] following policy


def _dp(problem: D002Problem, b) -> _Dp:
    H = b.horizon

    def e_r(t, s, a):
        """E[r_t | s_t=s, a_t=a] (r in {0,1}, so E[r^2] = E[r])."""
        p1 = transition_prob(b, t, s, a)
        return p1 * b.terminal_rewards[(s, a, 1)] + (1 - p1) * b.terminal_rewards[(s, a, 0)]

    q: dict = {}
    qpol: dict = {}
    q2: dict = {}
    q2pol: dict = {}
    for t in range(H - 1, -1, -1):
        for s in (0, 1):
            for a in (0, 1):
                p1 = transition_prob(b, t, s, a)
                if t == H - 1:
                    # terminal reward arrives at the last transition only
                    q[(t, s, a)] = e_r(t, s, a)
                    q2[(t, s, a)] = e_r(t, s, a)  # r in {0,1} -> E[r^2] = E[r]
                else:
                    e_future = sum(
                        (p1 if sp == 1 else 1 - p1) * qpol.get((t + 1, sp), 0.0) for sp in (0, 1)
                    )
                    e2_future = sum(
                        (p1 if sp == 1 else 1 - p1) * q2pol.get((t + 1, sp), 0.0) for sp in (0, 1)
                    )
                    q[(t, s, a)] = e_future
                    q2[(t, s, a)] = e2_future
            pa = action_prob(problem, b, t, s)
            qpol[(t, s)] = pa * q[(t, s, 1)] + (1 - pa) * q[(t, s, 0)]
            q2pol[(t, s)] = pa * q2[(t, s, 1)] + (1 - pa) * q2[(t, s, 0)]
    return _Dp(q=q, qpol=qpol, q2=q2, q2pol=q2pol)


def dense_bucket_moments(problem: D002Problem, b) -> BucketMoments:
    mean = np.zeros(3)
    mm = np.zeros((3, 3))
    for states, actions, p, r in enumerate_bucket(problem, b):
        scores = np.zeros(3)
        for t in range(b.horizon):
            j = b.param_map[t * 2 + states[t]]
            scores[j] += actions[t] - problem.logits[j]
        v = r * scores
        mean += p * v
        mm += p * np.outer(v, v)
    return BucketMoments(mean=mean, second_moment=mm)


def branching_bucket_moments(problem: D002Problem, b, d: int, width: int) -> BucketMoments:
    H = b.horizon
    Kb = max(1, width // 2)
    dp = _dp(problem, b)
    mean = np.zeros(3)
    mm = np.zeros((3, 3))
    for states, actions, p, r in enumerate_bucket(problem, b):
        s_d = states[d]
        j = b.param_map[d * 2 + s_d]
        p_j = problem.logits[j]
        a_d = actions[d]
        scores = np.zeros(3)
        for t in range(b.horizon):
            if t == d:
                continue
            jj = b.param_map[t * 2 + states[t]]
            scores[jj] += actions[t] - problem.logits[jj]
        dense_v = r * scores
        score_d = a_d - p_j
        q_a = dp.q[(d, s_d, a_d)]
        s2_a = max(dp.q2[(d, s_d, a_d)] - q_a * q_a, 0.0)
        for a_other in (0, 1):
            w_sib = p_j if a_other == 1 else (1.0 - p_j)
            if a_other == a_d:
                delta_mean = 0.0
                var_delta = 0.0
            else:
                q_op = dp.q[(d, s_d, a_other)]
                s2_op = max(dp.q2[(d, s_d, a_other)] - q_op * q_op, 0.0)
                delta_mean = q_a - q_op  # E[R(main) - R(other)]
                var_delta = (s2_a + s2_op) / Kb
            contrib = delta_mean * score_d
            mean += p * w_sib * dense_v
            mean[j] += p * w_sib * contrib
            mm += p * w_sib * np.outer(dense_v, dense_v)
            cross = dense_v * contrib
            mm[j, :] += p * w_sib * cross
            mm[:, j] += p * w_sib * cross
            mm[j, j] += p * w_sib * (var_delta + delta_mean * delta_mean) * score_d * score_d
    return BucketMoments(mean=mean, second_moment=mm)


def estimator_moments(problem: D002Problem, mapping: dict) -> BucketMoments:
    """Problem-level estimator moments for a mapping {bucket_id -> (depth, width)}.
    Dense per bucket is (None, 1) with cost = horizon. Buckets are independent
    MDPs, so E[g_a g_b^T] = mean_a mean_b^T for a != b (cross terms must be
    added explicitly)."""
    parts = []
    for b in problem.buckets:
        d, K = mapping[b.bucket_id]
        if K == 1:
            parts.append(dense_bucket_moments(problem, b))
        elif b.focal_reward is not None:
            parts.append(paired_replay_all_bucket_moments(problem, b, d))
        else:
            parts.append(ksample_bucket_moments(problem, b, d, K))
    mean = sum((part.mean for part in parts), np.zeros(3))
    mm = sum((part.second_moment for part in parts), np.zeros((3, 3)))
    for i, a in enumerate(parts):
        for j, c in enumerate(parts):
            if i != j:
                mm += np.outer(a.mean, c.mean)
    return BucketMoments(mean=mean, second_moment=mm)


def moments_to_stats(bm: BucketMoments, target: np.ndarray) -> dict:
    bias = bm.mean - np.asarray(target, dtype=np.float64)
    var_trace = float(np.trace(bm.second_moment - np.outer(bm.mean, bm.mean)))
    bias_sq = float(bias @ bias)
    return {"mean": bm.mean.tolist(), "bias": bias.tolist(), "bias_sq": bias_sq, "var_trace": var_trace, "mse": bias_sq + var_trace}


def ksample_bucket_moments(problem: D002Problem, b, d: int, K: int) -> BucketMoments:
    """K-sample continuation-averaging estimator (the D002-style branching
    structure): sample ONE prefix through depth d (state s_d included, action
    a_d excluded), then K INDEPENDENT continuations; estimator =
    (1/K) sum_k R(tau_k) score(tau_k). Variance reduces ~K at every decision
    while the shared prefix couples the samples.

    Exact moments: mean = dense mean; second moment =
        (1/K) mm_dense + (1 - 1/K) E[ mu(pre) mu(pre)^T ]
    where mu(pre) = E[g | prefix] = Q(s_d) s_pre + g_from(s_d)."""
    H = b.horizon
    dp = _dp(problem, b)
    # dense single-trajectory moments
    md = dense_bucket_moments(problem, b)
    mean = md.mean
    # per-decision gradient contribution G(t,s) = p(1-p)(Q1 - Q0)
    G = {}
    for t in range(H):
        for s in (0, 1):
            j = b.param_map[t * 2 + s]
            pj = problem.logits[j]
            G[(t, s)] = pj * (1 - pj) * (dp.q[(t, s, 1)] - dp.q[(t, s, 0)])
    # g_from(s_d) = expected gradient contribution of decisions t >= d given s_d
    g_from: dict[int, np.ndarray] = {}
    for s_d in (0, 1):
        reach = {s_d: 1.0}
        acc = np.zeros(3)
        for t in range(d, H):
            for s in (0, 1):
                pr = reach.get(s, 0.0)
                if pr <= 0:
                    continue
                j = b.param_map[t * 2 + s]
                acc[j] += pr * G[(t, s)]
            if t < H - 1:
                new_reach = {}
                for s in (0, 1):
                    pr = reach.get(s, 0.0)
                    if pr <= 0:
                        continue
                    pa = action_prob(problem, b, t, s)
                    for a in (0, 1):
                        p_tr = transition_prob(b, t, s, a)
                        for sp in (0, 1):
                            new_reach[sp] = new_reach.get(sp, 0.0) + pr * (pa if a == 1 else 1 - pa) * (p_tr if sp == 1 else 1 - p_tr)
                reach = new_reach
        g_from[s_d] = acc
    # E[mu(pre) mu(pre)^T] over the prefix distribution (states 0..d, actions 0..d-1)
    e_mumu = np.zeros((3, 3))
    for sbits in range(1 << (d + 1)):
        states = tuple((sbits >> (d - t)) & 1 for t in range(d + 1))
        if states[0] != b.initial_state:
            continue
        for abits in range(1 << d):
            actions = tuple((abits >> (d - 1 - t)) & 1 for t in range(d))
            p = 1.0
            s_pre = np.zeros(3)
            ok = True
            for t in range(d):
                s = states[t]
                a = actions[t]
                pa = action_prob(problem, b, t, s)
                p *= (pa if a == 1 else 1 - pa)
                p_tr = transition_prob(b, t, s, a)
                p *= p_tr if states[t + 1] == 1 else 1 - p_tr
                j = b.param_map[t * 2 + s]
                s_pre[j] += a - problem.logits[j]
            s_d = states[d]
            mu = dp.qpol[(d, s_d)] * s_pre + g_from[s_d]
            e_mumu += p * np.outer(mu, mu)
    mm = (1.0 / K) * md.second_moment + (1.0 - 1.0 / K) * e_mumu
    return BucketMoments(mean=mean, second_moment=mm)


def paired_replay_all_bucket_moments(problem: D002Problem, b, d: int) -> BucketMoments:
    """Paired-replay branching at every decision t >= d (decision log D9):
    the cycle samples a trajectory and, at each branched decision, draws the
    sibling action a'_t (policy probability) and forms the coupled contrast
    Delta_t = R(tau with a'_t) - R(tau). The reward depends only on the action
    sequence (focal world), so the contrast is exact and deterministic;
    noise-time decisions (zero target) contribute zero. Dense terms remain
    for t < d. Enumerated exactly over (trajectory, sibling-vector) pairs."""
    H = b.horizon
    cfg = b.focal_reward
    noise_times = cfg["noise_times"] if cfg else ()
    out_mean = np.zeros(3)
    out_mm = np.zeros((3, 3))
    paths = enumerate_bucket(problem, b)
    for states, actions, p, r in paths:
        # dense terms for t < d (reward r is the full-trajectory reward)
        scores_dense = np.zeros(3)
        for t in range(d):
            j = b.param_map[t * 2 + states[t]]
            scores_dense[j] += actions[t] - problem.logits[j]
        # sibling vectors over the branched decisions [d, H)
        branched = [t for t in range(d, H) if t not in noise_times]
        for sbits in range(1 << len(branched)):
            w_sib = 1.0
            sib_actions = {}
            for i, t in enumerate(branched):
                a_sib = (sbits >> (len(branched) - 1 - i)) & 1
                pj = problem.logits[b.param_map[t * 2 + states[t]]]
                w_sib *= pj if a_sib == 1 else (1.0 - pj)
                sib_actions[t] = a_sib
            delta_total = np.zeros(3)
            for t in branched:
                # single-flip counterfactual: only a_t replaced (paired replay);
                # contrast = main branch minus sibling branch (E = local effect)
                alt = list(actions)
                alt[t] = sib_actions[t]
                contrast = r - focal_reward_value(problem, b, tuple(alt))
                j = b.param_map[t * 2 + states[t]]
                delta_total[j] += contrast * (actions[t] - problem.logits[j])
            v = r * scores_dense + delta_total
            out_mean += p * w_sib * v
            out_mm += p * w_sib * np.outer(v, v)
    return BucketMoments(mean=out_mean, second_moment=out_mm)


def focal_reward_value(problem: D002Problem, b, actions: tuple[int, ...]) -> float:
    from credit_auditor.worlds.d002_shared_logits import focal_reward
    return focal_reward(problem, b, actions)


def dense_optimal_constant_moments(problem: D002Problem, b) -> BucketMoments:
    """Per-logit optimal constant c_j = E[R s_j^2]/E[s_j^2]; g_j = (R - c_j) s_j."""
    paths = enumerate_bucket(problem, b)
    num = np.zeros(3)
    den = np.zeros(3)
    for states, actions, p, r in enumerate_bucket(problem, b):
        scores = np.zeros(3)
        for t in range(b.horizon):
            j = b.param_map[t * 2 + states[t]]
            scores[j] += actions[t] - problem.logits[j]
        num += p * r * scores ** 2
        den += p * scores ** 2
    c = np.where(den > 0, num / np.maximum(den, 1e-30), 0.0)
    mean = np.zeros(3)
    mm = np.zeros((3, 3))
    for states, actions, p, r in paths:
        scores = np.zeros(3)
        for t in range(b.horizon):
            j = b.param_map[t * 2 + states[t]]
            scores[j] += actions[t] - problem.logits[j]
        v = (r - c) * scores
        mean += p * v
        mm += p * np.outer(v, v)
    return BucketMoments(mean=mean, second_moment=mm)


def root_rloo_moments(problem: D002Problem, b) -> BucketMoments:
    """Root-RLOO over 2-sample sets: g = 0.5[(R1-R2)s1 + (R2-R1)s2], exact
    moments via single-trajectory statistics (scores are mean-zero)."""
    paths = enumerate_bucket(problem, b)
    a2 = np.zeros((3, 3))  # E[R^2 s s^T]
    a1 = np.zeros((3, 3))  # E[R s s^T]
    e_r2 = 0.0
    e_r = 0.0
    ess = np.zeros((3, 3))  # E[s s^T]
    bvec = np.zeros(3)  # E[R s]
    for states, actions, p, r in paths:
        scores = np.zeros(3)
        for t in range(b.horizon):
            j = b.param_map[t * 2 + states[t]]
            scores[j] += actions[t] - problem.logits[j]
        outer = np.outer(scores, scores)
        a2 += p * r * r * outer
        a1 += p * r * outer
        e_r2 += p * r * r
        e_r += p * r
        ess += p * outer
        bvec += p * r * scores
    # E[g g^T] expansion for independent pairs with E[s] = 0:
    # g = (R1-R2)(s1-s2)/2 ; E[g g^T] = 0.25 * [2 A2 + 2 E[R^2] E[ss^T]
    #                          - 4 E[R] E[R s s^T] - 4 E[R s] E[R s]^T]
    mm = 0.25 * (2 * a2 + 2 * e_r2 * ess - 4 * e_r * a1 - 4 * np.outer(bvec, bvec))
    return BucketMoments(mean=bvec, second_moment=mm)


def mapping_cycle_cost(problem: D002Problem, mapping: dict) -> float:
    total = 0.0
    for b in problem.buckets:
        d, K = mapping[b.bucket_id]
        if K == 1:
            total += b.horizon
        elif b.focal_reward is not None:
            total += b.horizon + sum((b.horizon - t) + 1 for t in range(d, b.horizon))
        else:
            total += d + K * (b.horizon - d) + (K - 1)
    return total


def fixed_budget_mse_from_moments(mean: np.ndarray, mm: np.ndarray, target: np.ndarray, budget: int, cycle_cost: float) -> float | None:
    """MSE at a fixed budget (floor complete cycles). None when infeasible."""
    if cycle_cost <= 0 or budget < cycle_cost:
        return None
    n = int(budget // cycle_cost)
    bias = mean - np.asarray(target, dtype=np.float64)
    bias_sq = float(bias @ bias)
    var_trace = float(np.trace(mm - np.outer(mean, mean)))
    return bias_sq + var_trace / n
