"""Evidence bridge tests (Stage 2): the exact layer must agree with the
sampled layer (cycle MC), the MC reference must agree with the exact target,
and the exact predictor must track the measured fixed-budget MSE."""

from __future__ import annotations

import random
import random as _random

import pytest

from credit_auditor.estimators.bridge_estimators import (
    dense_estimate,
    exact_distribution,
    sample_cycle_records,
)
from credit_auditor.experiments.evidence_bridge import (
    SIBLING_T,
    _cycle_value,
    _estimator_variants,
    exact_layer,
    run_bridge,
)
from credit_auditor.oracles.mc_reference import mc_target
from credit_auditor.worlds.tool_agent import (
    EVIDENCE_CHAIN,
    TOOL_SELECTION,
    ToolAgentTask,
)


def _world():
    return ToolAgentTask(TOOL_SELECTION)


def test_exact_target_matches_mc():
    world = _world()
    exact = world.exact_target()
    mc = mc_target(world, n=200_000, seed=7)
    assert abs(mc["target"] - exact) <= 6 * mc["se"]


def test_exact_distribution_weights_sum_to_one():
    for estimator, kw in _estimator_variants(_world()):
        dist = exact_distribution(_world(), estimator, **kw)
        total = sum(w for w, _ in dist)
        assert abs(total - 1.0) < 1e-9, (estimator, total)


def test_dense_cycle_mc_matches_exact():
    world = _world()
    exact_mean = sum(w * v for w, v in exact_distribution(world, "dense"))
    rng = random.Random(42)
    n = 50_000
    acc = 0.0
    acc2 = 0.0
    for _ in range(n):
        records, _ = sample_cycle_records(world, rng, "dense")
        v = dense_estimate(records[0])
        acc += v
        acc2 += v * v
    mc_mean = acc / n
    se = max(acc2 / n - mc_mean**2, 0.0) ** 0.5 / n**0.5
    assert abs(mc_mean - exact_mean) <= 6 * se


@pytest.mark.parametrize("estimator", ["dense", "local_sibling", "paired_replay", "pc_rsg"])
def test_cycle_mc_matches_exact(estimator):
    world = _world()
    kw = (
        {"t": SIBLING_T}
        if estimator == "local_sibling"
        else ({"q": tuple(1.0 / world.horizon for _ in range(world.horizon))} if estimator == "pc_rsg" else {})
    )
    exact_mean = sum(w * v for w, v in exact_distribution(world, estimator, **kw))
    rng = random.Random(42)
    n = 30_000
    acc = 0.0
    acc2 = 0.0
    for _ in range(n):
        records, _ = sample_cycle_records(world, rng, estimator, **kw)
        v = _cycle_value(estimator, records, **kw)
        acc += v
        acc2 += v * v
    mc_mean = acc / n
    se = max(acc2 / n - mc_mean**2, 0.0) ** 0.5 / n**0.5
    assert abs(mc_mean - exact_mean) <= 6 * se


def test_dense_unbiased_and_local_sibling_biased_for_full_gradient():
    world = _world()
    target = world.exact_target()
    dense_mean = sum(w * v for w, v in exact_distribution(world, "dense"))
    sib_mean = sum(w * v for w, v in exact_distribution(world, "local_sibling", t=SIBLING_T))
    assert abs(dense_mean - target) < 1e-9
    assert abs(sib_mean - target) > 1e-3  # T003: local contrast is not the full gradient


def test_exact_layer_predicts_sampled_mse():
    """The exact predictor (var*cost/B + bias^2) must track the measured
    fixed-budget MSE within a factor of 2 on every estimator."""
    import statistics

    from credit_auditor.experiments.evidence_bridge import BUDGET_TRANSITIONS, REPLICATES

    world = _world()
    target, rows = exact_layer(world)
    for ex in rows:
        predicted = ex["var"] * ex["cost"] / BUDGET_TRANSITIONS + ex["bias"] ** 2
        cost = ex["cost"]
        n_cycles = max(1, int(BUDGET_TRANSITIONS // cost))
        ests = []
        for rep in range(REPLICATES):
            rng = _random.Random(20260824 + rep)
            acc = 0.0
            for _ in range(n_cycles):
                records, _ = sample_cycle_records(
                    world,
                    rng,
                    ex["estimator"],
                    **(
                        {"t": SIBLING_T}
                        if ex["estimator"] == "local_sibling"
                        else (
                            {"q": tuple(1.0 / world.horizon for _ in range(world.horizon))}
                            if ex["estimator"] == "pc_rsg"
                            else {}
                        )
                    ),
                )
                acc += _cycle_value(
                    ex["estimator"],
                    records,
                    **(
                        {"t": SIBLING_T}
                        if ex["estimator"] == "local_sibling"
                        else (
                            {"q": tuple(1.0 / world.horizon for _ in range(world.horizon))}
                            if ex["estimator"] == "pc_rsg"
                            else {}
                        )
                    ),
                )
            ests.append(acc / n_cycles)
        mse = statistics.mean((e - target) ** 2 for e in ests)
        assert 0.5 <= mse / predicted <= 2.0, (ex["estimator"], mse, predicted)


def test_bridge_run_writes_report_and_records(tmp_path):
    summary = run_bridge(tmp_path)
    assert (tmp_path / "REPORT.md").is_file()
    assert (tmp_path / "result.json").is_file()
    for task in summary["tasks"]:
        assert task["mc_agreement_gate"] in ("PASS", "MC_ONLY")
    # exported records must pass the trajectory-level audit (self-consistency)
    from credit_auditor.audit.trajectory_audit import audit_trajectory_dir

    for f in sorted(tmp_path.glob("*_records.jsonl")):
        a = audit_trajectory_dir(f)
        assert a["consistent"] is True, f


def test_records_export_schema():
    world = _world()
    rng = random.Random(1)
    actions, obs, r = world.sample_rollout(rng)
    rec = world.to_records([(actions, obs, r)])[0]
    for key in (
        "trajectory_id",
        "policy_version",
        "generated_tokens",
        "action_mask",
        "old_logprobs",
        "behavior_probs",
        "rewards",
    ):
        assert key in rec
    assert len(rec["generated_tokens"]) == world.horizon
    assert len(rec["behavior_probs"]) == world.horizon


def test_task_variants_are_distinct():
    a = ToolAgentTask(TOOL_SELECTION).exact_target()
    b = ToolAgentTask(EVIDENCE_CHAIN).exact_target()
    assert a != b
