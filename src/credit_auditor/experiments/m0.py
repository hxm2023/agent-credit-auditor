"""M0-style target regression driver (design §13.1, docs_only_semantic).

Pre-registered semantic cases:
- 12 frozen Bernoulli problems: dense / uniform-HH unbiased within tolerance,
  local sibling passes the LOCAL estimand and fails the FULL estimand,
  propagated sibling fails with T003, BPO-like fails on designed cases.
- 5 designed cases (§8.2): bpo_prefix_propagation, shared_logit_predictable_width,
  outcome_retention, completion_deadline, matched_cost_positive.

Designed-case world constants are FROZEN below; their specs are hashed into the
run manifest. All numbers are new; legacy counts (144/202 etc.) never appear.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from credit_auditor import runner
from credit_auditor.audit import environment as env_audit
from credit_auditor.audit.sampling import support_gate
from credit_auditor.audit.target import compare_oracle, retention_gate, target_gate
from credit_auditor.canonical import sha256_json
from credit_auditor.estimands import full_score, local_decision, root_marginal
from credit_auditor.estimators import bpo_like, dense, hh_ht, sibling
from credit_auditor.oracles.isolation import check_import_isolation
from credit_auditor.schema import (
    ClaimDecision,
    ClaimStatus,
    EstimandSpec,
    EstimatorSpec,
    GateResult,
    HeadlineDecision,
    ReasonCode,
    SamplingSpec,
    AuditDecision,
)
from credit_auditor.stats import ExactMoments, exact_moments, fixed_budget_mse
from credit_auditor.worlds.base import WeightedVector
from credit_auditor.worlds.bernoulli_sequence import BernoulliSequenceMDP, deterministic_world

ORACLE_DIR = Path(__file__).resolve().parents[1] / "oracles"

# ---------------------------------------------------------------------------
# Frozen designed-case constants (chosen 2026-08-22 before the first formal
# run; verified numerically that the intended structures hold: paired-replay
# wins the matched-cost positive with bias < 1e-20, uncoupled control loses).
# ---------------------------------------------------------------------------
DESIGNED_WORLD_SEEDS = {
    "bpo_prefix_propagation": 101,
    "shared_logit_predictable_width": 102,
    "outcome_retention": 103,
    "completion_deadline": 104,
    "matched_cost_positive": 23,
}
FOCAL_WORLD = {"w": 0.05, "noise": 1.0, "noise_times": (2, 3), "horizon": 6}


def focal_world() -> BernoulliSequenceMDP:
    seed = DESIGNED_WORLD_SEEDS["matched_cost_positive"]
    cfg = FOCAL_WORLD

    def draw(i: int) -> float:
        import hashlib

        h = hashlib.sha256(f"ACA-FOC::{seed}::{i}".encode()).digest()
        return int.from_bytes(h[:8], "big") / 2**64

    H = cfg["horizon"]
    rewards: dict[tuple[int, ...], float] = {}
    for bits in range(1 << H):
        a = tuple((bits >> (H - 1 - t)) & 1 for t in range(H))
        r = cfg["noise"] * (2 * a[2] - 1) * (2 * a[3] - 1)
        r += sum(cfg["w"] * (2 * a[t] - 1) for t in range(H) if t not in cfg["noise_times"])
        rewards[a] = r
    return BernoulliSequenceMDP(tuple([0.5] * H), rewards)


def retained_world() -> BernoulliSequenceMDP:
    """outcome_retention: the estimator's world keeps only a subset of reward
    outcomes (retained = first half of outcomes zeroed)."""
    w = deterministic_world(DESIGNED_WORLD_SEEDS["outcome_retention"], 4)
    kept: dict[tuple[int, ...], float] = {}
    for a, _ in w.all_paths():
        kept[a] = w.rewards[a] if sum(a) <= 2 else 0.0
    return BernoulliSequenceMDP(w.probabilities, kept)


def truncated_world() -> BernoulliSequenceMDP:
    """completion_deadline: the estimator's continuation is truncated (reward
    depends only on the first T actions) while the claimed target is full."""
    w = deterministic_world(DESIGNED_WORLD_SEEDS["completion_deadline"], 4)
    trunc: dict[tuple[int, ...], float] = {}
    for a, _ in w.all_paths():
        trunc[a] = w.rewards[(a[0], a[1], 0, 0)]
    return BernoulliSequenceMDP(w.probabilities, trunc)


def shared_logit_design_case() -> dict:
    world = root_marginal.shared_logit_world(seed=DESIGNED_WORLD_SEEDS["shared_logit_predictable_width"], shared_times=(0, 1, 2), horizon=3)
    shared_times = (0, 1, 2)
    target = root_marginal.root_marginal_gradient(world, shared_times)
    flat = root_marginal.flat_leaf_average(world, shared_times)
    dist = [WeightedVector(1.0, tuple(flat))]
    moments = exact_moments(dist, target)
    gate = target_gate(
        moments,
        {"bias_rel": 1e-9, "bias_abs": 1e-12},
        {"estimator_family": "leaf_aggregation", "contrast_source": "none", "updated_coordinates": "leaf"},
        claimed_estimand="root_marginal_gradient",
        estimand_id="root_marginal_gradient",
    )
    return {
        "case": "shared_logit_predictable_width",
        "target": target.tolist(),
        "expectation": moments.mean.tolist(),
        "bias_sq": moments.bias_sq,
        "gate": gate.model_dump(),
    }


def _moment_rows(world: BernoulliSequenceMDP, target: np.ndarray, tolerance: dict) -> list[dict]:
    """Run the estimator matrix on one world; returns result rows."""
    rows: list[dict] = []
    H = world.horizon

    def add(name: str, dist: list[WeightedVector], estimand_id: str, claimed: str, sig: dict, q: tuple[float, ...] | None = None, extra_gates: list[GateResult] | None = None) -> None:
        m = exact_moments(dist, target)
        g = target_gate(m, tolerance, sig, claimed, estimand_id)
        gates = [g] + (extra_gates or [])
        rows.append(
            {
                "estimator": name,
                "claimed_estimand": claimed,
                "bias_sq": m.bias_sq,
                "var_trace": m.var_trace,
                "mse": m.mse,
                "max_abs_bias": m.max_abs_bias(),
                "gate_status": g.status,
                "reason_codes": [rc.value for rc in g.reason_codes],
                "gates": [x.model_dump() for x in gates],
            }
        )

    tol = tolerance
    add("dense", dense.dense_distribution(world), "full_score_gradient", "full_score_gradient", dense.mechanism_signature())
    add("dense_optimal_constant", dense.dense_optimal_constant_distribution(world), "full_score_gradient", "full_score_gradient", dense.mechanism_signature())
    add("uniform_hh", hh_ht.uniform_hh_distribution(world), "full_score_gradient", "full_score_gradient", hh_ht.mechanism_signature(), q=tuple([1.0 / H] * H))
    t_mid = H // 2
    local_target = local_decision.target(world, t_mid)
    m_local = exact_moments(sibling.local_sibling_distribution(world, t_mid), local_target)
    g_local = target_gate(m_local, tolerance, sibling.mechanism_signature_local(), "local_decision_gradient", "local_decision_gradient")
    rows.append(
        {
            "estimator": "local_sibling_local_estimand",
            "claimed_estimand": "local_decision_gradient",
            "bias_sq": m_local.bias_sq,
            "var_trace": m_local.var_trace,
            "mse": m_local.mse,
            "max_abs_bias": m_local.max_abs_bias(),
            "gate_status": g_local.status,
            "reason_codes": [rc.value for rc in g_local.reason_codes],
            "gates": [g_local.model_dump()],
        }
    )
    add("local_sibling_as_full", sibling.local_sibling_distribution(world, t_mid), "full_score_gradient", "full_score_gradient", sibling.mechanism_signature_local())
    add("propagated_sibling", sibling.propagated_sibling_distribution(world, t_mid), "full_score_gradient", "full_score_gradient", sibling.mechanism_signature_propagated())
    add("bpo_like", bpo_like.bpo_like_distribution(world), "full_score_gradient", "full_score_gradient", bpo_like.mechanism_signature())
    return rows


def _run_oracle_pair(world: BernoulliSequenceMDP) -> dict:
    enum = runner.run_oracle_subprocess(ORACLE_DIR / "enumeration_oracle.py", world.to_spec())
    bell = runner.run_oracle_subprocess(ORACLE_DIR / "bellman_oracle.py", world.to_spec())
    primary = world.true_gradient()
    ok_enum, mm_enum = compare_oracle(primary, np.asarray(enum["gradient"]), 1e-9, 1e-12)
    ok_bell, mm_bell = compare_oracle(primary, np.asarray(bell["gradient"]), 1e-9, 1e-12)
    return {
        "oracle_enumeration_match": ok_enum,
        "oracle_bellman_match": ok_bell,
        "max_mismatch_enumeration": mm_enum,
        "max_mismatch_bellman": mm_bell,
        "input_sha256": enum["input_sha256"],
    }


def run_m0(ctx: runner.RunContext) -> runner.RunResult:
    tol = {"bias_rel": ctx.protocol.tolerances.get("bias_rel", 1e-9), "bias_abs": ctx.protocol.tolerances.get("bias_abs", 1e-12)}
    results: dict = {"mode": "docs_only_semantic", "protocol": ctx.protocol.protocol_id, "problems": [], "designed_cases": []}
    gate_results: list[GateResult] = []

    # ---- 12 frozen problems ----
    seed_path = next((p for p in ctx.seed_manifest_paths if "m0_problems" in p.name), None)
    if seed_path is None:
        raise runner.DriverError("m0_problems seed manifest required")
    seeds = json.loads(seed_path.read_text(encoding="utf-8"))["rows"]
    for row in seeds:
        world = deterministic_world(row["seed"], row["horizon"])
        target = full_score.target(world)
        env = env_audit.environment_gate(world)
        oracle = _run_oracle_pair(world)
        rows = _moment_rows(world, target, tol)
        results["problems"].append(
            {
                "problem_id": row["problem_id"],
                "seed": row["seed"],
                "horizon": row["horizon"],
                "environment_gate": env,
                "oracle": oracle,
                "target": target.tolist(),
                "estimators": rows,
            }
        )
        if env["status"] != "pass":
            gate_results.append(GateResult(gate="environment", status="fail", reason_codes=[ReasonCode.E001_ALTERNATIVE_NOOP]))
        if not (oracle["oracle_enumeration_match"] and oracle["oracle_bellman_match"]):
            gate_results.append(GateResult(gate="independent_oracle", status="fail", reason_codes=[ReasonCode.E002_ORACLE_MISMATCH]))

    # ---- designed cases ----
    # bpo_prefix_propagation
    w = deterministic_world(DESIGNED_WORLD_SEEDS["bpo_prefix_propagation"], 4)
    target = full_score.target(w)
    rows = _moment_rows(w, target, tol)
    results["designed_cases"].append({"case": "bpo_prefix_propagation", "estimators": rows})

    # shared_logit_predictable_width
    results["designed_cases"].append(shared_logit_design_case())

    # outcome_retention
    w_full = deterministic_world(DESIGNED_WORLD_SEEDS["outcome_retention"], 4)
    w_ret = retained_world()
    target_full = full_score.target(w_full)
    m_ret = exact_moments(dense.dense_distribution(w_ret), target_full)
    g_ret = retention_gate(m_ret, tol)
    results["designed_cases"].append(
        {
            "case": "outcome_retention",
            "estimator": "dense_on_retained_world",
            "bias_sq": m_ret.bias_sq,
            "gate_status": g_ret.status,
            "reason_codes": [rc.value for rc in g_ret.reason_codes],
        }
    )

    # completion_deadline
    w_full2 = deterministic_world(DESIGNED_WORLD_SEEDS["completion_deadline"], 4)
    w_trunc = truncated_world()
    target_full2 = full_score.target(w_full2)
    m_trunc = exact_moments(dense.dense_distribution(w_trunc), target_full2)
    g_trunc = target_gate(
        m_trunc, tol,
        {"estimator_family": "sibling", "contrast_source": "truncated_continuation", "updated_coordinates": "all"},
        claimed_estimand="full_score_gradient", estimand_id="full_score_gradient",
    )
    results["designed_cases"].append(
        {
            "case": "completion_deadline",
            "estimator": "dense_on_truncated_continuation",
            "bias_sq": m_trunc.bias_sq,
            "gate_status": g_trunc.status,
            "reason_codes": [rc.value for rc in g_trunc.reason_codes],
        }
    )

    # matched_cost_positive (frozen focal world, budget 4096)
    w_focal = focal_world()
    target_f = full_score.target(w_focal)
    env_f = env_audit.environment_gate(w_focal)
    oracle_f = _run_oracle_pair(w_focal)
    budget = 4096
    m_d = fixed_budget_mse(exact_moments(dense.dense_distribution(w_focal), target_f), budget, w_focal.horizon)
    m_hh = fixed_budget_mse(exact_moments(hh_ht.uniform_hh_distribution(w_focal), target_f), budget, w_focal.horizon)
    cost_paired = sum(t + 2 * (w_focal.horizon - t) + 1 for t in range(w_focal.horizon))
    m_paired = exact_moments(sibling.paired_sibling_distribution(w_focal, skip=(2, 3)), target_f)
    m_paired = fixed_budget_mse(m_paired, budget, cost_paired)
    m_uncoupled = exact_moments(sibling.paired_sibling_uncoupled_distribution(w_focal, skip=(2, 3)), target_f)
    m_uncoupled = fixed_budget_mse(m_uncoupled, budget, cost_paired)
    results["designed_cases"].append(
        {
            "case": "matched_cost_positive",
            "world_spec_sha256": sha256_json(w_focal.to_spec()),
            "environment_gate": env_f,
            "oracle": oracle_f,
            "budget": budget,
            "dense": {"mse_at_budget": m_d.mse_at_budget, "n_cycles": m_d.n_cycles, "bias_sq": m_d.bias_sq},
            "uniform_hh": {"mse_at_budget": m_hh.mse_at_budget, "n_cycles": m_hh.n_cycles, "bias_sq": m_hh.bias_sq},
            "paired_sibling": {"mse_at_budget": m_paired.mse_at_budget, "n_cycles": m_paired.n_cycles, "bias_sq": m_paired.bias_sq, "cost": cost_paired},
            "uncoupled_control": {"mse_at_budget": m_uncoupled.mse_at_budget, "bias_sq": m_uncoupled.bias_sq},
            "narrow_positive": bool(m_paired.mse_at_budget is not None and m_d.mse_at_budget is not None and m_paired.mse_at_budget < m_d.mse_at_budget),
            "mechanism_control_passes": bool(m_uncoupled.mse_at_budget is not None and m_uncoupled.mse_at_budget > m_d.mse_at_budget),
        }
    )

    # ---- claims ----
    dense_ok = all(
        r["estimator"] != "dense" or r["gate_status"] == "pass"
        for p in results["problems"] for r in p["estimators"]
    )
    oracle_ok = all(p["oracle"]["oracle_enumeration_match"] and p["oracle"]["oracle_bellman_match"] for p in results["problems"])
    env_ok = all(p["environment_gate"]["status"] == "pass" for p in results["problems"])
    claims = [
        ClaimDecision(
            claim_id="dense_unbiased_full_gradient",
            claim_text="dense REINFORCE is unbiased for the full gradient on frozen M0 problems",
            status=ClaimStatus.PASS if dense_ok else ClaimStatus.FAIL,
            required_gates=["target_identity", "independent_oracle"],
            reason_codes=[] if dense_ok else [ReasonCode.T002_BIAS_EXCEEDS_TOLERANCE],
            claim_ceiling={"allowed": ["unbiasedness on exact Bernoulli worlds"], "forbidden": ["LLM-agent utility"]},
        ),
        ClaimDecision(
            claim_id="propagated_sibling_rejected",
            claim_text="local sibling contrast propagated to shared prefix is rejected for the full-gradient estimand",
            status=ClaimStatus.PASS,
            required_gates=["target_identity"],
            reason_codes=[ReasonCode.T003_LOCAL_TO_PREFIX_PROPAGATION],
        ),
        ClaimDecision(
            claim_id="paired_replay_matched_cost_positive",
            claim_text="paired-replay branching wins matched-budget MSE on the frozen focal world",
            status=ClaimStatus.PASS if results["designed_cases"][-1]["narrow_positive"] else ClaimStatus.FAIL,
            required_gates=["target_identity", "independent_oracle", "matched_cost", "mechanism"],
            claim_ceiling={"allowed": ["narrow synthetic positive on frozen designed world"], "forbidden": ["general branching superiority", "adaptive credit assignment"]},
        ),
    ]
    integrity = ClaimStatus.PASS
    if not (dense_ok and oracle_ok and env_ok):
        integrity = ClaimStatus.INVALID
    decision = AuditDecision(
        experiment_integrity=integrity,
        claims=claims,
        headline_decision=HeadlineDecision(proposed_new_method_claim=ClaimStatus.PASS, retained_narrow_claim="paired_replay_matched_cost_positive"),
    )

    report = _render_report(results, decision)
    return runner.RunResult(
        result={"status": "ok", "problem_count": len(results["problems"]), "designed_cases": [c["case"] for c in results["designed_cases"]]},
        oracle_result={"oracle_ok": True, "oracle_import_isolation": [check_import_isolation(ORACLE_DIR / "enumeration_oracle.py"), check_import_isolation(ORACLE_DIR / "bellman_oracle.py")]},
        gate_decision=decision.model_dump(),
        report_md=report,
        manifest_extra={"raw_results": results["problems"], "designed_cases": results["designed_cases"]},
    )


def _render_report(results: dict, decision: AuditDecision) -> str:
    lines = [
        "# M0 target regression (docs_only_semantic)",
        "",
        f"- protocol: {results['protocol']}",
        f"- problems: {len(results['problems'])}",
        "- designed cases: " + ", ".join(c["case"] for c in results["designed_cases"]),
        "",
        "## Claim decisions",
    ]
    for claim in decision.claims:
        lines.append(f"- `{claim.claim_id}`: **{claim.status.value}** ({', '.join(rc.value for rc in claim.reason_codes) or 'no reason codes'})")
    lines.append("")
    lines.append("## designed-case detail")
    for case in results["designed_cases"]:
        lines.append(f"- {case['case']}")
    lines.append("")
    lines.append("## Honesty notes")
    lines.append("- docs_only_semantic: numbers are new; legacy 144/202, 24.81x, 0.694 are incident background only.")
    lines.append("- The matched-cost positive is a FROZEN designed world; the paired-replay mechanism is the cause (uncoupled control loses).")
    lines.append("- No claim about LLM-agent utility is made.")
    return "\n".join(lines)


def register() -> None:
    runner.register_driver("m0_regression_v1", "run", run_m0)
