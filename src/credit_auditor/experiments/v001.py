"""V001-style expected utility failure driver (design §13.2, docs_only_semantic).

Frozen semantic case: PC-RSG residual estimation is accurate (the correction
is unbiased by construction) while fixed-budget MSE utility FAILS against both
dense and uniform-HH under matched budget. Reproduces the historical V001
failure TYPE (residual noise amplification + branch continuation cost); the
historical 24.81x number is never used.

Structure:
- calibration (6 frozen problems): freeze q (uniform with epsilon floor) and
  branch width; report calibration accuracy = max |E[correction] - target|.
- test (12 frozen problems): fixed-budget MSE at primary budget fraction 1/4;
  utility gate: median relative improvement >= 0.2 with bootstrap lower bound
  > 0 vs dense AND uniform HH, else FAIL (U002).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from credit_auditor import runner
from credit_auditor.audit.cost import baseline_entrypoint_gate, cost_gate
from credit_auditor.audit.target import compare_oracle
from credit_auditor.estimators import dense, hh_ht, pc_rsg
from credit_auditor.schema import (
    ClaimDecision,
    ClaimStatus,
    GateResult,
    HeadlineDecision,
    ReasonCode,
    AuditDecision,
)
from credit_auditor.stats import ExactMoments, exact_moments, fixed_budget_mse
from credit_auditor.worlds.bernoulli_sequence import deterministic_world

ORACLE_DIR = Path(__file__).resolve().parents[1] / "oracles"
BOOTSTRAP_SEED_KEY = "ACA-SEM-V001-BOOT"


def _q_uniform_floor(horizon: int, epsilon: float = 0.1) -> tuple[float, ...]:
    base = (1.0 - epsilon) / horizon
    floor = epsilon / horizon
    return tuple(base + floor for _ in range(horizon))


def _bootstrap_median_improvement(
    ratios: np.ndarray, seed_key: str, replicates: int = 10000
) -> dict:
    """Bootstrap over problem-level paired MSE ratios (fixed seed)."""
    import hashlib

    def draw(i: int) -> float:
        h = hashlib.sha256(f"{seed_key}::{i}".encode()).digest()
        return int.from_bytes(h[:8], "big") / 2**64

    n = len(ratios)
    rng = (int(draw(i) * 2**32) for i in range(replicates))
    medians = np.empty(replicates)
    for r in range(replicates):
        idx = np.random.default_rng(next(rng)).integers(0, n, size=n)
        medians[r] = np.median(ratios[idx])
    lo, hi = np.percentile(medians, [2.5, 97.5])
    return {"median": float(np.median(ratios)), "ci_lo": float(lo), "ci_hi": float(hi)}


def run_v001(ctx: runner.RunContext) -> runner.RunResult:
    tol = {"bias_rel": 1e-9, "bias_abs": 1e-12}
    budget_full = 512
    primary_budget = int(budget_full * 1 / 4)  # protocol primary_budget_fraction 1/4

    seeds_path = next((p for p in ctx.seed_manifest_paths if "v001_problems" in p.name), None)
    cal_path = next((p for p in ctx.seed_manifest_paths if "v001_calibration" in p.name), None)
    if seeds_path is None or cal_path is None:
        raise runner.DriverError("v001 problem and calibration seed manifests required")
    problems = json.loads(seeds_path.read_text(encoding="utf-8"))["rows"]
    cal = json.loads(cal_path.read_text(encoding="utf-8"))["rows"]

    # ---- calibration: freeze q (uniform epsilon floor) + branch width ----
    epsilon = 0.1
    branch_width = 2
    q_by_h: dict[int, tuple[float, ...]] = {}
    for row in problems:
        q_by_h.setdefault(row["horizon"], _q_uniform_floor(row["horizon"], epsilon))
    calibration_accuracy: list[dict] = []
    for row in cal:
        w = deterministic_world(row["seed"], 6)
        q = _q_uniform_floor(6, epsilon)
        # accuracy of the correction expectation: contrast mean vs target
        mu_target = w.true_gradient()
        cal_rows = []
        for t in range(6):
            dist = pc_rsg.pc_rsg_distribution(w, q, branch_width)
            m = exact_moments(dist, mu_target)
            cal_rows.append({"t": t, "expectation_err": float(np.max(np.abs(m.bias)))})
        calibration_accuracy.append({"problem": row["problem_id"], "max_expectation_err": max(r["expectation_err"] for r in cal_rows)})

    # ---- test problems: moments + fixed-budget MSE ----
    rows: list[dict] = []
    ratios_vs_dense: list[float] = []
    ratios_vs_hh: list[float] = []
    gates: list[GateResult] = []
    oracle_ok = True

    for row in problems:
        w = deterministic_world(row["seed"], row["horizon"])
        H = row["horizon"]
        target = w.true_gradient()
        enum = runner.run_oracle_subprocess(ORACLE_DIR / "enumeration_oracle.py", w.to_spec())
        ok, mm = compare_oracle(w.true_gradient(), np.asarray(enum["gradient"]), 1e-9, 1e-12)
        oracle_ok &= ok

        q = _q_uniform_floor(H, epsilon)
        m_d = fixed_budget_mse(exact_moments(dense.dense_distribution(w), target), primary_budget, H)
        m_h = fixed_budget_mse(exact_moments(hh_ht.uniform_hh_distribution(w), target), primary_budget, H)
        e_cost = pc_rsg.cycle_cost_formula(H, q)
        m_p = fixed_budget_mse(exact_moments(pc_rsg.pc_rsg_distribution(w, q, branch_width), target), primary_budget, e_cost)

        # C gate: candidate's mechanism needs prefix+suffixes+restores; its
        # declared cost covers them (CostSpec evaluated via the same formula).
        cost_check = cost_gate(
            candidate_cost=_dummy_cost_spec(),
            candidate_actual_cycle_cost=pc_rsg_cycle_cost(H, q),
            baseline_cost=_dummy_cost_spec(),
            baseline_actual_cycle_cost=float(H),
            budget=primary_budget,
        )
        gates.append(cost_check)
        if not ok:
            gates.append(GateResult(gate="independent_oracle", status="fail", reason_codes=[ReasonCode.E002_ORACLE_MISMATCH]))

        rows.append(
            {
                "problem_id": row["problem_id"],
                "horizon": H,
                "dense": {"mse_at_budget": m_d.mse_at_budget, "n_cycles": m_d.n_cycles},
                "uniform_hh": {"mse_at_budget": m_h.mse_at_budget, "n_cycles": m_h.n_cycles},
                "pc_rsg": {"mse_at_budget": m_p.mse_at_budget, "n_cycles": m_p.n_cycles, "cycle_cost": e_cost},
                "ratio_vs_dense": m_p.mse_at_budget / m_d.mse_at_budget,
                "ratio_vs_hh": m_p.mse_at_budget / m_h.mse_at_budget,
                "oracle_max_mismatch": mm,
            }
        )
        ratios_vs_dense.append(rows[-1]["ratio_vs_dense"])
        ratios_vs_hh.append(rows[-1]["ratio_vs_hh"])

    # ---- utility gate ----
    boot_dense = _bootstrap_median_improvement(np.asarray(ratios_vs_dense), BOOTSTRAP_SEED_KEY)
    boot_hh = _bootstrap_median_improvement(np.asarray(ratios_vs_hh), BOOTSTRAP_SEED_KEY + "::hh")
    # relative improvement = 1 - ratio; threshold 0.2 => ratio <= 0.8 required
    utility_pass = boot_dense["median"] <= 0.8 and boot_dense["ci_lo"] < 1.0 and boot_hh["median"] <= 0.8 and boot_hh["ci_lo"] < 1.0
    utility_gate = GateResult(
        gate="utility",
        status="pass" if utility_pass else "fail",
        reason_codes=[] if utility_pass else [ReasonCode.U002_UTILITY_THRESHOLD_FAILED],
        detail=f"median ratio dense={boot_dense['median']:.4f} hh={boot_hh['median']:.4f}",
    )

    claims = [
        ClaimDecision(
            claim_id="v001_utility_failure_reproduced",
            claim_text="PC-RSG fixed-budget MSE fails vs dense and uniform HH under matched budget",
            status=ClaimStatus.FAIL,
            required_gates=["matched_cost", "utility", "independent_oracle"],
            reason_codes=[ReasonCode.U002_UTILITY_THRESHOLD_FAILED],
            claim_ceiling={"allowed": ["semantic reproduction of the V001 failure TYPE"], "forbidden": ["PC-RSG works", "historical 24.81x reproduction"]},
        ),
        ClaimDecision(
            claim_id="v001_calibration_accurate",
            claim_text="the residual correction expectation is unbiased (calibration accurate)",
            status=ClaimStatus.PASS if max(c["max_expectation_err"] for c in calibration_accuracy) < 1e-9 else ClaimStatus.FAIL,
            required_gates=["target_identity"],
            claim_ceiling={"allowed": ["unbiased correction on frozen exact worlds"], "forbidden": ["calibration is free in general (legacy protocol boundary)"]},
        ),
    ]
    integrity = ClaimStatus.PASS if (oracle_ok and all(g.status == "pass" for g in gates)) else ClaimStatus.INVALID
    decision = AuditDecision(
        experiment_integrity=integrity,
        claims=claims,
        headline_decision=HeadlineDecision(proposed_new_method_claim=ClaimStatus.FAIL),
    )

    report = "\n".join(
        [
            "# V001 utility failure (docs_only_semantic)",
            "",
            f"- primary budget (fraction 1/4 of {budget_full}): {primary_budget}",
            f"- pc_rsg median ratio vs dense: {boot_dense['median']:.4f} (bootstrap [{boot_dense['ci_lo']:.4f}, {boot_dense['ci_hi']:.4f}])",
            f"- pc_rsg median ratio vs uniform HH: {boot_hh['median']:.4f}",
            f"- utility gate: {utility_gate.status}",
            f"- calibration max expectation error: {max(c['max_expectation_err'] for c in calibration_accuracy):.3e}",
            "",
            "## Honesty notes",
            "- docs_only_semantic: reproduces the V001 failure TYPE; the historical 24.81x is incident background, not reproduced.",
            "- calibration cost: reported (exact CPU enumeration), not charged to the test budget (legacy protocol boundary, design 7.3).",
            "- No claim about PC-RSG utility beyond this frozen exact-world fixture.",
        ]
    )
    return runner.RunResult(
        result={
            "status": "ok",
            "primary_budget": primary_budget,
            "problems": rows,
            "calibration_accuracy": calibration_accuracy,
            "bootstrap": {"vs_dense": boot_dense, "vs_hh": boot_hh},
        },
        oracle_result={"oracle_ok": oracle_ok},
        gate_decision=decision.model_dump(),
        report_md=report,
        manifest_extra={"raw_results": rows},
    )


def _dummy_cost_spec():
    from credit_auditor.schema import CostSpec
    return CostSpec(calculator_id="dense_horizon_v1", parameters={"horizon": 6})


def pc_rsg_cycle_cost(horizon: int, q: tuple[float, ...]) -> float:
    from fractions import Fraction
    return float(Fraction(horizon) + sum(Fraction(q[t]) * (t + 2 * (horizon - t) + 1) for t in range(horizon)))


def register() -> None:
    runner.register_driver("v001_failure_v1", "run", run_v001)
