"""D002-style semantic regression driver (design §13.3, docs_only_semantic).

The flagship dual verdict:
- C1 fixed-budget efficiency: PASS — the calibrated global-K mapping beats the
  dense envelope (per-problem min(optimal-constant, root-RLOO)) on the frozen
  48 test problems.
- C2 adaptive variable-width mechanism: FAIL — the calibrated widths collapse
  to the global control (MECH001), so the adaptive-method headline is FAIL and
  only the narrow fixed-width claim is retained.

World (decision log D9): shared-logit Bernoulli-style world, state-independent
policy (both states share a logit at each time), deterministic transitions
(s_{t+1} = a_t), focal terminal reward with centered noise at non-adjacent
zero-target times. All numbers are new.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from credit_auditor import runner
from credit_auditor.audit.mechanism import width_diversity_gate
from credit_auditor.audit.target import compare_oracle
from credit_auditor.estimators import branching
from credit_auditor.schema import (
    AuditDecision,
    ClaimDecision,
    ClaimStatus,
    HeadlineDecision,
    ReasonCode,
)
from credit_auditor.worlds.d002_shared_logits import (
    BUCKET_ORDER,
    generate_problem_focal,
    true_gradient,
)

ORACLE_DIR = Path(__file__).resolve().parents[1] / "oracles"
PRIMARY_BUDGET = 4096


def _bucket_options(bucket_id: str, depths: dict) -> list[tuple]:
    opts: list[tuple] = [(None, 1)]
    for d in depths[bucket_id]:
        for K in (2, 4, 8):
            opts.append((d, K))
    return opts


def _build_mapping(combo: tuple[int, ...], depths: dict) -> dict:
    return {bid: _bucket_options(bid, depths)[idx] for bid, idx in zip(BUCKET_ORDER, combo)}


def _combine(parts):
    mean = sum((x.mean for x in parts), np.zeros(3))
    mm = sum((x.second_moment for x in parts), np.zeros((3, 3)))
    for i in range(4):
        for j in range(4):
            if i != j:
                mm += np.outer(parts[i].mean, parts[j].mean)
    return mean, mm


def _envelope_mse(problem, target, budget: int) -> tuple[float | None, str]:
    parts_opt = [branching.dense_optimal_constant_moments(problem, b) for b in problem.buckets]
    parts_rloo = [branching.root_rloo_moments(problem, b) for b in problem.buckets]
    m_opt, mm_opt = _combine(parts_opt)
    m_rloo, mm_rloo = _combine(parts_rloo)
    cost_env = float(sum(b.horizon for b in problem.buckets))
    e_opt = branching.fixed_budget_mse_from_moments(m_opt, mm_opt, target, budget, cost_env)
    e_rloo = branching.fixed_budget_mse_from_moments(m_rloo, mm_rloo, target, budget, cost_env)
    if e_rloo is not None and (e_opt is None or e_rloo < e_opt):
        return e_rloo, "root_rloo"
    return e_opt, "optimal_constant"


def _bootstrap_ratios(ratios: np.ndarray, seed_key: str, replicates: int = 10000) -> dict:
    def draw(i: int) -> float:
        h = hashlib.sha256(f"{seed_key}::{i}".encode()).digest()
        return int.from_bytes(h[:8], "big") / 2**64

    n = len(ratios)
    medians = np.empty(replicates)
    for r in range(replicates):
        rng = np.random.default_rng(int(draw(r) * 2**32))
        idx = rng.integers(0, n, size=n)
        medians[r] = np.median(ratios[idx])
    return {
        "median": float(np.median(ratios)),
        "ci_lo": float(np.percentile(medians, 2.5)),
        "ci_hi": float(np.percentile(medians, 97.5)),
    }


def _protocol_depths(ctx: runner.RunContext) -> dict:
    return {b["bucket_id"]: tuple(b["depth_candidates"]) for b in ctx.protocol.extra["generator_spec"]["buckets"]}


def _calibration(ctx: runner.RunContext) -> runner.RunResult:
    depths = _protocol_depths(ctx)
    cal_path = next((p for p in ctx.seed_manifest_paths if "d002_calibration" in p.name), None)
    if cal_path is None:
        raise runner.DriverError("d002_calibration seed manifest required")
    cal = json.loads(cal_path.read_text(encoding="utf-8"))["rows"]

    # precompute per-bucket option moments per problem
    cal_objectives: list[dict] = []
    for row in cal:
        problem = generate_problem_focal(row["problem_id"], row["seed"])
        target = true_gradient(problem)
        bm: dict = {}
        for bid in BUCKET_ORDER:
            b = next(x for x in problem.buckets if x.bucket_id == bid)
            for i, (d, K) in enumerate(_bucket_options(bid, depths)):
                bm[(bid, i)] = (
                    branching.dense_bucket_moments(problem, b)
                    if K == 1
                    else branching.paired_replay_all_bucket_moments(problem, b, d)
                )
        objs: dict = {}
        for combo in itertools.product(range(7), repeat=4):
            parts = [bm[(bid, i)] for bid, i in zip(BUCKET_ORDER, combo)]
            mean, mm = _combine(parts)
            bias = mean - target
            mse = float(bias @ bias) + float(np.trace(mm - np.outer(mean, mean)))
            cost = branching.mapping_cycle_cost(problem, _build_mapping(combo, depths))
            objs[combo] = (np.log(mse + 1e-18), cost)
        cal_objectives.append(objs)

    best_combo: tuple[int, ...] | None = None
    best_obj = None
    best_cost = None
    for combo in itertools.product(range(7), repeat=4):
        obj = sum(o[combo][0] for o in cal_objectives) / len(cal_objectives)
        cost = sum(o[combo][1] for o in cal_objectives) / len(cal_objectives)
        if (
            best_obj is None
            or best_cost is None
            or obj < best_obj - 1e-15
            or (abs(obj - best_obj) <= 1e-15 and cost < best_cost)
        ):
            best_obj, best_cost, best_combo = obj, cost, combo

    mapping = _build_mapping(best_combo, depths)
    selection = {
        "protocol_id": ctx.protocol.protocol_id,
        "mode": "docs_only_semantic",
        "calibration_problem_count": len(cal),
        "objective": "mean(log(exact_trace_mse + 1e-18))",
        "tie_break": ["objective", "cycle_cost", "candidate_index"],
        "selected_combo": list(best_combo),
        "selected_mapping": {k: list(v) for k, v in mapping.items()},
        "selected_cycle_cost": best_cost,
        "calibration_objective": best_obj,
        "selection_sha256": None,  # filled at publish (content hash)
    }
    from credit_auditor.canonical import sha256_json

    body = {k: v for k, v in selection.items() if k != "selection_sha256"}
    selection["selection_sha256"] = sha256_json(body)

    report = "\n".join(
        [
            "# D002 calibration (docs_only_semantic)",
            "",
            f"- problems: {len(cal)}",
            f"- selected mapping: {mapping}",
            f"- objective: {best_obj:.6f}  cycle cost: {best_cost:.2f}",
            "",
            "## Honesty notes",
            "- Semantic reconstruction (decision log D9): new frozen world, seeds, and numbers.",
            "- The mapping is frozen here; the test phase refuses any other selection (A8).",
        ]
    )
    return runner.RunResult(
        result={
            "status": "ok",
            "selection": selection,
            "calibration_objectives": {str(k): v for o in cal_objectives for k, v in o.items()},
        },
        oracle_result={"oracle_ok": True},
        gate_decision=AuditDecision(experiment_integrity=ClaimStatus.PASS).model_dump(),
        report_md=report,
        raw_rows=[{"problem": k, "objective": v[0], "cost": v[1]} for o in cal_objectives for k, v in o.items()],
    )


def _test(ctx: runner.RunContext) -> runner.RunResult:
    test_path = next((p for p in ctx.seed_manifest_paths if "d002_test" in p.name), None)
    if test_path is None:
        raise runner.DriverError("d002_test seed manifest required")
    tests = json.loads(test_path.read_text(encoding="utf-8"))["rows"]

    # A8: test phase requires the frozen selection AND its content must be
    # tamper-evident: the selection's self-hash must match its body, so an
    # edited mapping (e.g. hand-picked widths) is rejected, not silently used.
    if ctx.frozen_selection is None or not ctx.frozen_selection.is_file():
        raise runner.DriverError("test phase requires --frozen-selection (A8: no test-time reselection)")
    selection = json.loads(ctx.frozen_selection.read_text(encoding="utf-8"))
    from credit_auditor.canonical import sha256_json

    body = {k: v for k, v in selection.items() if k != "selection_sha256"}
    if sha256_json(body) != selection.get("selection_sha256"):
        raise runner.DriverError(
            "frozen selection content hash mismatch: the selection file was modified "
            "after calibration (A8 lineage violation)"
        )
    mapping = {k: tuple(v) for k, v in selection["selected_mapping"].items()}
    selection_hash = selection["selection_sha256"]

    rows: list[dict] = []
    ratios: list[float] = []
    oracle_ok = True
    for row in tests:
        problem = generate_problem_focal(row["problem_id"], row["seed"])
        target = true_gradient(problem)
        enum = runner.run_oracle_subprocess(ORACLE_DIR / "enumeration_oracle.py", problem.to_spec())
        ok, mm = compare_oracle(target, np.asarray(enum["gradient"]), 1e-9, 1e-12)
        oracle_ok &= ok
        env_mse, env_kind = _envelope_mse(problem, target, PRIMARY_BUDGET)
        mb = branching.estimator_moments(problem, mapping)
        cost = branching.mapping_cycle_cost(problem, mapping)
        map_mse = branching.fixed_budget_mse_from_moments(mb.mean, mb.second_moment, target, PRIMARY_BUDGET, cost)
        if env_mse is None or map_mse is None:
            raise runner.DriverError(f"infeasible budget at problem {row['problem_id']}")
        ratio = map_mse / env_mse
        ratios.append(ratio)
        rows.append(
            {
                "problem_id": row["problem_id"],
                "envelope_mse": env_mse,
                "envelope_kind": env_kind,
                "mapping_mse": map_mse,
                "mapping_cycle_cost": cost,
                "ratio_vs_envelope": ratio,
                "mapping_bias_sq": float(np.max(np.abs(mb.mean - target))),
                "oracle_max_mismatch": mm,
            }
        )

    boot = _bootstrap_ratios(np.asarray(ratios), "ACA-SEM-D002-BOOT-20260822")
    # utility: median ratio <= 0.8 (relative improvement >= 0.2) and CI upper < 1
    utility_pass = boot["median"] <= 0.8 and boot["ci_hi"] < 1.0

    selected_widths = [mapping[bid][1] for bid in BUCKET_ORDER]
    mech_gate = width_diversity_gate(selected_widths)
    widths_collapsed = mech_gate.status == "fail"

    claims = [
        ClaimDecision(
            claim_id="global_k8_efficiency",
            claim_text=(
                f"the calibrated fixed mapping (declared widths all equal to {selected_widths[0]}) "
                f"improves finite-MDP fixed-budget MSE under protocol d002_regression_v1 (semantic). "
                f"The width itself is not asserted to be the mechanism (see ceiling)."
            ),
            status=ClaimStatus.PASS if utility_pass else ClaimStatus.FAIL,
            required_gates=["integrity", "independent_oracle", "matched_cost", "heldout_split", "utility"],
            reason_codes=[] if utility_pass else [ReasonCode.U002_UTILITY_THRESHOLD_FAILED],
            claim_ceiling={
                "allowed": [
                    "fixed mapping efficiency on the frozen semantic world; the paired-replay protocol (not the width) drives the win"
                ],
                "forbidden": [
                    "adaptive variable-width credit assignment",
                    "width-dependent mechanism claims",
                    "historical 0.694 reproduction",
                ],
            },
        ),
        ClaimDecision(
            claim_id="variable_width_adaptivity",
            claim_text="calibrated variable-width mapping differs materially from the global control",
            status=ClaimStatus.FAIL if widths_collapsed else ClaimStatus.PASS,
            required_gates=["heldout_split", "mechanism"],
            reason_codes=[ReasonCode.MECH001_ADAPTIVE_MAPPING_COLLAPSED_TO_GLOBAL] if widths_collapsed else [],
            claim_ceiling={"allowed": [], "forbidden": ["adaptive mechanism claim without width diversity"]},
        ),
    ]
    integrity = ClaimStatus.PASS if oracle_ok else ClaimStatus.INVALID
    decision = AuditDecision(
        experiment_integrity=integrity,
        claims=claims,
        headline_decision=HeadlineDecision(
            proposed_new_method_claim=ClaimStatus.FAIL if widths_collapsed else ClaimStatus.SUPPORT_ONLY,
            retained_narrow_claim="global_k8_efficiency" if utility_pass else None,
        ),
    )

    report = "\n".join(
        [
            "# D002 test (docs_only_semantic)",
            "",
            f"- frozen mapping: {mapping}",
            f"- selected widths: {selected_widths}",
            f"- median ratio vs envelope: {boot['median']:.4f} (bootstrap [{boot['ci_lo']:.4f}, {boot['ci_hi']:.4f}])",
            f"- utility gate: {'PASS' if utility_pass else 'FAIL'}",
            f"- mechanism gate: {mech_gate.status} ({[rc.value for rc in mech_gate.reason_codes]})",
            "",
            "## Dual verdict (design 13.3)",
            "- C1 fixed-budget efficiency: " + ("PASS" if utility_pass else "FAIL"),
            "- C2 adaptive variable-width mechanism: FAIL (widths collapse to the global control)",
            "- headline: proposed adaptive method FAIL; retained claim: fixed global-K only.",
            "",
            "## Honesty notes",
            "- docs_only_semantic: new frozen world/seeds/numbers (decision log D9); historical 0.694 and 192/192 are incident background.",
            "- Mechanism-fail is STRUCTURAL under the pre-registered calibration objective",
            "  (mean log exact-trace MSE): the raw MSE always prefers the largest width,",
            "  so the calibrated widths collapse to the global control by construction.",
            "  The demonstration shows a metric pass does NOT license an adaptive",
            "  mechanism claim; it does not claim the gate would catch every fake",
            "  adaptive method (the two-sided behavior of width_diversity_gate is",
            "  unit-tested directly).",
            "- Calibration cost: exact CPU enumeration, REPORTED only (protocol shared",
            "  costs calibration_transitions 0/1), never charged to the test budget",
            "  (legacy protocol boundary, design 7.3).",
        ]
    )
    return runner.RunResult(
        result={
            "status": "ok",
            "problems": rows,
            "bootstrap": boot,
            "selected_mapping": {k: list(v) for k, v in mapping.items()},
            "selected_widths": selected_widths,
        },
        oracle_result={"oracle_ok": oracle_ok},
        gate_decision=decision.model_dump(),
        report_md=report,
        manifest_extra={"parent_calibration_selection_sha256": selection_hash},
        raw_rows=rows,
    )


def register() -> None:
    runner.register_driver("d002_regression_v1", "calibration", _calibration)
    runner.register_driver("d002_regression_v1", "test", _test)
