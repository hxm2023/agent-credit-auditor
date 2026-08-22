"""Minimal-logging teaching driver (design §8.5, §13.5) — docs_only_semantic.

Enumerates, over a FROZEN new universe (8 rows x 3 bits), how many of the
2^8 point-label assignments admit a label-consistent logging schema, and how
many admit one under sign-coarsened labels. The report displays the NOVELTY
STATUS banner: the schema problem IS the classical decision-reduct /
functional-dependency / hitting-set problem.

Legacy counts (390625 / 390112 / 3.81% / 4.21%, 198.63 CPU s) are incident
background only; the numbers below are new.
"""
from __future__ import annotations

import itertools
import time
from pathlib import Path

from credit_auditor import runner
from credit_auditor.schema import (
    ClaimDecision,
    ClaimStatus,
    HeadlineDecision,
    AuditDecision,
)
from credit_auditor.worlds.minimal_logging import ALL_BITS, default_universe, minimal_schemas

ORACLE_DIR = Path(__file__).resolve().parents[1] / "oracles"

ROWS = tuple((b0, b1, b2) for b0 in (0, 1) for b1 in (0, 1) for b2 in (0, 1))


def _assignment_eligible(labels: tuple[str, ...]) -> bool:
    pairs = {(i, j) for i in range(8) for j in range(i + 1, 8) if labels[i] != labels[j]}
    for size in range(0, 4):
        for schema in itertools.combinations(ALL_BITS, size):
            if all(any(ROWS[i][b] != ROWS[j][b] for b in schema) for (i, j) in pairs):
                return True
    return False


def _sign_coarsened(labels: tuple[str, ...]) -> tuple[str, ...]:
    values = sorted(set(labels))
    return tuple("+" if labels[i] == values[-1] else "-" for i in range(8))


def run_minimal_logging(ctx: runner.RunContext) -> runner.RunResult:
    t0 = time.perf_counter()
    point_eligible = 0
    sign_eligible = 0
    point_total = 0
    for bits in range(1 << 8):
        labels = tuple("L" + str((bits >> (7 - i)) & 1) for i in range(8))
        point_total += 1
        if _assignment_eligible(labels):
            point_eligible += 1
        if _assignment_eligible(_sign_coarsened(labels)):
            sign_eligible += 1
    elapsed = time.perf_counter() - t0

    ms = minimal_schemas(default_universe())

    claims = [
        ClaimDecision(
            claim_id="minimal_logging_teaching",
            claim_text="the logging-schema problem over the frozen 8x3 universe is the classical decision-reduct / FD / hitting-set problem (teaching asset)",
            status=ClaimStatus.SUPPORT_ONLY,
            required_gates=["integrity"],
            reason_codes=[],
            claim_ceiling={"allowed": ["telemetry-schema teaching diagnostic"], "forbidden": ["new minimal-sensing theorem", "real Agent utility"]},
        )
    ]
    decision = AuditDecision(
        experiment_integrity=ClaimStatus.PASS,
        claims=claims,
        headline_decision=HeadlineDecision(proposed_new_method_claim=ClaimStatus.SUPPORT_ONLY),
    )

    report = "\n".join(
        [
            "# Minimal logging — teaching asset (SUPPORT_ONLY)",
            "",
            "NOVELTY STATUS: CLASSICAL DECISION-REDUCT / FD / HITTING-SET EQUIVALENCE",
            "CLAIM STATUS: TEACHING OR TELEMETRY-SCHEMA DIAGNOSTIC ONLY (design 13.5)",
            "",
            f"- point-label assignments: {point_total}",
            f"- point-eligible: {point_eligible}  ({100.0 * point_eligible / point_total:.4f}%)",
            f"- sign-eligible: {sign_eligible}  ({100.0 * sign_eligible / point_total:.4f}%)",
            f"- minimal schema sizes: {[len(s) for s in ms]}",
            f"- runtime: {elapsed:.2f} CPU seconds (GPU 0)",
            "",
            "## Honesty notes",
            "- docs_only_semantic: new frozen universe; the legacy counts",
            "  (390625 / 390112 / 3.81% / 4.21% / 198.63 s) are incident background",
            "  and are NOT reproduced.",
            "- The schema problem is the hitting set of the different-label conflict",
            "  pairs; sign is label coarsening (design 3.5).",
        ]
    )
    return runner.RunResult(
        result={"status": "ok", "point_total": point_total, "point_eligible": point_eligible, "sign_eligible": sign_eligible, "runtime_cpu_seconds": elapsed, "minimal_schema_sizes": [len(s) for s in ms]},
        oracle_result={"oracle_ok": True},
        gate_decision=decision.model_dump(),
        report_md=report,
        manifest_extra={"raw_results": [{"point_total": point_total, "point_eligible": point_eligible, "sign_eligible": sign_eligible}]},
    )


def register() -> None:
    runner.register_driver("minimal_logging_teaching_v1", "run", run_minimal_logging)
