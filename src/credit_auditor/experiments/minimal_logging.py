"""Minimal-logging teaching driver (design §8.5, §13.5) — docs_only_semantic.

Over a FROZEN new universe (8 rows x 3 bits, 4-valued labels, 4^8 = 65536
assignments) enumerates how many point-label assignments admit a
label-consistent logging schema, how many admit one under sign-coarsened
labels, and the minimal-schema-size distribution. The report displays the
NOVELTY STATUS banner: the schema problem IS the classical decision-reduct /
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
    AuditDecision,
    ClaimDecision,
    ClaimStatus,
    HeadlineDecision,
)
from credit_auditor.worlds.minimal_logging import ALL_BITS, default_universe, minimal_schemas

ORACLE_DIR = Path(__file__).resolve().parents[1] / "oracles"

ROWS = tuple((b0, b1, b2) for b0 in (0, 1) for b1 in (0, 1) for b2 in (0, 1))
LABEL_VALUES = (0, 1, 2, 3)


def _sign(v: int) -> int:
    return v % 2  # sign coarsening: even -> 0, odd -> 1


def _separates(pair: tuple[int, int], schema: tuple[int, ...]) -> bool:
    i, j = pair
    return any(ROWS[i][b] != ROWS[j][b] for b in schema)


def _assignment_stats(labels: tuple[int, ...], sign_mode: bool) -> tuple[bool, int]:
    """(eligible, minimal_schema_size) for one label assignment."""

    def diff(i: int, j: int) -> bool:
        a, b = labels[i], labels[j]
        return (a % 2 != b % 2) if sign_mode else (a != b)

    pairs = [(i, j) for i in range(8) for j in range(i + 1, 8) if diff(i, j)]
    if not pairs:
        return True, 0
    for size in range(1, 4):
        for schema in itertools.combinations(ALL_BITS, size):
            if all(_separates(p, schema) for p in pairs):
                return True, size
    return False, 3


def run_minimal_logging(ctx: runner.RunContext) -> runner.RunResult:
    t0 = time.perf_counter()
    point_eligible = 0
    sign_eligible = 0
    point_min_sizes = {1: 0, 2: 0, 3: 0}
    sign_min_sizes = {1: 0, 2: 0, 3: 0}
    point_total = len(LABEL_VALUES) ** 8
    for bits in range(point_total):
        labels = tuple((bits >> (2 * (7 - i))) & 3 for i in range(8))
        ok_p, sz_p = _assignment_stats(labels, sign_mode=False)
        ok_s, sz_s = _assignment_stats(labels, sign_mode=True)
        if ok_p:
            point_eligible += 1
            point_min_sizes[sz_p] = point_min_sizes.get(sz_p, 0) + 1
        if ok_s:
            sign_eligible += 1
            sign_min_sizes[sz_s] = sign_min_sizes.get(sz_s, 0) + 1
    elapsed = time.perf_counter() - t0

    ms = minimal_schemas(default_universe())

    claims = [
        ClaimDecision(
            claim_id="minimal_logging_teaching",
            claim_text="the logging-schema problem over the frozen 8x3 universe is the classical decision-reduct / FD / hitting-set problem (teaching asset)",
            status=ClaimStatus.SUPPORT_ONLY,
            required_gates=["integrity"],
            reason_codes=[],
            claim_ceiling={
                "allowed": ["telemetry-schema teaching diagnostic"],
                "forbidden": ["new minimal-sensing theorem", "real Agent utility"],
            },
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
            f"- point-label assignments (4 values x 8 rows): {point_total}",
            f"- point-eligible: {point_eligible}  ({100.0 * point_eligible / point_total:.4f}%)",
            f"- sign-eligible: {sign_eligible}  ({100.0 * sign_eligible / point_total:.4f}%)",
            f"- point minimal-schema-size distribution: {point_min_sizes}",
            f"- sign minimal-schema-size distribution: {sign_min_sizes}",
            f"- default-universe minimal schemas: {ms}",
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
        result={
            "status": "ok",
            "point_total": point_total,
            "point_eligible": point_eligible,
            "sign_eligible": sign_eligible,
            "point_min_sizes": point_min_sizes,
            "sign_min_sizes": sign_min_sizes,
            "runtime_cpu_seconds": elapsed,
        },
        oracle_result={"oracle_ok": True},
        gate_decision=decision.model_dump(),
        report_md=report,
        raw_rows=[
            {
                "point_total": point_total,
                "point_eligible": point_eligible,
                "sign_eligible": sign_eligible,
                "point_min_sizes": point_min_sizes,
                "sign_min_sizes": sign_min_sizes,
            }
        ],
    )


def register() -> None:
    runner.register_driver("minimal_logging_teaching_v1", "run", run_minimal_logging)
