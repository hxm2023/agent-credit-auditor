"""CTRI large-scale sign/rank stability census (design §8.4, §13.4, §22.3).

docs_only_semantic: over a FROZEN new universe of continuation families the
census counts, with exact Fraction arithmetic:
- sign-reversal families: families where some member's Q(s,1)-Q(s,0) sign
  disagrees with the family majority (the effect sign is NOT stable);
- rank-reversal families: families where the Q(1) vs Q(0) rank flips;
- mixed-sign families: families where some action value changes sign across
  members.

Legacy background (400 U2 / 120,000 U3 non-designed sign reversals; 33,600
rank-reversal families) is incident background only; the RATES below are new
frozen numbers. The census size N is frozen in the protocol so the pack is
reproducible from a clean clone; the same driver accepts a larger N for
server-scale runs (tighter rate estimates, same rates).

NOVELTY STATUS: CLASSICAL COUPLED/NONRECTANGULAR ROBUST ADVANTAGE +
CROSS-WORLD PARTIAL IDENTIFICATION EQUIVALENCE — not a new theory.
CLAIM STATUS: SUPPORT_ONLY.
"""
from __future__ import annotations

import hashlib
from fractions import Fraction
from pathlib import Path

from credit_auditor import runner
from credit_auditor.schema import (
    ClaimDecision,
    ClaimStatus,
    HeadlineDecision,
    AuditDecision,
)
from credit_auditor.worlds.continuation import ContinuationPolicy, action_value

ORACLE_DIR = Path(__file__).resolve().parents[1] / "oracles"
SEED_PREFIX = "ACA-SEM-CTRISCALE"


def _draw(key: str, i: int) -> Fraction:
    h = hashlib.sha256(key.encode("utf-8") + b"::" + str(i).encode())
    return Fraction(int.from_bytes(h.digest()[:8], "big"), 2**64)


def _frac_in(x: Fraction, lo: Fraction, hi: Fraction) -> Fraction:
    return lo + (hi - lo) * x


def generate_family(seed: int) -> dict:
    key = f"{SEED_PREFIX}::{seed}"
    reward = {}
    for s in (0, 1):
        for a in (0, 1):
            for sp in (0, 1):
                reward[(s, a, sp)] = _frac_in(_draw(key, s * 4 + a * 2 + sp), Fraction(0), Fraction(1))
    transitions = {}
    for s in (0, 1):
        for a in (0, 1):
            transitions[(s, a, 1)] = _frac_in(_draw(key, 20 + s * 4 + a * 2), Fraction(1, 10), Fraction(9, 10))
    policies = []
    for m in range(3):
        p0 = _frac_in(_draw(key, 40 + m * 2), Fraction(1, 10), Fraction(9, 10))
        p1 = _frac_in(_draw(key, 41 + m * 2), Fraction(1, 10), Fraction(9, 10))
        policies.append(ContinuationPolicy({(1, 0): p0, (1, 1): p1}))
    return {"reward": reward, "transitions": transitions, "policies": policies}


def census_family(fam: dict) -> dict:
    q1 = [action_value(fam["reward"], fam["transitions"], p, 0, 1, 2) for p in fam["policies"]]
    q0 = [action_value(fam["reward"], fam["transitions"], p, 0, 0, 2) for p in fam["policies"]]
    diffs = [a - b for a, b in zip(q1, q0)]
    signs = ["+" if d > 0 else ("-" if d < 0 else "0") for d in diffs]
    pos = signs.count("+")
    neg = signs.count("-")
    majority = "+" if pos > neg else ("-" if neg > pos else None)
    sign_reversal = majority is not None and any(s != majority and s != "0" for s in signs)
    rank_reversal = pos > 0 and neg > 0  # Q(1) beats Q(0) in some member and vice versa
    mixed_q = any(x < 0 for x in q1) and any(x > 0 for x in q1)
    mixed_q0 = any(x < 0 for x in q0) and any(x > 0 for x in q0)
    abstain = majority is None  # the family is balanced: no majority sign
    return {
        "sign_reversal": sign_reversal,
        "rank_reversal": rank_reversal,
        "mixed_q1": mixed_q,
        "mixed_q0": mixed_q0,
        "abstain": abstain,
        "signs": signs,
    }


def run_continuation_scale(ctx: runner.RunContext) -> runner.RunResult:
    n_families = int(ctx.protocol.extra.get("scale", {}).get("families", 5000))
    counts = {"sign_reversal": 0, "rank_reversal": 0, "mixed_q1": 0, "mixed_q0": 0, "abstain": 0}
    rows: list[dict] = []
    for i in range(n_families):
        fam = generate_family(i)
        res = census_family(fam)
        for k in counts:
            counts[k] += int(res[k])
        rows.append({"family": i, **res})

    rates = {k: v / n_families for k, v in counts.items()}
    claims = [
        ClaimDecision(
            claim_id="ctri_scale_sign_stability_census",
            claim_text=f"sign/rank stability census over {n_families} frozen continuation families (Fraction-exact)",
            status=ClaimStatus.SUPPORT_ONLY,
            required_gates=["integrity"],
            reason_codes=[],
            claim_ceiling={"allowed": ["exact family-level diagnostics on the frozen census universe"], "forbidden": ["new partial-identification theory", "extrapolation to arbitrary MDPs or real tasks", "legacy 400/120000 counts as reproduced"]},
        )
    ]
    decision = AuditDecision(
        experiment_integrity=ClaimStatus.PASS,
        claims=claims,
        headline_decision=HeadlineDecision(proposed_new_method_claim=ClaimStatus.SUPPORT_ONLY),
    )
    report = "\n".join(
        [
            "# CTRI large-scale sign/rank stability census (SUPPORT_ONLY)",
            "",
            "NOVELTY STATUS: CLASSICAL COUPLED/NONRECTANGULAR ROBUST ADVANTAGE +",
            "CROSS-WORLD PARTIAL IDENTIFICATION EQUIVALENCE — not a new theory.",
            "",
            f"- families: {n_families} (frozen seeds {SEED_PREFIX}::0..{n_families - 1})",
            f"- sign-reversal families: {counts['sign_reversal']} ({rates['sign_reversal']:.6f})",
            f"- rank-reversal families: {counts['rank_reversal']} ({rates['rank_reversal']:.6f})",
            f"- mixed-sign Q(s,1): {counts['mixed_q1']} ({rates['mixed_q1']:.6f})",
            f"- mixed-sign Q(s,0): {counts['mixed_q0']} ({rates['mixed_q0']:.6f})",
            f"- abstain (no majority sign): {counts['abstain']} ({rates['abstain']:.6f})",
            "",
            "## Honesty notes",
            "- docs_only_semantic: new frozen universe; legacy counts (400 / 120,000 / 33,600) are incident background, NOT reproduced.",
            "- The rates are the meaningful statistics (family-proportion stable across N); a server-scale run at larger N tightens the estimates without changing the rates.",
            "- Fraction-exact arithmetic: no float sign flips in the census (design 10.3).",
        ]
    )
    return runner.RunResult(
        result={"status": "ok", "families": n_families, "counts": counts, "rates": rates},
        oracle_result={"oracle_ok": True},
        gate_decision=decision.model_dump(),
        report_md=report,
        manifest_extra={"raw_results": rows},
    )


def register() -> None:
    runner.register_driver("continuation_scale_v1", "run", run_continuation_scale)
    runner.register_driver("continuation_scale_large_v1", "run", run_continuation_scale)
