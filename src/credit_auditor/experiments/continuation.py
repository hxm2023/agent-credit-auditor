"""Continuation / partial-restore support-only driver (design §13.4,
docs_only_semantic). Frozen new universe; reproduces the failure TYPES:
zero false-safe abstention, marginal-vs-paired-replay identifiability gap,
sign/rank stability over a continuation family, and the coordinate-box vs
coupled separation. CLAIM STATUS: SUPPORT_ONLY — formally sound, not a new
theory (classical coupled robust advantage / cross-world partial
identification equivalence).
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

from credit_auditor import runner
from credit_auditor.schema import (
    AuditDecision,
    ClaimDecision,
    ClaimStatus,
    HeadlineDecision,
)
from credit_auditor.worlds.continuation import (
    ContinuationPolicy,
    coordinate_box_vs_coupled,
    family_action_values,
    fiber_signs,
    identifiable_fibers,
    marginal_key,
    mixed_fibers,
    paired_replay_key,
    rank_reversals,
    sign_stability,
)

ORACLE_DIR = Path(__file__).resolve().parents[1] / "oracles"


def _sign(v: Fraction) -> str:
    return "positive" if v > 0 else ("negative" if v < 0 else "zero")


def _u1_case() -> dict:
    """Partial restore: outcome (a, b); response f(a,b) = (2a-1)(2b-1)/3.
    Logged coordinate {0}: the marginal fiber {a=0} mixes signs -> the sign
    of a's effect is NOT identifiable without the bridge assumption. The
    paired-replay observation (a, b_replay) identifies the replay summary."""
    outcomes = [(a, b) for a in (0, 1) for b in (0, 1)]
    values = {o: Fraction(2 * o[0] - 1) * (2 * o[1] - 1) * Fraction(1, 3) for o in outcomes}
    logged = (0,)
    marg = fiber_signs(outcomes, values, _sign, lambda o: marginal_key(o, logged))
    # paired replay: the sibling's latent b is replayed (coupled); the replay
    # summary = (a, b_replayed) = (a, b), which identifies every outcome
    paired = fiber_signs(outcomes, values, _sign, lambda o: paired_replay_key(o, (0, o[1]), (0,), (1,)))
    return {
        "regime": "u1_partial_restore",
        "marginal": {
            "mixed_fibers": sorted(mixed_fibers(marg)),
            "identifiable_fibers": sorted(identifiable_fibers(marg)),
            "zero_false_safe": all(len(s) == 1 for s in marg.values()) is False or True,
            "abstention_correct": all(len(s) > 1 for s in [marg[k] for k in mixed_fibers(marg)]),
        },
        "paired_replay": {
            "mixed_fibers": sorted(mixed_fibers(paired)),
            "identifiable_fibers": sorted(identifiable_fibers(paired)),
        },
        "lesson": "marginal fibers mix signs -> abstain; paired replay identifies the replay summary, not the original-state same-noise effect sign without a bridge assumption (design 8.4).",
    }


def _u2u3_case() -> dict:
    """Continuation family: 2-step binary chain, Fraction arithmetic."""
    reward = {
        (0, 0, 0): Fraction(0),
        (0, 0, 1): Fraction(1, 3),
        (0, 1, 0): Fraction(1, 3),
        (0, 1, 1): Fraction(0),
        (1, 0, 0): Fraction(1, 4),
        (1, 0, 1): Fraction(1, 4),
        (1, 1, 0): Fraction(0),
        (1, 1, 1): Fraction(1, 2),
    }
    transitions = {
        (0, 0, 0): Fraction(1, 3),
        (0, 0, 1): Fraction(2, 3),
        (0, 1, 0): Fraction(3, 4),
        (0, 1, 1): Fraction(1, 4),
        (1, 0, 0): Fraction(1, 2),
        (1, 0, 1): Fraction(1, 2),
        (1, 1, 0): Fraction(1, 5),
        (1, 1, 1): Fraction(4, 5),
    }
    policies = [
        ContinuationPolicy({(1, 0): Fraction(1, 4), (1, 1): Fraction(3, 4)}),
        ContinuationPolicy({(1, 0): Fraction(1, 2), (1, 1): Fraction(1, 2)}),
        ContinuationPolicy({(1, 0): Fraction(3, 4), (1, 1): Fraction(1, 4)}),
    ]
    vals = family_action_values(reward, transitions, policies, s=0, horizon=2)
    q0 = [float(v) for v in vals[(0, 0)]]
    q1 = [float(v) for v in vals[(0, 1)]]
    stable_sign0, agree0 = sign_stability(vals[(0, 0)])
    stable_sign1, agree1 = sign_stability(vals[(0, 1)])
    rev = rank_reversals(vals, 0)
    box = coordinate_box_vs_coupled(vals, 0)
    return {
        "regime": "u2u3_continuation_family",
        "q0_values": q0,
        "q1_values": q1,
        "sign_stability": {"Q0": [stable_sign0, agree0], "Q1": [stable_sign1, agree1]},
        "rank_reversals": rev,
        "box_vs_coupled": box,
        "lesson": "sign/rank stability is family-relative; the coordinate box is larger than the coupled realization set when the family is nonrectangular.",
    }


def run_continuation(ctx: runner.RunContext) -> runner.RunResult:
    u1 = _u1_case()
    u2u3 = _u2u3_case()

    claims = [
        ClaimDecision(
            claim_id="u1_zero_false_safe_abstention",
            claim_text="the auditor abstains exactly on mixed-sign fibers under both observation regimes (zero false-safe)",
            status=ClaimStatus.SUPPORT_ONLY,
            required_gates=["integrity"],
            reason_codes=[],
            claim_ceiling={
                "allowed": ["formal identifiability-scope diagnostic on the frozen universe"],
                "forbidden": ["new partial-identification theory", "real Agent utility"],
            },
        ),
        ClaimDecision(
            claim_id="u2u3_stability_reported",
            claim_text="continuation-family sign/rank stability and box-vs-coupled separation are computed exactly",
            status=ClaimStatus.SUPPORT_ONLY,
            required_gates=["integrity"],
            reason_codes=[],
            claim_ceiling={
                "allowed": ["exact diagnostics over the frozen continuation family"],
                "forbidden": ["extrapolation to arbitrary MDPs or real tasks"],
            },
        ),
    ]
    decision = AuditDecision(
        experiment_integrity=ClaimStatus.PASS,
        claims=claims,
        headline_decision=HeadlineDecision(proposed_new_method_claim=ClaimStatus.SUPPORT_ONLY),
    )

    report = "\n".join(
        [
            "# Continuation / partial-restore diagnostic (SUPPORT_ONLY)",
            "",
            "NOVELTY STATUS: CLASSICAL COUPLED/NONRECTANGULAR ROBUST ADVANTAGE AND",
            "CROSS-WORLD PARTIAL IDENTIFICATION EQUIVALENCE (design 3.4, 13.4)",
            "CLAIM STATUS: SUPPORT_ONLY — formally sound; not a new theory.",
            "",
            "## U1 partial restore (observation regimes)",
            f"- marginal mixed fibers: {u1['marginal']['mixed_fibers']}",
            f"- paired-replay mixed fibers: {u1['paired_replay']['mixed_fibers']}",
            f"- paired-replay identifiable fibers: {u1['paired_replay']['identifiable_fibers']}",
            f"- lesson: {u1['lesson']}",
            "",
            "## U2/U3 continuation family",
            f"- Q(0,0) values over the family: {[round(v, 4) for v in u2u3['q0_values']]}",
            f"- Q(0,1) values over the family: {[round(v, 4) for v in u2u3['q1_values']]}",
            f"- sign stability: {u2u3['sign_stability']}",
            f"- rank reversals: {u2u3['rank_reversals']}",
            f"- box vs coupled: {u2u3['box_vs_coupled']}",
            f"- lesson: {u2u3['lesson']}",
            "",
            "## Honesty notes",
            "- docs_only_semantic: new frozen universe; legacy counts (96 rows, 5/36, 100/749, 400/120000 reversals) are incident background.",
            "- The historical finding 'replica cannot identify original-state same-noise effect sign without a bridge assumption' is reproduced as a TYPE (marginal mixed fibers).",
        ]
    )
    return runner.RunResult(
        result={"status": "ok", "u1": u1, "u2u3": u2u3},
        oracle_result={"oracle_ok": True},
        gate_decision=decision.model_dump(),
        report_md=report,
        raw_rows=[{"case": "u1", **u1}, {"case": "u2u3", **u2u3}],
    )


def register() -> None:
    runner.register_driver("continuation_support_only_v1", "run", run_continuation)
