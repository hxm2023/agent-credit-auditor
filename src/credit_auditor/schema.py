"""Core schema for Agent-RL Credit Auditor (design doc §7).

All formal objects are Pydantic v2 models. Float results keep a high-precision
string alongside the float where the value is exact or rational; cost arithmetic
is `fractions.Fraction` end-to-end.

An experiment may produce multiple ClaimDecisions; a single status never covers
all claim levels (§7.5 aggregation rules).
"""

from __future__ import annotations

from enum import Enum
from fractions import Fraction
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

CanonicalModel = BaseModel

# --------------------------------------------------------------------------
# Status semantics (§12) — status belongs to a claim_id, not to a directory.
# --------------------------------------------------------------------------


class ClaimStatus(str, Enum):
    PASS = "pass"
    SUPPORT_ONLY = "support_only"
    FAIL = "fail"
    INVALID = "invalid"


class ExperimentStatus(str, Enum):
    PASS = "pass"
    SUPPORT_ONLY = "support_only"
    FAIL = "fail"
    INVALID = "invalid"


# --------------------------------------------------------------------------
# Reason codes (§11)
# --------------------------------------------------------------------------


class ReasonCode(str, Enum):
    # Target gate
    T001_ESTIMAND_UNSPECIFIED = "T001_ESTIMAND_UNSPECIFIED"
    T002_BIAS_EXCEEDS_TOLERANCE = "T002_BIAS_EXCEEDS_TOLERANCE"
    T003_LOCAL_TO_PREFIX_PROPAGATION = "T003_LOCAL_TO_PREFIX_PROPAGATION"
    T004_CONTINUATION_TARGET_MISMATCH = "T004_CONTINUATION_TARGET_MISMATCH"
    T005_CLIPPING_SCOPE_MISMATCH = "T005_CLIPPING_SCOPE_MISMATCH"
    # Sampling / support gate
    S001_ZERO_SUPPORT = "S001_ZERO_SUPPORT"
    S002_Q_NOT_LOGGED = "S002_Q_NOT_LOGGED"
    S003_WRONG_HH_HT_CORRECTION = "S003_WRONG_HH_HT_CORRECTION"
    S004_OUTCOME_ADAPTIVE_UNDECLARED = "S004_OUTCOME_ADAPTIVE_UNDECLARED"
    # Cost gate
    C001_UNMATCHED_TRANSITION_BUDGET = "C001_UNMATCHED_TRANSITION_BUDGET"
    C002_CALIBRATION_COST_OMITTED = "C002_CALIBRATION_COST_OMITTED"
    C003_BASELINE_ENTRYPOINT_UNFAITHFUL = "C003_BASELINE_ENTRYPOINT_UNFAITHFUL"
    # Utility gate
    U001_PRIMARY_THRESHOLD_MET = "U001_PRIMARY_THRESHOLD_MET"
    U002_UTILITY_THRESHOLD_FAILED = "U002_UTILITY_THRESHOLD_FAILED"
    # Mechanism gate
    MECH001_ADAPTIVE_MAPPING_COLLAPSED_TO_GLOBAL = "MECH001_ADAPTIVE_MAPPING_COLLAPSED_TO_GLOBAL"
    MECH002_ROOT_VS_LEAF_NOT_MATERIAL = "MECH002_ROOT_VS_LEAF_NOT_MATERIAL"
    # Data / split gate
    D001_SPLIT_OVERLAP = "D001_SPLIT_OVERLAP"
    D002_TEST_TIME_RESElection = "D002_TEST_TIME_RESElection"
    D003_OUTPUT_OVERWRITE = "D003_OUTPUT_OVERWRITE"
    # Environment / oracle gate
    E001_ALTERNATIVE_NOOP = "E001_ALTERNATIVE_NOOP"
    E002_ORACLE_MISMATCH = "E002_ORACLE_MISMATCH"
    E003_GROUP_VARIANCE_ZERO = "E003_GROUP_VARIANCE_ZERO"
    # Novelty / claim gate
    N001_NEAR_ZERO_SIGN = "N001_NEAR_ZERO_SIGN"
    # Provenance / evidence gate
    P001_EVIDENCE_INCOMPLETE = "P001_EVIDENCE_INCOMPLETE"


# --------------------------------------------------------------------------
# 7.1 EstimandSpec
# --------------------------------------------------------------------------


class EstimandSpec(CanonicalModel):
    estimand_id: str
    world_family: str
    policy_parameterization: str
    reward_semantics: str
    continuation_policy: str = "current_policy"
    state_conditioning: str = "on_policy_marginal"
    clipping: str = "none"
    discount: float = 1.0
    coordinate_map_sha256: str | None = None


# --------------------------------------------------------------------------
# 7.2 SamplingSpec
# --------------------------------------------------------------------------


class DecisionSampling(CanonicalModel):
    replacement: Literal["with_replacement", "without_replacement"]
    probabilities: list[float] | None = None
    probability_source: str = "frozen_protocol"
    minimum_support: float = 1e-6


class Restore(CanonicalModel):
    state_identity: str = "exact"
    latent_noise_coupling: str = "independent"


class Continuation(CanonicalModel):
    policy_identity: str = "current_policy"
    samples_per_branch: int = 1


class Correction(CanonicalModel):
    name: Literal["none", "hansen_hurwitz", "horvitz_thompson"] = "none"
    version: str = "v1"


class SamplingSpec(CanonicalModel):
    decision_sampling: DecisionSampling
    restore: Restore = Field(default_factory=Restore)
    continuation: Continuation = Field(default_factory=Continuation)
    correction: Correction = Field(default_factory=Correction)


# --------------------------------------------------------------------------
# 7.3 CostSpec — Fraction arithmetic, registered calculators only.
# --------------------------------------------------------------------------


class CostTerm(CanonicalModel):
    term_id: str
    quantity: str  # Fraction string "a/b"
    unit_cost: str  # Fraction string "a/b"
    subtotal: str  # Fraction string "a/b"

    def subtotal_fraction(self) -> Fraction:
        return _frac(self.quantity) * _frac(self.unit_cost)


class CostBreakdown(CanonicalModel):
    primary_unit: str = "environment_transition"
    terms: list[CostTerm]
    total: str  # Fraction string "a/b"

    @model_validator(mode="after")
    def _total_must_equal_sum(self) -> CostBreakdown:
        # Auditor re-sums term subtotals and checks units; a plugin cannot
        # return an undecomposable total (§7.3).
        total = _frac(self.total)
        if total != sum((t.subtotal_fraction() for t in self.terms), Fraction(0)):
            raise ValueError("total != sum(term subtotals)")
        return self

    def total_fraction(self) -> Fraction:
        return _frac(self.total)


def _frac(s: str) -> Fraction:
    return Fraction(s)


class _RegisteredCalculator:
    """Registry of cost calculators. No free-form expression strings; v0.1 only
    ships tested calculator enums (§7.3)."""

    def __init__(self) -> None:
        self._registry: dict[str, Any] = {}
        self.register("dense_horizon_v1", _dense_horizon_v1)
        self.register("d002_branching_v1", _d002_branching_v1)
        self.register("paired_replay_all_v1", _paired_replay_all_v1)

    def register(self, calculator_id: str, fn: Any) -> None:
        self._registry[calculator_id] = fn

    def has(self, calculator_id: str) -> bool:
        return calculator_id in self._registry

    def evaluate(self, calculator_id: str, **params: Any) -> CostBreakdown:
        if calculator_id not in self._registry:
            raise ValueError(f"unregistered cost calculator: {calculator_id}")
        return self._registry[calculator_id](**params)


def _dense_horizon_v1(**params: Any) -> CostBreakdown:
    h = params.get("horizon")
    if not isinstance(h, int) or h <= 0:
        raise ValueError("dense_horizon_v1 requires integer horizon > 0")
    return CostBreakdown(
        primary_unit="environment_transition",
        terms=[CostTerm(term_id="full_rollout", quantity=f"{h}/1", unit_cost="1/1", subtotal=f"{h}/1")],
        total=f"{h}/1",
    )


def _d002_branching_v1(**params: Any) -> CostBreakdown:
    # c(h, d, K) = c_prefix*d + c_suffix*K*(H-d) + c_restore*(K-1)  (§7.3, §8.3)
    horizon = params.get("horizon")
    depth = params.get("depth")
    width = params.get("width")
    c_prefix = _frac(params.get("prefix_transition_cost", "1/1"))
    c_suffix = _frac(params.get("suffix_transition_cost", "1/1"))
    c_restore = _frac(params.get("restore_overhead_per_extra_suffix", "1/1"))
    if not (isinstance(horizon, int) and isinstance(depth, int) and isinstance(width, int)):
        raise ValueError("d002_branching_v1 requires integer horizon/depth/width")
    if not (horizon > 0 and 0 < depth < horizon and width >= 1):
        raise ValueError("d002_branching_v1 domain: H>0, 0<d<H, K>=1")
    prefix = c_prefix * depth
    suffixes = c_suffix * width * (horizon - depth)
    restores = c_restore * (width - 1)
    total = prefix + suffixes + restores
    return CostBreakdown(
        primary_unit="environment_transition",
        terms=[
            CostTerm(term_id="prefix", quantity=f"{depth}/1", unit_cost=_fstr(c_prefix), subtotal=_fstr(prefix)),
            CostTerm(
                term_id="suffixes",
                quantity=f"{width * (horizon - depth)}/1",
                unit_cost=_fstr(c_suffix),
                subtotal=_fstr(suffixes),
            ),
            CostTerm(
                term_id="restores", quantity=f"{width - 1}/1", unit_cost=_fstr(c_restore), subtotal=_fstr(restores)
            ),
        ],
        total=_fstr(total),
    )


def _paired_replay_all_v1(**params: Any) -> CostBreakdown:
    """Semantic D002 cost (decision log D9): the cycle samples the full
    trajectory (h) plus, for each branched decision t >= d, one sibling
    continuation of length (h-t) and one restore. Width-free (the paired
    contrast is deterministic in the focal world)."""
    h = params.get("horizon")
    d = params.get("depth")
    if not (isinstance(h, int) and isinstance(d, int)):
        raise ValueError("paired_replay_all_v1 requires integer horizon/depth")
    if not (h > 0 and 0 <= d < h):
        raise ValueError("paired_replay_all_v1 domain: H>0, 0<=d<H")
    roll = Fraction(h, 1)
    cont = sum((Fraction(h - t, 1) for t in range(d, h)), Fraction(0))
    rest = Fraction(h - d, 1)
    return CostBreakdown(
        primary_unit="environment_transition",
        terms=[
            CostTerm(term_id="full_rollout", quantity=f"{h}/1", unit_cost="1/1", subtotal=_fstr(roll)),
            CostTerm(
                term_id="sibling_continuations",
                quantity=f"{cont.numerator}/{cont.denominator}",
                unit_cost="1/1",
                subtotal=_fstr(cont),
            ),
            CostTerm(
                term_id="restores",
                quantity=f"{rest.numerator}/{rest.denominator}",
                unit_cost="1/1",
                subtotal=_fstr(rest),
            ),
        ],
        total=_fstr(roll + cont + rest),
    )


def _fstr(f: Fraction) -> str:
    return f"{f.numerator}/{f.denominator}"


COST_CALCULATORS = _RegisteredCalculator()


class SharedCosts(CanonicalModel):
    calibration_transitions: str = "0/1"
    calibration_cpu_seconds: str = "report_only"
    backbone_policy: str = "included_in_cycle_formula"


class Amortization(CanonicalModel):
    mode: Literal["none", "predeclared"] = "none"
    denominator: int | None = None
    deployments: int | None = None


class CostSpec(CanonicalModel):
    primary_unit: str = "environment_transition"
    arithmetic: Literal["rational", "float64"] = "rational"
    calculator_id: str
    calculator_code_sha256: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    shared_costs: SharedCosts = SharedCosts()
    amortization: Amortization = Amortization()
    total_budget_grid: list[int] = Field(default_factory=list)
    rounding: Literal["floor_complete_cycles"] = "floor_complete_cycles"
    infeasible_if_budget_below_cycle_cost: bool = True
    leftover_policy: str = "reserved_dummy_no_gradient_work"

    def evaluate_cycle(self, **kwargs: Any) -> CostBreakdown:
        params = {**self.parameters, **kwargs}
        return COST_CALCULATORS.evaluate(self.calculator_id, **params)

    def n_complete_cycles(self, budget: int, **kwargs: Any) -> tuple[int, Fraction]:
        """floor(budget / cycle_cost); returns (n, unused). INFEASIBLE when
        budget < cycle_cost (§7.3)."""
        cycle = self.evaluate_cycle(**kwargs).total_fraction()
        if cycle <= 0:
            raise ValueError("cycle cost must be positive")
        if budget < cycle:
            return 0, Fraction(budget, 1)
        n = budget // cycle
        return n, Fraction(budget, 1) - n * cycle


# --------------------------------------------------------------------------
# 7.4 EstimatorSpec
# --------------------------------------------------------------------------


class EstimatorSpec(CanonicalModel):
    estimator_id: str
    version: str = "v1"
    claimed_estimand: str
    required_observations: list[str] = Field(default_factory=list)
    required_assumptions: list[str] = Field(default_factory=list)
    sampling_spec_sha256: str | None = None
    cost_spec_sha256: str | None = None


# --------------------------------------------------------------------------
# 7.5 AuditDecision / ClaimDecision
# --------------------------------------------------------------------------


class GateResult(CanonicalModel):
    gate: str  # integrity | target_identity | independent_oracle | sampling_support | matched_cost | heldout_split | utility | mechanism | environment | provenance | novelty
    status: Literal["pass", "fail", "invalid", "skipped"]
    reason_codes: list[ReasonCode] = Field(default_factory=list)
    detail: str | None = None


class ClaimCeiling(CanonicalModel):
    allowed: list[str] = Field(default_factory=list)
    forbidden: list[str] = Field(default_factory=list)


class ClaimDecision(CanonicalModel):
    claim_id: str
    claim_text: str
    status: ClaimStatus
    required_gates: list[str] = Field(default_factory=list)
    gate_results: list[GateResult] = Field(default_factory=list)
    reason_codes: list[ReasonCode] = Field(default_factory=list)
    claim_ceiling: ClaimCeiling = Field(default_factory=ClaimCeiling)


class HeadlineDecision(CanonicalModel):
    proposed_new_method_claim: ClaimStatus
    retained_narrow_claim: str | None = None


class AuditDecision(CanonicalModel):
    experiment_integrity: ClaimStatus = ClaimStatus.PASS
    claims: list[ClaimDecision] = Field(default_factory=list)
    headline_decision: HeadlineDecision | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _integrity_dominates(self) -> AuditDecision:
        # §7.5: integrity INVALID makes every dependent claim INVALID.
        if self.experiment_integrity == ClaimStatus.INVALID:
            for claim in self.claims:
                claim.status = ClaimStatus.INVALID
        return self


# --------------------------------------------------------------------------
# Moments (§5.5) — exact bias, variance trace, MSE, fixed-budget MSE.
# --------------------------------------------------------------------------


class MomentResult(CanonicalModel):
    estimand_id: str
    estimator_id: str
    target: list[float]
    expectation: list[float]
    bias: list[float]
    bias_sq: float
    var_trace: float
    mse: float
    single_cycle_cost: str | None = None  # Fraction string, if costed
    n_cycles_at_budget: int | None = None
    mse_at_budget: float | None = None
    mean_abs_target: float
    max_abs_bias: float
    near_zero_target: bool = False


# --------------------------------------------------------------------------
# Protocol document (top-level frozen config)
# --------------------------------------------------------------------------


class Protocol(CanonicalModel):
    protocol_id: str
    protocol_version: str
    description: str | None = None
    reconstruction_mode: Literal["docs_only_semantic", "legacy_exact"]
    frozen_at: str
    world_family: str
    policy_parameterization: str
    reward_semantics: str
    primary_estimand: str | None = None
    tolerances: dict[str, float] = Field(default_factory=dict)
    gates: dict[str, Any] = Field(default_factory=dict)
    claims: list[Any] = Field(default_factory=list)
    expected_outcome: dict[str, Any] = Field(default_factory=dict)
    fault_injection_expected: dict[str, str] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _collect_extra(cls, data: Any) -> Any:
        # Allow protocols with per-experiment fields not listed here; they are
        # preserved under `extra`. This keeps the frozen JSON files literal.
        if isinstance(data, dict):
            known = {
                "protocol_id",
                "protocol_version",
                "description",
                "reconstruction_mode",
                "frozen_at",
                "world_family",
                "policy_parameterization",
                "reward_semantics",
                "primary_estimand",
                "tolerances",
                "gates",
                "claims",
                "expected_outcome",
                "fault_injection_expected",
            }
            extras = {k: v for k, v in data.items() if k not in known}
            if extras:
                data = {**data, "extra": extras}
        return data
