# M0 target regression (docs_only_semantic)

- protocol: m0_regression_v1
- problems: 12
- designed cases: bpo_prefix_propagation, shared_logit_predictable_width, outcome_retention, completion_deadline, matched_cost_positive
- fraction-exact oracle alignment (mismatch == 0): True

## Claim decisions
- `dense_unbiased_full_gradient`: **pass** (no reason codes)
- `propagated_sibling_rejected`: **pass** (T003_LOCAL_TO_PREFIX_PROPAGATION)
- `paired_replay_matched_cost_positive`: **pass** (no reason codes)
- `fraction_exact_oracle_alignment`: **pass** (no reason codes)

## designed-case detail
- bpo_prefix_propagation
- shared_logit_predictable_width
- outcome_retention
- completion_deadline
- matched_cost_positive

## Honesty notes
- docs_only_semantic: numbers are new; legacy 144/202, 24.81x, 0.694 are incident background only.
- The matched-cost positive is a FROZEN designed world; the paired-replay mechanism is the cause (uncoupled control loses).
- No claim about LLM-agent utility is made.