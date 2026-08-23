# Agent-RL Credit Auditor — v0.1.0 release report

## Claim decisions

### CONT  (integrity: pass, exit: ok)
- `u1_zero_false_safe_abstention`: **support_only**
- `u2u3_stability_reported`: **support_only**
- headline: {'proposed_new_method_claim': 'support_only', 'retained_narrow_claim': None}

### CSCALE  (integrity: pass, exit: ok)
- `ctri_scale_sign_stability_census`: **support_only**
- headline: {'proposed_new_method_claim': 'support_only', 'retained_narrow_claim': None}

### CSCALE_LARGE  (integrity: pass, exit: ok)
- `ctri_scale_sign_stability_census`: **support_only**
- headline: {'proposed_new_method_claim': 'support_only', 'retained_narrow_claim': None}

### D002_cal  (integrity: pass, exit: ok)

### D002_test  (integrity: pass, exit: ok)
- `global_k8_efficiency`: **pass**
- `variable_width_adaptivity`: **fail**
- headline: {'proposed_new_method_claim': 'fail', 'retained_narrow_claim': 'global_k8_efficiency'}

### guard_demo  (integrity: fail, exit: None)

### M0  (integrity: pass, exit: ok)
- `dense_unbiased_full_gradient`: **pass**
- `propagated_sibling_rejected`: **pass**
- `paired_replay_matched_cost_positive`: **pass**
- `fraction_exact_oracle_alignment`: **pass**
- headline: {'proposed_new_method_claim': 'pass', 'retained_narrow_claim': 'paired_replay_matched_cost_positive'}

### ML  (integrity: pass, exit: ok)
- `minimal_logging_teaching`: **support_only**
- headline: {'proposed_new_method_claim': 'support_only', 'retained_narrow_claim': None}

### real_scenario_demo  (integrity: fail, exit: None)

### real_training  (integrity: fail, exit: None)

### real_training_audit  (integrity: fail, exit: None)

### scale_supplement  (integrity: fail, exit: None)

### SELFAUDIT  (integrity: pass, exit: ok)
- `self_audit_tpr_fpr`: **pass**
- headline: {'proposed_new_method_claim': 'pass', 'retained_narrow_claim': None}

### V001  (integrity: pass, exit: ok)
- `v001_utility_failure_reproduced`: **fail**
- `v001_calibration_accurate`: **pass**
- headline: {'proposed_new_method_claim': 'fail', 'retained_narrow_claim': None}

## Claim ceilings and forbidden extrapolations (design §23)

- Allowed (docs_only_semantic): exact finite-MDP unbiasedness checks,
  matched-budget MSE comparisons, dual-verdict (metric pass / mechanism fail)
  demonstrations, and the narrow fixed-width synthetic efficiency claim.
- FORBIDDEN: 'proposed an effective credit method'; legacy success curves;
  rho=0.735; 'CPC works'; detection-rate overclaims; any historical number
  (144/202, 24.81x, 0.694, 192/192) presented as reproduced.
- No claim about real LLM-agent downstream utility is made; exact finite-MDP
  results never represent real task distributions.

## Honesty notes

- reconstruction_mode=docs_only_semantic (decision log D1): no legacy bundle
  exists on this machine; the 56 legacy tests' semantics were migrated,
  their numbers were not.
- The D002 'global-K efficiency' pass holds only on the frozen semantic world
  with the paired-replay protocol; the adaptive mechanism claim failed
  (widths collapsed to [2,2,2,2]) and is NOT claimed.
- Every number above traces to the artifact dirs + git commit + SHA256SUMS.
- Known limitations (design 17.2): the world spec JSON is produced by the
  primary-side world code; a wrong estimand definition embedded in the spec
  would align oracle and primary. This is mitigated by (a) two oracles using
  different algorithms, (b) import isolation tests, and (c) the non-degeneracy
  pre-gate; it is not a formal proof (no formality claim is made).