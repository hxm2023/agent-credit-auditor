# Predefined Fault Mutation Regression Suite (design 14)

- instances per fault type: 200 (runner-based types: 30)
- elapsed: 2.1s

| fault template | n | regression TPR | TPR CI | control FPR |
|---|---:|---:|---:|---:|
| A1_LOCAL_AS_FULL | 200 | 1.000 | [np.float64(0.9812), 1.0] | 0.000 |
| A2_PROPAGATED | 200 | 1.000 | [np.float64(0.9812), 1.0] | 0.000 |
| A3_ZERO_SUPPORT | 200 | 1.000 | [np.float64(0.9812), 1.0] | 0.000 |
| A4_WR_HT | 200 | 1.000 | [np.float64(0.9812), 1.0] | 0.000 |
| A5_COST_OMITTED | 200 | 1.000 | [np.float64(0.9812), 1.0] | 0.000 |
| A6_WEAK_BASELINE | 200 | 1.000 | [np.float64(0.9812), 1.0] | 0.000 |
| A7_SPLIT_OVERLAP | 30 | 1.000 | [np.float64(0.8865), 1.0] | 0.000 |
| A9_WIDTH_COLLAPSE | 200 | 1.000 | [np.float64(0.9812), 1.0] | 0.000 |
| A10_ORACLE_IMPORT | 30 | 1.000 | [np.float64(0.8865), 1.0] | 0.000 |
| A11_NOOP_ALT | 200 | 1.000 | [np.float64(0.9812), 1.0] | 0.000 |
| A12_EVIDENCE_MISSING | 30 | 1.000 | [np.float64(0.8865), 1.0] | 0.000 |
| A13_OVERWRITE | 30 | 1.000 | [np.float64(0.8865), 1.0] | 0.000 |
| A14_NEAR_ZERO_SIGN | 200 | 1.000 | [np.float64(0.9812), 1.0] | 0.000 |

## Verdict
- all templates trigger their expected reason code: True
- no-fault controls clean (FPR == 0.0): True
- underperforming templates: none

## Honesty notes (scope of this suite)
- This is a PREDEFINED fault-mutation regression suite: each fault template and its
  expected reason code are co-constructed, so the numbers measure software-regression
  detection on frozen templates, NOT a general TPR/FPR against unknown real faults.
- The templates mirror the failures the Auditor was built to catch (legacy route
  failures and GRPO-Guard online faults) on frozen random instances.
- A missed template (TPR < 1.0) is a real Auditor bug and would be fixed, not reported away.