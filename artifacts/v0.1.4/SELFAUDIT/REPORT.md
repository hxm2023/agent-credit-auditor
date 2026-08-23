# Self-audit: Auditor's fault matrix characterized (design 14)

- instances per fault type: 200 (heavy types: 30)
- elapsed: 1.4s

| fault | n | TPR | TPR CI | FPR |
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
- all TPR == 1.0: True
- all FPR == 0.0: True
- underperforming types: none

## Honesty notes
- The fault patterns mirror the failures the Auditor was built to catch (legacy route failures and GRPO-Guard online faults), run at scale on frozen random instances.
- A TPR < 1.0 would be a real Auditor bug and would be fixed, not reported away.