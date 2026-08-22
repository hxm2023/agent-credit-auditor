# D002 test (docs_only_semantic)

- frozen mapping: {'early_sensitive': (0, 2), 'late_reusable': (0, 2), 'medium_mixed': (0, 2), 'short_mixed': (0, 2)}
- selected widths: [2, 2, 2, 2]
- median ratio vs envelope: 0.2054 (bootstrap [0.1766, 0.2289])
- utility gate: PASS
- mechanism gate: fail (['MECH001_ADAPTIVE_MAPPING_COLLAPSED_TO_GLOBAL'])

## Dual verdict (design 13.3)
- C1 fixed-budget efficiency: PASS
- C2 adaptive variable-width mechanism: FAIL (widths collapse to the global control)
- headline: proposed adaptive method FAIL; retained claim: fixed global-K only.

## Honesty notes
- docs_only_semantic: new frozen world/seeds/numbers (decision log D9); historical 0.694 and 192/192 are incident background.
- Mechanism-fail is STRUCTURAL under the pre-registered calibration objective
  (mean log exact-trace MSE): the raw MSE always prefers the largest width,
  so the calibrated widths collapse to the global control by construction.
  The demonstration shows a metric pass does NOT license an adaptive
  mechanism claim; it does not claim the gate would catch every fake
  adaptive method (the two-sided behavior of width_diversity_gate is
  unit-tested directly).
- Calibration cost: exact CPU enumeration, REPORTED only (protocol shared
  costs calibration_transitions 0/1), never charged to the test budget
  (legacy protocol boundary, design 7.3).