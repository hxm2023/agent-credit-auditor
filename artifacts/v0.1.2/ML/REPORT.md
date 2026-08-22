# Minimal logging — teaching asset (SUPPORT_ONLY)

NOVELTY STATUS: CLASSICAL DECISION-REDUCT / FD / HITTING-SET EQUIVALENCE
CLAIM STATUS: TEACHING OR TELEMETRY-SCHEMA DIAGNOSTIC ONLY (design 13.5)

- point-label assignments (4 values x 8 rows): 65536
- point-eligible: 65536  (100.0000%)
- sign-eligible: 65536  (100.0000%)
- point minimal-schema-size distribution: {1: 36, 2: 684, 3: 64812, 0: 4}
- sign minimal-schema-size distribution: {1: 1536, 2: 7680, 3: 55808, 0: 512}
- default-universe minimal schemas: [(0, 1, 2)]
- runtime: 1.88 CPU seconds (GPU 0)

## Honesty notes
- docs_only_semantic: new frozen universe; the legacy counts
  (390625 / 390112 / 3.81% / 4.21% / 198.63 s) are incident background
  and are NOT reproduced.
- The schema problem is the hitting set of the different-label conflict
  pairs; sign is label coarsening (design 3.5).