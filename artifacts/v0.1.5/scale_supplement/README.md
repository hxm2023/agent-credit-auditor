# Server-scale census supplement (autodl2, 2026-08-23)

This supplement extends the frozen CTRI census packs (`CSCALE`, N=5,000 and
`CSCALE_LARGE`, N=100,000) with a server-scale run at **N = 10,000,000**
continuation families, executed on the autodl2 host (208 CPU cores, 1 TB RAM;
CPU-only, 0 GPU, bounded parallelism — no contention with GPU projects).

## Execution evidence

- script: `scripts/census_server_standalone.py` (stdlib-only, runs with zero
  installed packages on the shared host)
- consistency check: the standalone script reproduces the local package's
  counts EXACTLY at N=1,000 (verified before the server run)
- host: autodl2, workers: 48, elapsed: 88.4 s
- arithmetic: exact fractions.Fraction (no float sign flips, design §10.3)
- result: `census_result.json` (host, workers, counts, rates, elapsed_s)

## Rate convergence across scales (frozen seeds, same derivation)

| N | sign-reversal rate | SE (approx) |
|---|---:|---:|
| 5,000 (CSCALE) | 0.02880 | ±0.24 pp |
| 100,000 (CSCALE_LARGE) | 0.03244 | ±0.06 pp |
| 10,000,000 (this supplement) | 0.032744 | ±0.006 pp |

The three estimates agree within sampling error and converge; the
server-scale estimate pins the sign/rank-reversal family rate at
**3.2744% ± 0.006 pp** (sign reversal and rank reversal are the same event:
the Q(s,1)-Q(s,0) sign flips within the family).

## Honesty notes

- docs_only_semantic: the legacy counts (400 / 120,000 / 33,600) remain
  incident background, not reproduced.
- The supplement is an artifact, not a protocol pack: the canonical frozen
  packs (CSCALE / CSCALE_LARGE) are the reproducible-from-clean-clone
  evidence; this file adds the tighter large-sample estimate obtained with
  the server compute granted by the project owner.
