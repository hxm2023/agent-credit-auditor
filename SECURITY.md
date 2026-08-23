# Security policy

## Scope

Agent-RL Credit Auditor is a CPU-only, deterministic audit tool. It never
loads model weights, never executes remote code, and never requires GPU or
external services to run the frozen protocol packs. The GRPO-Guard adapter
reads serialized envelope JSON only and never writes back to Guard artifacts
(design §25).

## Reporting a vulnerability

This project is a personal engineering portfolio. For security issues
affecting the tool itself (e.g. a canonical-hashing bypass, an oracle
independence violation, or unsafe handling of artifact paths):

1. Do NOT open a public issue for exploitable vulnerabilities.
2. Open a GitHub Security Advisory (private) at
   https://github.com/hxm2023/agent-credit-auditor/security/advisories/new
   or email the maintainer via the GitHub profile.

## Trust model

- `docs_only_semantic` results are self-contained; no external data is
  trusted except the frozen protocol/seed files (content-hashed).
- `legacy_exact` mode is only reachable with a signed, out-of-band-anchored
  migration bundle (`credit-auditor validate-legacy-bundle`); a bundle's own
  SHA256SUMS is never self-anchoring (design §13.6).
- Guard envelopes are validated fail-closed: unknown schema majors, unknown
  required extensions, and missing content hashes are rejected.
