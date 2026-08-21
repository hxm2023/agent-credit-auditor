# Agent-RL Credit Auditor

CPU-first audit and exact-benchmark tool for credit estimators in agent RL.

**Not** a new credit-assignment algorithm. It forces every candidate method to
answer four questions (§0 of the design handbook):

1. **Estimand** — what exactly is being estimated (full policy gradient, a local
   decision gradient, root-marginal, a continuation-specific effect)?
2. **Bias** — against an independent oracle in finite, exactly-enumerable worlds,
   is the estimator unbiased or is the bias explainable?
3. **Cost** — under a matched transition/token/intervention budget, does
   fixed-budget MSE beat strong baselines?
4. **Mechanism** — is a positive result really produced by the claimed
   adaptive/local/causal mechanism, or did it degenerate into fixed
   hyperparameters?

## Design authority

The authoritative project design is
[`Agent-RL-Credit-Auditor_详细项目设计与旧项目迁移手册.md`](Agent-RL-Credit-Auditor_详细项目设计与旧项目迁移手册.md)
(v1.0, 2026-08-22). It locks the formal objects, schema, exact worlds, estimator
plugin contract, independent oracle, audit gates, regression package,
protocol-first runner, 7-day plan, budget and claims policy.

**Reconstruction mode: `docs_only_semantic`.** The legacy
`grpo-credit-assignment` code (credit_v2 / credit_transport / minimal_logging)
and its signed migration bundle are not available on this machine, so per §13.6
the release reproduces failure *types* (target mismatch, utility failure,
metric-pass/mechanism-fail) with new frozen protocols, seeds, and numbers.
Historical numbers (144/202, 24.81×, 0.694, 192/192, ρ=0.735, …) are incident
background only and never appear as reproduced results. See
[`docs/decision_log.md`](docs/decision_log.md) (D1).

## Quick start

```bash
uv sync --frozen
uv run credit-auditor validate-protocol configs/protocols/m0_regression_v1.json
uv run credit-auditor run \
  --protocol configs/protocols/m0_regression_v1.json \
  --output artifacts/local/M0 \
  --seed configs/seeds/m0_problems.json
uv run credit-auditor audit --artifact-dir artifacts/local/M0
```

## Architecture

```
configs/         frozen protocols + seed manifests (pre-registered)
src/credit_auditor/
  schema.py      EstimandSpec/SamplingSpec/CostSpec/EstimatorSpec/ClaimDecision (§7)
  canonical.py   deterministic JSON + SHA-256 content hashing
  runner.py      protocol-first pipeline, no-overwrite, atomic publish (§16)
  worlds/        exact finite MDP/SCM enumerators (§8)
  estimands/     formal targets, independent of estimators (§5.2)
  estimators/    plugin implementations (§9)
  oracles/       self-contained Bellman/enumeration oracles in separate processes (§10)
  audit/         T/S/C/U/M/D/E/N gates (§11) + A1-A14 fault matrix (§14)
  experiments/   m0 / v001 / d002 semantic regression drivers (§13)
  report.py      claim ceilings, reason codes, release report
tests/           math units, oracle independence, protocol/evidence, fault injection
artifacts/       canonical run outputs (result/manifest/report, no-overwrite)
```

## Status semantics

Per claim, not per directory (§12): `PASS` (all required gates), `SUPPORT_ONLY`
(formally sound but insufficient novelty/external validity), `FAIL`
(pre-registered gate failed), `INVALID` (oracle mismatch / split pollution /
missing artifacts). `FAIL` is a correct audit result, not a software error.

## Claims policy (§23)

Only the allowed claims in §23.1 may be used in public materials. Forbidden:
“proposed an effective credit method”, legacy success curves, ρ=0.735,
“CPC works”, detection-rate overclaims. Every published number traces to an
artifact directory + git commit + SHA256SUMS entry.

## License

MIT (see [LICENSE](LICENSE)).
