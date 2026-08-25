# Agent-RL Credit Auditor

[![CI](https://img.shields.io/github/actions/workflow/status/hxm2023/agent-credit-auditor/ci.yml?branch=main&label=CI)](https://github.com/hxm2023/agent-credit-auditor/actions)
[![Release](https://img.shields.io/github/v/release/hxm2023/agent-credit-auditor?label=release)](https://github.com/hxm2023/agent-credit-auditor/releases)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12-blue)](.python-version)

CPU-first audit and exact-benchmark tool for credit estimators in agent RL.

**Not** a new credit-assignment algorithm. It forces every candidate method to
answer four questions (§0 of the design handbook) — and refuses to let a
pretty learning curve answer them:

| # | Question | What it kills | Gate |
|---|---|---|---|
| 1 | **Estimand** — what exactly are you estimating (full policy gradient, a local decision gradient, root-marginal, a continuation-specific effect)? | local-contrast methods that update shared prefixes without proving they still estimate the full gradient | T001-T005 |
| 2 | **Bias** — against an independent oracle in finite, exactly-enumerable worlds, is the estimator unbiased or is the bias explainable? | any estimator whose "old logprobs" were never produced by the behavior policy | S001-S004 |
| 3 | **Cost** — under a matched transition/token/intervention budget, does fixed-budget MSE beat strong baselines? | methods that buy their wins with extra oracle calls or branch rollouts | C001-C003 |
| 4 | **Mechanism** — is a positive result really produced by the claimed adaptive/local/causal mechanism, or did it degenerate into fixed hyperparameters? | "adaptive" methods whose calibrated widths collapse to the global control | MECH001-MECH002 |

Real cases it caught (legacy incident background, `docs/caught_fakes.md`):
CPC's fake adaptivity (widths collapsed to [2,2,2,2]), the ρ=0.735 fake
variance (computed on no-op alternatives with undefined variance), and the
86% no-op fake success curve (static rollout policy).

## Design authority

The authoritative project design is
[`Agent-RL-Credit-Auditor_详细项目设计与旧项目迁移手册.md`](Agent-RL-Credit-Auditor_详细项目设计与旧项目迁移手册.md)
(v1.0, 2026-08-22). It locks the formal objects, schema, exact worlds, estimator
plugin contract, independent oracle, audit gates, regression package,
protocol-first runner, 7-day plan, budget and claims policy. All deviations and
semantic-reconstruction decisions are recorded in
[`docs/decision_log.md`](docs/decision_log.md) (D1-D9).

**Reconstruction mode: `docs_only_semantic`.** The legacy
`grpo-credit-assignment` code (credit_v2 / credit_transport / minimal_logging)
and its signed migration bundle are not available on this machine, so per §13.6
the release reproduces failure *types* (target mismatch, utility failure,
metric-pass/mechanism-fail) with new frozen protocols, seeds, and numbers.
Historical numbers (144/202, 24.81×, 0.694, 192/192, ρ=0.735, …) are incident
background only and never appear as reproduced results.

## Quick start

```bash
uv sync --extra dev --frozen

# one-command reproduction of the full release (all nine packs + report)
bash scripts/reproduce_all.sh artifacts/v0.1.6

# or run the packs individually
bash scripts/run_m0.sh      # target audit
bash scripts/run_v001.sh    # expected utility failure
bash scripts/run_d002.sh    # calibration + frozen test
uv run credit-auditor audit --artifact-dir artifacts/local/M0   # claims + ceilings
```

The runner never overwrites canonical outputs; re-runs must target new
directories, and every package carries `run_manifest.json` + `raw_rows.jsonl.zst`
+ `SHA256SUMS`. Frozen protocols are content-hashed; any edit is a
decision-logged version bump.

## Headline results (docs_only_semantic, all numbers new)

| Experiment | Verdict | What it demonstrates |
|---|---|---|
| M0 target audit | dense/HH unbiased; propagated sibling & BPO-like rejected (T003); paired-replay narrow positive passes | the Auditor approves unbiased estimators and rejects wrong targets/propagation |
| V001 utility failure | calibration accurate (err ~1e-16) but fixed-budget MSE FAILS (median 26.5x vs dense) | "calibration is accurate" does not imply utility; cost accounting matters |
| D002 dual verdict | calibrated mapping **PASS** vs the dense envelope (median ratio 0.21, CI [0.18, 0.23]); adaptive variable-width mechanism **FAIL** (calibrated widths collapse to [2,2,2,2]) | a metric pass does not license an adaptive-mechanism claim (§13.3) |
| CTRI-style continuation diagnostics (support_only) | zero false-safe abstention on mixed-sign fibers; marginal regime cannot identify the sign without a bridge assumption, paired replay identifies the replay summary | formal scope + classical coupled/nonrectangular robust-advantage mapping; not a new theory (§13.4) |
| CTRI large-scale census (support_only) | Fraction-exact sign/rank stability over 5k + 100k frozen continuation families (reversal rate ≈ 3.2%, scale-stable) | family-level diagnostics only; legacy 400/120,000 counts are background, not reproduced |
| CTRI census server supplement | autodl2 run at N=10⁷ (48 CPU workers, 88 s, 0 GPU): sign-reversal family rate **3.2744% ± 0.006 pp**, converging across 5k → 100k → 10⁷ | supplement artifact with execution evidence in `artifacts/scale_supplement/`; canonical packs remain the reproducible-from-clone evidence |
| Minimal-logging teaching asset (support_only) | 8×3 universe eligibility enumeration (point vs sign labels), minimal schemas | classical decision-reduct / FD / hitting-set equivalence; teaching only (§13.5) |

Claim ceilings: the D002 pass is a *fixed mapping efficiency* claim on a frozen
designed world with the paired-replay protocol — the width itself is not the
mechanism (the pre-registered raw-MSE calibration objective structurally
prefers the largest width, so the collapse to [2,2,2,2] is expected and
documented in the run's REPORT.md). It says nothing about adaptive methods,
LLM agents, or the historical 0.694.

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
  adapters/      GRPO-Guard envelope validation (§25, v0.2-prep, fail-closed)
  experiments/   m0 / v001 / d002 / continuation / minimal_logging drivers (§13)
  report.py      claim ceilings, reason codes, release report
scripts/         run_m0.sh / run_v001.sh / run_d002.sh / reproduce_all.sh
tests/           math units, oracle independence, protocol/evidence, fault injection
artifacts/       canonical run outputs (result/manifest/report, no-overwrite)
```

## The two-project trust chain (GRPO-Guard ↔ Credit Auditor)

These two repos are one story: **GRPO-Guard validates the online trajectory
chain** (policy/token/mask identity, canonical hashing, envelope lifecycle),
**Credit Auditor validates the offline estimator claims** (estimand, target,
cost, mechanism) — and they consume each other:

- **This repo audits trajectories from GRPO-Guard**: real Guard-issued
  trajectory envelopes flow through the Auditor's `CreditAuditBundle`
  validation (hash-only references, fail-closed on the pinned
  `grpo-guard-envelope-1.0` schema, §25), and the trajectory-level audit
  (`audit-trajectories`, v0.2-prep) checks the optimizer-consumed records
  for the Guard online faults that are offline-detectable (mask shift →
  T005, misbound logprob → S002, stale policy → T004).
- **The other direction**: GRPO-Guard's envelope contract is the online half
  of the same failure taxonomy — the `static_rollout`/`mask_shift` incidents
  that the Auditor catches offline are the ones Guard catches online
  (`docs/online_offline_fault_map.md`).
- **Cross-project demo**: `scripts/run_guard_demo.sh` (envelope → bundle) +
  `scripts/run_real_scenario_demo.sh` (Guard fault patterns injected into
  Auditor artifacts) + the Stage-3 real loop (18 Guard-supervised runs whose
  trajectories the Auditor then audits).

The online/offline split is the point: a trajectory can be perfectly
Guard-validated and still be audited for what its estimator claims.

## CTRI census at scale — N = 10⁷

The sign/rank stability census scales across three frozen sizes with the
same convergence behavior:

| N (families) | 5,000 | 100,000 | **10,000,000** |
|---|---:|---:|---:|
| sign-reversal rate | ≈ 3.2% | ≈ 3.2% | **3.2744% ± 0.006 pp** |
| hardware | laptop CPU | laptop CPU | autodl2, 48 CPU workers, 88 s, 0 GPU |
| evidence | `artifacts/v0.1.6/CSCALE` | `artifacts/v0.1.6/CSCALE_LARGE` | `artifacts/scale_supplement/` |

The 10⁷ run (Fraction-exact) was executed on the shared autodl2 server's CPU
cores with zero GPU use, in parallel with the GPU projects — the Auditor is
CPU-first by design. `scripts/run_on_autodl2.sh` reproduces it.

## v0.2-prep additions — Stage 3: matched-budget real closed loop (jindun)

The review's Stage 3 executed end-to-end: 18 Guard-supervised real GRPO runs
(2 tool-use tasks x 3 credit estimators x 3 seeds, Qwen3-4B LoRA, 32 prompts
x 8 gens x 3 epochs) on the jindun A800 server, one GPU at a time, yielding
to concurrent user jobs. Results (`stage3/REPORT.md`, pre-registered
predictions in `stage3/PREDICTIONS.md`, raw metrics + trajectory records in
`artifacts/stage3_jindun/`):

- dense/local produce real Guard-validated updates every epoch; the
  paired-branch reliability gate abstained in ALL 9 cts_order runs (zero
  credit -> zero updates) — a conservative-gate finding, not a bug
- tau2_retail: base Qwen3-4B produced no valid tool calls (invalid_rate
  1.0) — all-zero rewards, an honest task/model negative
- prediction P1 VOID (no final-success differentiation at this update
  scale), P2 CONFIRMED (paired abstains), P3 INCONCLUSIVE, P4 partial
- execution limits: vLLM 0.26 unavailable on both servers' GPUs (SM 12.x
  on autodl2; env mismatch on jindun) — all rollouts used the trainer's own
  sampler with Guard `exact_behavior_scorer` mode (schema 7.5)

## v0.2-prep additions — real-trajectory bridge, Stages 1-2

The external review direction: the Auditor must audit the data the optimizer
actually consumes, not just manifests.

**Stage 1 — trajectory-level audit** (`audit/trajectory_audit.py` +
`adapters/trajectory_bundle.py` + CLI `audit-trajectories`):

- Rollout trajectory records (tokens + action mask + old logprobs + rewards)
  are checked for the Guard online faults that are offline-detectable at
  record level — mask_shift → T005, misbound_logprob → S002,
  retokenization → T005, stale/mixed policy_version → T004, and
  estimator-vs-optimizer mask drift → T005. `scripts/run_trajectory_demo.sh`
  injects all six fault types into frozen fixture records; every detector
  fires, clean baseline stays clean.
- Hash-anchored trajectory bundles: sha256-only refs under a pinned schema
  (`aca-trajectory-bundle-1.0`), fail-closed on mutation.

**Stage 2 — exact-to-real evidence bridge** (`worlds/tool_agent.py` +
`estimators/bridge_estimators.py` + `oracles/mc_reference.py` +
`experiments/evidence_bridge.py`, `docs/evidence_bridge.md`):

- Controllable tool-agent tasks (observation-dependent tool-use MDP, two
  frozen H=4 tasks + two H=12 twins).
- The SAME estimators as the exact worlds (dense / local sibling /
  paired-replay / pc-rsg) run on both layers: exact enumeration (bias,
  intrinsic cycle variance) and trajectory-record consumption under matched
  transition budgets.
- **Result**: the exact-layer predictor var·cost/B + bias² reproduces the
  sampled fixed-budget MSE with ratios 0.87–1.07 on all estimator-task pairs;
  the exact layer flags that the designed-world unbiasedness of
  paired-replay/pc-rsg does NOT transfer to observation-dependent worlds
  (the coupled contrast misses the indirect effect of a_t through future
  observations/actions). `scripts/run_evidence_bridge.sh` reproduces it.
- Honest boundary: the record format is the Auditor's own frozen fixture
  spec; estimators consume records only, so real LLM trajectories drop in
  unchanged (Stage 3 harness, gated on the Guard trajectory schema package,
  §20.2 gate). No claim about real LLM agent performance.

## v0.1.4 additions

- **Self-audit (audit of the auditor)**: every fault type A1-A14 injected
  into frozen random instances (9 types N=200, 4 runner-based types N=30):
  all 13 predefined fault templates trigger their expected reason code
  (regression TPR=1.0) with zero control false positives. The suite is a
  Predefined Fault Mutation Regression Suite (templates and expected codes are
  co-constructed; no general TPR/FPR claim). The self-audit
  itself caught and fixed 3 control-group bugs during development.
- **Mechanism theory**: [`docs/mechanism_theory.md`](docs/mechanism_theory.md)
  — closed-form derivations (paired-replay variance → why the win is
  structural; the K-sample prefix floor → why the historical 0.694 is not
  reconstructable; the HH 1/q amplification) each verified == 0 diff against
  exact enumeration; MECH001 now carries a pre-registered statistical test.
- **Real-scenario fault injection**:
  [`docs/online_offline_fault_map.md`](docs/online_offline_fault_map.md)
  maps GRPO-Guard online faults → offline-detectable signals → Auditor
  gates; `scripts/run_real_scenario_demo.sh` injects Guard fault patterns
  into the Auditor's own artifacts (split leakage refused, artifact mutation
  hash-failed, event-reorder manifest flagged) — answering "has this project
  really been used?" with concrete detectors firing.

## v0.1.3 additions

- **GRPO-Guard real-trajectory integration (§25 bridge)**: REAL Guard-issued
  trajectory envelopes (frozen fixtures copied from the GRPO-Guard repo) now
  flow through the Auditor's `CreditAuditBundle` validation — hash-only
  references, fail-closed pin on `grpo-guard-envelope-1.0` (the Guard repo's
  own schema version), no write-back. `scripts/run_guard_demo.sh` runs the
  demo; this is the exact-toy → real-toolchain connection.
- **Interview narrative**: [`docs/tech_narrative.md`](docs/tech_narrative.md)
  — the 10-minute technical story with the honest designed-vs-discovered
  distinction and the §24.4 FAQ answers.

## v0.1.2 additions

- **Fraction-exact cross-validation**: M0's frozen problems are verified with
  exact `fractions.Fraction` arithmetic — the primary and BOTH oracles align
  with **mismatch == 0** (no float rounding in the enumeration; §10.3).
- **CTRI large-scale census**: Fraction-exact sign/rank stability census over
  frozen continuation families (canonical N=5,000 + large-sample N=100,000);
  sign-reversal rate ≈ 3.2%, stable across scales. Server-scale N≥10⁷ via
  `scripts/run_on_autodl2.sh` (CPU-only, user-executed).
- **legacy_exact readiness**: `validate-legacy-bundle` checks the §13.6 bundle
  structure, content hashes, and the out-of-band root anchor; mode gating
  fails closed (legacy_exact only with a protocol flag AND an anchored bundle).
- **Dev experience**: `scripts/run_smoke.sh` (fast set, ~85 s), `report
  --skip-tests`, and a GitHub Actions CI (sync → validate → smoke →
  fresh-clone M0 → full suite → guard CLI).

## v0.1.1 additions

- `scripts/reproduce_all.sh`: one-command reproduction of the full release.
- GRPO-Guard envelope adapter (§25): `CreditAuditBundle` references Guard
  envelopes by sha256 only; validation fails closed on unknown schema major,
  unknown required extensions, or missing hashes; CPU-only, no Guard server.
- Protocol validation rejects unknown gate names / reason codes / claim gates
  at validate time (typos cannot silently enter frozen configs).
- `credit-auditor audit` prints per-claim status + ceiling and exits nonzero
  when evidence integrity fails.

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
