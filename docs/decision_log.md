# Agent-RL Credit Auditor — Decision Log

> ARIS execution discipline: every decision recorded with
> **Decision · Evidence · Alternative · Why rejected · Falsification**,
> before the first formal run. Deviations from the design doc are marked DEVIATION.
>
> Project: Agent-RL-Credit-Auditor (clean repo, CPU-first, docs_only_semantic)
> Design doc: `Agent-RL-Credit-Auditor_详细项目设计与旧项目迁移手册.md` v1.0 (2026-08-22)
> Start: 2026-08-22

---

## D1 — Reconstruction mode: `docs_only_semantic`

- **Decision**: Run v0.1 in `reconstruction_mode=docs_only_semantic`. All frozen
  protocols, seeds, thresholds and golden fixtures are new. Old numbers
  (144/202, 24.81×, 0.694, 192/192, 390112, ρ=0.735, 36.5%→63.5%, …) are
  historical incident background only and never appear in the new README's
  reproduced-results table or in claims (§13.6, §23).
- **Evidence**: The local `grpo-credit-assignment` checkout (C:\Users\w1828\repos)
  contains no `src/credit_v2`, `src/credit_transport`, `src/minimal_logging`, no
  `configs/v001_phase_diagram_20260821_*.json`, no `d002_*_seeds_*.json`, no
  `deep-experiment-logs/M0_FORMAL_VALIDATION/` dirs, and no
  `legacy_bundle_manifest.json` / `SHA256SUMS`. A filesystem-wide search under
  C:\Users\w1828\repos confirmed the modules do not exist anywhere on this
  machine. Design §13.6: without the signed bundle, the only authorized mode is
  `docs_only_semantic`, and missing legacy files are NOT an engineering blocker.
- **Alternative**: `legacy_exact` — requires the one-time migration bundle with
  content hashes and a trusted out-of-band trust anchor; not available.
- **Why rejected**: The design doc explicitly forbids claiming exact reproduction
  without the bundle; fabricating legacy hashes is forbidden (§21.3: never
  invent legacy hashes; `migration_kind=clean_reimplementation_from_design_doc`).
- **Falsification**: If a signed `legacy_bundle_manifest.json` with a root SHA-256
  arrives via the design-doc update or trusted channel, mode may be upgraded to
  `legacy_exact` in a NEW protocol version; old semantic results stay as-is.

## D2 — Repo identity: current directory is the clean new repo (DEVIATION, cosmetic)

- **Decision**: The current git repo `Agent-RL-Credit-Auditor` (created 2026-08-22,
  commits: scaffold constitution + compute note) is the clean new repo. Python
  package name: `credit_auditor` (per design §15 tree). Repo keeps the human-chosen
  name instead of renaming to `agent-credit-auditor`.
- **Evidence**: Design §15 shows repo dir `agent-credit-auditor/` with package
  `src/credit_auditor/`. The human established this repo before this run with
  CLAUDE.md + design doc committed; §0/§21's intent is a clean repo distinct from
  the legacy `grpo-credit-assignment` museum — satisfied.
- **Alternative**: Create a new repo dir named `agent-credit-auditor` and move.
- **Why rejected**: Pure churn; the design's hard requirement is "clean new repo,
  never rename-and-continue the old repo". Package name follows §15 exactly.
- **Falsification**: If the legacy repo content appears inside this repo, or this
  repo is a rename of `grpo-credit-assignment`, the decision is void.

## D3 — Environment

- **Decision**: Local laptop (Windows 11, Git Bash, Python 3.12.10, uv 0.10.10),
  CPU-only, 0 GPU. No server reservation needed for v0.1 (design §20: v0.1 is
  0 GPU·h; laptop sufficient per CLAUDE.md).
- **Evidence**: Design §20.1: CPU 8-32 cores, runs of seconds-to-minutes, full
  D002 < 2 CPU·h target, artifacts < 2 GB. CLAUDE.md: "Local laptop also
  sufficient for Auditor if server CPU is busy."
- **Alternative**: Run on autodl2 CPU cores (shared with GRPO-Guard/agent-ttrl).
- **Why rejected**: Not needed; avoid contention with GPU projects. Revisit only
  if a full D002-style sweep exceeds laptop time budget.
- **Falsification**: If any v0.1 experiment needs > 2 CPU·h or > 8 GB RAM on the
  laptop, re-evaluate.

## D4 — Legacy 56 tests: semantic migration only

- **Decision**: The 56 legacy CPU tests (credit_v2 28 + credit_transport 12 +
  minimal_logging 16) are NOT runnable here (code absent). Their validated
  *semantics* (§22.2–§22.4 function inventories, §17.1 math properties) are
  migrated into the new suite as fresh tests. Target: ≥ 70 tests (§17.4).
  No old test file is byte-copied; no old number is reused.
- **Evidence**: Modules absent on machine (D1). Design §17.4: "迁移旧 56 tests 的
  有效语义，不追求逐文件机械复制"; §13.6: missing legacy files are not a blocker.
- **Alternative**: Block Phase 0 until a bundle is produced from the old machine.
- **Why rejected**: Design explicitly says don't treat missing bundle as an
  engineering blocker; the semantic property set (§17.1) is fully specified in
  the design doc.
- **Falsification**: If new tests do not cover the §17.1 property list, Phase 2
  is not complete.

## D5 — D002 semantic reconstruction: new frozen generator, no historical parity

- **Decision**: Implement the shared-logit world per §8.3 *semantics* (3 shared
  logits, binary states/actions, terminal reward tables, bucket strata,
  calibration/test disjointness, mapping space 7^4, cycle cost
  c(h,d,K)=d+K(h-d)+(K-1)r with r=1, budget grid {512,1024,2048,4096}, bootstrap
  10,000 paired problem-level ratios) but with NEW frozen seeds and problem
  counts (calibration 12, test 48). The historical generator algorithm
  (§8.3.1) is not byte-reproducible without golden fixtures; using it would
  create unverifiable pseudo-legacy numbers.
- **Evidence**: §13.6: docs_only_semantic only requires reproducing failure
  *types*; §8.3: "若没有 legacy generator/source bundle，只能建立新的 semantic
  reconstruction protocol，不能宣称复现了上述历史数值".
- **Alternative**: Transcribe §8.3.1 and claim semantic parity.
- **Why rejected**: Unverifiable; violates claim honesty (§23.1 forbids old
  numbers under docs_only_semantic).
- **Falsification**: If any README/artifact number coincides with a historical
  legacy number by chance, it is flagged and re-derived from new artifacts
  (identical numbers are possible by coincidence but must never be *cited* as
  legacy reproduction).

## D6 — V001 expected-fail fixture design

- **Decision**: Build at least one frozen case where residual estimation is
  *well-calibrated* (its mean/q estimates accurate) but fixed-budget MSE utility
  FAILS (residual noise amplification + branch continuation cost), per
  §13.2/§24.2. Pre-registered threshold: median relative improvement ≥ 0.2 with
  bootstrap lower bound > 0 vs both dense and uniform-HH baselines, else FAIL.
  New protocol `v001_failure_v1`, new numbers only.
- **Evidence**: §13.2: "Docs-only semantic 模式只要求预先构造并冻结至少一个
  'calibration 准确但 fixed-budget utility 失败' 的 case".
- **Alternative**: Port historical V001 numbers/params for semantic alignment.
- **Why rejected**: Historical numbers require legacy-exact (§13.6, §23.1).
- **Falsification**: If the frozen fixture ever passes utility after protocol
  changes without a decision-logged protocol version bump, integrity is broken.

## D7 — Claim policy enforcement

- **Decision**: Only §23.1 docs_only_semantic claims may appear in README/resume.
  §23.2 forbidden claims are checked by the release gate and by review.
  Every published number must trace to artifact dir + git commit + SHA256SUMS
  entry (§23.1, CLAUDE.md honesty rules).
- **Evidence**: §23 claim table; CLAUDE.md honesty rules.
- **Alternative**: Include legacy numbers "for context" in README.
- **Why rejected**: §13.6/§23 explicitly forbid; would poison the audit tool's
  own claim hygiene.
- **Falsification**: If any release artifact contains a §23.2 claim or an
  untraceable number, release is blocked.

## D8 — Pre-run protocol refinement (M0, before first formal run)

- **Decision**: Amend `m0_regression_v1` (hash changes; the version stays v1
  because NO formal run exists yet — the amendment predates the first artifact):
  1. `bpo_like` reason code in `bpo_prefix_propagation` changed from
     `T002` to `T003` (structural classification: BPO-like both selects
     uncorrected AND propagates; the gate's mechanism-aware code assignment
     is more precise than the initial draft).
  2. `matched_cost_positive` case pinned: frozen focal world (H=6, w=0.05,
     noise=1.0 at coordinates 2,3), paired-replay sibling with skip at
     zero-target coordinates; expected `narrow_positive=true` and
     `uncoupled_control_loses=true`.
- **Evidence**: Numerical exploration (2026-08-22, before any formal run):
  paired-replay sibling is unbiased (bias² < 1e-29) and wins fixed-budget MSE
  57× (ratio 0.017) on the focal world; the uncoupled control loses 14× —
  the win genuinely requires the paired-replay mechanism. The initial draft
  of the case (HH-style sampling) could not win under D002 cost conventions
  (analytical + numeric dead ends recorded during exploration).
- **Alternative**: Leave the protocol untouched and let the case fail.
- **Why rejected**: The designed case is required by §8.2 case 5; a case that
  cannot be satisfied under the cost convention would make the Auditor
  vacuous ("everything fails"). The chosen world is pre-registered, frozen,
  and disclosed as designed (claim ceiling limits it to a narrow synthetic
  positive).
- **Falsification**: If the paired-replay estimator fails the target gate
  (bias > tolerance) on the frozen focal world, the case is broken and the
  whole M0 run is INVALID.

## D9 — D002 semantic world design (pre-run, before the first formal run)

- **Decision**: The D002 semantic reconstruction uses a NEW frozen world within
  the `d002_shared_logits_mdp` family (decision D5's generator was amended
  pre-run):
  1. **State-independent policy**: both states at each time share one logit
     (valid shared-logit MDP), so decisions are independent Bernoulli draws
     and the centered-noise zero-target property holds exactly.
  2. **Deterministic transitions** s_{t+1} = a_t (semantic choice; the §8.3.1
     clip range cannot represent exact determinism).
  3. **Focal terminal reward** with CENTERED noise (a_{n1}-p)(a_{n2}-p) at
     non-adjacent zero-target times; focal effects w(2a_t - 1) elsewhere.
  4. **Estimator**: paired-replay branching from depth d onward (skip the
     zero-target noise times); width K enters the candidate space but the
     deterministic contrast makes it cost- and variance-neutral.
  5. **Cost calculator**: new registered `paired_replay_all_v1`
     (h + Σ_{t≥d}((h-t)+1)); protocol `cost_spec.calculator_id` amended.
  6. **Candidate depths** amended to {0,2}/{0,2}/{0,1}/{0,3} so the
     branch-from-depth-0 option exists; claim text made width-generic.
- **Evidence**: Extensive pre-run numerical exploration (recorded during
  development): the K-sample continuation-averaging estimator (the natural
  reading of "branch width K") has an unavoidable shared-prefix variance floor
  Var(E[g|prefix]) that makes it lose to the dense envelope in every
  terminal-reward configuration tried (scale sweeps x1/x2/x4; depth sweeps;
  no reconstruction of the historical 0.694 — reported honestly as a
  non-reproducibility). The paired-replay structure is the only verified
  unbiased winner (bias < 1e-16; 4.9x test MSE win with CI [0.18, 0.23]).
- **Alternative**: Port the historical generator byte-for-byte from §8.3.1.
- **Why rejected**: §13.6 forbids claiming reproduction without golden
  fixtures; the summary-level generator has unresolvable ambiguities (reward
  semantics, estimator definition, coupling).
- **Falsification**: If the frozen focal world ever fails the oracle alignment
  (enum or Bellman mismatch > 1e-9), the D002 pack is INVALID. If the
  calibrated widths ever show diversity >= 2, the mechanism-fail claim is void.

## D10 — v0.1.5 packs removed from the tree; v0.1.6 regenerated with LF (2026-08-23)

- **Decision**: The v0.1.5 artifact packs committed with CRLF line endings
  were removed from the working tree (kept in git history and the v0.1.5
  GitHub release) and the full release regenerated as `artifacts/v0.1.6`:
  (a) `.gitattributes` (`* text=auto eol=lf`) + canonical LF writes fix the
  cross-platform checksum root cause; (b) the strict provenance validator now
  fails on empty SHA256SUMS and enforces expected-file coverage; (c) a CI
  release gate (`scripts/validate_artifacts.sh`) runs the Auditor's own
  validation + `sha256sum -c` on every checked-in pack.
- **Evidence**: External review (docs/gpt_review_round1.md, P0-1) showed
  73/98 SHA256SUMS mismatches on the v0.1.5 packs when checked out on Linux —
  Windows had hashed CRLF bytes while git stores LF. `credit-auditor audit`
  on the v0.1.5 packs returned integrity=fail.
- **Alternative**: Patch the v0.1.5 checksums in place.
- **Why rejected**: That would hash the rewritten bytes post-hoc, which is
  exactly the "files rewritten but checksums not regenerated" failure the
  review called out; it would not fix the root cause.
- **Falsification**: If any regenerated v0.1.6 pack fails `sha256sum -c` on a
  Linux checkout or fails the strict provenance validation, the release gate
  is broken and the pack is INVALID.

## D11 — Stage 1 trajectory bridge: the Auditor's own record format (2026-08-23)

- **Decision**: The trajectory-level audit consumes `aca-trajectory-record-1.0`
  records — the Auditor's OWN frozen fixture spec (`audit/trajectory_audit.py`,
  `adapters/trajectory_bundle.py`, CLI `audit-trajectories`). Real
  GRPO-Guard trajectories keep flowing through the envelope adapter
  (`guard_integration.py`, schema pinned to the Guard repo, §25); the record
  format is NOT claimed to be Guard's schema.
- **Evidence**: External review direction "audit the data the optimizer
  consumes, not just manifests" (docs/gpt_review_round1.md); the six
  trajectory faults (mask_shift/misbound_logprob/retokenization/stale/mixed
  policy/silent mask drift) all map to existing reason codes (T005/S002/T004).
- **Alternative**: Wait for Guard's trajectory schema package before building
  the offline audit.
- **Why rejected**: The offline audit must exist independently of Guard's
  release schedule; the adapter layer keeps the two decoupled (§20.2 gate
  intact for real Guard records).
- **Falsification**: If a real Guard record cannot be expressed in the
  record format via the adapter without semantic loss, the bridge claim is
  void.

## D12 — Stage 2 evidence bridge: controllable tool-agent worlds (2026-08-24)

- **Decision**: The exact-to-real bridge uses a NEW controllable tool-agent
  world family (`worlds/tool_agent.py`: observation-dependent tool-use MDP,
  two H=4 exact tasks + two H=12 sampling twins). The exact-layer predictor
  `var_cycle * cost / B + bias^2` is the pre-registered quantitative claim;
  the estimators are the SAME definitions as the M0 exact worlds.
- **Evidence**: Prediction ratios 0.87-1.07 across all estimator-task pairs;
  the transfer finding (paired-replay unbiasedness does NOT hold in
  observation-dependent worlds — the coupled contrast misses the indirect
  effect of the decision through future observations/actions) is a REAL
  exact-layer discovery, verified by cycle-MC agreement (6-sigma) for every
  estimator.
- **Alternative**: Extend the existing Bernoulli/shared-logit worlds with
  observation dependence.
- **Why rejected**: The tool-agent structure (call/observe/decide) is the
  minimal structure that exhibits the indirect-effect phenomenon; reusing
  the old families would not have surfaced it.
- **Falsification**: If the exact predictor fails (ratio outside [0.5, 2.0])
  on any estimator-task pair at fixed budget, the bridge claim is void.

## D13 — Stage 3 real closed loop: execution deviations (2026-08-24/25)

- **Decision**:
  1. All rollouts use the TRAINER'S OWN sampler with Guard
     `behavior_logprob_source="exact_behavior_scorer"` (schema §7.5) — vLLM
     0.26 cannot initialize on either server (autodl2's Blackwell SM 12.x;
     jindun's env mismatch). Recorded in every run's metrics as a documented
     limitation; the generation-service mode is NOT exercised.
  2. The 18-run matrix was executed on jindun (js3.blockelite.cn, 8x A800),
     one GPU at a time, free-card-only, yielding to concurrent user jobs —
     autodl2's partial 3/18 dense runs (artifacts/stage3/) were superseded.
  3. Paired-branch credit uses a (decision slots x 2 branches) utility
     matrix with the reliability gate — CRN coupling is NOT reproduced from
     the exact world (real rollouts are independent draws); documented in
     stage3/credit.py.
  4. Malformed tool calls raise in the CTS reward path on jindun's
     agent_ttrl copy; the trainer now catches reward exceptions (reward 0 +
     recorded error) instead of crashing.
- **Evidence**: 18/18 runs with Guard-validated updates where applicable
  (dense/local every epoch; paired gate abstained 9/9 — zero credit, zero
  updates); pre-registered predictions (stage3/PREDICTIONS.md) compared in
  stage3/REPORT.md (P1 VOID, P2 CONFIRMED, P3 INCONCLUSIVE, P4 partial);
  trajectory records audited offline (18 x 768 records, zero per-record
  findings).
- **Alternative**: Fix vLLM on the servers; run on autodl2 as planned.
- **Why rejected**: vLLM 0.26's SM 12.x incompatibility is a binary/runtime
  issue outside the project's control; jindun had free A800 cards and a
  shared-host discipline that matches the "yield to others" compute policy.
- **Falsification**: If any of the 18 runs fails the offline trajectory
  audit or its Guard events are inconsistent, the closed-loop claim is
  INVALID.

---

*Log opened 2026-08-22 before the first formal run. Append-only; new entries
numbered sequentially. Deviations are marked; design doc remains authoritative.*
