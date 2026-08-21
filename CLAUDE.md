# Agent-RL Credit Auditor — Credit-Estimator Audit & Exact Benchmark (Engineering Project)

**Goal**: Build Agent-RL Credit Auditor — a CPU-first audit and exact-benchmark tool
for credit estimators in agent RL. NOT a new credit-assignment algorithm: it forces
every candidate method to answer four questions (estimand? unbiased vs independent
oracle? MSE under matched budget? mechanism real or degenerated?). Engineering
flagship for job-market (post-training / RL / eval reliability), not a paper project.

**AUTHORITATIVE REQUIREMENTS**: `Agent-RL-Credit-Auditor_详细项目设计与旧项目迁移手册.md`
(project root, v1.0 2026-08-22) is the single source of truth (formal objects,
schema, exact worlds, estimator plugin contract, independent oracle, audit gates,
regression package, protocol-first runner, 7-day plan, claims policy). This CLAUDE.md
adds ARIS execution discipline; it does NOT redesign the project.

## What's Locked (from the design doc — non-negotiable)

- **Clean new repo** (`agent-credit-auditor`); old grpo-credit-assignment repo is
  legacy-evidence museum only — never rename-and-continue.
- **CPU-first**: exact finite-MDP worlds, independent oracle (never imports the
  estimator under test), matched transition/token/intervention budget, frozen
  calibration/test splits (protocol-first, pre-registered).
- **Four audit questions** (§0) + Audit Gates (§11): estimand contract; unbiasedness
  vs oracle (or explainable bias); matched-cost MSE vs strong baselines; mechanism
  verification (adaptive mappings must not degenerate to fixed hyperparameters).
- **Old failures as fixed regression cases** (§3, §22): CPC / PC-RSG / RMTPG / CTRI /
  minimal-logging become pass/fail fixtures proving the Auditor actively rejects
  wrong targets, unfair budgets, degenerated mechanisms, degraded environments and
  stale novelty.
- **Core formal objects** (§5) and schema (§7) as specified; estimator plugins obey
  the contract (§9); exact worlds per §8; oracle per §10 (non-degeneracy pre-gate:
  alternatives must change state, group variance non-zero).
- **Protocol-first runner** (§16): gate + result/manifest/report structure;
  no-overwrite canonical outputs; content hashes.
- **v0.1 regression package** (§13) + Auditor's own fault-injection matrix (§14);
  tests per §17; result package format per §18.
- **7-day implementation plan** (§19); budget §20 (CPU-first, light);
  open-source/migration strategy §21.
- **Claims policy (§23)**: only the allowed claims may appear on the resume;
  forbidden: "proposed effective credit method", old success curves, ρ=0.735,
  "CPC works", detection-rate overclaims. Legacy hashes (§22.4) preserved as
  protocol aliases, not re-run as results.
- **Definition of Done** = design doc §23 claims + §19 plan completion + gates.

## ARIS Role (adapted for an engineering/audit project)

- **No Phase 0-1 re-discovery**: design is authoritative and locked. ARIS provides:
  execution discipline (gates, decision logs), independent review at each gate
  (cross-model per `shared-references/reviewer-fallback.md`), regression-run
  management, claim-honesty enforcement.
- **Review gates**: before each audit gate, run an independent review targeting:
  oracle independence (no shared code with estimators), budget matching (no
  asymmetric compute), split freezing (calibration/test disjoint, frozen before
  runs), mechanism-check completeness (degeneration probes), claim honesty.
- **Decision log**: every deviation recorded with Decision · Evidence · Alternative ·
  Why rejected · Falsification — before first formal run.
- **Honesty rules**: no post-hoc gate loosening; regression fixtures frozen
  (updates create new versions, old results kept); published numbers traceable to
  artifact + commit + checksum.

## Compute (server TBD — reserved, do not bind)

- CPU-first: most work runs anywhere (local laptop sufficient for exact-world
  audits). GPU only if a future extension needs it — server choice PENDING
  (candidates: jindun / autodl1 / local). Record choice here when made; GPU
  contention rules of the chosen box apply.
- Budget: per design doc §20 (light); checkpoint/resume default.

## Pipeline (adapted)

- Phase 0: re-read design doc + old-asset index (§22) + verify 56 legacy CPU tests
  are reproducible as regression seeds (design doc §22.3 — current-tree numbers
  only prove CPU module self-consistency).
- Phase 1: frozen design confirmation + environment + protocol freeze.
- Phase 2 (audit loop): 7-day plan execution with per-day review; each run produces
  gate + result/manifest/report (no-overwrite).
- Phase 3: release package (README, regression results, LICENSE, claims-limited
  resume bullets per §23-24).
- Compliance: human owns the project; AI participation disclosed where required.
<!-- ARIS:BEGIN -->
## ARIS Skill Scope
ARIS skills installed in this project: 108 entries.
Manifest: `.aris/installed-skills.txt` (lists every skill ARIS installed and its upstream target).
For ARIS workflows, prefer the project-local skills under `.claude/skills/` over global skills.
Do not modify or delete files inside any skill that is a symlink (symlinks point into `/c/Users/w1828/repos/aris_repo`).
Update with: `bash /c/Users/w1828/repos/aris_repo/tools/install_aris.sh`  (re-runnable; reconciles new/removed skills).
<!-- ARIS:END -->
