# 3-5 minute demo script

## Setup (before the demo)

```bash
cd agent-credit-auditor   # clean clone
uv sync --frozen
uv run credit-auditor validate-protocol configs/protocols/m0_regression_v1.json
```

## Part 1 — the four audit questions (30s)

> Credit Auditor 不是新算法，它逼每个候选方法回答四个问题：estimand 是什么、
> 相对独立 oracle 是否无偏、matched budget 下 MSE 是否更好、正结果是否来自声称
> 的机制。开场先跑 M0 看 target audit 怎么工作。

## Part 2 — M0 target audit (60s)

```bash
uv run credit-auditor run \
  --protocol configs/protocols/m0_regression_v1.json \
  --output artifacts/local/M0 \
  --seed configs/seeds/m0_problems.json
cat artifacts/local/M0/REPORT.md
```

Show: dense/HH unbiased PASS; `propagated_sibling` and `bpo_like` rejected
(T003); the matched-cost positive (paired-replay) PASSES with the uncoupled
control failing — the positive is mechanism-driven.

## Part 3 — V001: calibration accurate ≠ utility (60s)

```bash
uv run credit-auditor run \
  --protocol configs/protocols/v001_failure_v1.json \
  --output artifacts/local/V001 \
  --seed configs/seeds/v001_problems.json \
  --seed configs/seeds/v001_calibration.json
cat artifacts/local/V001/REPORT.md
```

Show: calibration error ~1e-16 (PASS) while fixed-budget MSE fails 26.5x —
"the residual is unbiased but the cost accounting kills it."

## Part 4 — D002 dual verdict (90s)

```bash
uv run credit-auditor run \
  --protocol configs/protocols/d002_regression_v1.json \
  --phase calibration --output artifacts/local/D002_cal \
  --seed configs/seeds/d002_calibration.json
uv run credit-auditor run \
  --protocol configs/protocols/d002_regression_v1.json \
  --phase test --output artifacts/local/D002_test \
  --frozen-selection artifacts/local/D002_cal/selection.json \
  --seed configs/seeds/d002_test.json
cat artifacts/local/D002_test/REPORT.md
```

Show: `global_k8_efficiency` PASS (median ratio 0.21) + `variable_width_adaptivity`
FAIL (widths [2,2,2,2] = global control) + headline FAIL with the narrow claim
retained. This is the flagship: total metrics cannot mask mechanism failure.

## Part 5 — evidence discipline (30s)

```bash
uv run credit-auditor audit --artifact-dir artifacts/local/M0
cat artifacts/local/M0/SHA256SUMS | head
```

Show: no-overwrite, run_manifest, SHA256SUMS; every published number traces to
artifact + commit + checksum.

## Talking points for questions

- Local vs full gradient: a sibling contrast estimates the local effect at one
  decision; propagating it to a shared prefix changes the estimand (T003).
- HH vs HT: WR sampling uses selection probabilities (Hansen-Hurwitz); WOR uses
  inclusion probabilities (Horvitz-Thompson); mismatching them is S003.
- Biased can still win on fixed-budget MSE — the Auditor reports both.
- The D002 dual verdict: metric pass says "global-K is efficient on this frozen
  world"; mechanism fail says "there is no adaptive mechanism" — two claims,
  two verdicts.
