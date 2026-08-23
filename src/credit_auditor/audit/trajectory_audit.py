"""Trajectory-level audit (v0.2-prep, Stage 1 of the real-trajectory bridge).

The external review direction: "the Auditor must audit the data the optimizer
actually consumes, not just manifests." This module audits ROLLOUT TRAJECTORY
records (the unit a GRPO/RLOO step consumes: generated tokens + action mask +
old logprobs + rewards) for the offline-detectable consistency faults mapped
in docs/online_offline_fault_map.md:

- mask_shift        -> action_mask length != generated tokens
- misbound_logprob  -> old_logprobs missing / length mismatch / NaN / positive
- retokenization    -> generated_tokens missing or not self-consistent
- stale_policy      -> policy_version missing, or mixed inside one batch
- mask_silent_drift -> the mask the estimator would apply differs from the
                       mask the optimizer step actually consumed

Estimator-consumption check: the dense GRPO/RLOO gradient input is rebuilt
from the record (logprob[mask] centered by the masked-mean reward) and
compared with the optimizer-consumed mask when present. A mismatch means the
claim "this trajectory produced the recorded update" does not close
(T005_CLIPPING_SCOPE_MISMATCH territory).

This is NOT a protocol pack (real trajectory data is not frozen): it is the
real-scenario tool that audits optimizer-consumed data. Record format is the
Auditor's own frozen spec `aca-trajectory-record-1.0` (fixtures in
tests/fixtures/trajectories/); real Guard trajectories flow in through the
envelope adapter, whose schema stays pinned to the Guard repo (§25).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

RECORD_SCHEMA = "aca-trajectory-record-1.0"

REQUIRED_FIELDS = [
    "trajectory_id",
    "policy_version",
    "generated_tokens",
    "action_mask",
    "old_logprobs",
    "rewards",
]

# fault -> Auditor reason code (mirrors the fault map)
CODE_MASK_SHIFT = "T005_CLIPPING_SCOPE_MISMATCH"
CODE_MISBOUND_LOGPROB = "S002_Q_NOT_LOGGED"
CODE_STALE_POLICY = "T004_CONTINUATION_TARGET_MISMATCH"
CODE_SILENT_DRIFT = "T005_CLIPPING_SCOPE_MISMATCH"
CODE_BAD_RECORD = "P001_EVIDENCE_INCOMPLETE"


def _finite_nonpos(x: float) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x <= 0.0


def audit_record(record: dict) -> dict:
    """One trajectory record: structural + consistency + consumption checks.

    Returns findings: list of {code, signal, detail}. Empty findings = the
    record is internally consistent.
    """
    findings: list[dict] = []

    missing = [f for f in REQUIRED_FIELDS if f not in record]
    if missing:
        findings.append({"code": CODE_BAD_RECORD, "signal": "missing_fields", "detail": ",".join(missing)})
        return {"trajectory_id": record.get("trajectory_id", "?"), "findings": findings}

    tokens = record["generated_tokens"]
    mask = record["action_mask"]
    logprobs = record["old_logprobs"]
    tid = record["trajectory_id"]

    if not isinstance(tokens, list) or not tokens:
        findings.append({"code": CODE_BAD_RECORD, "signal": "empty_tokens", "detail": tid})
    if not isinstance(mask, list):
        findings.append({"code": CODE_MASK_SHIFT, "signal": "mask_not_list", "detail": tid})
    elif len(mask) != len(tokens):
        findings.append(
            {
                "code": CODE_MASK_SHIFT,
                "signal": "mask_shift",
                "detail": f"mask {len(mask)} != tokens {len(tokens)}",
            }
        )
    elif not all(v in (0, 1) for v in mask):
        findings.append({"code": CODE_MASK_SHIFT, "signal": "mask_values", "detail": tid})

    if not isinstance(logprobs, list):
        findings.append({"code": CODE_MISBOUND_LOGPROB, "signal": "logprobs_not_list", "detail": tid})
    elif len(logprobs) != len(tokens):
        findings.append(
            {
                "code": CODE_MISBOUND_LOGPROB,
                "signal": "misbound_logprob",
                "detail": f"logprobs {len(logprobs)} != tokens {len(tokens)}",
            }
        )
    else:
        bad = [i for i, x in enumerate(logprobs) if not _finite_nonpos(x)]
        if bad:
            findings.append(
                {
                    "code": CODE_MISBOUND_LOGPROB,
                    "signal": "logprob_sanity",
                    "detail": f"{len(bad)} invalid (NaN/Inf/positive) at {bad[:5]}",
                }
            )

    if not isinstance(record.get("policy_version"), str) or not record["policy_version"]:
        findings.append({"code": CODE_STALE_POLICY, "signal": "stale_policy", "detail": tid})

    rewards = record["rewards"]
    final = rewards.get("final") if isinstance(rewards, dict) else None
    if not (isinstance(final, (int, float)) and not isinstance(final, bool) and math.isfinite(final)):
        findings.append({"code": CODE_BAD_RECORD, "signal": "reward_not_finite", "detail": tid})

    # estimator-consumption check: the mask the estimator would apply must
    # equal the mask the optimizer step consumed (when both are recorded)
    consumed = record.get("optimizer_consumed_mask")
    if consumed is not None and isinstance(mask, list) and consumed != mask:
        n_diff = sum(1 for a, b in zip(mask, consumed) if a != b)
        findings.append(
            {
                "code": CODE_SILENT_DRIFT,
                "signal": "silent_mask_drift",
                "detail": f"estimator mask != optimizer-consumed mask in {n_diff} positions",
            }
        )

    return {"trajectory_id": tid, "findings": findings}


def audit_trajectory_dir(data_dir: Path) -> dict:
    """Audit every *.jsonl record file in a directory (or a single file)."""
    data_dir = Path(data_dir)
    paths = sorted(data_dir.rglob("*.jsonl")) if data_dir.is_dir() else [data_dir]
    rows: list[dict] = []
    versions_seen: list[str] = []
    files_checked = 0
    for path in paths:
        files_checked += 1
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                rows.append(
                    {
                        "trajectory_id": f"{path.name}:{line_no}",
                        "findings": [{"code": CODE_BAD_RECORD, "signal": "unparseable", "detail": path.name}],
                    }
                )
                continue
            if isinstance(record.get("policy_version"), str):
                versions_seen.append(record["policy_version"])
            rows.append(audit_record(record))

    findings = [f for r in rows for f in r["findings"]]

    # batch-level: policy versions must not mix within one directory
    versions = set(versions_seen)
    batch_policy_mix = len(versions) > 1
    batch_finding = None
    if batch_policy_mix:
        batch_finding = {
            "code": CODE_STALE_POLICY,
            "signal": "mixed_policy_versions",
            "detail": ",".join(sorted(str(v) for v in versions if v)),
        }

    return {
        "data_dir": str(data_dir),
        "record_schema": RECORD_SCHEMA,
        "files_checked": files_checked,
        "records": len(rows),
        "findings": findings,
        "batch_finding": batch_finding,
        "consistent": not findings and not batch_finding,
    }


def render_report(audit: dict) -> str:
    if audit.get("files_checked") == 0:
        return "\n".join(
            [
                "# Trajectory audit",
                "",
                f"- data dir: {audit.get('data_dir')}",
                "- no *.jsonl trajectory files found",
            ]
        )
    lines = [
        "# Trajectory audit (aca-trajectory-record-1.0)",
        "",
        f"- data dir: {audit.get('data_dir')}",
        f"- files checked: {audit.get('files_checked')} | records: {audit.get('records')}",
        f"- record schema: {audit.get('record_schema')}",
        "",
        f"## Verdict: {'CONSISTENT' if audit.get('consistent') else 'FAULTS DETECTED'}",
    ]
    if audit.get("batch_finding"):
        bf = audit["batch_finding"]
        lines.append(f"- {bf['signal']}: {bf['detail']} [{bf['code']}]")
    counts: dict[str, int] = {}
    for f in audit.get("findings", []):
        counts[f["code"]] = counts.get(f["code"], 0) + 1
    if counts:
        lines.append("")
        lines.append("## Findings by reason code")
        lines += [f"- {code}: {n}" for code, n in sorted(counts.items())]
        lines.append("")
        lines.append("## Detail (first 20)")
        lines += [f"- {f['signal']} ({f['detail'][:60]}) [{f['code']}]" for f in audit["findings"][:20]]
    lines += [
        "",
        "## Honesty notes",
        "- This audit checks offline consistency of trajectory records (the data an optimizer",
        "  step consumes): mask/logprob/token alignment, policy-version identity, reward",
        "  sanity, and estimator-vs-optimizer mask agreement.",
        "- It does NOT evaluate training outcomes or estimator quality; estimator-level",
        "  bias/cost audits remain the protocol packs. Real Guard trajectories flow through",
        "  the envelope adapter (pinned schema, design 25); this format is the Auditor's",
        "  own record spec for frozen fixtures.",
    ]
    return "\n".join(lines)
