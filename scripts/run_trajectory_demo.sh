#!/usr/bin/env bash
# Trajectory-level fault demo (v0.2-prep, Stage 1 of the real-trajectory
# bridge): inject the Guard online faults that are offline-detectable at the
# TRAJECTORY level (the data an optimizer step consumes) and show the Auditor's
# detectors firing:
#
#   mask_shift         action_mask length != generated tokens      -> T005
#   misbound_logprob   old_logprobs length/sanity mismatch        -> S002
#   retokenization     token ids inconsistent with the prompt      -> T005
#   stale_policy       policy_version missing / mixed in a batch   -> T004
#   silent_mask_drift  estimator mask != optimizer-consumed mask   -> T005
#
# Usage: bash scripts/run_trajectory_demo.sh [OUTPUT_DIR]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-artifacts/trajectory_demo}"
rm -rf "$OUT"
mkdir -p "$OUT"
uv run python - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

from credit_auditor.adapters.trajectory_bundle import (
    BUNDLE_SCHEMA,
    trajectory_to_bundle,
    validate_trajectory_bundle,
)
from credit_auditor.audit.trajectory_audit import audit_trajectory_dir

out_dir = Path(sys.argv[1])
fixture = Path("tests/fixtures/trajectories/clean_trajectories.jsonl")
records = [json.loads(line) for line in fixture.read_text(encoding="utf-8").splitlines() if line.strip()]

def write(name, recs):
    (out_dir / name).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n", encoding="utf-8", newline="\n"
    )

def codes_of(audit):
    return sorted({f["code"] for f in audit["findings"]})

def run_case(name, recs, note=""):
    write(name, recs)
    a = audit_trajectory_dir(out_dir / name)
    detected = not a["consistent"]
    print(f"{name:28s} -> consistent={a['consistent']} codes={codes_of(a) or '-'} {note}")
    return a

results = []

a = run_case("clean_trajectories.jsonl", records, "(baseline)")
results.append(("clean copy (no fault)", a["consistent"], "none"))

mut = json.loads(json.dumps(records[0]))
mut["action_mask"] = mut["action_mask"][:-1]
a = run_case("f1_mask_shift.jsonl", [mut] + records[1:])
results.append(("f1 mask_shift", not a["consistent"], codes_of(a)))

mut = json.loads(json.dumps(records[1]))
mut["old_logprobs"] = mut["old_logprobs"] + [-0.9]
a = run_case("f2_misbound_logprob.jsonl", [mut])
results.append(("f2 misbound_logprob", not a["consistent"], codes_of(a)))

mut = json.loads(json.dumps(records[0]))
mut["generated_tokens"] = mut["generated_tokens"][1:]
a = run_case("f3_retokenization.jsonl", [mut])
results.append(("f3 retokenization", not a["consistent"], codes_of(a)))

mut = json.loads(json.dumps(records[2]))
mut["policy_version"] = ""
a = run_case("f4_stale_policy.jsonl", [mut])
results.append(("f4 stale_policy", not a["consistent"], codes_of(a)))

mut = json.loads(json.dumps(records[3]))
mut["policy_version"] = "v2"
a = run_case("f5_mixed_policy.jsonl", [mut] + records[:3])
results.append(("f5 mixed_policy", a["batch_finding"] is not None, [a["batch_finding"]["code"]] if a["batch_finding"] else []))

mut = json.loads(json.dumps(records[0]))
mut["optimizer_consumed_mask"] = [0, 1, 1]
a = run_case("f6_silent_mask_drift.jsonl", [mut])
results.append(("f6 silent_mask_drift", not a["consistent"], codes_of(a)))

# bundle anchoring: hash-only refs; a mutation must REJECT
write("bundle_records.jsonl", records)
bundle = trajectory_to_bundle(out_dir)
v = validate_trajectory_bundle(bundle.to_dict(), out_dir)
print(f"bundle validation        -> {v['status']} (refs={v['records']})")
(out_dir / "bundle_records.jsonl").write_text(
    json.dumps({"trajectory_id": "tampered"}), encoding="utf-8", newline="\n"
)
v2 = validate_trajectory_bundle(bundle.to_dict(), out_dir)
print(f"bundle after mutation    -> {v2['status']} reasons={v2['reasons'][:1]}")

all_detected = all(ok for _, ok, _ in results)

lines = [
    "# Trajectory-level fault demo (v0.2-prep, Stage 1)",
    "",
    "- record schema: aca-trajectory-record-1.0 | bundle schema: " + BUNDLE_SCHEMA,
    "- fixtures: tests/fixtures/trajectories/clean_trajectories.jsonl (frozen, committed)",
    "- bundle: hash-only refs, fail-closed on mutation",
    "",
    "| injected fault | detected | reason code(s) |",
    "|---|---|---|",
]
lines += [f"| {name} | {ok} | {','.join(c) if c else '-'} |" for name, ok, c in results]
lines += [
    "",
    f"## Verdict: {'ALL DETECTORS FIRE' if all_detected else 'MISSED DETECTIONS'}"
    + f" (clean baseline consistent={results[0][1]})",
    "",
    "## Honesty notes",
    "- The detectors check offline consistency of optimizer-consumed trajectory records;",
    "  they do not evaluate training outcomes. Real Guard trajectories still flow through",
    "  the envelope adapter (pinned to the Guard schema, design 25); this record format is",
    "  the Auditor's own fixture spec for the trajectory-level audit.",
]
(out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
print("report written:", out_dir / "REPORT.md")
print("VERDICT:", "ALL_DETECTORS_FIRE" if all_detected else "MISSED_DETECTIONS")
PY