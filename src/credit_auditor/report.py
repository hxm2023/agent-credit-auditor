"""Release report builder (design §15 tree, §18).

Scans a canonical artifact root (e.g. artifacts/v0.1.0), verifies every
experiment directory against the package format, and emits:
- result_index.json: per-experiment status/claims/headline
- environment.json: platform/deps
- TEST_LOG.txt: fresh test-suite summary (must be re-run at release time)
- REPORT.md: claim ceilings + forbidden extrapolations + honesty notes
Failed runs still enter the index (design §17.3).
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

from credit_auditor.audit.provenance import REQUIRED_PACKAGE_FILES, audit_artifact_dir
from credit_auditor.canonical import atomic_write_json, atomic_write_text, sha256_file, sha256_tree


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_release_report(artifact_root: Path, output: Path | None = None, run_tests: bool = True) -> Path:
    artifact_root = artifact_root.resolve()
    if not artifact_root.is_dir():
        raise FileNotFoundError(f"artifact root not found: {artifact_root}")
    output = (output or artifact_root).resolve()
    output.mkdir(parents=True, exist_ok=True)

    index: list[dict] = []
    for exp_dir in sorted(p for p in artifact_root.iterdir() if p.is_dir() and not p.name.endswith(".tmp")):
        audit = audit_artifact_dir(exp_dir)
        entry: dict = {
            "experiment": exp_dir.name,
            "integrity": audit["integrity"],
            "errors": audit["errors"],
            "protocol_id": None,
            "exit_status": None,
            "claims": [],
            "headline": None,
        }
        manifest_path = exp_dir / "run_manifest.json"
        gd_path = exp_dir / "gate_decision.json"
        if manifest_path.is_file():
            man = _load(manifest_path)
            entry["protocol_id"] = man.get("protocol_id")
            entry["exit_status"] = man.get("exit_status")
            entry["protocol_sha256"] = man.get("protocol_sha256")
        if gd_path.is_file():
            gd = _load(gd_path)
            entry["claims"] = [{"claim_id": c["claim_id"], "status": c["status"]} for c in gd.get("claims", [])]
            entry["headline"] = gd.get("headline_decision")
            entry["integrity"] = gd.get("experiment_integrity", audit["integrity"])
        index.append(entry)

    atomic_write_json(
        {
            "artifact_root": str(artifact_root),
            "build_date": "2026-08-22",
            "experiments": index,
        },
        output / "result_index.json",
    )
    atomic_write_json(
        {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "processor": platform.processor(),
            "source_commit": _git_commit(),
            "dirty": _git_dirty(),
        },
        output / "environment.json",
    )

    if run_tests:
        test_log = _run_tests()
    else:
        test_log = "TEST_LOG skipped (run_tests=False; release build runs it)\n"
    (output / "TEST_LOG.txt").write_text(test_log, encoding="utf-8")

    report = _render_report(index)
    atomic_write_text(report, output / "REPORT.md")
    sums = {k: v for k, v in sha256_tree(output).items() if k != "SHA256SUMS"}
    (output / "SHA256SUMS").write_text("\n".join(f"{v}  {k}" for k, v in sorted(sums.items())) + "\n", encoding="utf-8")
    return output


def _run_tests() -> str:
    """Run the test suite for TEST_LOG.txt, EXCLUDING the release-report
    tests (marked `release_report`) to avoid infinite recursion (a report
    build inside a report build)."""
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--tb=no", "-m", "not release_report"],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        return f"pytest exit={out.returncode}\n{out.stdout[-2000:]}\n{out.stderr[-2000:]}"
    except Exception as e:  # noqa: BLE001
        return f"pytest failed to run: {e}"


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _git_dirty() -> bool:
    try:
        out = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        return bool(out.stdout.strip())
    except Exception:  # noqa: BLE001
        return True


def _render_report(index: list[dict]) -> str:
    lines = [
        "# Agent-RL Credit Auditor — v0.1.0 release report",
        "",
        "## Claim decisions",
        "",
    ]
    for entry in index:
        lines.append(f"### {entry['experiment']}  (integrity: {entry['integrity']}, exit: {entry['exit_status']})")
        for c in entry["claims"]:
            lines.append(f"- `{c['claim_id']}`: **{c['status']}**")
        if entry["headline"]:
            lines.append(f"- headline: {entry['headline']}")
        lines.append("")
    lines += [
        "## Claim ceilings and forbidden extrapolations (design §23)",
        "",
        "- Allowed (docs_only_semantic): exact finite-MDP unbiasedness checks,",
        "  matched-budget MSE comparisons, dual-verdict (metric pass / mechanism fail)",
        "  demonstrations, and the narrow fixed-width synthetic efficiency claim.",
        "- FORBIDDEN: 'proposed an effective credit method'; legacy success curves;",
        "  rho=0.735; 'CPC works'; detection-rate overclaims; any historical number",
        "  (144/202, 24.81x, 0.694, 192/192) presented as reproduced.",
        "- No claim about real LLM-agent downstream utility is made; exact finite-MDP",
        "  results never represent real task distributions.",
        "",
        "## Honesty notes",
        "",
        "- reconstruction_mode=docs_only_semantic (decision log D1): no legacy bundle",
        "  exists on this machine; the 56 legacy tests' semantics were migrated,",
        "  their numbers were not.",
        "- The D002 'global-K efficiency' pass holds only on the frozen semantic world",
        "  with the paired-replay protocol; the adaptive mechanism claim failed",
        "  (widths collapsed to [2,2,2,2]) and is NOT claimed.",
        "- Every number above traces to the artifact dirs + git commit + SHA256SUMS.",
        "- Known limitations (design 17.2): the world spec JSON is produced by the",
        "  primary-side world code; a wrong estimand definition embedded in the spec",
        "  would align oracle and primary. This is mitigated by (a) two oracles using",
        "  different algorithms, (b) import isolation tests, and (c) the non-degeneracy",
        "  pre-gate; it is not a formal proof (no formality claim is made).",
    ]
    return "\n".join(lines)
