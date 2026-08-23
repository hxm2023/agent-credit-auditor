"""Protocol-first runner (design §16). Fixed order, no-overwrite, atomic publish.

Steps:
1. Parse protocol
2. Validate schema and prerequisites
3. Hash source/config/seed manifests
4. Refuse existing canonical output
5. Validate calibration/test disjointness
6. Generate or load exact worlds
7. Run primary estimators
8. Spawn independent oracle process
9. Compare targets and moments
10. Apply cost, utility, mechanism gates
11. Write result.json to temporary directory
12. Write run_manifest.json and REPORT.md
13. Compute SHA256SUMS
14. Atomically publish output directory

Steps 8-10 failures still produce an INVALID/FAIL result package. Prerequisite,
hash, or no-overwrite failures terminate before canonical output creation (§16).
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from credit_auditor.canonical import (
    atomic_write_json,
    atomic_write_text,
    refuse_existing,
    sha256_file,
    sha256_json,
    sha256_tree,
)
from credit_auditor.schema import Protocol


class DriverError(Exception):
    """Experiment driver failure that must still produce a result package."""


@dataclass
class RunContext:
    protocol: Protocol
    protocol_path: Path
    phase: str  # "run" | "calibration" | "test"
    output_dir: Path
    frozen_selection: Path | None = None
    seed_manifest_paths: list[Path] = field(default_factory=list)
    argv: list[str] = field(default_factory=list)
    source_hashes: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    result: dict[str, Any] = field(default_factory=dict)
    oracle_result: dict[str, Any] = field(default_factory=dict)
    gate_decision: dict[str, Any] = field(default_factory=dict)
    report_md: str = ""
    exit_status: str = "ok"  # ok | driver_failed | invalid | gate_failed
    manifest_extra: dict[str, Any] = field(default_factory=dict)
    raw_rows: list[Any] = field(default_factory=list)  # §18 raw rows -> zst only


_DRIVERS: dict[str, dict[str, Callable[[RunContext], RunResult]]] = {}


def register_driver(protocol_id: str, phase: str, fn: Callable[[RunContext], RunResult]) -> None:
    _DRIVERS.setdefault(protocol_id, {})[phase] = fn


def get_driver(protocol_id: str, phase: str) -> Callable[[RunContext], RunResult]:
    try:
        return _DRIVERS[protocol_id][phase]
    except KeyError:
        raise DriverError(f"no driver for protocol_id={protocol_id!r} phase={phase!r} (registered: {sorted(_DRIVERS)})")


def _source_hashes() -> dict[str, str]:
    src = Path(__file__).resolve().parent
    return sha256_tree(src)


def _lock_file_sha() -> str:
    """SHA-256 of the uv.lock file (P0-5: previously the literal string
    'uv.lock' was recorded instead of the file digest)."""
    lock = Path(__file__).resolve().parents[2] / "uv.lock"
    if lock.is_file():
        return sha256_file(lock)
    return "missing-uv.lock"


def _seed_manifest_hashes(paths: list[Path]) -> dict[str, str]:
    return {p.name: sha256_file(p) for p in paths}


KNOWN_GATES = {
    "integrity",
    "target_identity",
    "independent_oracle",
    "sampling_support",
    "matched_cost",
    "heldout_split",
    "utility",
    "mechanism",
    "environment",
    "provenance",
    "numerical_margin",
    "novelty",
}


def validate_protocol(protocol_path: Path) -> Protocol:
    """Parse + validate a frozen protocol, then check gate/reason-code
    consistency: unknown gate names, unknown reason codes, or unknown claim
    gates fail fast at validate time (no silent typos in frozen configs)."""
    from credit_auditor.schema import ReasonCode

    data = json.loads(protocol_path.read_text(encoding="utf-8"))
    proto = Protocol.model_validate(data)

    known_reason_codes = {rc.value for rc in ReasonCode}
    gates = proto.gates or {}
    unknown_gates = [g for g in gates if g not in KNOWN_GATES]
    if unknown_gates:
        raise ValueError(f"protocol {proto.protocol_id}: unknown gate(s) {unknown_gates}")
    for gate, spec in gates.items():
        codes = spec.get("reason_codes", []) if isinstance(spec, dict) else []
        bad = [c for c in codes if c not in known_reason_codes]
        if bad:
            raise ValueError(f"protocol {proto.protocol_id} gate {gate}: unknown reason code(s) {bad}")
    for claim in proto.claims or []:
        req = claim.get("required_gates", []) if isinstance(claim, dict) else []
        bad = [g for g in req if g not in KNOWN_GATES]
        if bad:
            raise ValueError(
                f"protocol {proto.protocol_id} claim {claim.get('claim_id')}: unknown required gate(s) {bad}"
            )
    return proto


def check_split_disjoint(cal_path: Path | None, test_path: Path | None) -> None:
    """Step 5: calibration and test seed sets must be disjoint (§11.6)."""
    if cal_path is None or test_path is None:
        return
    cal = {r["seed"] for r in json.loads(cal_path.read_text(encoding="utf-8"))["rows"]}
    test = {r["seed"] for r in json.loads(test_path.read_text(encoding="utf-8"))["rows"]}
    overlap = cal & test
    if overlap:
        raise DriverError(f"split overlap: {len(overlap)} seeds in both calibration and test")


def run_oracle_subprocess(oracle_script: Path, world_spec: dict[str, Any]) -> dict[str, Any]:
    """Step 8: spawn an independent oracle process. The script is a
    self-contained program reading one JSON world spec on stdin and writing
    one JSON result on stdout (§10.1)."""
    payload = json.dumps(world_spec, separators=(",", ":"), ensure_ascii=False)
    proc = subprocess.run(
        [sys.executable, str(oracle_script)],
        input=payload,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    if proc.returncode != 0:
        raise DriverError(f"oracle subprocess failed rc={proc.returncode}: {proc.stderr[-2000:]}")
    out = json.loads(proc.stdout)
    if not isinstance(out, dict):
        raise DriverError("oracle output must be a JSON object")
    return out


def run(
    protocol_path: Path,
    output_dir: Path,
    phase: str = "run",
    frozen_selection: Path | None = None,
    seed_manifests: list[Path] | None = None,
    argv: list[str] | None = None,
) -> Path:
    """Execute the §16 pipeline. Returns the canonical output dir."""
    protocol_path = protocol_path.resolve()
    output_dir = output_dir.resolve()

    # 1-2. parse + validate
    proto = validate_protocol(protocol_path)
    proto_hash = sha256_json(proto.model_dump(mode="json"))

    # 3. hash sources + seed manifests
    src_hashes = _source_hashes()
    seed_paths = seed_manifests or []
    seed_hashes = _seed_manifest_hashes(seed_paths)

    # 4. refuse existing canonical output (A13)
    refuse_existing(output_dir)

    # 5. disjointness
    cal = next((p for p in seed_paths if "calibration" in p.name), None)
    test = next((p for p in seed_paths if "test" in p.name), None)
    check_split_disjoint(cal, test)

    # 6-13. driver (generates worlds, estimators, oracles, gates)
    family = proto.protocol_id
    ctx = RunContext(
        protocol=proto,
        protocol_path=protocol_path,
        phase=phase,
        output_dir=output_dir,
        frozen_selection=frozen_selection,
        seed_manifest_paths=seed_paths,
        argv=argv or [],
        source_hashes=src_hashes,
    )
    try:
        driver = get_driver(family, phase)
        run_result = driver(ctx)
    except DriverError as e:
        run_result = RunResult(
            result={"status": "driver_failed", "error": str(e)},
            exit_status="driver_failed",
        )
    except Exception as e:  # noqa: BLE001 — still produce an evidence package
        run_result = RunResult(
            result={"status": "invalid", "error": f"{type(e).__name__}: {e}"},
            exit_status="invalid",
        )

    # 11-13. write temp package then publish
    tmp_dir = output_dir.parent / (output_dir.name + ".tmp")
    import shutil

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    try:
        _write_package(tmp_dir, proto, proto_hash, seed_hashes, src_hashes, run_result, ctx)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    os.replace(tmp_dir, output_dir)
    return output_dir


def _write_package(
    dirpath: Path,
    proto: Protocol,
    proto_hash: str,
    seed_hashes: dict[str, str],
    src_hashes: dict[str, str],
    run_result: RunResult,
    ctx: RunContext | None,
) -> None:
    start = _dt.datetime.now(_dt.timezone.utc)
    manifest = {
        "protocol_id": proto.protocol_id,
        "protocol_sha256": proto_hash,
        "phase": ctx.phase if ctx else None,
        "utc_start": start.isoformat(),
        "source_commit": _git_commit(),
        "dirty": _git_dirty(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "dependency_lock_sha": _lock_file_sha(),
        "seed_manifest_hashes": seed_hashes,
        "source_hashes": src_hashes,
        "argv": ctx.argv if ctx else [],
        "exit_status": run_result.exit_status,
    }
    # §18: raw rows live ONLY in the zst archive (v0.1.6 infra fix); the
    # manifest carries a count + hash reference, never the full rows
    # (previously the embedded raw_results bloated manifests to 10+ MB and
    # duplicated the zst content).
    raw = run_result.raw_rows or run_result.manifest_extra.get("raw_results") or []
    if isinstance(raw, list) and raw:
        from credit_auditor.canonical import write_jsonl_zst

        write_jsonl_zst(raw, dirpath / "raw_rows.jsonl.zst")
        manifest["raw_rows_count"] = len(raw)
        manifest["raw_rows_sha256"] = sha256_file(dirpath / "raw_rows.jsonl.zst")
    manifest.update(run_result.manifest_extra)

    atomic_write_json(proto.model_dump(mode="json"), dirpath / "protocol.json")
    atomic_write_json(run_result.result, dirpath / "result.json")
    atomic_write_json(run_result.oracle_result, dirpath / "oracle_result.json")
    atomic_write_json(run_result.gate_decision, dirpath / "gate_decision.json")
    atomic_write_json(manifest, dirpath / "run_manifest.json")
    atomic_write_text(run_result.report_md, dirpath / "REPORT.md")
    if run_result.result.get("selection") is not None:
        atomic_write_json(run_result.result["selection"], dirpath / "selection.json")

    sums = {k: v for k, v in sha256_tree(dirpath).items() if k != "SHA256SUMS"}
    lines = [f"{v}  {k}" for k, v in sorted(sums.items())]
    (dirpath / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    run_result.manifest_extra["output_hashes"] = sums


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _git_dirty() -> bool:
    """Dirty = uncommitted changes to TRACKED source files OR untracked
    source files (P0-5, GPT review: untracked sources are dirty). The
    artifacts output tree is excluded: it is a generated output whose tracked
    state is managed by the release commit, not by the run."""
    try:
        out = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", ".", ":(exclude)artifacts"],
            capture_output=True,
            text=True,
        )
        if out.returncode != 0:
            return True
        # untracked files outside artifacts/ make the tree dirty
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", ".", ":(exclude)artifacts"],
            capture_output=True,
            text=True,
        )
        return bool(untracked.stdout.strip())
    except Exception:  # noqa: BLE001
        return True
