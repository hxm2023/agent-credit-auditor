"""Protocol/evidence tests (§17.3): frozen configs, seed disjointness,
no-overwrite, config-hash lineage, failed runs in index."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from credit_auditor import runner
from credit_auditor.canonical import NoOverwriteError, sha256_json

PROTOCOLS = Path(__file__).resolve().parents[2] / "configs" / "protocols"
SEEDS = Path(__file__).resolve().parents[2] / "configs" / "seeds"


def _load_seeds(name: str) -> set[int]:
    return {r["seed"] for r in json.loads((SEEDS / name).read_text(encoding="utf-8"))["rows"]}


def test_all_frozen_protocols_validate():
    for f in ("m0_regression_v1", "v001_failure_v1", "d002_regression_v1"):
        proto = runner.validate_protocol(PROTOCOLS / f"{f}.json")
        assert proto.reconstruction_mode == "docs_only_semantic"
        assert proto.frozen_at == "2026-08-22"


def test_frozen_protocol_hashes_are_stable():
    """Frozen config change must trigger lineage mismatch — snapshot the hashes
    (first freeze 2026-08-22). Any change requires a decision-logged protocol
    version bump, never an in-place edit."""
    hashes = {f.name: sha256_json(runner.validate_protocol(f).model_dump(mode="json")) for f in sorted(PROTOCOLS.glob("*.json"))}
    assert hashes == {
        "d002_regression_v1.json": "7d138af0016c7e1a76ca83416fb0106f2106a1475001cbd18a03932af95a1d1e",
        "m0_regression_v1.json": "8bd2866c4a8dba8e8bfae8d4b551c8cc4bfdd82335cf58f26d11009b692e8af6",
        "v001_failure_v1.json": "37260cca02a399335943999b1cc703c208a0c18d07f727adc5c550d7700de26a",
    }


def test_seed_manifests_disjoint_d002():
    cal = _load_seeds("d002_calibration.json")
    test = _load_seeds("d002_test.json")
    assert len(cal) == 12
    assert len(test) == 48
    assert cal & test == set()


def test_seed_manifests_disjoint_v001():
    cal = _load_seeds("v001_calibration.json")
    test = _load_seeds("v001_problems.json")
    assert len(cal) == 6
    assert len(test) == 12
    assert cal & test == set()


def test_check_split_disjoint_detects_overlap(tmp_path: Path):
    cal = tmp_path / "cal.json"
    test = tmp_path / "test.json"
    rows = [{"problem_id": "x", "seed": 1}]
    cal.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    test.write_text(json.dumps({"rows": rows + [{"problem_id": "y", "seed": 2}]}), encoding="utf-8")
    with pytest.raises(Exception):
        runner.check_split_disjoint(cal, test)


def test_runner_refuses_existing_output(tmp_path: Path):
    out = tmp_path / "M0"
    out.mkdir()
    with pytest.raises(NoOverwriteError):
        runner.run(
            protocol_path=PROTOCOLS / "m0_regression_v1.json",
            output_dir=out,
            seed_manifests=[SEEDS / "m0_problems.json"],
        )


def test_runner_missing_driver_produces_evidence_package(tmp_path: Path):
    """Steps 8-10 failure must still yield a result package with exit status
    (not a bare traceback) — here the driver is missing entirely."""
    proto = json.loads((PROTOCOLS / "m0_regression_v1.json").read_text(encoding="utf-8"))
    proto["protocol_id"] = "no_driver_test"
    proto["world_family"] = "no_such_family"
    fake = tmp_path / "nofamily.json"
    fake.write_text(json.dumps(proto), encoding="utf-8")
    out = runner.run(
        protocol_path=fake,
        output_dir=tmp_path / "M0_nodriver",
        seed_manifests=[SEEDS / "m0_problems.json"],
    )
    assert (out / "result.json").is_file()
    assert (out / "run_manifest.json").is_file()
    assert (out / "SHA256SUMS").is_file()
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["exit_status"] == "driver_failed"
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["status"] in ("driver_failed", "invalid")


def test_runner_second_run_no_overwrite(tmp_path: Path):
    out = tmp_path / "M0"
    runner.run(
        protocol_path=PROTOCOLS / "m0_regression_v1.json",
        output_dir=out,
        seed_manifests=[SEEDS / "m0_problems.json"],
    )
    with pytest.raises(NoOverwriteError):
        runner.run(
            protocol_path=PROTOCOLS / "m0_regression_v1.json",
            output_dir=out,
            seed_manifests=[SEEDS / "m0_problems.json"],
        )


def test_published_package_hashes_match(tmp_path: Path):
    out = runner.run(
        protocol_path=PROTOCOLS / "m0_regression_v1.json",
        output_dir=tmp_path / "M0",
        seed_manifests=[SEEDS / "m0_problems.json"],
    )
    sums = {}
    for line in (out / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        h, _, rel = line.partition("  ")
        sums[rel] = h
    assert "result.json" in sums
    assert "run_manifest.json" in sums
    assert "REPORT.md" in sums
    from credit_auditor.canonical import sha256_file
    for rel, h in sums.items():
        assert sha256_file(out / rel) == h


def test_protocol_family_registration():
    with pytest.raises(Exception):
        runner.get_driver("no_such_family", "run")
