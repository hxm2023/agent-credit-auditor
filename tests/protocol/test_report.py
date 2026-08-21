"""Release report builder tests (§18): result index, environment, TEST_LOG,
failed runs enter the index."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from credit_auditor import runner, report
from credit_auditor.experiments import m0 as m0_exp

pytestmark = pytest.mark.release_report

ROOT = Path(__file__).resolve().parents[2]


def test_release_report_builds(tmp_path):
    m0_exp.register()
    exp = runner.run(
        protocol_path=ROOT / "configs/protocols/m0_regression_v1.json",
        output_dir=tmp_path / "artifacts" / "M0",
        seed_manifests=[ROOT / "configs/seeds/m0_problems.json"],
    )
    root = tmp_path / "artifacts"
    out = report.build_release_report(root, run_tests=False)
    assert (out / "result_index.json").is_file()
    assert (out / "environment.json").is_file()
    assert (out / "TEST_LOG.txt").is_file()
    assert (out / "REPORT.md").is_file()
    assert (out / "SHA256SUMS").is_file()
    index = json.loads((out / "result_index.json").read_text(encoding="utf-8"))
    assert len(index["experiments"]) == 1
    entry = index["experiments"][0]
    assert entry["integrity"] == "pass"
    assert entry["exit_status"] == "ok"
    assert entry["protocol_id"] == "m0_regression_v1"


def test_release_report_failed_run_enters_index(tmp_path):
    """§17.3: failed runs still enter the index."""
    proto = json.loads((ROOT / "configs/protocols/m0_regression_v1.json").read_text(encoding="utf-8"))
    proto["protocol_id"] = "no_driver_2"
    proto["world_family"] = "no_such_family"
    fake = tmp_path / "nofamily.json"
    fake.write_text(json.dumps(proto), encoding="utf-8")
    runner.run(protocol_path=fake, output_dir=tmp_path / "artifacts" / "M0_fail", seed_manifests=[ROOT / "configs/seeds/m0_problems.json"])
    out = report.build_release_report(tmp_path / "artifacts", run_tests=False)
    index = json.loads((out / "result_index.json").read_text(encoding="utf-8"))
    entry = next(e for e in index["experiments"] if e["experiment"] == "M0_fail")
    assert entry["exit_status"] == "driver_failed"


def test_release_report_claim_ceilings_present(tmp_path):
    m0_exp.register()
    runner.run(
        protocol_path=ROOT / "configs/protocols/m0_regression_v1.json",
        output_dir=tmp_path / "artifacts" / "M0",
        seed_manifests=[ROOT / "configs/seeds/m0_problems.json"],
    )
    out = report.build_release_report(tmp_path / "artifacts", run_tests=False)
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "FORBIDDEN" in text
    assert "docs_only_semantic" in text
    # historical numbers appear ONLY inside the forbidden-claims list, never as results
    for num in ("144/202", "24.81x", "0.694", "192/192"):
        idx = text.find(num)
        assert idx != -1, num  # present as forbidden context
        assert "FORBIDDEN" in text[:idx]  # only mentioned inside the forbidden section
