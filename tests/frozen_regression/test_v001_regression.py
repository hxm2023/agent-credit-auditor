"""V001 frozen regression: expected-fail must stay a stable FAIL while the
calibration-accuracy claim stays PASS."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.full

from credit_auditor import runner
from credit_auditor.experiments import v001 as v001_exp

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def v001_run(tmp_path_factory):
    v001_exp.register()
    out = runner.run(
        protocol_path=ROOT / "configs/protocols/v001_failure_v1.json",
        output_dir=tmp_path_factory.mktemp("v001") / "V001",
        seed_manifests=[ROOT / "configs/seeds/v001_problems.json", ROOT / "configs/seeds/v001_calibration.json"],
    )
    return out


def test_v001_utility_failure_is_stable(v001_run):
    gd = json.loads((v001_run / "gate_decision.json").read_text(encoding="utf-8"))
    statuses = {c["claim_id"]: c["status"] for c in gd["claims"]}
    assert statuses["v001_utility_failure_reproduced"] == "fail"
    assert statuses["v001_calibration_accurate"] == "pass"
    assert gd["experiment_integrity"] == "pass"


def test_v001_pc_rsg_loses_on_every_problem(v001_run):
    res = json.loads((v001_run / "result.json").read_text(encoding="utf-8"))
    for row in res["problems"]:
        assert row["ratio_vs_dense"] > 1.0, row["problem_id"]
        assert row["ratio_vs_hh"] > 1.0, row["problem_id"]


def test_v001_calibration_accuracy(v001_run):
    res = json.loads((v001_run / "result.json").read_text(encoding="utf-8"))
    max_err = max(c["max_expectation_err"] for c in res["calibration_accuracy"])
    assert max_err < 1e-9


def test_v001_oracle_ok(v001_run):
    res = json.loads((v001_run / "result.json").read_text(encoding="utf-8"))
    for row in res["problems"]:
        assert row["oracle_max_mismatch"] < 1e-12
