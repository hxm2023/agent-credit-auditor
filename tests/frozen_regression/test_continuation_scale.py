"""CTRI large-scale census frozen regression: stable rates across the two
frozen scales, support_only verdicts, banners."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from credit_auditor import runner
from credit_auditor.experiments import continuation_scale as cs

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.full


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    cs.register()
    return runner.run(
        protocol_path=ROOT / "configs/protocols/continuation_scale_v1.json",
        output_dir=tmp_path_factory.mktemp("cscale_small") / "CSCALE",
    )


@pytest.fixture(scope="module")
def large_run(tmp_path_factory):
    cs.register()
    return runner.run(
        protocol_path=ROOT / "configs/protocols/continuation_scale_large_v1.json",
        output_dir=tmp_path_factory.mktemp("cscale_large") / "CSCALE_LARGE",
    )


def test_small_scale_verdict(small_run):
    gd = json.loads((small_run / "gate_decision.json").read_text(encoding="utf-8"))
    assert gd["claims"][0]["status"] == "support_only"
    assert gd["experiment_integrity"] == "pass"


def test_small_scale_counts(small_run):
    res = json.loads((small_run / "result.json").read_text(encoding="utf-8"))
    assert res["families"] == 5000
    assert all(0 <= v <= 1 for v in res["rates"].values())
    # sign reversal == rank reversal (same event: Q(1)-Q(0) sign flip)
    assert res["counts"]["sign_reversal"] == res["counts"]["rank_reversal"]


def test_large_scale_rates_close_to_small(large_run, small_run):
    res_l = json.loads((large_run / "result.json").read_text(encoding="utf-8"))
    res_s = json.loads((small_run / "result.json").read_text(encoding="utf-8"))
    assert res_l["families"] == 100000
    # the rate statistics must agree within sampling error (~1.5pp at N=5000)
    for k in ("sign_reversal", "rank_reversal"):
        assert abs(res_l["rates"][k] - res_s["rates"][k]) < 0.02, (k, res_l["rates"][k], res_s["rates"][k])


def test_large_scale_banner(large_run):
    text = (large_run / "REPORT.md").read_text(encoding="utf-8")
    assert "SUPPORT_ONLY" in text
    assert "not a new theory" in text
    assert "incident background" in text


def test_census_family_derivation_deterministic():
    fam1 = cs.generate_family(42)
    fam2 = cs.generate_family(42)
    assert fam1 == fam2
    assert cs.census_family(fam1) == cs.census_family(fam2)
