"""stage3_report prediction-verdict logic tests: the VOID (all-equal) and
INCONCLUSIVE (abstention-confounded) verdicts must fire on synthetic runs."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from stage3_report import _prediction_1, _prediction_2, _prediction_3  # noqa: E402


def _run(estimator: str, final_success: float, grad_l2: list[float], kl: float | None) -> dict:
    return {
        "task": "cts_order",
        "estimator": estimator,
        "metrics": {
            "final_eval": {"success_rate": final_success},
            "epoch_metrics": [{"grad_l2": g} for g in grad_l2],
            "kl_drift_vs_base": kl,
        },
    }


def test_p1_void_when_all_equal():
    runs = {
        "d": _run("dense", 0.0, [1.0], 0.001),
        "l": _run("local", 0.0, [1.0], 0.001),
        "p": _run("paired", 0.0, [1.0], 0.001),
    }
    verdict, _ = _prediction_1(runs)
    assert verdict == "VOID"


def test_p1_confirmed_when_paired_leads():
    runs = {
        "d": _run("dense", 0.4, [1.0], 0.001),
        "l": _run("local", 0.2, [1.0], 0.001),
        "p": _run("paired", 0.6, [1.0], 0.001),
    }
    verdict, evidence = _prediction_1(runs)
    assert verdict == "CONFIRMED"
    assert evidence.startswith("cts_order: paired")


def test_p2_void_when_all_zero():
    runs = {
        "d": _run("dense", 0.0, [0.0, 0.0], 0.0),
        "l": _run("local", 0.0, [0.0, 0.0], 0.0),
        "p": _run("paired", 0.0, [0.0, 0.0], 0.0),
    }
    verdict, evidence = _prediction_2(runs)
    assert verdict == "VOID"
    assert "no signal" in evidence


def test_p2_confirmed_paired_abstains():
    runs = {
        "d": _run("dense", 0.0, [5.0, 5.0], 0.001),
        "l": _run("local", 0.0, [5.0, 5.0], 0.001),
        "p": _run("paired", 0.0, [0.0, 0.0], 0.0),
    }
    verdict, evidence = _prediction_2(runs)
    assert verdict == "CONFIRMED"
    assert "paired abstains" in evidence


def test_p3_inconclusive_when_paired_zero():
    runs = {
        "d": _run("dense", 0.0, [1.0], 0.0010),
        "l": _run("local", 0.0, [1.0], 0.0005),
        "p": _run("paired", 0.0, [0.0], 0.0),
    }
    verdict, evidence = _prediction_3(runs)
    assert verdict == "INCONCLUSIVE"
    assert "abstention" in evidence
