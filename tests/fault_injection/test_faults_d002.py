"""Faults A7-A9, A11 (§14) on the D002 machinery."""
from __future__ import annotations

from credit_auditor.audit.environment import environment_gate
from credit_auditor.audit.mechanism import width_diversity_gate
from credit_auditor.schema import ReasonCode
from credit_auditor.worlds.d002_shared_logits import generate_problem_focal


def test_A7_split_overlap_detected(tmp_path):
    """A7: calibration/test seed overlap -> split invalid (runner refuses)."""
    from credit_auditor import runner
    rows = [{"problem_id": "x", "seed": 1}]
    cal = tmp_path / "d002_calibration.json"
    test = tmp_path / "d002_test.json"
    cal.write_text(__import__("json").dumps({"rows": rows}), encoding="utf-8")
    test.write_text(__import__("json").dumps({"rows": rows + [{"problem_id": "y", "seed": 2}]}), encoding="utf-8")
    try:
        runner.check_split_disjoint(cal, test)
        assert False, "overlap must raise"
    except Exception:
        pass


def test_A8_test_time_reselection_refused(tmp_path):
    """A8: test phase without the frozen selection -> refused."""
    from credit_auditor import runner
    from credit_auditor.experiments import d002 as d002_exp
    d002_exp.register()
    try:
        runner.run(
            protocol_path=tmp_path.parents[2] / "configs/protocols/d002_regression_v1.json",
            output_dir=tmp_path / "D002_t",
            phase="test",
            seed_manifests=[tmp_path.parents[2] / "configs/seeds/d002_test.json"],
        )
        assert False, "must refuse unfrozen test"
    except Exception:
        pass


def test_A9_width_collapse_detected():
    """A9: variable widths forced equal but labeled adaptive -> MECH001."""
    gate = width_diversity_gate([8, 8, 8, 8])
    assert gate.status == "fail"
    assert ReasonCode.MECH001_ADAPTIVE_MAPPING_COLLAPSED_TO_GLOBAL in gate.reason_codes


def test_A9b_global_control_equality_detected():
    gate = width_diversity_gate([2, 2, 2, 2], global_control_width=2)
    assert gate.status == "fail"
    assert ReasonCode.MECH001_ADAPTIVE_MAPPING_COLLAPSED_TO_GLOBAL in gate.reason_codes


def test_A9c_diverse_widths_pass():
    gate = width_diversity_gate([2, 4, 2, 8])
    assert gate.status == "pass"


def test_A11_noop_alternative_detected():
    """A11: an alternative that never changes reward is a no-op."""
    from credit_auditor.worlds.bernoulli_sequence import BernoulliSequenceMDP
    world = BernoulliSequenceMDP(
        probabilities=(0.5, 0.5),
        rewards={(0, 0): 1.0, (0, 1): 1.0, (1, 0): 2.0, (1, 1): 2.0},
    )
    gate = environment_gate(world)
    assert gate["status"] == "fail"
    assert any("E001_ALTERNATIVE_NOOP" in rc for rc in gate["reason_codes"])


def test_focal_world_oracle_alignment():
    from credit_auditor import runner
    from credit_auditor.audit.target import compare_oracle
    from credit_auditor.worlds.d002_shared_logits import true_gradient
    import numpy as np
    from pathlib import Path
    p = generate_problem_focal("unit_test_problem", 424242)
    enum = runner.run_oracle_subprocess(
        Path(__file__).resolve().parents[2] / "src/credit_auditor/oracles/enumeration_oracle.py",
        p.to_spec(),
    )
    ok, mm = compare_oracle(true_gradient(p), np.asarray(enum["gradient"]), 1e-9, 1e-12)
    assert ok, mm
    bell = runner.run_oracle_subprocess(
        Path(__file__).resolve().parents[2] / "src/credit_auditor/oracles/bellman_oracle.py",
        p.to_spec(),
    )
    ok2, mm2 = compare_oracle(true_gradient(p), np.asarray(bell["gradient"]), 1e-9, 1e-12)
    assert ok2, mm2
