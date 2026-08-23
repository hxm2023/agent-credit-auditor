"""Fraction-exact world + oracle alignment tests (design §10.3)."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np

from credit_auditor import runner
from credit_auditor.worlds.bernoulli_sequence import BernoulliSequenceMDP, deterministic_world
from credit_auditor.worlds.fraction_world import BernoulliFractionMDP

ORACLE_DIR = Path(__file__).resolve().parents[2] / "src" / "credit_auditor" / "oracles"


def _frac_world() -> BernoulliFractionMDP:
    return BernoulliFractionMDP(
        probabilities=(Fraction(1, 2), Fraction(1, 3), Fraction(3, 4)),
        rewards={
            (0, 0, 0): Fraction(1, 3),
            (0, 0, 1): Fraction(2, 5),
            (0, 1, 0): Fraction(1, 7),
            (0, 1, 1): Fraction(3, 8),
            (1, 0, 0): Fraction(5, 6),
            (1, 0, 1): Fraction(1, 9),
            (1, 1, 0): Fraction(2, 3),
            (1, 1, 1): Fraction(4, 7),
        },
    )


def test_fraction_path_probs_sum_to_one_exactly():
    w = _frac_world()
    total = sum((p for _, p in w.all_paths()), Fraction(0))
    assert total == 1


def test_fraction_gradient_matches_float():
    w = _frac_world()
    g_f = [float(x) for x in w.true_gradient()]
    # build the float equivalent world
    probs = [float(p) for p in w.probabilities]
    rewards = {a: float(r) for a, r in w.rewards.items()}
    w2 = BernoulliSequenceMDP(tuple(probs), rewards)
    np.testing.assert_allclose(g_f, w2.true_gradient(), rtol=1e-15, atol=1e-15)


def test_fraction_spec_roundtrip():
    w = _frac_world()
    spec = w.to_spec()
    assert spec["world"] == "bernoulli_fraction_mdp"
    assert spec["probabilities"][0] == "1/2"
    # The float->Fraction conversion is deterministic and consistent with the
    # float world (the exact ARITHMETIC self-consistency is tested separately
    # by test_fraction_oracles_align_exactly*).
    fw_float = BernoulliSequenceMDP(
        tuple(float(p) for p in w.probabilities), {a: float(r) for a, r in w.rewards.items()}
    )
    w2 = BernoulliFractionMDP.from_float_world(fw_float)
    g2 = [float(x) for x in w2.true_gradient()]
    np.testing.assert_allclose(g2, fw_float.true_gradient(), rtol=1e-12, atol=1e-12)
    # deterministic: the same input produces the same exact world
    assert BernoulliFractionMDP.from_float_world(fw_float).true_gradient() == w2.true_gradient()


def test_fraction_oracles_align_exactly():
    """The exact cross-validation: primary vs both oracles with mismatch == 0."""
    from fractions import Fraction as F

    w = _frac_world()
    target = w.true_gradient()
    enum = runner.run_oracle_subprocess(ORACLE_DIR / "enumeration_oracle.py", w.to_spec())
    bell = runner.run_oracle_subprocess(ORACLE_DIR / "bellman_oracle.py", w.to_spec())
    assert enum["precision"] == "exact_fraction"
    assert bell["precision"] == "exact_fraction"
    for oracle_out in (enum, bell):
        got = tuple(F(x) for x in oracle_out["gradient"])
        assert got == target, (oracle_out["oracle"], got, target)


def test_fraction_oracles_align_exactly_on_seeded_worlds():
    """The seeded deterministic worlds are rational; exact alignment must hold."""
    from fractions import Fraction as F

    for seed, horizon in ((1, 4), (2, 5), (3, 3)):
        world = deterministic_world(seed, horizon)
        fw = BernoulliFractionMDP.from_float_world(world)
        target = fw.true_gradient()
        enum = runner.run_oracle_subprocess(ORACLE_DIR / "enumeration_oracle.py", fw.to_spec())
        bell = runner.run_oracle_subprocess(ORACLE_DIR / "bellman_oracle.py", fw.to_spec())
        for oracle_out in (enum, bell):
            got = tuple(F(x) for x in oracle_out["gradient"])
            assert got == target


def test_fraction_world_rejects_bad_probs():
    import pytest

    with pytest.raises(ValueError):
        BernoulliFractionMDP(
            probabilities=(Fraction(1, 2), Fraction(3, 2)),
            rewards={(0, 0): Fraction(1), (0, 1): Fraction(1), (1, 0): Fraction(1), (1, 1): Fraction(1)},
        )
