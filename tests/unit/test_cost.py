"""Cost calculator tests (§7.3 — hand-computed fixtures, no eval, Fraction math)."""

from __future__ import annotations

from fractions import Fraction

import pytest

from credit_auditor.schema import CostSpec


def test_dense_horizon_v1():
    for h in (1, 3, 6, 10):
        c = CostSpec(calculator_id="dense_horizon_v1", parameters={"horizon": h})
        bd = c.evaluate_cycle()
        assert bd.total == f"{h}/1"
        assert bd.total_fraction() == Fraction(h, 1)


def test_d002_branching_v1_hand_computed():
    # §8.3: c(h,d,K)=d+K(h-d)+(K-1)r, r=1. Historical examples:
    #   c(6,4,8)=4+8*2+7=27 ; c(3,1,8)=1+8*2+7=24 ; c(5,3,8)=3+8*2+7=26
    cases = [(6, 4, 8, 27), (3, 1, 8, 24), (5, 3, 8, 26), (6, 2, 4, 2 + 16 + 3)]
    for h, d, k, expected in cases:
        c = CostSpec(calculator_id="d002_branching_v1", parameters={"horizon": h, "depth": d, "width": k})
        assert c.evaluate_cycle().total_fraction() == Fraction(expected, 1)
        assert c.evaluate_cycle().total_fraction().denominator == 1


def test_d002_branching_terms_decompose():
    c = CostSpec(calculator_id="d002_branching_v1", parameters={"horizon": 6, "depth": 4, "width": 8})
    bd = c.evaluate_cycle()
    terms = {t.term_id: t for t in bd.terms}
    assert terms["prefix"].subtotal == "4/1"
    assert terms["suffixes"].quantity == "16/1"  # K*(H-d)=8*2
    assert terms["restores"].quantity == "7/1"  # K-1


def test_d002_branching_restore_overhead_fractional():
    c = CostSpec(
        calculator_id="d002_branching_v1",
        parameters={"horizon": 6, "depth": 4, "width": 8, "restore_overhead_per_extra_suffix": "1/2"},
    )
    bd = c.evaluate_cycle()
    assert bd.total_fraction() == Fraction(4, 1) + 16 + Fraction(7, 2)


def test_unknown_calculator_rejected():
    with pytest.raises(ValueError):
        CostSpec(calculator_id="eval_free_formula", parameters={"expr": "2*3"}).evaluate_cycle()


def test_invalid_domain_rejected():
    for kwargs in (
        {"horizon": 0, "depth": 0, "width": 1},
        {"horizon": 6, "depth": 6, "width": 1},
        {"horizon": 6, "depth": 4, "width": 0},
    ):
        c = CostSpec(calculator_id="d002_branching_v1", parameters=kwargs)
        with pytest.raises(ValueError):
            c.evaluate_cycle()


def test_non_integer_domain_rejected():
    c = CostSpec(calculator_id="d002_branching_v1", parameters={"horizon": "6", "depth": 4, "width": 8})
    with pytest.raises(ValueError):
        c.evaluate_cycle()


def test_floor_complete_cycles_and_unused_budget():
    c = CostSpec(calculator_id="d002_branching_v1", parameters={"horizon": 6, "depth": 4, "width": 8})
    n, unused = c.n_complete_cycles(4096)
    assert n == 4096 // 27
    assert unused == Fraction(4096 - (4096 // 27) * 27, 1)


def test_infeasible_budget_below_cycle_cost():
    c = CostSpec(calculator_id="dense_horizon_v1", parameters={"horizon": 6})
    n, unused = c.n_complete_cycles(5)
    assert n == 0
    assert unused == Fraction(5, 1)


def test_fractional_cycle_cost():
    c = CostSpec(
        calculator_id="d002_branching_v1",
        parameters={"horizon": 6, "depth": 4, "width": 8, "prefix_transition_cost": "1/2"},
    )
    # cost = 2 + 16 + 7 = 25/1 -> with prefix 1/2: 4*(1/2)=2 ... total 25
    assert c.evaluate_cycle().total_fraction() == Fraction(25, 1)


def test_budget_multiple_of_cost_exact():
    c = CostSpec(calculator_id="dense_horizon_v1", parameters={"horizon": 4})
    n, unused = c.n_complete_cycles(512)
    assert n == 128
    assert unused == 0
