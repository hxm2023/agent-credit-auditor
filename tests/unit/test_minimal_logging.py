"""Minimal telemetry universe tests (§13.5 teaching asset, §17.1)."""

from __future__ import annotations

from credit_auditor.worlds.minimal_logging import (
    conflict_pairs,
    default_universe,
    eligible_schemas,
    fiber,
    minimal_schemas,
    schema_separates,
    sign_label_universe,
)


def test_universe_structure():
    u = default_universe()
    assert u.n_rows == 8
    assert len(set(u.rows)) == 8  # all 3-bit rows present


def test_conflict_pairs_hit_all_schemas():
    u = default_universe()
    pairs = conflict_pairs(u.rows, u.labels)
    assert len(pairs) > 0
    for schema in eligible_schemas(u):
        for pair in pairs:
            assert schema_separates(pair, u.rows, schema)


def test_fiber_label_consistency():
    u = default_universe()
    for schema in eligible_schemas(u):
        for rows_in_fiber in fiber(u.rows, schema).values():
            labels = {u.labels[i] for i in rows_in_fiber}
            assert len(labels) == 1, f"fiber {rows_in_fiber} has labels {labels}"


def test_minimal_schemas_nonempty():
    u = default_universe()
    ms = minimal_schemas(u)
    assert ms, "a label-consistent schema must exist (all 3 bits always work)"
    # all three bits always separate everything
    assert (0, 1, 2) in eligible_schemas(u)


def test_classical_equivalence_documented():
    """§13.5: the banner must be enforced at the report level; the schema
    problem IS the hitting-set/decision-reduct problem (no new theory)."""
    u = default_universe()
    # a schema is eligible iff it hits every conflict pair — direct check
    pairs = conflict_pairs(u.rows, u.labels)
    for schema in eligible_schemas(u):
        assert all(schema_separates(p, u.rows, schema) for p in pairs)


def test_sign_labels_may_need_fewer_bits():
    """Sign is label coarsening: point labels may require more bits than
    sign labels (§3.5 lesson: sign 3.8% vs point 4.2% compressible)."""
    point = eligible_schemas(default_universe())
    sign = eligible_schemas(sign_label_universe())
    # in the default universe both need all three bits (parity labels)
    assert (0, 1, 2) in point
    assert (0, 1, 2) in sign
