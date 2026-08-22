"""Oracle independence tests (§17.2): different algorithms, independent
process, import-graph isolation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from credit_auditor import runner
from credit_auditor.oracles.isolation import check_import_isolation, check_process_import_isolation
from credit_auditor.worlds.bernoulli_sequence import deterministic_world

ORACLE_DIR = Path(__file__).resolve().parents[2] / "src" / "credit_auditor" / "oracles"


def _oracle_result(script: str, world) -> dict:
    return runner.run_oracle_subprocess(ORACLE_DIR / script, world.to_spec())


def test_enumeration_oracle_matches_primary():
    world = deterministic_world(seed=11, horizon=5)
    out = _oracle_result("enumeration_oracle.py", world)
    assert out["oracle"] == "enumeration"
    np.testing.assert_allclose(out["gradient"], world.true_gradient(), rtol=1e-12, atol=1e-15)
    assert out["n_paths"] == 32


def test_bellman_oracle_matches_primary():
    world = deterministic_world(seed=12, horizon=6)
    out = _oracle_result("bellman_oracle.py", world)
    assert out["oracle"] == "bellman"
    np.testing.assert_allclose(out["gradient"], world.true_gradient(), rtol=1e-12, atol=1e-15)


def test_two_oracles_agree():
    world = deterministic_world(seed=13, horizon=5)
    a = _oracle_result("enumeration_oracle.py", world)
    b = _oracle_result("bellman_oracle.py", world)
    np.testing.assert_allclose(a["gradient"], b["gradient"], rtol=1e-12, atol=1e-15)
    assert a["input_sha256"] == b["input_sha256"]


def test_oracle_input_hash_changes_with_world():
    w1 = deterministic_world(seed=14, horizon=4)
    w2 = deterministic_world(seed=15, horizon=4)
    h1 = _oracle_result("bellman_oracle.py", w1)["input_sha256"]
    h2 = _oracle_result("bellman_oracle.py", w2)["input_sha256"]
    assert h1 != h2


def test_ast_import_isolation_both_oracles():
    import ast
    import sys
    from types import ModuleType

    stdlib = {m for m in sys.stdlib_module_names}
    for script in ("enumeration_oracle.py", "bellman_oracle.py"):
        bad = check_import_isolation(ORACLE_DIR / script)
        assert bad == [], f"{script} imports {bad}"
        tree_imports = []
        tree = ast.parse((ORACLE_DIR / script).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                tree_imports.append(node.module)
            elif isinstance(node, ast.Import):
                tree_imports.extend(alias.name for alias in node.names)
        # stdlib-only: no credit_auditor, no third-party (fractions/json/hashlib
        # are stdlib; oracle scripts stay self-contained)
        non_stdlib = [m for m in tree_imports if m and m.split(".")[0] not in stdlib]
        assert non_stdlib == [], (script, non_stdlib)


def test_process_import_isolation_both_oracles():
    for script in ("enumeration_oracle.py", "bellman_oracle.py"):
        res = check_process_import_isolation(ORACLE_DIR / script)
        assert res["isolated"] is True, res


def test_monkeypatching_primary_does_not_change_oracle():
    """§17.2: patching the primary path enumeration must not affect the oracle."""
    world = deterministic_world(seed=16, horizon=4)
    out = _oracle_result("enumeration_oracle.py", world)
    baseline = list(out["gradient"])
    # sabotage the primary-side helper
    import credit_auditor.worlds.bernoulli_sequence as bs
    orig = bs.BernoulliSequenceMDP.true_gradient
    bs.BernoulliSequenceMDP.true_gradient = lambda self: np.zeros(self.horizon)
    try:
        out2 = _oracle_result("enumeration_oracle.py", world)
    finally:
        bs.BernoulliSequenceMDP.true_gradient = orig
    assert baseline == out2["gradient"]


def test_oracle_rejects_bad_spec():
    import pytest
    with pytest.raises(Exception):
        runner.run_oracle_subprocess(
            ORACLE_DIR / "enumeration_oracle.py",
            {"world": "bernoulli_sequence_mdp", "probabilities": [0.5], "rewards": {"0": "nan"}},
        )


def test_oracle_output_json_roundtrip():
    world = deterministic_world(seed=17, horizon=3)
    out = _oracle_result("bellman_oracle.py", world)
    assert json.loads(json.dumps(out)) == out
    assert set(out) == {"gradient", "oracle", "input_sha256", "precision", "n_states"}
