"""Canonical serialization + hashing tests."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest

from credit_auditor.canonical import (
    NoOverwriteError,
    atomic_write_json,
    canonical_json,
    canonicalize,
    refuse_existing,
    sha256_file,
    sha256_json,
    sha256_tree,
)


def test_canonical_order_independent():
    a = canonical_json({"b": 1, "a": [2, {"f": Fraction(1, 2)}]})
    b = canonical_json({"a": [2, {"f": Fraction(1, 2)}], "b": 1})
    assert a == b
    assert "1/2" in a


def test_canonical_fraction_vs_string_distinct():
    # Fraction is explicitly typed as {"__fraction__": ...}; a plain "1/2"
    # string must NOT hash equal (type discipline, §7.3).
    assert canonical_json({"f": Fraction(1, 2)}) != canonical_json({"f": "1/2"})


def test_canonical_nested_and_types():
    obj = {"x": (1, 2), "frac": Fraction(3, 7), "none": None, "flag": True}
    s = canonical_json(obj)
    assert json.loads(s) == canonicalize({"x": [1, 2], "frac": {"__fraction__": "3/7"}, "none": None, "flag": True})


def test_canonical_float_stability():
    s1 = canonical_json({"v": 0.1})
    s2 = canonical_json({"v": 0.1})
    assert s1 == s2
    assert sha256_json({"v": 0.1}) == sha256_json({"v": 0.1})


def test_canonical_rejects_nan():
    with pytest.raises(ValueError):
        canonical_json({"v": float("nan")})
    with pytest.raises(ValueError):
        canonical_json({"v": float("inf")})


def test_sha256_json_deterministic():
    assert sha256_json({"a": 1}) == sha256_json({"a": 1})
    assert sha256_json({"a": 1}) != sha256_json({"a": 2})


def test_refuse_existing_raises(tmp_path: Path):
    p = tmp_path / "out"
    p.mkdir()
    with pytest.raises(NoOverwriteError):
        refuse_existing(p)
    refuse_existing(tmp_path / "newdir")  # non-existing is fine


def test_atomic_write_and_read(tmp_path: Path):
    p = tmp_path / "r.json"
    atomic_write_json({"k": [1, 2, {"f": "1/3"}]}, p)
    assert json.loads(p.read_text(encoding="utf-8")) == {"k": [1, 2, {"f": "1/3"}]}
    assert not p.with_name("r.json.tmp").exists()


def test_sha256_tree_and_file(tmp_path: Path):
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"\x00\x01")
    tree = sha256_tree(tmp_path)
    assert set(tree) == {"a.txt", "sub/b.bin"}
    assert sha256_file(tmp_path / "a.txt") == tree["a.txt"]
