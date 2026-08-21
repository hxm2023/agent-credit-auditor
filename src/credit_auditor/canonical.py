"""Canonical serialization and content hashing.

Every published number must be traceable to artifact + commit + checksum
(design §23, CLAUDE.md honesty rules). This module defines ONE canonical JSON
form: keys sorted, compact separators, tuples/fractions as explicit strings,
no float bit-exactness claims across platforms (floats are written with
repr().lower() so the hash is stable within a platform; near-zero markers are
handled by the numerical-margin gate, not by hashing tricks).
"""
from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any


def canonical_float(x: float) -> Any:
    """Serialize a float deterministically. NaN/Inf are rejected for hashing."""
    if x != x or x in (float("inf"), float("-inf")):
        raise ValueError(f"non-finite float cannot be hashed: {x!r}")
    # 17 significant digits round-trip float64 exactly.
    return json.loads(f"{x:.17g}") if abs(x) < 1e300 else str(x)


def canonicalize(obj: Any) -> Any:
    """Normalize an object graph into hashable, order-independent JSON values."""
    if isinstance(obj, dict):
        return {str(k): canonicalize(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [canonicalize(v) for v in obj]
    if isinstance(obj, Fraction):
        return {"__fraction__": f"{obj.numerator}/{obj.denominator}"}
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float)):
        return canonical_float(float(obj)) if isinstance(obj, float) else int(obj)
    if isinstance(obj, (str, type(None))):
        return obj
    if hasattr(obj, "model_dump"):
        return canonicalize(obj.model_dump(mode="json"))
    if hasattr(obj, "__dataclass_fields__"):
        return canonicalize(vars(obj))
    raise TypeError(f"cannot canonicalize {type(obj)!r}")


def canonical_json(obj: Any) -> str:
    return json.dumps(canonicalize(obj), separators=(",", ":"), ensure_ascii=False)


def sha256_hex(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_hex(canonical_json(obj))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_tree(root: Path) -> dict[str, str]:
    """Content hashes of every file under root, relative path -> sha256."""
    out: dict[str, str] = {}
    if not root.exists():
        return out
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[p.relative_to(root).as_posix()] = sha256_file(p)
    return out


class NoOverwriteError(FileExistsError):
    """Raised when a canonical output path already exists (§16 step 4, A13)."""


def refuse_existing(path: Path) -> None:
    if path.exists():
        raise NoOverwriteError(f"canonical output already exists: {path}")


def atomic_write_json(obj: Any, path: Path) -> None:
    """Write canonical JSON atomically (temp file + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(canonical_json(obj) + "\n", encoding="utf-8")
    tmp.replace(path)


def atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
