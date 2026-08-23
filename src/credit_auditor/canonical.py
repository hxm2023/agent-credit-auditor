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
    """Write canonical JSON atomically (temp file + rename), LF line endings
    (see atomic_write_text)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(canonical_json(obj) + "\n")
    tmp.replace(path)


def atomic_write_text(text: str, path: Path) -> None:
    """Write text with explicit LF line endings (newline='\n') so the bytes
    are platform-independent and the SHA256SUMS hashes reproduce on any
    checkout (the v0.1.5 cross-platform checksum failure root cause)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    tmp.replace(path)


def write_jsonl_zst(rows: list[Any], path: Path) -> None:
    """§18 package format: raw rows as zstd-compressed JSONL (one object per
    line), written atomically."""
    import zstandard as zstd

    payload = "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows) + "\n"
    compressed = zstd.ZstdCompressor().compress(payload.encode("utf-8"))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(compressed)
    tmp.replace(path)


def read_jsonl_zst(path: Path) -> list[Any]:
    import zstandard as zstd

    data = zstd.ZstdDecompressor().decompress(path.read_bytes())
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
