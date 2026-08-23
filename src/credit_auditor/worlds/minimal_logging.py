"""Minimal telemetry universe (design §8.5, §13.5) — teaching asset only.

8 model rows with 3-bit observations; a logging schema is a subset of the
three bits. Point/sign labels over rows; an observation fiber is the set of
rows sharing the same logged-bit values. The minimal-cost schema that makes
every fiber label-consistent is a hitting set of the different-label conflict
pairs (equivalently a decision reduct / functional-dependency key).

NOVELTY STATUS: CLASSICAL DECISION-REDUCT / FD / HITTING-SET EQUIVALENCE
CLAIM STATUS: TEACHING OR TELEMETRY-SCHEMA DIAGNOSTIC ONLY  (§13.5)
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

ALL_BITS = (0, 1, 2)


@dataclass(frozen=True)
class TelemetryUniverse:
    rows: tuple[tuple[int, int, int], ...]  # 8 rows x 3 bits
    labels: tuple[str, ...]  # per-row point label

    @property
    def n_rows(self) -> int:
        return len(self.rows)


def default_universe() -> TelemetryUniverse:
    rows = tuple((b0, b1, b2) for b0 in (0, 1) for b1 in (0, 1) for b2 in (0, 1))
    labels = tuple("A" if (r[0] + r[1] + r[2]) % 2 == 0 else "B" for r in rows)
    return TelemetryUniverse(rows=rows, labels=labels)


def fiber(rows: tuple[tuple[int, int, int], ...], schema: tuple[int, ...]) -> dict:
    """Observation fiber: rows grouped by their logged-bit values."""
    out: dict = {}
    for i, r in enumerate(rows):
        key = tuple(r[b] for b in schema)
        out.setdefault(key, []).append(i)
    return out


def conflict_pairs(rows, labels) -> set[frozenset[int]]:
    """Different-label row pairs (the set a logging schema must hit)."""
    pairs: set[frozenset[int]] = set()
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if labels[i] != labels[j]:
                pairs.add(frozenset((i, j)))
    return pairs


def schema_separates(pair: frozenset[int], rows, schema: tuple[int, ...]) -> bool:
    i, j = tuple(pair)
    return any(rows[i][b] != rows[j][b] for b in schema)


def eligible_schemas(universe: TelemetryUniverse) -> list[tuple[int, ...]]:
    """Schemas whose every fiber is label-consistent (all conflict pairs hit)."""
    pairs = conflict_pairs(universe.rows, universe.labels)
    out = []
    for size in range(0, len(ALL_BITS) + 1):
        for schema in itertools.combinations(ALL_BITS, size):
            if all(schema_separates(p, universe.rows, schema) for p in pairs):
                out.append(schema)
    return out


def minimal_schemas(universe: TelemetryUniverse) -> list[tuple[int, ...]]:
    """Minimum-cardinality schemas (hitting-set / decision-reduct solutions)."""
    elig = eligible_schemas(universe)
    if not elig:
        return []
    min_size = min(len(s) for s in elig)
    return [s for s in elig if len(s) == min_size]


def sign_label_universe() -> TelemetryUniverse:
    """Sign-coarsened labels (only the sign of the label value matters)."""
    u = default_universe()
    labels = tuple("+" if label == "A" else "-" for label in u.labels)
    return TelemetryUniverse(rows=u.rows, labels=labels)
