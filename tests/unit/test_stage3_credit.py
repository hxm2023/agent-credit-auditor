"""Stage 3 pure-function tests (no GPU / no server deps): the credit scheme
math (decision positions, advantage formulas, paired gate handling) and the
task parsing helpers must be covered locally — the GPU training loop itself
runs on the servers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

STAGE3 = Path(__file__).resolve().parents[2] / "stage3"
sys.path.insert(0, str(STAGE3))

from credit import decision_positions, dense_credit, local_credit, paired_credit  # noqa: E402
from tasks import parse_tool_calls  # noqa: E402


class _Tok:
    """Toy tokenizer whose decode is the concatenation of the token texts."""

    def __init__(self, tokens: list[str]):
        self._tokens = tokens

    def convert_ids_to_tokens(self, ids):
        return self._tokens

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self._tokens)


@pytest.fixture()
def fake_gate(monkeypatch):
    """Inject a controllable fake for agent_ttrl's paired_credit gate so the
    wrapper's contract is testable without the server package."""

    class FakeModule:
        def paired_credit(self, U):
            U = np.asarray(U, dtype=float)
            if U.std() < 1e-9:
                from types import SimpleNamespace

                return SimpleNamespace(status="DEGENERATE_GROUP", reason_code="ALL_SAME_OUTCOME", rows=[])
            # 2x2 with a clear winner: action 0 good, action 1 bad
            from types import SimpleNamespace

            rows = [
                SimpleNamespace(credit=1.0, raw_credit=0.8, gate_passed=True),
                SimpleNamespace(credit=-1.0, raw_credit=-0.8, gate_passed=True),
            ]
            return SimpleNamespace(status="OK", reason_code="", rows=rows)

    monkeypatch.setitem(sys.modules, "agent_ttrl", type(sys)("agent_ttrl"))
    monkeypatch.setitem(sys.modules, "agent_ttrl.credit", type(sys)("agent_ttrl.credit"))
    monkeypatch.setitem(sys.modules, "agent_ttrl.credit.paired_credit", FakeModule())
    yield


def test_parse_tool_calls_json_list():
    text = '{"tool": "charge", "call": {"amount_cents": 100}}'
    calls = parse_tool_calls(f"[{text}]")
    assert len(calls) == 1 and calls[0]["tool"] == "charge"


def test_parse_tool_calls_invalid_returns_empty():
    assert parse_tool_calls("no json here") == []
    assert parse_tool_calls("[not json]") == []


def test_dense_credit_group_centering():
    u = np.array([[0.0, 0.5, 1.0, 0.5], [1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    adv = dense_credit(u)
    # group 2 is degenerate (all 1.0) -> advantage 0
    assert adv[4:].sum() == pytest.approx(0.0)
    # group 1 is centered with population std = sqrt(0.125)
    assert adv[:4].sum() == pytest.approx(0.0)
    assert adv[0] == pytest.approx(-0.5 / (np.sqrt(0.125) + 1e-3), abs=1e-4)


def test_local_credit_same_as_dense():
    u = np.array([[0.0, 1.0], [0.2, 0.8]], dtype=np.float32)
    assert np.allclose(local_credit(u), dense_credit(u))


def test_paired_credit_closed_gate_zero(fake_gate):
    # degenerate U (all rows identical) -> gate closes -> zero credit
    U = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32)
    credits, info = paired_credit(U)
    assert info["status"] == "DEGENERATE_GROUP"
    assert np.all(credits == 0.0)


def test_paired_credit_opens_on_signal(fake_gate):
    # one action clearly better in both branches -> gate opens
    U = np.array([[0.9, 0.9], [0.1, 0.1]], dtype=np.float32)
    credits, info = paired_credit(U)
    assert info["status"] == "OK"
    assert credits[0] > 0 and credits[1] < 0


def test_decision_positions_finds_tool_names():
    tokens = ['{"', "tool", '": "', "charge", '"}']
    tok = _Tok(tokens)
    pos = decision_positions("", tok, [0, 1, 2, 3, 4])
    assert isinstance(pos, list) and len(pos) >= 1


def test_decision_positions_empty_completion():
    tok = _Tok([])
    assert decision_positions("", tok, []) == []
