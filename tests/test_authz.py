"""Pillar 2, the policy loads from JSON, enforces the limits, and the kill switch halts trading."""

from corral.action import ActionType, AgentAction
from corral.authz.gate import PreTradePolicyGate
from corral.authz.policy import TradingPolicy


def _order(instrument, side, qty):
    return AgentAction("bot", 0.0, ActionType.ORDER, instrument=instrument, side=side, quantity=qty)


def test_policy_json_round_trip():
    policy = TradingPolicy(allowed_instruments=frozenset({"AAPL", "MSFT"}), max_order_quantity=1000,
                           allow_short=False)
    assert TradingPolicy.from_dict(policy.to_dict()) == policy


def test_policy_from_dict_enforces():
    policy = TradingPolicy.from_dict({
        "allowed_instruments": ["AAPL"],
        "restricted_instruments": ["GME"],
        "max_order_quantity": 100,
    })
    gate = PreTradePolicyGate(policy)
    assert gate.authorize(_order("AAPL", "buy", 50)).allowed
    assert not gate.authorize(_order("AAPL", "buy", 500)).allowed
    assert not gate.authorize(_order("GME", "buy", 10)).allowed


def test_kill_switch_halts_and_resets():
    gate = PreTradePolicyGate(TradingPolicy(allowed_instruments=frozenset({"AAPL", "MSFT"})))
    good = _order("AAPL", "buy", 10)
    assert gate.authorize(good).allowed

    gate.kill("AAPL")
    assert not gate.authorize(good).allowed                    # AAPL halted
    assert gate.authorize(_order("MSFT", "buy", 10)).allowed    # MSFT still fine

    gate.reset("AAPL")
    assert gate.authorize(good).allowed

    gate.kill()  # halt everything
    assert not gate.authorize(_order("MSFT", "buy", 10)).allowed
    gate.reset()
    assert gate.authorize(_order("MSFT", "buy", 10)).allowed
