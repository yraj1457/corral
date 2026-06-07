"""Load a trading policy from a JSON file, the way a risk team keeps it, check some orders against
it, then hit the kill switch. Run from the repo root:

    PYTHONPATH=src python examples/policy_gate.py
"""

from pathlib import Path

from corral.action import ActionType, AgentAction
from corral.authz.gate import PreTradePolicyGate
from corral.authz.policy import load_policy


def order(bot, instrument, side, qty):
    return AgentAction(bot, 0.0, ActionType.ORDER, instrument=instrument, side=side, quantity=qty)


def main():
    policy = load_policy(Path(__file__).parent / "policy.json")
    gate = PreTradePolicyGate(policy)

    checks = [
        order("bot1", "AAPL", "buy", 50),     # fine
        order("bot2", "AAPL", "buy", 5000),   # over the size limit
        order("bot3", "GME", "buy", 10),      # restricted name
        order("bot4", "AAPL", "sell", 50),    # short, not allowed here
    ]
    for o in checks:
        d = gate.authorize(o)
        verdict = "ALLOW" if d.allowed else "DENY"
        why = ", ".join(d.reasons)
        print(f"{verdict:5} {o.agent_id} {o.side} {o.quantity:g} {o.instrument}  {why}")

    # crisis, pull the kill switch on AAPL
    print("\nkill switch on AAPL")
    gate.kill("AAPL")
    d = gate.authorize(order("bot1", "AAPL", "buy", 50))
    verdict = "ALLOW" if d.allowed else "DENY"
    print(f"{verdict:5} the same fine order  {', '.join(d.reasons)}")


if __name__ == "__main__":
    main()
