"""Quickstart. Log agent actions tamper-evidently, gate them against policy, then score the fleet
for herding. Run it with `python examples/quickstart.py` from the repo root."""

import numpy as np

from corral import AgentAction, ActionLog, PreTradePolicyGate, TamperEvidentAuditLog
from corral.action import ActionType
from corral.authz.policy import TradingPolicy
from corral.herding.detector import HerdingDetector


def main() -> None:
    # the deny-by-default gate goes in front of every order
    gate = PreTradePolicyGate(
        TradingPolicy(allowed_instruments=frozenset({"AAPL", "MSFT"}), max_order_quantity=1000)
    )
    # the hash-chained log records what actually went through
    audit = TamperEvidentAuditLog()

    log = ActionLog()
    rng = np.random.default_rng(7)
    for t in range(40):
        for i in range(15):
            order = AgentAction(
                f"agent{i}", float(t), ActionType.ORDER, instrument="AAPL",
                side="buy" if rng.random() < 0.5 else "sell",
                quantity=float(rng.integers(1, 100)),
            )
            if gate.authorize(order).allowed:  # only authorized orders are logged and counted
                audit.append(order)
                log.append(order)

    print(f"actions logged: {len(audit)}   chain valid: {audit.verify()}")

    # score the fleet against a chance baseline (this one is independent, so it stays low)
    X, agents, times = log.to_array(features=("signed_quantity",))
    det = HerdingDetector(random_state=0).fit(X)
    print(f"herding score (excess over chance): {float(det.decision_function(X)[0]):.2f}")


if __name__ == "__main__":
    main()
