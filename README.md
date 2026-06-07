# corral

A safety layer for autonomous AI agents in finance. You corral a herd to keep it from stampeding, and that's the job here. Watch a fleet of black-box trading agents and keep it inside safe bounds.

Three pieces, and every one treats the agent as a black box. It reads what an agent *did* (orders, cancels, fills), never its weights or its prompts, because the action stream is the one thing that generalizes across architectures and the one thing a supervisor actually gets to see.

- **Audit**, an append-only, hash-chained record of every action with its decision context and the authorization behind it. Each entry binds the hash of the one before it, so editing history breaks every hash after it and the tampering shows. Any agent's behavior can be reconstructed and verified after the fact.
- **Authorization**, a deny-by-default gate that checks each order against declarative policy before it runs, against allowed instruments, order-size and notional limits, position caps, and restricted lists. Block on violation, or just log it and watch.
- **Herding detection**, a scikit-learn-style detector that scores how far a fleet is converging on the same move, measured against an explicit null so it reports real crowding and not noise, and flags it before it turns into a liquidity event.

## Status

Early, `0.0.1`. Real and tested today, the action contract, the hash-chained audit log, the policy gate, a first herding detector, the absorption-ratio early-warning signal, and the validation statistics. Stubbed and on the roadmap, Merkle proofs and RFC 3161 time-anchoring, OPA/Cedar policy adapters, the fuller herding menu (RMT, CSAD, transfer entropy, Kuramoto, copula tail dependence, policy similarity), and the market-simulator validation harness.

## Install

```bash
pip install -e ".[dev]"
```

Python 3.10+. The core needs numpy and scipy; scikit-learn is optional and used when it's present.

## A quick look

```python
import numpy as np
from corral import AgentAction, ActionLog, TamperEvidentAuditLog, PreTradePolicyGate
from corral.action import ActionType
from corral.authz.policy import TradingPolicy
from corral.herding.detector import HerdingDetector

# the deny-by-default gate goes in front of every order
gate = PreTradePolicyGate(TradingPolicy(allowed_instruments=frozenset({"AAPL"}),
                                        max_order_quantity=1000))
# the hash-chained log records what actually went through
audit = TamperEvidentAuditLog()

log = ActionLog()
rng = np.random.default_rng(0)
for t in range(40):
    for i in range(15):
        order = AgentAction(f"agent{i}", float(t), ActionType.ORDER, instrument="AAPL",
                            side="buy" if rng.random() < 0.5 else "sell",
                            quantity=float(rng.integers(1, 100)))
        if gate.authorize(order).allowed:
            audit.append(order)
            log.append(order)

print("logged:", len(audit), "chain valid:", audit.verify())

# score the fleet against a chance baseline
X, agents, times = log.to_array(features=("signed_quantity",))
det = HerdingDetector(random_state=0).fit(X)
print("herding score:", round(float(det.decision_function(X)[0]), 2))
```

## License

BSD-3-Clause. See [LICENSE](LICENSE).
