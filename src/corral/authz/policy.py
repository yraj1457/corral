"""Pillar 2, the policy a trade is checked against.

Declarative limits, the kind regulators already wrote down (SEC Rule 15c3-5, MiFID II RTS 6):
which instruments are allowed, how large a single order can be, how much notional it can carry,
how big a net position an agent may hold. This module is the rule set; gate.py runs the check.
The engine is deny-by-default and most-conservative-wins, so leaving a field as None means "no
limit on this axis," not "anything goes", the other rules still apply.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TradingPolicy:
    allowed_instruments: frozenset[str] | None = None  # None = no allow-list restriction
    restricted_instruments: frozenset[str] = field(default_factory=frozenset)
    max_order_quantity: float | None = None
    max_order_notional: float | None = None  # quantity * price ceiling for a single order
    max_position: float | None = None  # absolute net position per (agent, instrument)
    allow_short: bool = True
