"""Pillar 2, the policy a trade is checked against.

Declarative limits, the kind regulators already wrote down (SEC Rule 15c3-5, MiFID II RTS 6). Which
instruments are allowed, how large a single order can be, how much notional it can carry, how big a
net position an agent may hold. This module is the rule set, gate.py runs the check. The engine is
deny-by-default and most-conservative-wins, so leaving a field as None means no limit on that axis,
not anything goes, the other rules still apply. A policy can be written as a plain JSON file and
loaded, so risk and compliance can author and version the rules without touching code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TradingPolicy:
    """The rule set a trade is checked against. None on any field means no limit on that axis,
    not anything goes, the other rules still apply."""

    allowed_instruments: frozenset[str] | None = None  # None = no allow-list restriction
    restricted_instruments: frozenset[str] = field(default_factory=frozenset)
    max_order_quantity: float | None = None
    max_order_notional: float | None = None  # quantity * price ceiling for a single order
    max_position: float | None = None  # absolute net position per (agent, instrument)
    allow_short: bool = True

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TradingPolicy:
        """Build a policy from a dict, the kind you keep in a JSON file. Instrument lists become
        sets, and anything left out keeps its default of no limit on that axis."""
        allowed = d.get("allowed_instruments")
        return cls(
            allowed_instruments=frozenset(allowed) if allowed is not None else None,
            restricted_instruments=frozenset(d.get("restricted_instruments", ())),
            max_order_quantity=d.get("max_order_quantity"),
            max_order_notional=d.get("max_order_notional"),
            max_position=d.get("max_position"),
            allow_short=d.get("allow_short", True),
        )

    def to_dict(self) -> dict[str, Any]:
        """Back to a plain dict, sets written as sorted lists, ready to dump as JSON."""
        allowed = self.allowed_instruments
        return {
            "allowed_instruments": sorted(allowed) if allowed is not None else None,
            "restricted_instruments": sorted(self.restricted_instruments),
            "max_order_quantity": self.max_order_quantity,
            "max_order_notional": self.max_order_notional,
            "max_position": self.max_position,
            "allow_short": self.allow_short,
        }


def load_policy(path) -> TradingPolicy:
    """Read a trading policy from a JSON file, so the rules live in version control next to the code
    that enforces them, owned by the people who write them."""
    with open(path) as f:
        return TradingPolicy.from_dict(json.load(f))
