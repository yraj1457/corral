"""Pillar 2, the pre-trade gate.

Every order an agent wants to send goes through authorize() first. It returns ALLOW or DENY with the
reasons, deny-by-default, if a check cannot be evaluated the order does not go. `mode` picks hard
enforcement (block) or soft monitoring (record the violation, let it through), so the same gate
serves a firm that wants to block and a supervisor that only watches. A kill switch sits above
all of it, a halted gate denies every order in scope whatever the mode, the RTS 6 emergency stop.

OPA/Rego and Cedar adapters for externally-authored policy come later. This gate evaluates the
built-in TradingPolicy directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from corral.authz.policy import TradingPolicy


class Effect(str, Enum):
    ALLOW = "allow"
    DENY = "deny"


class Mode(str, Enum):
    ENFORCE = "enforce"  # block on violation
    MONITOR = "monitor"  # record the violation, allow the order through


@dataclass(frozen=True)
class Decision:
    effect: Effect
    reasons: tuple[str, ...] = ()

    @property
    def allowed(self) -> bool:
        return self.effect is Effect.ALLOW


class PreTradePolicyGate:
    """Checks each order against a TradingPolicy before it executes, with a kill switch on top."""

    def __init__(self, policy: TradingPolicy, mode: Mode = Mode.ENFORCE) -> None:
        self.policy = policy
        self.mode = mode
        self._positions: dict[tuple[str, str], float] = {}  # (agent, instrument) -> net qty
        self._halted: set[str] = set()  # instruments halted by name
        self._halt_all = False

    def kill(self, scope: str | None = None) -> None:
        """Halt trading. With no scope, halt everything, with an instrument, halt just that one. A
        halted gate denies every order in scope until reset, the RTS 6 emergency stop."""
        if scope is None:
            self._halt_all = True
        else:
            self._halted.add(scope)

    def reset(self, scope: str | None = None) -> None:
        """Lift a halt. With no scope, lift the whole halt and clear every named instrument."""
        if scope is None:
            self._halt_all = False
            self._halted.clear()
        else:
            self._halted.discard(scope)

    def authorize(self, order: Any, *, position: float | None = None) -> Decision:
        """Check one order. `order` is an AgentAction, or anything exposing instrument / side /
        quantity / price / agent_id. `position` overrides the gate's tracked net position for the
        position-cap check when the caller keeps its own book. Every failed check adds a reason, and
        any reason at all means DENY unless the gate is in MONITOR mode. A halted instrument is
        blocked regardless of mode.
        """
        instrument = getattr(order, "instrument", None)
        if self._halt_all or (instrument is not None and instrument in self._halted):
            return Decision(Effect.DENY, ("trading halted by the kill switch",))

        p = self.policy
        reasons: list[str] = []
        side = getattr(order, "side", None)
        qty = float(getattr(order, "quantity", 0.0) or 0.0)
        price = getattr(order, "price", None)

        if instrument is None:
            reasons.append("order has no instrument, cannot evaluate")  # deny-by-default
        else:
            if instrument in p.restricted_instruments:
                reasons.append(f"{instrument} is on the restricted list")
            if p.allowed_instruments is not None and instrument not in p.allowed_instruments:
                reasons.append(f"{instrument} is not on the allowed list")

        if p.max_order_quantity is not None and qty > p.max_order_quantity:
            reasons.append(f"order quantity {qty:g} over the {p.max_order_quantity:g} limit")
        if (
            p.max_order_notional is not None
            and price is not None
            and qty * price > p.max_order_notional
        ):
            reasons.append(
                f"order notional {qty * price:g} over the {p.max_order_notional:g} limit"
            )

        signed = qty if side == "buy" else -qty if side == "sell" else 0.0
        if instrument is not None and (p.max_position is not None or not p.allow_short):
            key = (getattr(order, "agent_id", ""), instrument)
            base = position if position is not None else self._positions.get(key, 0.0)
            projected = base + signed
            if not p.allow_short and projected < 0:
                reasons.append("short selling is not permitted")
            if p.max_position is not None and abs(projected) > p.max_position:
                reasons.append(
                    f"net position {projected:g} would breach the {p.max_position:g} cap"
                )

        effect = Effect.DENY if reasons else Effect.ALLOW
        if reasons and self.mode is Mode.MONITOR:
            effect = Effect.ALLOW  # monitor mode keeps the reasons but lets the order through
        return Decision(effect, tuple(reasons))

    def commit(self, order: Any) -> None:
        """Update the tracked net position after an order is accepted or filled, so the next
        position-cap check sees where the agent actually stands."""
        instrument = getattr(order, "instrument", None)
        if instrument is None:
            return
        side = getattr(order, "side", None)
        qty = float(getattr(order, "quantity", 0.0) or 0.0)
        signed = qty if side == "buy" else -qty if side == "sell" else 0.0
        key = (getattr(order, "agent_id", ""), instrument)
        self._positions[key] = self._positions.get(key, 0.0) + signed
