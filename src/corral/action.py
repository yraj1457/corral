"""The black-box agent action, the only thing corral ever sees.

corral does not look inside an agent, not its weights, prompts, or chain-of-thought. It works
entirely off a stream of actions, because that is the one thing that generalizes across
architectures and the one thing a supervisor actually gets to observe. Audit, authorization,
and herding detection are all defined over these records.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class ActionType(str, Enum):
    ORDER = "order"
    CANCEL = "cancel"
    FILL = "fill"
    POSITION = "position"
    HOLD = "hold"


@dataclass(frozen=True)
class AgentAction:
    """One action by one agent at one instant.

    The fields are the small, model-agnostic set every trading agent exposes whatever it is
    built from. `context` holds the decision context (features, signals, anything), and
    `authorization` holds who or what let the action through, both kept so the action can be
    reconstructed and explained later.
    """

    agent_id: str
    timestamp: float  # the event's own time in unix seconds, not wall-clock when it was logged
    action_type: ActionType
    instrument: str | None = None
    side: str | None = None  # "buy", "sell", or None for non-directional events
    quantity: float = 0.0
    price: float | None = None
    context: dict[str, Any] = field(default_factory=dict)
    authorization: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["action_type"] = self.action_type.value
        return d


# how each action turns into a scalar channel of the action_dim axis
_FEATURES = {
    "side_sign": lambda a: 1.0 if a.side == "buy" else (-1.0 if a.side == "sell" else 0.0),
    "quantity": lambda a: float(a.quantity),
    "signed_quantity": lambda a: (
        (1.0 if a.side == "buy" else -1.0 if a.side == "sell" else 0.0) * float(a.quantity)
    ),
    "price": lambda a: float(a.price) if a.price is not None else 0.0,
}


class ActionLog:
    """An ordered collection of AgentAction records, plus the conversions the rest of the
    library needs, most importantly the (n_agents, n_timesteps, action_dim) array the herding
    detectors read.
    """

    def __init__(self, actions: list[AgentAction] | None = None) -> None:
        self._actions: list[AgentAction] = list(actions or [])

    def __len__(self) -> int:
        return len(self._actions)

    def __iter__(self):
        return iter(self._actions)

    def append(self, action: AgentAction) -> None:
        self._actions.append(action)

    @property
    def agents(self) -> list[str]:
        return sorted({a.agent_id for a in self._actions})

    def to_array(self, features: tuple[str, ...] = ("signed_quantity",), fill: float = 0.0):
        """Lay the log out as a dense (n_agents, n_timesteps, action_dim) array.

        Timesteps are the sorted unique timestamps, agents the sorted unique ids. A missing
        (agent, time) cell means the agent did nothing then, which is not the same as trading,
        so it is set to `fill` (0.0 by default). `features` picks which scalars of each action
        become the channels. Returns the array together with the agent and time labels so the
        axes stay interpretable. If an agent acts more than once at the same timestamp, the
        last action wins.
        """
        agents = self.agents
        times = sorted({a.timestamp for a in self._actions})
        a_idx = {a: i for i, a in enumerate(agents)}
        t_idx = {t: j for j, t in enumerate(times)}
        funcs = [_FEATURES[f] for f in features]

        X = np.full((len(agents), len(times), len(funcs)), fill, dtype=float)
        for act in self._actions:
            i, j = a_idx[act.agent_id], t_idx[act.timestamp]
            X[i, j, :] = [fn(act) for fn in funcs]
        return X, agents, times
