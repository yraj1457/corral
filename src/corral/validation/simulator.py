"""A small, controllable cascade simulator for validating the herding detectors.

Real markets never hand you a label that says the herd formed here. A simulator does, because we
control the shock. Each run is a fleet of agents emitting buy/sell orders over time, written as a
(n_agents, n_timesteps, 1) signed-quantity panel, the same shape the detectors read. In a cascade
run some of the agents start herding, with a strength that ramps from 0 at the onset to w_max at the
event, and the event is the moment we call the most fragile point. In a normal run nobody herds and
the agents act on their own.

There are two kinds of herd. A "directional" one is the obvious case, the herders all pile onto the
same side, so net order flow goes one way. A "balanced" one is the hidden case, the herders split
into two groups crowding opposite sides of a shared factor, so net flow looks calm while the market
is really two crowded opposing trades. The two are caught by different detectors, which is the whole
point of comparing them.
"""

from __future__ import annotations

import numpy as np


def simulate_run(n_agents=40, n_timesteps=140, cascade=True, onset=50, event=90, w_max=0.9,
                 herd_mode="directional", herd_fraction=1.0, max_qty=100.0, rng=None):
    """One fleet run. Returns (X, event_time), where X is (n_agents, n_timesteps, 1) signed
    quantities and event_time is the cascade timestep, or None for a normal run. herd_mode is
    "directional" or "balanced" (see the module docstring), and herd_fraction is how much of the
    fleet joins the crowd. The sign of each entry is the side, buy positive and sell negative, and
    the magnitude is a random order size that the sign-based detectors ignore for now but that keeps
    the panel shaped like real order flow.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_herd = int(round(herd_fraction * n_agents))
    herders = rng.permutation(n_agents)[:n_herd]
    group = np.zeros(n_agents)  # +1 and -1 mark the two herding groups, 0 marks a non-herder
    if herd_mode == "balanced":
        half = n_herd // 2
        group[herders[:half]] = 1.0
        group[herders[half:]] = -1.0
    else:
        group[herders] = 1.0  # everyone leans the same way

    X = np.zeros((n_agents, n_timesteps, 1))
    span = max(1, event - onset)
    for t in range(n_timesteps):
        w = w_max * min(1.0, (t - onset) / span) if (cascade and t >= onset) else 0.0
        # a balanced herd follows a factor that moves each step, a directional one a steady sell
        shared = rng.standard_normal() if herd_mode == "balanced" else -1.0
        latent = (1.0 - w) * rng.standard_normal(n_agents) + w * group * shared
        side = np.where(latent >= 0.0, 1.0, -1.0)
        qty = rng.uniform(1.0, max_qty, size=n_agents)
        X[:, t, 0] = side * qty
    return X, (event if cascade else None)


def make_dataset(n_normal, n_cascade, n_agents=40, n_timesteps=140, herd_mode="directional",
                 herd_fraction=1.0, rng=None):
    """A list of (X, event_time) runs, n_normal labeled None and n_cascade with the onset and event
    jittered so the alarm time is not identical on every run. The cascade runs use the given herd
    mode and fraction.
    """
    if rng is None:
        rng = np.random.default_rng()
    runs = [simulate_run(n_agents, n_timesteps, cascade=False, rng=rng) for _ in range(n_normal)]
    for _ in range(n_cascade):
        onset = int(rng.integers(45, 60))
        event = min(onset + int(rng.integers(28, 42)), n_timesteps - 5)
        runs.append(simulate_run(n_agents, n_timesteps, cascade=True, onset=onset, event=event,
                                 herd_mode=herd_mode, herd_fraction=herd_fraction, rng=rng))
    return runs
