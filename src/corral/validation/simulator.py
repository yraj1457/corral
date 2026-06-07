"""A small, controllable cascade simulator for validating the herding detector.

Real markets never hand you a label that says "the herd formed here." A simulator does, because we
control the shock. Each run is a fleet of agents emitting buy/sell orders over time, written as a
(n_agents, n_timesteps, 1) signed-quantity panel, the same shape the detector reads. In a cascade
run a shared sell signal is mixed into every agent's decision with a weight that ramps from 0 at the
onset to w_max at the event, so the fleet crowds onto one side and the crowding peaks at the event,
the moment we call the liquidity break. In a normal run that weight stays 0 and the agents act on
their own. The detector's job is to see the crowding building and fire before the event.
"""

from __future__ import annotations

import numpy as np


def simulate_run(n_agents=40, n_timesteps=140, cascade=True, onset=50, event=90, w_max=0.9,
                 max_qty=100.0, rng=None):
    """One fleet run. Returns (X, event_time). X is (n_agents, n_timesteps, 1) signed quantities,
    event_time is the cascade timestep, or None for a normal run. The sign of each entry is the side
    (buy +, sell -); the magnitude is a random order size, which the sign-based detector ignores for
    now but which keeps the panel shaped like real order flow.
    """
    if rng is None:
        rng = np.random.default_rng()
    common = -1.0  # the coordinated sell, the direction a cascade runs in
    X = np.zeros((n_agents, n_timesteps, 1))
    span = max(1, event - onset)
    for t in range(n_timesteps):
        if cascade and t >= onset:
            w = w_max * min(1.0, (t - onset) / span)
        else:
            w = 0.0
        # each agent mixes its own idiosyncratic draw with the shared signal; as w rises the shared
        # sell takes over and the fleet ends up on the same side
        latent = (1.0 - w) * rng.standard_normal(n_agents) + w * common
        side = np.where(latent >= 0.0, 1.0, -1.0)
        qty = rng.uniform(1.0, max_qty, size=n_agents)
        X[:, t, 0] = side * qty
    return X, (event if cascade else None)


def make_dataset(n_normal, n_cascade, n_agents=40, n_timesteps=140, rng=None):
    """A list of (X, event_time) runs, n_normal labeled None and n_cascade with the onset and event
    jittered so the alarm time is not identical on every run.
    """
    if rng is None:
        rng = np.random.default_rng()
    runs = [simulate_run(n_agents, n_timesteps, cascade=False, rng=rng) for _ in range(n_normal)]
    for _ in range(n_cascade):
        onset = int(rng.integers(45, 60))
        event = min(onset + int(rng.integers(28, 42)), n_timesteps - 5)
        runs.append(
            simulate_run(n_agents, n_timesteps, cascade=True, onset=onset, event=event, rng=rng)
        )
    return runs
