"""Null models, the discipline that keeps a herding signal honest.

Raw concentration or correlation is not evidence; agents reacting to the same market look alike
even when nothing is coordinated, so the appearance of herding is the null, not the finding. Every
metric is therefore reported against an explicit chance baseline. The default null shuffles each
agent's actions across time independently, which destroys any cross-agent timing while preserving
each agent's own action mix, the difference between the observed metric and this baseline is the
part that is not chance.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def surrogate_shuffle(X, rng: np.random.Generator) -> np.ndarray:
    """Return a copy of X (n_agents, n_timesteps, action_dim) with each agent's timeline
    independently permuted. Breaks synchrony across agents, preserves each agent's marginals."""
    X = np.asarray(X, dtype=float)
    out = np.empty_like(X)
    n_agents, n_t = X.shape[0], X.shape[1]
    for i in range(n_agents):
        out[i] = X[i, rng.permutation(n_t)]
    return out


def null_distribution(
    X,
    statistic: Callable[[np.ndarray], float],
    n_resamples: int = 200,
    random_state: int | None = None,
) -> np.ndarray:
    """Sample `statistic` under the shuffle null. Returns an array of n_resamples values; compare
    the observed statistic(X) against this to read off the excess over chance."""
    rng = np.random.default_rng(random_state)
    return np.array([statistic(surrogate_shuffle(X, rng)) for _ in range(n_resamples)])
