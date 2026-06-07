"""Concentration of the action space, the simplest herding signal.

When a fleet herds, its actions pile onto a few choices. HHI and entropy both read that pile-up
from the distribution of actions across choices: HHI = sum_k s_k^2 rises toward 1 as one action
dominates, Shannon entropy H = -sum_k s_k log s_k falls toward 0. They are cheap and model-free,
and they are the first thing the detector looks at.
"""

from __future__ import annotations

import numpy as np


def _shares(counts) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        raise ValueError("need at least one action to measure concentration")
    return counts / total


def hhi(counts) -> float:
    """Herfindahl-Hirschman index of an action distribution: 1/k for a uniform spread across k
    actions, up to 1 when everything is one action."""
    s = _shares(counts)
    return float(np.sum(s**2))


def shannon_entropy(counts, base: float = np.e) -> float:
    """Entropy of the action distribution, in nats by default. Falls as the fleet concentrates."""
    s = _shares(counts)
    s = s[s > 0]
    return float(-np.sum(s * (np.log(s) / np.log(base))))


def normalized_entropy(counts) -> float:
    """Entropy scaled to [0, 1] by the number of distinct actions, so windows with different
    action counts compare on the same scale, 1 is maximally spread, 0 is total herding."""
    s = _shares(counts)
    k = int(np.count_nonzero(s))
    if k <= 1:
        return 0.0
    return shannon_entropy(counts) / np.log(k)
