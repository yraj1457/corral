"""Pillar 3, the market-mode detector.

This reads a herd from the correlation structure of the fleet's actions instead of its net
direction. It builds the correlation matrix of the agents over a window and takes the largest
eigenvalue, the strength of the single common factor the fleet is moving on. Independent agents
produce a spread of small eigenvalues, the noise floor that random matrix theory describes, and a
crowd pokes one big eigenvalue above it. This catches the hidden case the alignment detector
misses, two groups crowding opposite sides of one factor, where net order flow looks calm but the
market is really two crowded opposing trades.

Same fit / decision_function / predict contract as HerdingDetector, so the two drop into the same
benchmark and can be compared head to head.
"""

from __future__ import annotations

import numpy as np

from corral.herding.nulls import null_distribution

try:
    from sklearn.base import BaseEstimator, OutlierMixin
except ImportError:  # scikit-learn is optional

    class BaseEstimator:  # type: ignore[no-redef]
        pass

    class OutlierMixin:  # type: ignore[no-redef]
        pass


def market_mode(X) -> float:
    """Largest eigenvalue of the agents' correlation matrix over the window. Agents that never move
    in the window carry no correlation and are dropped so the matrix stays well defined."""
    A = np.asarray(X, dtype=float)[..., 0]  # (n_agents, n_timesteps)
    active = A.std(axis=1) > 1e-12
    if int(active.sum()) < 2:
        return 0.0
    A = A[active]
    A = (A - A.mean(axis=1, keepdims=True)) / A.std(axis=1, keepdims=True)
    corr = (A @ A.T) / A.shape[1]
    return float(np.linalg.eigvalsh(corr)[-1])


class MarketModeDetector(BaseEstimator, OutlierMixin):
    """Flag a herd by how much of the fleet's movement collapses onto one shared factor.

    Black-box and action-only, X is (n_agents, n_timesteps, action_dim). fit() learns how big the
    top eigenvalue looks under the shuffle null, where the cross-agent correlation is destroyed, and
    decision_function() returns the observed eigenvalue's excess over that baseline in null-spread
    units. predict() thresholds it into +1 (herding) / -1 (normal), the scikit-learn outlier
    convention.
    """

    def __init__(self, *, null_resamples=200, threshold=3.0, random_state=None):
        self.null_resamples = null_resamples
        self.threshold = threshold
        self.random_state = random_state

    def fit(self, X, y=None):
        null = null_distribution(X, market_mode, self.null_resamples, self.random_state)
        self.null_mean_ = float(null.mean())
        self.null_std_ = float(null.std()) or 1e-12
        return self

    def decision_function(self, X) -> np.ndarray:
        return np.array([(market_mode(X) - self.null_mean_) / self.null_std_])

    def predict(self, X) -> np.ndarray:
        return np.where(self.decision_function(X) >= self.threshold, 1, -1)
