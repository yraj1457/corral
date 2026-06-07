"""Pillar 3, the herding detector.

Black-box and action-only. X is (n_agents, n_timesteps, action_dim). fit() learns the chance
baseline, how aligned the fleet looks under the shuffle null, and decision_function() returns
the observed alignment's excess over that baseline, in units of the null's own spread, so a high
score means the fleet is more aligned than chance would explain. predict() thresholds that into
+1 (herding) / -1 (normal), following scikit-learn's outlier convention.

This first cut scores directional alignment against a surrogate-shuffle null. The fuller menu
(RMT-denoised market mode, CSAD, transfer entropy, Kuramoto, copula tail dependence, policy
similarity) plugs in through the same fit / decision_function contract, see PLAN.md, Pillar 3.
"""

from __future__ import annotations

import numpy as np

from corral.herding.nulls import null_distribution

try:
    from sklearn.base import BaseEstimator, OutlierMixin
except ImportError:  # scikit-learn is optional; fall back to plain bases when it is absent

    class BaseEstimator:  # type: ignore[no-redef]
        pass

    class OutlierMixin:  # type: ignore[no-redef]
        pass


def _fleet_alignment(X) -> float:
    """Mean directional alignment of the fleet over time. At each timestep, |mean sign of action|
    is 0 when buys and sells cancel and 1 when everyone leans the same way; average over time.
    Uses the first action channel. This is the scalar the first detector scores."""
    X = np.asarray(X, dtype=float)
    signs = np.sign(X[..., 0])  # (n_agents, n_timesteps)
    per_t = np.abs(signs.mean(axis=0))  # (n_timesteps,)
    return float(per_t.mean())


class HerdingDetector(BaseEstimator, OutlierMixin):
    """Flag a window where an agent fleet is herding.

    A "sample" here is a whole fleet panel X of shape (n_agents, n_timesteps, action_dim), scored
    as one window; sliding the window across a longer stream is left to the caller for now.

    Parameters
    ----------
    null_resamples : int
        How many shuffle draws to build the chance baseline in fit().
    threshold : float
        Excess-over-null score (in null-spread units) at which a window is called herding.
    random_state : int | None
        Seed for the shuffle null.
    """

    def __init__(self, *, null_resamples: int = 200, threshold: float = 3.0,
                 random_state: int | None = None) -> None:
        self.null_resamples = null_resamples
        self.threshold = threshold
        self.random_state = random_state

    def fit(self, X, y=None):
        null = null_distribution(X, _fleet_alignment, self.null_resamples, self.random_state)
        self.null_mean_ = float(null.mean())
        self.null_std_ = float(null.std()) or 1e-12  # guard a degenerate (zero-spread) null
        return self

    def decision_function(self, X) -> np.ndarray:
        # excess over chance, in units of the null's own spread, the same idea as a z-score
        obs = _fleet_alignment(X)
        return np.array([(obs - self.null_mean_) / self.null_std_])

    def predict(self, X) -> np.ndarray:
        score = self.decision_function(X)
        return np.where(score >= self.threshold, 1, -1)
