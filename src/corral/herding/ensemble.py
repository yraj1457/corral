"""Pillar 3, the combined herding watchdog.

One detector catches one kind of herd and goes blind to the others, which is no good when you do
not know in advance which kind is forming. This runs the whole menu at once and fires when any
member crosses its own line, so any kind of herd trips the alarm. Each member is calibrated to its
share of the false-alarm budget, so the overall rate stays at the budget as the menu grows.

It follows the same decision_function / predict contract as the single detectors, so it drops into
the same benchmark. Calibrate each member first, then the ensemble fires on whichever one is over.
"""

from __future__ import annotations

import numpy as np

try:
    from sklearn.base import BaseEstimator, OutlierMixin
except ImportError:  # scikit-learn is optional

    class BaseEstimator:  # type: ignore[no-redef]
        pass

    class OutlierMixin:  # type: ignore[no-redef]
        pass


class HerdingEnsemble(BaseEstimator, OutlierMixin):
    """Run several herding detectors and fire when any one crosses its own threshold.

    The score is how far the loudest member sits over its own line, so 0 or more means one of them
    fired. Calibrate each member to its share of the budget (target_far / number of members)
    before using it, which keeps the combined false-alarm rate at the budget. Mixing members on
    different scales by their raw scores does not work, the heavier-tailed one drowns out the rest.
    """

    def __init__(self, detectors, *, threshold=0.0):
        self.detectors = detectors
        self.threshold = threshold

    def fit(self, X, y=None):
        for d in self.detectors:
            d.fit(X)
        return self

    def decision_function(self, X) -> np.ndarray:
        return np.array([max(d.decision_function(X)[0] - d.threshold for d in self.detectors)])

    def predict(self, X) -> np.ndarray:
        return np.where(self.decision_function(X) >= self.threshold, 1, -1)
