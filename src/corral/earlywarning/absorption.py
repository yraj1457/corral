"""Kritzman-Li absorption ratio, a fast coupling alarm.

The fraction of a market's total variance captured by its top few principal components. When it
rises, returns are being driven by fewer common factors, so the market is more tightly coupled and
more fragile. It is cheap to compute from a returns matrix, which is why it works as a real-time
coupling signal. The operational trigger is a standardized *shift* in the ratio, not its raw level
(see PLAN.md, early-warning), this function returns the level; the shift is built on top of it.
"""

from __future__ import annotations

import numpy as np


def absorption_ratio(returns, n_components: int | None = None) -> float:
    """returns: (n_periods, n_assets). Fraction of total variance in the top n_components principal
    components. n_components defaults to ~1/5 of the assets, as in Kritzman-Li."""
    R = np.asarray(returns, dtype=float)
    if R.ndim != 2:
        raise ValueError("returns must be 2D (n_periods, n_assets)")
    n_assets = R.shape[1]
    if n_components is None:
        n_components = max(1, n_assets // 5)
    cov = np.cov(R, rowvar=False)
    eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
    total = eig.sum()
    if total <= 0:
        raise ValueError("degenerate covariance (zero total variance)")
    return float(eig[:n_components].sum() / total)
