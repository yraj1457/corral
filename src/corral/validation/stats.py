"""The honest-statistics spine for validating detectors.

This is the same machinery a leaderboard needs and the same shape the cascade-detection evaluation
needs: a standard error that does not pretend clustered items are independent, a bootstrap null for
the gap between two methods that resamples *items* so the shared item-difficulty cancels in the
difference, Benjamini-Hochberg over the many comparisons, and effective-N as a descriptive read on
how much independent information is really there. See PLAN.md, section 9.
"""

from __future__ import annotations

import numpy as np


def binomial_se(p: float, n: int) -> float:
    """Standard error of a mean of 0/1 outcomes under independence: sqrt(p(1-p)/n)."""
    if n <= 0:
        raise ValueError("n must be positive")
    return float(np.sqrt(p * (1 - p) / n))


def cluster_se(values, clusters) -> float:
    """Standard error of a mean when items cluster (same repo, same regime, same seed family).
    Widens the naive SE so correlated items are not counted as independent information: it sums the
    squared within-cluster residual totals, which keeps the covariance between items in the same
    cluster instead of throwing it away."""
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    n = values.size
    if n == 0:
        raise ValueError("need at least one value")
    resid = values - values.mean()
    var = 0.0
    for c in np.unique(clusters):
        s = resid[clusters == c].sum()
        var += s * s
    return float(np.sqrt(var) / n)


def effective_n(rho_bar: float, n: int) -> float:
    """Effective number of independent comparisons given the average pairwise correlation rho_bar:
    N / (1 + (N-1)*rho_bar). Identical items drive it toward 1, independent items toward N."""
    if n <= 0:
        raise ValueError("n must be positive")
    return float(n / (1.0 + (n - 1) * rho_bar))


def bootstrap_gap(scores_a, scores_b, n_resamples: int = 2000, random_state: int | None = None):
    """Bootstrap the gap mean(a) - mean(b) by resampling *items*, the shared axis, so both methods
    are recomputed on the same resampled items each draw and the covariance between them is
    reproduced by construction, no independence assumed, no covariance estimated separately.
    Returns (gap, se, ci_low, ci_high), the interval at 95%."""
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("a and b must be scored on the same items (same shape)")
    rng = np.random.default_rng(random_state)
    n = a.size
    gaps = np.empty(n_resamples)
    for k in range(n_resamples):
        idx = rng.integers(0, n, n)  # the SAME item draw for both -> covariance preserved
        gaps[k] = a[idx].mean() - b[idx].mean()
    gap = float(a.mean() - b.mean())
    return gap, float(gaps.std()), float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))


def benjamini_hochberg(pvalues, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg step-up. Returns a boolean array, True where the hypothesis is rejected
    at false-discovery rate alpha, the expected fraction of false positives among the steps
    we call real."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    order = np.argsort(p)
    thresh = (np.arange(1, m + 1) / m) * alpha
    passed = p[order] <= thresh
    out = np.zeros(m, dtype=bool)
    hits = np.nonzero(passed)[0]
    if hits.size:
        cutoff = hits.max()  # reject everything up to the largest rank that passes
        out[order[: cutoff + 1]] = True
    return out
