"""Validation, honest statistics now, the market-simulator harness later."""

from corral.validation.stats import (
    benjamini_hochberg,
    binomial_se,
    bootstrap_gap,
    cluster_se,
    effective_n,
)

__all__ = ["binomial_se", "cluster_se", "effective_n", "bootstrap_gap", "benjamini_hochberg"]
