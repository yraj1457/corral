"""Honest statistics, the cascade simulator, and the detector benchmark."""

from corral.validation.benchmark import alarm_time, calibrate_threshold, evaluate
from corral.validation.simulator import make_dataset, simulate_run
from corral.validation.stats import (
    benjamini_hochberg,
    binomial_se,
    bootstrap_ci,
    bootstrap_gap,
    cluster_se,
    effective_n,
)

__all__ = [
    "binomial_se",
    "cluster_se",
    "effective_n",
    "bootstrap_gap",
    "bootstrap_ci",
    "benjamini_hochberg",
    "simulate_run",
    "make_dataset",
    "calibrate_threshold",
    "evaluate",
    "alarm_time",
]
