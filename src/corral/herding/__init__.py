"""Pillar 3, correlated-behavior / herding detection."""

from corral.herding.concentration import hhi, normalized_entropy, shannon_entropy
from corral.herding.detector import HerdingDetector
from corral.herding.ensemble import HerdingEnsemble
from corral.herding.nulls import null_distribution, surrogate_shuffle
from corral.herding.rmt import MarketModeDetector, market_mode

__all__ = [
    "HerdingDetector",
    "HerdingEnsemble",
    "MarketModeDetector",
    "market_mode",
    "hhi",
    "shannon_entropy",
    "normalized_entropy",
    "surrogate_shuffle",
    "null_distribution",
]
