"""The combined watchdog catches every kind of herd at one controlled false-alarm rate, where a
single detector only catches its own kind."""

import numpy as np

from corral.herding.detector import HerdingDetector
from corral.herding.ensemble import HerdingEnsemble
from corral.herding.rmt import MarketModeDetector
from corral.validation.benchmark import calibrate_threshold, evaluate
from corral.validation.simulator import simulate_run


def _cascades(rng, n, mode):
    return [
        simulate_run(30, 140, cascade=True, onset=50, event=90, herd_mode=mode,
                     herd_fraction=0.7, w_max=0.8, rng=rng)
        for _ in range(n)
    ]


def test_combined_catches_both_herds_at_budget():
    rng = np.random.default_rng(0)
    normal = [simulate_run(30, 140, cascade=False, rng=rng) for _ in range(120)]
    directional = _cascades(rng, 50, "directional")
    balanced = _cascades(rng, 50, "balanced")
    panel = normal[0][0]

    members = []
    for cls in (HerdingDetector, MarketModeDetector):
        d = cls(null_resamples=80, random_state=0).fit(panel)
        d.threshold = calibrate_threshold(d, normal, window=25, target_far=0.025)
        members.append(d)
    combined = HerdingEnsemble(members)

    assert evaluate(combined, directional, window=25)["detection_rate"] > 0.7
    assert evaluate(combined, balanced, window=25)["detection_rate"] > 0.7
    assert evaluate(combined, normal, window=25)["false_alarm_rate"] < 0.2
