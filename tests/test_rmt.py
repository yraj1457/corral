"""The market-mode detector reads a herd from correlation structure, so it catches the hidden,
balanced crowd that the net-direction detector goes blind to."""

import numpy as np

from corral.herding.detector import HerdingDetector, _fleet_alignment
from corral.herding.rmt import MarketModeDetector, market_mode
from corral.validation.benchmark import calibrate_threshold, compare_detectors
from corral.validation.simulator import simulate_run


def test_balanced_herd_hides_from_direction_but_not_correlation():
    rng = np.random.default_rng(5)
    X, _ = simulate_run(30, 140, cascade=True, onset=50, event=90, herd_mode="balanced",
                        herd_fraction=0.8, w_max=0.9, rng=rng)
    late = X[:, 75:90, :]
    early = X[:, 10:25, :]
    assert _fleet_alignment(late) < 0.4  # net direction stays calm, the two groups cancel
    assert market_mode(late) > market_mode(early) + 1.0  # the shared factor is strong


def test_market_mode_beats_alignment_on_hidden_herd():
    rng = np.random.default_rng(11)
    normal = [simulate_run(30, 140, cascade=False, rng=rng) for _ in range(50)]
    cascades = [
        simulate_run(30, 140, cascade=True, onset=50, event=90, herd_mode="balanced",
                     herd_fraction=0.6, w_max=0.8, rng=rng)
        for _ in range(60)
    ]
    align = HerdingDetector(null_resamples=60, random_state=0).fit(normal[0][0])
    align.threshold = calibrate_threshold(align, normal, window=25, target_far=0.05)
    rmt = MarketModeDetector(null_resamples=60, random_state=0).fit(normal[0][0])
    rmt.threshold = calibrate_threshold(rmt, normal, window=25, target_far=0.05)

    c = compare_detectors(rmt, align, cascades, window=25, random_state=0)
    assert c["a_detection"] > c["b_detection"]  # rmt (a) beats alignment (b)
    assert c["gap"] > 0
