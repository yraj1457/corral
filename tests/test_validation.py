"""The validation loop: the simulator produces labeled runs, and the detector catches the seeded
cascades with few false alarms and real lead time."""

import numpy as np

from corral.herding.detector import HerdingDetector, _fleet_alignment
from corral.validation.benchmark import calibrate_threshold, evaluate
from corral.validation.simulator import simulate_run
from corral.validation.stats import bootstrap_ci


def test_cascade_run_has_rising_alignment():
    rng = np.random.default_rng(0)
    X, event = simulate_run(40, 140, cascade=True, onset=50, event=90, rng=rng)
    assert event == 90
    early = _fleet_alignment(X[:, 10:25, :])
    late = _fleet_alignment(X[:, 75:90, :])
    assert late > early + 0.2  # the fleet is far more aligned heading into the event


def test_normal_run_stays_low_and_unlabeled():
    rng = np.random.default_rng(1)
    X, event = simulate_run(40, 140, cascade=False, rng=rng)
    assert event is None
    assert _fleet_alignment(X) < 0.4  # independent fleet, little alignment


def test_detector_catches_cascades_with_few_false_alarms():
    rng = np.random.default_rng(7)
    normal = [simulate_run(40, 140, cascade=False, rng=rng) for _ in range(80)]
    cascades = []
    for _ in range(80):
        onset = int(rng.integers(45, 60))
        event = onset + int(rng.integers(28, 40))
        cascades.append(simulate_run(40, 140, cascade=True, onset=onset, event=event, rng=rng))
    calib, test_normal = normal[:40], normal[40:]

    det = HerdingDetector(null_resamples=80, random_state=0).fit(calib[0][0])
    det.threshold = calibrate_threshold(det, calib, window=15, target_far=0.05)

    m = evaluate(det, test_normal + cascades, window=15)
    assert m["detection_rate"] > 0.7
    assert m["false_alarm_rate"] < 0.15
    assert m["mean_lead"] > 0


def test_bootstrap_ci_brackets_the_mean():
    mean, lo, hi = bootstrap_ci([5, 7, 6, 8, 5, 9, 7, 6], random_state=0)
    assert lo <= mean <= hi
