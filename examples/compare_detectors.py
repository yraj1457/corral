"""Put two herding detectors head to head on the hidden, balanced kind of crowd, where net order
flow looks calm and the net-direction detector goes blind. Run from the repo root:

    PYTHONPATH=src python examples/compare_detectors.py
"""

import numpy as np

from corral.herding.detector import HerdingDetector
from corral.herding.rmt import MarketModeDetector
from corral.validation.benchmark import calibrate_threshold, compare_detectors, evaluate
from corral.validation.simulator import simulate_run


def make(rng, n, cascade, n_agents, n_timesteps):
    runs = []
    for _ in range(n):
        if cascade:
            onset = int(rng.integers(45, 60))
            event = min(onset + int(rng.integers(28, 42)), n_timesteps - 5)
            runs.append(simulate_run(n_agents, n_timesteps, cascade=True, onset=onset, event=event,
                                     herd_mode="balanced", herd_fraction=0.6, w_max=0.8, rng=rng))
        else:
            runs.append(simulate_run(n_agents, n_timesteps, cascade=False, rng=rng))
    return runs


def main():
    rng = np.random.default_rng(20260607)
    n_agents, n_timesteps, window = 30, 140, 25

    normal = make(rng, 120, False, n_agents, n_timesteps)
    cascades = make(rng, 120, True, n_agents, n_timesteps)
    calib, test_normal = normal[:60], normal[60:]

    align = HerdingDetector(null_resamples=120, random_state=0).fit(calib[0][0])
    align.threshold = calibrate_threshold(align, calib, window, target_far=0.05)

    rmt = MarketModeDetector(null_resamples=120, random_state=0).fit(calib[0][0])
    rmt.threshold = calibrate_threshold(rmt, calib, window, target_far=0.05)

    print("hidden 'balanced' herd, where net order flow stays calm:")
    for name, det in [("alignment (direction)", align), ("market-mode (correlation)", rmt)]:
        m = evaluate(det, test_normal + cascades, window)
        print(f"  {name:26} detection {m['detection_rate']:.0%}, "
              f"false alarms {m['false_alarm_rate']:.0%}")

    c = compare_detectors(rmt, align, cascades, window, random_state=0)
    lo, hi = c["ci95"]
    print()
    print(f"market-mode minus alignment, detection gap: {c['gap']:+.0%} "
          f"(95% CI {lo:+.0%} to {hi:+.0%}, resampling runs)")


if __name__ == "__main__":
    main()
