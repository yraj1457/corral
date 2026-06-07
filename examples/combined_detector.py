"""Show the combined watchdog catching every kind of herd at one controlled false-alarm rate, while
each single detector only catches its own kind. Run from the repo root:

    PYTHONPATH=src python examples/combined_detector.py
"""

import numpy as np

from corral.herding.detector import HerdingDetector
from corral.herding.ensemble import HerdingEnsemble
from corral.herding.rmt import MarketModeDetector
from corral.validation.benchmark import calibrate_threshold, evaluate
from corral.validation.simulator import simulate_run


def cascades(rng, n, mode, n_agents, n_timesteps):
    runs = []
    for _ in range(n):
        onset = int(rng.integers(45, 60))
        event = min(onset + int(rng.integers(28, 42)), n_timesteps - 5)
        runs.append(simulate_run(n_agents, n_timesteps, cascade=True, onset=onset, event=event,
                                 herd_mode=mode, herd_fraction=0.7, w_max=0.8, rng=rng))
    return runs


def fitted(cls, panel, calib, window, far):
    det = cls(null_resamples=80, random_state=0).fit(panel)
    det.threshold = calibrate_threshold(det, calib, window, target_far=far)
    return det


def main():
    rng = np.random.default_rng(20260607)
    n_agents, n_timesteps, window = 30, 140, 25

    normal = [simulate_run(n_agents, n_timesteps, cascade=False, rng=rng) for _ in range(300)]
    directional = cascades(rng, 80, "directional", n_agents, n_timesteps)
    balanced = cascades(rng, 80, "balanced", n_agents, n_timesteps)
    calib, test_normal = normal[:200], normal[200:]
    panel = calib[0][0]

    align = fitted(HerdingDetector, panel, calib, window, 0.05)
    rmt = fitted(MarketModeDetector, panel, calib, window, 0.05)
    # the combined watchdog splits the budget across its two members, then fires on either
    combined = HerdingEnsemble([
        fitted(HerdingDetector, panel, calib, window, 0.025),
        fitted(MarketModeDetector, panel, calib, window, 0.025),
    ])

    print("detector       directional  balanced  overall  false-alarms")
    for name, det in [("alignment", align), ("market-mode", rmt), ("combined", combined)]:
        d = evaluate(det, directional, window)["detection_rate"]
        b = evaluate(det, balanced, window)["detection_rate"]
        o = evaluate(det, directional + balanced, window)["detection_rate"]
        far = evaluate(det, test_normal, window)["false_alarm_rate"]
        print(f"{name:14} {d:>10.0%}  {b:>7.0%}  {o:>6.0%}  {far:>11.0%}")


if __name__ == "__main__":
    main()
