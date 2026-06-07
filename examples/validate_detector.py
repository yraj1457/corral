"""Validate the herding detector against synthetic cascades, end to end. Run from the repo root:

    PYTHONPATH=src python examples/validate_detector.py
"""

import numpy as np

from corral.herding.detector import HerdingDetector
from corral.validation.benchmark import calibrate_threshold, evaluate
from corral.validation.simulator import simulate_run
from corral.validation.stats import bootstrap_ci


def main():
    rng = np.random.default_rng(20260607)
    n_agents, n_timesteps, window = 40, 140, 15

    normal = [simulate_run(n_agents, n_timesteps, cascade=False, rng=rng) for _ in range(120)]
    cascades = []
    for _ in range(120):
        onset = int(rng.integers(45, 60))
        event = min(onset + int(rng.integers(28, 42)), n_timesteps - 5)
        cascades.append(
            simulate_run(n_agents, n_timesteps, cascade=True, onset=onset, event=event, rng=rng)
        )
    calib, test_normal = normal[:60], normal[60:]

    # learn the chance baseline, then set the threshold for a 5% false-alarm rate
    det = HerdingDetector(null_resamples=120, random_state=0).fit(calib[0][0])
    det.threshold = calibrate_threshold(det, calib, window, target_far=0.05)

    m = evaluate(det, test_normal + cascades, window)
    dr, dr_lo, dr_hi = bootstrap_ci(m["detected_flags"], random_state=0)
    lead, lead_lo, lead_hi = bootstrap_ci(m["leads"], random_state=0)

    print(f"cascades: {m['n_cascade']}   normal held out: {m['n_normal']}")
    print(f"detection rate:    {dr:.0%}  (95% CI {dr_lo:.0%} to {dr_hi:.0%})")
    print(f"false-alarm rate:  {m['false_alarm_rate']:.0%}  (target 5%)")
    print(f"lead time, steps:  mean {lead:.1f}  (95% CI {lead_lo:.1f} to {lead_hi:.1f})")


if __name__ == "__main__":
    main()
