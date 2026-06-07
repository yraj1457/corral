"""Measuring the herding detector against simulated cascades.

A safety tool that cannot prove it works is a liability, so this is where the detector earns its
keep. We slide it across each run, set the alarm threshold on normal runs to fix the false-alarm
rate, then on cascade runs ask the two questions that matter, does it fire before the break, and
how much warning does it give.
"""

from __future__ import annotations

import numpy as np


def _windows(n_timesteps, window):
    for end in range(window, n_timesteps + 1):
        yield end - window, end


def alarm_time(detector, X, window):
    """First window-end index whose detector score crosses detector.threshold, or None if it never
    does. Set the threshold with calibrate_threshold first.
    """
    n_timesteps = X.shape[1]
    for start, end in _windows(n_timesteps, window):
        if detector.decision_function(X[:, start:end, :])[0] >= detector.threshold:
            return end
    return None


def calibrate_threshold(detector, normal_runs, window, target_far=0.05):
    """Pick the score threshold that holds the per-run false-alarm rate near target_far on normal
    runs. A normal run false-alarms when its top-scoring window crosses the threshold, so the
    threshold is the (1 - target_far) quantile of those per-run maxima. The false-alarm rate is set
    by construction, not wished for.
    """
    run_max = []
    for X, _ in normal_runs:
        n_timesteps = X.shape[1]
        scores = (
            detector.decision_function(X[:, a:b, :])[0]
            for a, b in _windows(n_timesteps, window)
        )
        run_max.append(max(scores))
    return float(np.quantile(run_max, 1.0 - target_far))


def evaluate(detector, runs, window):
    """Run the detector (threshold already set) over every run and collect the numbers. A cascade
    run is caught only if the alarm fires at or before the event. Lead time is event minus alarm,
    and any alarm on a normal run is a false alarm. The per-run arrays come back too, so a
    confidence interval can be bootstrapped over runs.
    """
    detected_flags, leads, normal_alarms = [], [], []
    for X, event in runs:
        t = alarm_time(detector, X, window)
        if event is None:
            normal_alarms.append(1.0 if t is not None else 0.0)
        else:
            hit = t is not None and t <= event
            detected_flags.append(1.0 if hit else 0.0)
            if hit:
                leads.append(float(event - t))
    detected_flags = np.asarray(detected_flags)
    leads = np.asarray(leads)
    normal_alarms = np.asarray(normal_alarms)
    return {
        "n_cascade": int(detected_flags.size),
        "n_normal": int(normal_alarms.size),
        "detection_rate": float(detected_flags.mean()) if detected_flags.size else float("nan"),
        "false_alarm_rate": float(normal_alarms.mean()) if normal_alarms.size else float("nan"),
        "mean_lead": float(leads.mean()) if leads.size else float("nan"),
        "median_lead": float(np.median(leads)) if leads.size else float("nan"),
        "detected_flags": detected_flags,
        "leads": leads,
        "normal_alarms": normal_alarms,
    }
