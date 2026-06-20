---
title: "corral, an action-only safety layer for fleets of autonomous trading agents"
tags:
  - Python
  - financial markets
  - AI safety
  - autonomous agents
  - market microstructure
  - anomaly detection
authors:
  - name: Yashraj Behera
    orcid: 0009-0005-9976-2404
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 19 June 2026
bibliography: paper.bib
---

# Summary

`corral` is an open-source, `scikit-learn`-compatible Python library that acts as a
safety layer over fleets of autonomous trading agents. It treats each agent as a black
box, reading only the *action stream* the agent emits, namely its orders, cancellations,
and fills, and never the agent's model weights, prompts, or internal state. Because the
action stream is the one interface that generalises across agent architectures, a
supervisor built on it remains valid as agents are replaced or upgraded.

The library is organised around three components. An **audit** component maintains an
append-only, hash-chained log of every action; each entry binds the cryptographic hash of
its predecessor, so any later tampering is detectable, and Merkle inclusion and
consistency proofs let a third party verify that a specific action was recorded without
trusting whoever keeps the log. An **authorization** component is a deny-by-default
pre-trade gate that checks every order against declarative policy (allowed instruments,
order-size and notional limits, position caps, restricted lists) before release, with a
fleet-wide kill switch. A **detection** component scores fleet-wide herding against an
explicit chance baseline. It builds a surrogate-shuffle null that permutes each agent's own timeline
independently, destroying cross-agent synchrony while preserving each agent's marginal
behaviour, so that ordinary co-movement is not mistaken for coordination. A second,
correlation-structure detector based on random-matrix theory is provided alongside it, and
both expose the same interface.

The audit log, the authorization gate and kill switch, and both herding detectors are
implemented and covered by an automated test suite, and the herding detector is exercised
against synthetic action panels from an included cascade simulator. Because the same
project supplies both the data generator and the detector, this evaluation demonstrates
internal consistency and sensitivity rather than performance on real markets; evaluation in
a realistic limit-order-book environment such as ABIDES [@byrdetal2020], and ultimately on
live market data, is left to future work, as is a temporally-aware null that preserves each
agent's own autocorrelation. The audit log's guarantees are cryptographic
(tamper-evidence and verifiability) rather than a guarantee of the truthfulness of what an
agent reports, and the authorization gate is only as safe as the policy it is given.

# Statement of need

Autonomous, learning-based agents are increasingly deployed to place orders in financial
markets, often as fleets of many agents from different vendors, built on different model
families and updated on different schedules. Whoever is accountable for such a fleet, a
firm, a venue, or a regulator, rarely has access to each agent's internals, yet remains
responsible for the fleet's aggregate conduct and its effect on market stability. This
creates two needs that per-agent tooling does not address. The first is the ability to reconstruct and
*prove* after the fact what each agent did, and the second is the ability to tell whether a fleet is
converging on the same move (herding) rather than reacting independently to a common
signal.

`corral` addresses both from a deliberately minimal vantage point, the action stream
alone, which keeps it agnostic to how agents are constructed. Its herding detection pairs
a directional-alignment statistic with an explicit null model, so that a flag means the
fleet is more aligned than an independent-agent baseline can explain rather than merely
co-moving; the random-matrix detector follows established approaches to separating
market-wide modes from sampling noise in correlation matrices
[@lalouxetal1999; @plerouetal2002], and the directional measures connect to the empirical
literature on herding and return dispersion [@christiehuang1995; @changetal2000]. The
library is intended for researchers studying coordination and systemic risk in
agent-driven markets, and for practitioners building supervision and compliance tooling
around opaque trading agents. To our knowledge it is the first open-source package to
combine tamper-evident auditing, deny-by-default authorization, and baseline-scored
herding detection behind a single black-box action contract.

`corral` is released under the BSD-3-Clause licence and distributed on PyPI as
`corral-fleet` (importable as `corral`), so it can be installed with

```
pip install corral-fleet
```

It is developed at <https://github.com/yraj1457/corral> and archived at version 0.1.0
with DOI [10.5281/zenodo.20750017](https://doi.org/10.5281/zenodo.20750017).

# Acknowledgements

The author received no external funding for this work.

# References
