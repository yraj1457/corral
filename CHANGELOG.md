# Changelog

## 0.1.1
Documentation only. Corrected the README version and added the PyPI install command. No code changes.

## 0.1.0
First public release. A black-box safety layer for fleets of autonomous trading agents.

- Audit: append-only hash-chained log with RFC 6962 Merkle inclusion and consistency proofs.
- Authorization: deny-by-default policy gate with a kill switch.
- Detection: net-direction and RMT market-mode herding detectors, combined under one false-alarm budget.
- Early warning: Kritzman-Li absorption ratio.
- Validation: cascade simulator, detector benchmark, honest statistics.
