# Contributing to corral

Thanks for your interest in contributing to corral. Contributions of all
kinds are welcome, including bug reports, feature requests, documentation
improvements, and code.

## Installing corral

If you just want to use the library, install the published package from PyPI.

```
pip install corral-fleet
```

It imports as `corral`. The development install described below is only needed
if you intend to modify the code.

## Reporting issues

If you find a bug or unexpected behaviour, please open an issue on the
[issue tracker](https://github.com/yraj1457/corral/issues). A useful report
includes the following.

- what you did (ideally a minimal code snippet that reproduces the problem),
- what you expected to happen,
- what actually happened (including the full traceback, if any),
- your Python version and `corral` version (`pip show corral-fleet`).

## Requesting features

Feature requests are welcome as issues. Please describe the use case you have
in mind, not only the proposed solution, so the design can be discussed in
context.

## Contributing code

1. Fork the repository and create a branch for your change.
2. Set up a development environment.

   ```bash
   git clone https://github.com/<your-username>/corral.git
   cd corral
   python -m venv .venv && source .venv/bin/activate
   pip install -e ".[dev]"
   ```

3. Make your change. Please keep the public API consistent with the existing
   `scikit-learn`-style interface (`fit` / `decision_function` and the like),
   and add or update docstrings for anything user-facing.
4. Add tests covering your change.
5. Run the test suite and make sure it passes.

   ```bash
   pytest
   ```

6. Commit, push to your fork, and open a pull request describing the change
   and the motivation for it.

## Coding conventions

- Follow the style of the surrounding code.
- Public functions and classes should have docstrings describing parameters,
  return values, and behaviour.
- New functionality should come with tests; bug fixes should come with a test
  that fails before the fix and passes after.

## Scope and safety note

corral is a safety and supervision layer. Its herding detection has so far been
evaluated on synthetic data only; contributions that improve or extend
validation (for example, evaluation harnesses against realistic market
simulators) are particularly welcome. Please do not overstate detector
performance in documentation or examples beyond what the evaluation supports.

## Getting help

If you have a question that is not a bug report, you can open an issue with the
"question" label on the [issue tracker](https://github.com/yraj1457/corral/issues).
For anything else, contact the maintainer at the address listed in the
repository's `CITATION.cff`.

## Code of conduct

Please be respectful and constructive in all project spaces. Harassment or
abusive behaviour will not be tolerated.
