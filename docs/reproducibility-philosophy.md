# Reproducibility philosophy

This project treats reproducibility as a first-class software requirement.

## Core principles

1. **Run isolation**: every execution is scoped to `runs/<run-id>/`.
2. **Determinism**: synthetic generation is controlled by explicit seeds.
3. **Stable schema**: thermodynamic outputs use consistent field names.
4. **Side-effect boundaries**: pure API functions are separated from CLI file I/O.
5. **Cross-platform behavior**: smoke scripts and CLI are usable on Linux and Windows.

## Why this matters

Scientific claims are only as strong as their ability to be reproduced. By organizing outputs per run, fixing inputs and seeds, and keeping thermodynamic logic testable, the workflow supports:

- repeatable experiments
- easier debugging
- robust collaboration
- CI automation
