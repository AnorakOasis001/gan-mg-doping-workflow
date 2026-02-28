# Results demo

This workflow builds reproducible thermodynamic outputs for Mg-doped GaN by deriving mixing datasets, computing Gibbs quantities over temperature, estimating bootstrap uncertainty bands, and converting vn-vs-mgi comparisons into interpretable preference maps.

Key output artifacts (example run-local paths):

- `runs/<id>/figures/overlay_dGmix_vs_doping_multiT_ci.png`
- `runs/<id>/figures/phase_map_preference.png` (or `runs/<id>/figures/crossover_map_uncertainty.png` if using crossover-centric view)
- `repropack/<id>.zip`

Interpretation guide:

- A negative `ΔG = Gmix(vn) - Gmix(mgi)` indicates vn is thermodynamically preferred; positive indicates mgi.
- Robust preference requires uncertainty bounds that do **not** straddle zero.
- Regions labeled `uncertain` indicate overlap in confidence bounds, so neither mechanism is conclusively favored.
- Low effective sample size (ESS) and high `weight_max` indicate finite-size sensitivity (few microstates dominate).
- Use the phase-map view together with the CI overlay to separate trend direction from statistical confidence.
