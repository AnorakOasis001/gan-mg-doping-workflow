# Output artifact contracts

This page defines lightweight, versioned artifact contracts used to prevent accidental breaking schema changes.

All contracts currently use `schema_version: 1.0` and follow additive compatibility: required fields must exist, while extra fields are allowed.

| Artifact | Schema version | Required fields | Optional fields |
|---|---|---|---|
| `derived/gibbs_summary.csv` | `1.0` | `GIBBS_SUMMARY_COLUMNS` from `gan_mg.science.gibbs` | _(none)_ |
| `derived/mechanism_crossover.csv` | `1.0` | `CROSSOVER_COLUMNS` from `gan_mg.analysis.crossover` | _(none)_ |
| `derived/gibbs_uncertainty.csv` | `1.0` | `UNCERTAINTY_COLUMNS` from `gan_mg.science.uncertainty` | _(none)_ |
| `derived/crossover_uncertainty.csv` | `1.0` | `CROSSOVER_UNCERTAINTY_COLUMNS` from `gan_mg.analysis.crossover_uncertainty` | _(none)_ |
| `derived/phase_map.csv` | `1.0` | `T_K`, `x_mg_cation`, `doping_level_percent`, `winner_mechanism`, `delta_G_eV` | `delta_G_ci_low_eV`, `delta_G_ci_high_eV`, `robust` |
| `derived/phase_boundary.csv` | `1.0` | `PHASE_BOUNDARY_COLUMNS` from `gan_mg.analysis.phase_boundary` | _(none)_ |
| `outputs/metrics.json` | `1.0` | `temperature_K`, `num_configurations`, `partition_function`, `free_energy_mix_eV` | `mixing_energy_min_eV`, `mixing_energy_avg_eV`, `created_at`, `reproducibility_hash`, `provenance` |
| `outputs/diagnostics_T*.json` | `1.0` | `temperature_K`, `num_configurations`, `expected_energy_eV`, `energy_variance_eV2`, `energy_std_eV`, `p_min`, `effective_sample_size`, `logZ_shifted`, `logZ_absolute`, `notes` | `provenance` |
| `derived/repro_manifest.json` | `1.0` | _(none; best-effort contract)_ | `created_at`, `reproducibility_hash`, `run_id`, `python_version`, `platform`, `git_commit`, `input_hash` |
| `reports/<run-id>/manifest.json` | `1.0` | `run_id`, `generated_at`, `included_files`, `sha256` | `source_run_dir`, `files`, `file_hashes` |

See `gan_mg.contracts` for the executable contract registry and `gan_mg.validation` for contract check helpers.
