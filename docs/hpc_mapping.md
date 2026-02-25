# Mapping to Real Mg-doped GaN + MACE on HPC

This page explains how to connect this workflow to a real HPC relaxation/energy pipeline for Mg-doped GaN.

## Where Janus-core / MACE plugs in

In this repository, the thermodynamics steps (`analyze`, `sweep`) consume a canonical table at:

- `runs/<id>/inputs/results.csv`

For a real workflow, Janus-core + MACE is the **energy + relaxation backend** that produces those per-configuration energies.

### Practical integration point

1. Generate candidate Mg-doped GaN structures (locally or on cluster).
2. Run structure relaxation and final energy evaluation on HPC using Janus-core/MACE.
3. Export one row per configuration with at least:
   - `structure_id`
   - `mechanism`
   - `energy_eV`
4. Import this table via:

```bash
ganmg import --run-id <id> --results <hpc-results.csv>
```

After import, the existing CLI analysis is unchanged:

```bash
ganmg analyze --run-id <id> --temperature 1000
ganmg sweep --run-id <id> --tmin 300 --tmax 1200 --tstep 100
```

## What artifacts come from HPC

Typical HPC outputs that should be retained and mapped into the run directory:

- Relaxed structures (for example, POSCAR/CIF/EXTXYZ per config or combined file)
- Final total energies used for thermodynamics (`energy_eV`)
- Job/runtime metadata (SLURM job id, array index/task id, walltime, node/partition)
- Calculator metadata (MACE model/version, cutoff, precision/device, software versions)

At minimum, the imported `results.csv` must contain required thermodynamics columns. Extra metadata can be preserved in additional columns and/or sidecar files.

## Mapping HPC jobs to the run system

The local run layout remains the source of truth:

- `runs/<id>/inputs/results.csv`: canonical table used by `analyze` and `sweep`
- `runs/<id>/inputs/external_results.<ext>`: source copy from HPC import
- `runs/<id>/inputs/import_metadata.json`: import provenance
- `runs/<id>/run.json`: run-level metadata
- `runs/<id>/outputs/metrics.json`: standardized result metrics and reproducibility hash

A good mental model is:

- One workflow run (`<id>`) = one experiment definition
- One SLURM array task = one configuration (or one batch chunk) evaluated on HPC
- Array outputs are merged into one `results.csv`, then imported into `runs/<id>/inputs/`

## Reproducibility hash and SLURM arrays

`reproducibility_hash` is computed from:

- input CSV bytes (`results.csv` content)
- analysis temperature grid
- code version

For HPC array usage, keep these stable mapping keys to make runs reproducible and traceable:

- `run_id`: experiment identifier
- `seed`: structure generation/randomization seed (if used)
- `config_set`: deterministic set/version of configurations dispatched to the array

Recommended pattern:

- Treat `(run_id, seed, config_set)` as the logical array campaign identity.
- Record array-specific fields (`slurm_job_id`, `slurm_array_task_id`) per row or in sidecar metadata.
- Recreate identical `results.csv` ordering/content for strict reproducibility-hash stability.

## Real workflow outline

1. **Generate configurations** locally or on the cluster.
2. **Relax and evaluate energies on HPC** with Janus-core/MACE (often via SLURM arrays).
3. **Import results back** into `runs/<id>/inputs/results.csv` with `ganmg import`.
4. **Run `analyze` / `sweep`** locally or in a CI-like environment for reporting and regression checks.

## Data contract for `results.csv`

| Column | Required | Type | Description |
|---|---|---|---|
| `structure_id` | Yes | string | Stable identifier for each Mg-doped GaN configuration. |
| `mechanism` | Yes | string | Label/category for configuration origin or mechanism. |
| `energy_eV` | Yes | float | Final relaxed energy in eV used by thermodynamics. |
| `seed` | Optional | integer | Seed used to generate/scramble the configuration (if applicable). |
| `config_set` | Optional | string | Deterministic config-set/version tag used for the HPC campaign. |
| `slurm_job_id` | Optional | string/integer | SLURM job id that produced the record. |
| `slurm_array_task_id` | Optional | integer | SLURM array task index that produced the record. |
| `relaxed_structure_ref` | Optional | string | Public path/URI or artifact key for the relaxed structure file. |
| `model_name` | Optional | string | MACE model identifier used for this energy. |
| `model_version` | Optional | string | Model version/checkpoint tag used for this energy. |
| `calc_metadata_json` | Optional | string (JSON) | Serialized calculator/runtime metadata for traceability. |

Notes:

- Extra columns are allowed; thermodynamics currently requires the three required columns above.
- Keep units explicit (`energy_eV`) and identifiers stable across reruns.
