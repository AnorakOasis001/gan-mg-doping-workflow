# Output contract validation

The repository includes lightweight, deterministic JSON output validation in `gan_mg.validation`.

## What is validated

- Required keys for each output kind (`thermo_summary`, `metrics`, `diagnostics`, `run_manifest`)
- JSON primitive/object/array types
- Finite numeric values (no `NaN`/`inf`)
- Universal invariants already implied by the workflow, including:
  - `num_configurations >= 1`
  - `partition_function > 0`
  - probability-like diagnostics field `p_min` in `[0, 1]`

Unknown/additive fields are allowed by design so existing contracts can evolve additively.

## Integration points

- Golden generator (`scripts/generate_golden.py`) validates each JSON payload before writing.
- Config-driven runs (`gan_mg.run_config`) validate produced JSON files.
- CLI `analyze` can validate outputs additively with:
  - `--validate-output`
  - `GAN_MG_VALIDATE=1`

This keeps default CLI behavior unchanged unless opt-in validation is enabled.
