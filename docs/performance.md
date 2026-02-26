# Performance & scaling

This page summarizes the cost model for thermodynamic sweeps, memory behavior, and practical HPC sizing.

## Computational complexity

For a sweep over `N_configs` doped configurations and `N_temperatures` temperature points, the dominant work is evaluating Boltzmann factors and reductions for each `(config, T)` pair.

\[
\text{Time} = O(N_{\text{configs}} \times N_{\text{temperatures}})
\]

A useful engineering approximation is:

- total floating-point work grows linearly with number of configurations
- total floating-point work grows linearly with number of temperatures
- doubling both multiplies runtime by ~4x

## Memory considerations

Memory has two common regimes:

1. **Streaming / chunked reduction (preferred):**
   - store energies (`O(N_configs)`) and one working temperature slice
   - effective memory near `O(N_configs)`

2. **Fully materialized sweep arrays:**
   - store full matrix of intermediates/results over `(config, T)`
   - memory `O(N_configs × N_temperatures)`

In practice, materializing all intermediates can dominate RAM long before compute becomes the bottleneck.

## Lightweight benchmark

Use the built-in benchmark command to time synthetic thermo sweeps in-memory:

```bash
ganmg bench thermo --n 5000 --nT 50
```

By default it writes JSON to `outputs/bench.json` (or a custom path using `--out`). The command prints a timing summary and records parameters, timings, a small result snapshot, and environment metadata.

## Mapping to HPC batch sizes

You can estimate job sizes directly from the complexity model:

- **Single-job work unit** ≈ `N_configs × N_temperatures`
- keep work units similar across batch jobs for better queue predictability
- increase `N_temperatures` for denser phase-resolution studies, and split `N_configs` across array tasks when wall-time limits are strict

A practical batching strategy:

1. run a local benchmark to measure `time_per_temperature_ms` at representative `N_configs`
2. convert cluster wall-time budget into a max work unit
3. partition configuration lists into chunks that fit the budget
4. launch array jobs where each task processes one chunk over the same temperature grid

This keeps throughput steady while preserving reproducibility of each batch artifact.
