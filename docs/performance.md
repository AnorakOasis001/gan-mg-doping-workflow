# Performance & scaling

This page summarizes the cost model for thermodynamic sweeps and practical scaling guidance.

## Computational complexity

For a sweep over `N_configs` doped configurations and `N_temperatures` temperature points, the dominant work is evaluating Boltzmann factors and reductions for each `(config, T)` pair.

\[
\text{Time} = O(N_{\text{configs}} \times N_{\text{temperatures}})
\]

A useful engineering approximation is:

- total floating-point work grows linearly with number of configurations
- total floating-point work grows linearly with number of temperatures
- doubling both multiplies runtime by ~4x

## Memory scaling

Memory has two common regimes:

1. **Streaming / chunked reduction (preferred):**
   - store energies (`O(N_configs)`) and one working temperature slice
   - effective memory near `O(N_configs)`

2. **Fully materialized sweep arrays:**
   - store full matrix of intermediates/results over `(config, T)`
   - memory `O(N_configs × N_temperatures)`

In practice, materializing all intermediates can dominate RAM long before compute becomes the bottleneck.

## Scaling outlook to 10k configurations

At `N_configs = 10,000`, scaling remains tractable if the implementation stays vectorized and avoids unnecessary materialization.

Practical expectations:

- runtime increases ~10x relative to a 1k-config baseline at fixed temperature grid
- memory remains moderate in streaming mode
- memory can become large if keeping dense per-temperature/per-config intermediates

Operational guidance:

- keep result tensors minimal (only persist outputs needed for analysis)
- process configurations in chunks if RAM pressure appears
- benchmark representative `N_temperatures` (e.g., 50 vs 200) because grid density is a first-order runtime driver

## Vectorization and parallelization strategies

Prioritize in this order:

1. **NumPy vectorization over configurations**
   - replace Python loops with array ops for Boltzmann weights and partition-function reductions
2. **Chunked vectorization**
   - batch configs into fixed-size blocks to balance cache locality and memory footprint
3. **Multi-core parallelism over independent axes**
   - split by configuration chunks or temperature blocks using multiprocessing/joblib
4. **Distributed sweeps for very large campaigns**
   - partition independent sweeps across nodes (each node handles a subset; merge summary outputs)

Rules of thumb:

- vectorization usually yields the largest single-node gain
- parallelize only after vectorized baseline is efficient
- prefer coarse-grained tasks to minimize inter-process communication overhead
