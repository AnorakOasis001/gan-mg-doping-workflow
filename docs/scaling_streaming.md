# Streaming thermodynamics for large CSVs

For very large `results.csv` files, loading every energy value into memory can become the dominant bottleneck.
The `analyze` command now supports a chunked pathway that processes the file incrementally and keeps memory usage effectively constant with respect to file size.

## Why chunked streaming is useful

- **Scalability:** works with CSVs that are larger than available memory.
- **Stability:** computes partition statistics with numerically stable log-sum-exp updates.
- **Compatibility:** keeps the same CLI command and output schema (`ThermoResult`).

## How streaming log-sum-exp works

The canonical partition function uses:

\[
Z = \sum_i e^{x_i}, \quad x_i = -\Delta E_i / (k_B T)
\]

Directly summing `exp(x_i)` can overflow/underflow. The streaming implementation tracks two values:

- `m`: running maximum value seen so far
- `s`: scaled sum such that `logZ = m + log(s)`

For each chunk with local maximum `m2` and local scaled sum `s2 = sum(exp(x_chunk - m2))`:

- if `m2 <= m`: `s = s + exp(m2 - m) * s2`
- else: `s = exp(m - m2) * s + s2`, then `m = m2`

This yields the same `logZ` as in-memory log-sum-exp while avoiding large intermediate exponentials.

## CLI usage

Use the existing command with an optional chunk size:

```bash
gan-mg analyze --chunksize 200000
```

You can combine it with existing options such as `--csv`, `--T`, and `--energy-col`.

By default, analysis reads the `energy_eV` column. If your CSV uses a different column name (for example legacy `mixing_energy_eV`), pass it explicitly with `--energy-col mixing_energy_eV`.

## Result equivalence

Streaming and in-memory analysis are mathematically identical for the same inputs. The difference is only in I/O and numerical evaluation pathway (chunked vs whole-array loading).
