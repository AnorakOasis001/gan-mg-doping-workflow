from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np

from gan_mg.analysis.thermo import boltzmann_thermo_from_csv, thermo_from_csv_streaming

STABLE_FIELDS = (
    "temperature_K",
    "num_configurations",
    "mixing_energy_min_eV",
    "mixing_energy_avg_eV",
    "partition_function",
    "free_energy_mix_eV",
)


def _result_to_dict(result: Any) -> dict[str, float | int]:
    return {
        "temperature_K": result.temperature_K,
        "num_configurations": result.num_configurations,
        "mixing_energy_min_eV": result.mixing_energy_min_eV,
        "mixing_energy_avg_eV": result.mixing_energy_avg_eV,
        "partition_function": result.partition_function,
        "free_energy_mix_eV": result.free_energy_mix_eV,
    }


def _assert_results_close(
    in_memory: dict[str, float | int],
    streaming: dict[str, float | int],
    *,
    rtol: float = 1e-12,
    atol: float = 1e-15,
) -> None:
    for field in STABLE_FIELDS:
        in_memory_value = in_memory[field]
        streaming_value = streaming[field]

        if isinstance(in_memory_value, int):
            if not isinstance(streaming_value, int) or in_memory_value != streaming_value:
                raise ValueError(
                    f"Parity check failed for '{field}': "
                    f"in-memory={in_memory_value!r}, streaming={streaming_value!r}"
                )
            continue

        if not math.isclose(
            float(in_memory_value),
            float(streaming_value),
            rel_tol=rtol,
            abs_tol=atol,
        ):
            raise ValueError(
                f"Parity check failed for '{field}': "
                f"in-memory={in_memory_value!r}, streaming={streaming_value!r}, "
                f"rtol={rtol}, atol={atol}"
            )


def _write_synthetic_csv(csv_path: Path, rows: int, rng: np.random.Generator) -> None:
    # A light-tailed Gaussian around realistic eV-scale values keeps parity stable.
    energies = rng.normal(loc=-1.0, scale=0.25, size=rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["structure_id", "mechanism", "energy_eV"])
        for idx, energy in enumerate(energies):
            writer.writerow([f"struct_{idx:08d}", "synthetic", f"{float(energy):.16e}"])


def _timed_call(func: Any, *, measure_memory: bool) -> tuple[Any, float, float | None]:
    if measure_memory:
        tracemalloc.start()
    start = time.perf_counter()
    result = func()
    elapsed_s = time.perf_counter() - start
    peak_mib: float | None = None

    if measure_memory:
        _, peak_bytes = tracemalloc.get_traced_memory()
        peak_mib = peak_bytes / (1024.0 * 1024.0)
        tracemalloc.stop()

    return result, elapsed_s, peak_mib


def _plot_runtime(records: list[dict[str, Any]], output_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [int(record["rows"]) for record in records]
    in_memory_times = [float(record["time_in_memory_s"]) for record in records]
    streaming_times = [float(record["time_streaming_s"]) for record in records]

    plt.figure()
    plt.plot(rows, in_memory_times, marker="o", label="in-memory")
    plt.plot(rows, streaming_times, marker="o", label="streaming")
    plt.xlabel("Rows")
    plt.ylabel("Runtime (s)")
    plt.title("Thermodynamics runtime vs dataset size")
    plt.legend()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark in-memory vs streaming thermodynamic analysis on synthetic CSV inputs."
        )
    )
    parser.add_argument("--rows", nargs="+", type=int, default=[10_000, 100_000, 1_000_000])
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=Path, default=Path("results/benchmarks"))
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--measure-memory", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if any(rows <= 0 for rows in args.rows):
        raise ValueError("All --rows values must be > 0.")
    if args.temperature <= 0:
        raise ValueError("--temperature must be > 0.")
    if args.chunksize <= 0:
        raise ValueError("--chunksize must be > 0.")

    rng = np.random.default_rng(args.seed)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "thermo_benchmarks.jsonl"

    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="thermo_bench_") as temp_dir:
        temp_root = Path(temp_dir)

        for rows in args.rows:
            csv_path = temp_root / f"synthetic_{rows}.csv"
            _write_synthetic_csv(csv_path, rows, rng)

            in_memory_result, time_in_memory_s, in_memory_peak_mib = _timed_call(
                lambda: boltzmann_thermo_from_csv(csv_path, T=args.temperature, energy_col="energy_eV"),
                measure_memory=args.measure_memory,
            )
            streaming_result, time_streaming_s, streaming_peak_mib = _timed_call(
                lambda: thermo_from_csv_streaming(
                    csv_path,
                    temperature_K=args.temperature,
                    energy_column="energy_eV",
                    chunksize=args.chunksize,
                ),
                measure_memory=args.measure_memory,
            )

            in_memory_dict = _result_to_dict(in_memory_result)
            streaming_dict = _result_to_dict(streaming_result)
            _assert_results_close(in_memory_dict, streaming_dict)

            speedup = time_in_memory_s / time_streaming_s if time_streaming_s > 0 else float("inf")
            record: dict[str, Any] = {
                "rows": rows,
                "temperature_K": args.temperature,
                "chunksize": args.chunksize,
                "time_in_memory_s": time_in_memory_s,
                "time_streaming_s": time_streaming_s,
                "speedup": speedup,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "parity_passed": True,
            }
            if args.measure_memory:
                record["peak_memory_in_memory_mib"] = in_memory_peak_mib
                record["peak_memory_streaming_mib"] = streaming_peak_mib

            records.append(record)

    with summary_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    if args.plot:
        _plot_runtime(records, outdir / "runtime_vs_rows.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
