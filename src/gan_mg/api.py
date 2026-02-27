from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from gan_mg.analysis.thermo import (
    ThermoResult,
    boltzmann_thermo_from_csv,
    sweep_thermo_from_csv,
    write_thermo_vs_T_csv,
)

logger = logging.getLogger(__name__)


def analyze_from_csv(csv_path: Path, temperature_K: float, energy_col: str = "energy_eV") -> ThermoResult:
    """Analyze a results CSV at a single temperature."""
    csv_path = Path(csv_path)
    logger.debug("Analyzing thermodynamics from CSV: path=%s temperature_K=%s", csv_path, temperature_K)
    return boltzmann_thermo_from_csv(csv_path, T=temperature_K, energy_col=energy_col)


def sweep_from_csv(
    csv_path: Path,
    temperatures_K: Sequence[float],
    energy_col: str = "energy_eV",
) -> list[ThermoResult]:
    """Analyze a results CSV over multiple temperatures."""
    csv_path = Path(csv_path)
    sorted_rows = sweep_thermo_from_csv(
        csv_path,
        [float(value) for value in temperatures_K],
        energy_col=energy_col,
    )
    logger.debug(
        "Swept thermodynamics from CSV: path=%s n_temperatures=%d",
        csv_path,
        len(sorted_rows),
    )
    return [
        ThermoResult(
            temperature_K=row["temperature_K"],
            num_configurations=row["num_configurations"],
            mixing_energy_min_eV=row["mixing_energy_min_eV"],
            mixing_energy_avg_eV=row["mixing_energy_avg_eV"],
            partition_function=row["partition_function"],
            free_energy_mix_eV=row["free_energy_mix_eV"],
        )
        for row in sorted_rows
    ]


def analyze_run(
    run_dir: Path,
    run_id: str,
    temperature_K: float,
    energy_col: str = "energy_eV",
) -> ThermoResult:
    """Analyze a run folder by loading ``<run_dir>/<run_id>/inputs/results.csv``."""
    csv_path = Path(run_dir) / run_id / "inputs" / "results.csv"
    logger.debug("Analyzing run thermodynamics: run_dir=%s run_id=%s", run_dir, run_id)
    return analyze_from_csv(csv_path=csv_path, temperature_K=temperature_K, energy_col=energy_col)


def sweep_run(
    run_dir: Path,
    run_id: str,
    temperatures_K: Sequence[float],
    energy_col: str = "energy_eV",
) -> Path:
    """Sweep temperatures for a run and write ``thermo_vs_T.csv`` in run outputs."""
    run_root = Path(run_dir) / run_id
    csv_path = run_root / "inputs" / "results.csv"
    out_csv = run_root / "outputs" / "thermo_vs_T.csv"

    rows = sweep_thermo_from_csv(
        csv_path,
        [float(value) for value in temperatures_K],
        energy_col=energy_col,
    )
    write_thermo_vs_T_csv(rows, out_csv)
    logger.info("Wrote thermo sweep CSV: %s", out_csv)
    return out_csv
