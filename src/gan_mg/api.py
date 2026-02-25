from __future__ import annotations

from pathlib import Path
from typing import Sequence

from gan_mg.analysis.thermo import ThermoResult, boltzmann_thermo_from_csv, sweep_thermo_from_csv


def analyze_from_csv(path: Path, temperature_K: float) -> ThermoResult:
    """Analyze a results CSV at a single temperature with no side effects."""
    return boltzmann_thermo_from_csv(path, T=temperature_K)


def sweep_from_csv(path: Path, temperatures: Sequence[float]) -> list[ThermoResult]:
    """Analyze a results CSV over multiple temperatures with no side effects."""
    rows = sweep_thermo_from_csv(path, [float(value) for value in temperatures])
    return [
        ThermoResult(
            temperature_K=row["temperature_K"],
            num_configurations=row["num_configurations"],
            mixing_energy_min_eV=row["mixing_energy_min_eV"],
            mixing_energy_avg_eV=row["mixing_energy_avg_eV"],
            partition_function=row["partition_function"],
            free_energy_mix_eV=row["free_energy_mix_eV"],
        )
        for row in rows
    ]
