from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from gan_mg.analysis.thermo import (
    ThermoResult,
    boltzmann_thermo_from_csv,
    sweep_thermo_from_csv,
)


@dataclass(frozen=True)
class AnalyzeResponse:
    csv_path: Path
    energy_col: str
    result: ThermoResult


@dataclass(frozen=True)
class SweepResponse:
    csv_path: Path
    energy_col: str
    results: tuple[ThermoResult, ...]


def analyze_from_csv(
    csv_path: str | Path,
    temperature_K: float,
    energy_col: str = "energy_eV",
) -> AnalyzeResponse:
    """Analyze a results CSV at a single temperature.

    This API is side-effect free: it does not write files, print, or configure logging.
    """
    path = Path(csv_path)
    result = boltzmann_thermo_from_csv(path, T=temperature_K, energy_col=energy_col)
    return AnalyzeResponse(csv_path=path, energy_col=energy_col, result=result)


def sweep_from_csv(
    csv_path: str | Path,
    temperatures_K: Iterable[float],
    energy_col: str = "energy_eV",
) -> SweepResponse:
    """Analyze a results CSV over multiple temperatures.

    This API is side-effect free: it does not write files, print, or configure logging.
    """
    path = Path(csv_path)
    t_values = [float(value) for value in temperatures_K]
    rows = sweep_thermo_from_csv(path, t_values, energy_col=energy_col)

    results = tuple(
        ThermoResult(
            temperature_K=row["temperature_K"],
            num_configurations=row["num_configurations"],
            mixing_energy_min_eV=row["mixing_energy_min_eV"],
            mixing_energy_avg_eV=row["mixing_energy_avg_eV"],
            partition_function=row["partition_function"],
            free_energy_mix_eV=row["free_energy_mix_eV"],
        )
        for row in rows
    )

    return SweepResponse(csv_path=path, energy_col=energy_col, results=results)
