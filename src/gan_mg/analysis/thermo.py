from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from gan_mg.artifacts import write_thermo_vs_T_csv as write_thermo_vs_t_artifact_csv
from gan_mg.science.constants import K_B_EV_PER_K
from gan_mg.science.streaming import LogSumExpAccumulator, RunningStats, ScaledExpSumAccumulator
from gan_mg.science.thermo import (
    LOGGER,
    ThermoDiagnostics,
    ThermoResult,
    ThermoSweepRow,
    boltzmann_diagnostics_from_energies,
    boltzmann_thermo_from_energies,
    free_energy_from_logZ,
    log_partition_function,
)

if TYPE_CHECKING:
    import pandas as pd

REQUIRED_RESULTS_COLUMNS = ("structure_id", "mechanism", "energy_eV")

__all__ = [
    "K_B_EV_PER_K",
    "LOGGER",
    "LogSumExpAccumulator",
    "RunningStats",
    "ScaledExpSumAccumulator",
    "ThermoDiagnostics",
    "ThermoResult",
    "ThermoSweepRow",
    "REQUIRED_RESULTS_COLUMNS",
    "log_partition_function",
    "free_energy_from_logZ",
    "validate_results_dataframe",
    "read_energies_csv",
    "boltzmann_thermo_from_energies",
    "boltzmann_diagnostics_from_energies",
    "boltzmann_thermo_from_csv",
    "diagnostics_from_csv_streaming",
    "thermo_from_csv_streaming",
    "write_thermo_txt",
    "sweep_thermo_from_csv",
    "write_thermo_vs_T_csv",
    "plot_thermo_vs_T",
]


def validate_results_dataframe(df: "pd.DataFrame") -> None:
    from gan_mg.io.results_csv import validate_results_dataframe as impl

    return impl(df)


def read_energies_csv(csv_path: Path, energy_col: str = "energy_eV") -> list[float]:
    from gan_mg.io.results_csv import read_energies_csv as impl

    return impl(csv_path=csv_path, energy_col=energy_col)


def boltzmann_thermo_from_csv(csv_path: Path, T: float, energy_col: str = "energy_eV") -> ThermoResult:
    energies = read_energies_csv(csv_path=csv_path, energy_col=energy_col)
    return boltzmann_thermo_from_energies(energies=energies, T=T)


def diagnostics_from_csv_streaming(
    csv_path: Path,
    temperature_K: float,
    energy_column: str = "energy_eV",
    chunksize: int = 200_000,
) -> ThermoDiagnostics:
    from gan_mg.io.results_csv import diagnostics_from_csv_streaming as impl

    return impl(
        csv_path=csv_path,
        temperature_K=temperature_K,
        energy_column=energy_column,
        chunksize=chunksize,
    )


def thermo_from_csv_streaming(
    csv_path: Path,
    temperature_K: float,
    energy_column: str = "energy_eV",
    chunksize: int = 200_000,
) -> ThermoResult:
    from gan_mg.io.results_csv import thermo_from_csv_streaming as impl

    return impl(
        csv_path=csv_path,
        temperature_K=temperature_K,
        energy_column=energy_column,
        chunksize=chunksize,
    )


def write_thermo_txt(result: ThermoResult, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_path.write_text(
        f"temperature_K = {result.temperature_K}\n"
        f"num_configurations = {result.num_configurations}\n"
        f"mixing_energy_min_eV = {result.mixing_energy_min_eV:.6f}\n"
        f"partition_function = {result.partition_function:.6e}\n"
        f"mixing_energy_avg_eV = {result.mixing_energy_avg_eV:.6f}\n"
        f"free_energy_mix_eV = {result.free_energy_mix_eV:.6f}\n",
        encoding="utf-8",
    )


def sweep_thermo_from_csv(
    csv_path: Path,
    T_values: list[float],
    energy_col: str = "energy_eV",
) -> list[ThermoSweepRow]:
    """
    Run boltzmann_thermo_from_csv for each temperature and return list of dict rows.
    """
    csv_path = Path(csv_path)
    energies = read_energies_csv(csv_path=csv_path, energy_col=energy_col)
    rows: list[ThermoSweepRow] = []

    for T in T_values:
        res = boltzmann_thermo_from_energies(energies=energies, T=T)

        rows.append(
            {
                "temperature_K": float(T),
                "num_configurations": res.num_configurations,
                "mixing_energy_min_eV": res.mixing_energy_min_eV,
                "mixing_energy_avg_eV": res.mixing_energy_avg_eV,
                "partition_function": res.partition_function,
                "free_energy_mix_eV": res.free_energy_mix_eV,
            }
        )

    rows.sort(key=lambda r: r["temperature_K"])
    return rows


def write_thermo_vs_T_csv(rows: list[ThermoSweepRow], out_csv: Path) -> None:
    write_thermo_vs_t_artifact_csv(rows, out_csv)


def plot_thermo_vs_T(rows: list[ThermoSweepRow], out_png: Path) -> None:
    from gan_mg.viz.thermo import plot_thermo_vs_T as _plot_thermo_vs_t

    _plot_thermo_vs_t(rows=rows, out_png=out_png)
