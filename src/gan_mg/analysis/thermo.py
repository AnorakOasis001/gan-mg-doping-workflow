from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    import pandas as pd

K_B_EV_PER_K = 8.617333262e-5  # Boltzmann constant in eV/K


@dataclass(frozen=True)
class ThermoResult:
    temperature_K: float
    num_configurations: int
    mixing_energy_min_eV: float
    mixing_energy_avg_eV: float
    partition_function: float
    free_energy_mix_eV: float


class ThermoSweepRow(TypedDict):
    temperature_K: float
    num_configurations: int
    mixing_energy_min_eV: float
    mixing_energy_avg_eV: float
    partition_function: float
    free_energy_mix_eV: float


REQUIRED_RESULTS_COLUMNS = ("structure_id", "mechanism", "energy_eV")


def validate_results_dataframe(df: "pd.DataFrame") -> None:
    """
    Validate the expected thermodynamic input table.

    Checks:
    - required columns are present
    - at least one row exists
    - no NaN values in required columns
    - energy_eV is numeric
    """
    import pandas as pd

    if df.empty:
        raise ValueError("results.csv must contain at least 1 row.")

    missing_columns = [column for column in REQUIRED_RESULTS_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "results.csv is missing required columns: "
            f"{', '.join(missing_columns)}"
        )

    required_frame = df.loc[:, REQUIRED_RESULTS_COLUMNS]
    nan_columns = required_frame.columns[required_frame.isna().any()].tolist()
    if nan_columns:
        raise ValueError(
            "results.csv contains NaN values in required columns: "
            f"{', '.join(nan_columns)}"
        )

    energy_as_numeric = pd.to_numeric(required_frame["energy_eV"], errors="coerce")
    if energy_as_numeric.isna().any():
        raise ValueError("Column 'energy_eV' must contain numeric values.")


def read_energies_csv(csv_path: Path, energy_col: str = "energy_eV") -> list[float]:
    """
    Read energies (eV) from a CSV file.
    Expects a column named `energy_col` (default: energy_eV).
    """
    import pandas as pd

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]
        fieldnames = [] if reader.fieldnames is None else list(reader.fieldnames)

    df = pd.DataFrame(rows, columns=fieldnames)
    validate_results_dataframe(df)

    if energy_col not in df.columns:
        raise ValueError(f"results.csv must contain column '{energy_col}'.")

    energy_series = pd.to_numeric(df[energy_col], errors="coerce")
    if energy_series.isna().any():
        raise ValueError(f"Column '{energy_col}' must contain numeric values and no NaN entries.")

    return [float(value) for value in energy_series.astype(float).tolist()]


def boltzmann_thermo_from_energies(energies: list[float], T: float) -> ThermoResult:
    """
    Compute basic canonical thermodynamics from a list of energies (eV) at temperature T (K).
    Uses a numerically-stable energy shift by Emin.

    Returns:
      - partition_function = sum_i exp(-beta*(Ei - Emin))
      - mixing_energy_avg_eV in eV
      - free_energy_mix_eV in eV, where
        free_energy_mix_eV = Emin - (1/beta)*ln(partition_function)
    """
    if T <= 0:
        raise ValueError("Temperature T must be > 0.")

    beta = 1.0 / (K_B_EV_PER_K * T)
    emin = min(energies)

    shifted = [e - emin for e in energies]
    weights = [math.exp(-beta * e) for e in shifted]
    partition_function = sum(weights)

    probs = [w / partition_function for w in weights]
    mixing_energy_avg_eV = sum(p * e for p, e in zip(probs, energies))

    free_energy_mix_eV = emin - (1.0 / beta) * math.log(partition_function)

    return ThermoResult(
        temperature_K=T,
        num_configurations=len(energies),
        mixing_energy_min_eV=emin,
        mixing_energy_avg_eV=mixing_energy_avg_eV,
        partition_function=partition_function,
        free_energy_mix_eV=free_energy_mix_eV,
    )


def boltzmann_thermo_from_csv(csv_path: Path, T: float, energy_col: str = "energy_eV") -> ThermoResult:
    energies = read_energies_csv(csv_path, energy_col=energy_col)
    return boltzmann_thermo_from_energies(energies, T=T)


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
    energies = read_energies_csv(csv_path, energy_col=energy_col)
    rows: list[ThermoSweepRow] = []

    for T in T_values:
        res = boltzmann_thermo_from_energies(energies, T=T)

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
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "temperature_K",
        "num_configurations",
        "mixing_energy_min_eV",
        "mixing_energy_avg_eV",
        "partition_function",
        "free_energy_mix_eV",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_thermo_vs_T(rows: list[ThermoSweepRow], out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")  # headless backend (CI-safe)
    import matplotlib.pyplot as plt

    temperature_K = [r["temperature_K"] for r in rows]
    free_energy_mix_eV = [r["free_energy_mix_eV"] for r in rows]
    mixing_energy_avg_eV = [r["mixing_energy_avg_eV"] for r in rows]

    plt.figure()
    plt.plot(temperature_K, free_energy_mix_eV, label="free_energy_mix_eV")
    plt.plot(temperature_K, mixing_energy_avg_eV, label="mixing_energy_avg_eV")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Energy (eV)")
    plt.legend()

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
