from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

K_B_EV_PER_K = 8.617333262e-5  # Boltzmann constant in eV/K


@dataclass(frozen=True)
class ThermoResult:
    temperature_K: float
    num_configurations: int
    mixing_energy_min_eV: float
    mixing_energy_avg_eV: float
    partition_function: float
    free_energy_mix_eV: float


def read_energies_csv(csv_path: Path, energy_col: str = "energy_eV") -> list[float]:
    """
    Read energies (eV) from a CSV file.
    Expects a column named `energy_col` (default: energy_eV).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    energies: list[float] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or energy_col not in reader.fieldnames:
            raise ValueError(f"CSV must contain column '{energy_col}'. Found: {reader.fieldnames}")

        for row in reader:
            energies.append(float(row[energy_col]))

    if not energies:
        raise ValueError("No energies found in CSV.")
    return energies


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
) -> list[dict]:
    """
    Run boltzmann_thermo_from_csv for each temperature and return list of dict rows.
    """
    csv_path = Path(csv_path)
    rows: list[dict] = []

    for T in T_values:
        res = boltzmann_thermo_from_csv(csv_path, T=T, energy_col=energy_col)

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


def write_thermo_vs_T_csv(rows: list[dict], out_csv: Path) -> None:
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


def plot_thermo_vs_T(rows: list[dict], out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")  # headless backend (CI-safe)
    import matplotlib.pyplot as plt

    required = {"temperature_K", "free_energy_mix_eV", "mixing_energy_avg_eV"}
    missing = [required - set(r.keys()) for r in rows[:1]]
    if missing and missing[0]:
        raise KeyError(f"Row is missing keys {missing[0]}. Row keys are: {list(rows[0].keys())}")

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
