from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import pandas as pd

K_B_EV_PER_K = 8.617333262e-5  # Boltzmann constant in eV/K
LOGGER = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class ThermoDiagnostics:
    temperature_K: float
    num_configurations: int
    expected_energy_eV: float
    energy_variance_eV2: float
    energy_std_eV: float
    p_min: float
    effective_sample_size: float
    logZ_shifted: float
    logZ_absolute: float
    notes: list[str]


REQUIRED_RESULTS_COLUMNS = ("structure_id", "mechanism", "energy_eV")


class LogSumExpAccumulator:
    def __init__(self) -> None:
        self._m: float | None = None
        self._s: float = 0.0

    def update(self, x: NDArray[np.float64]) -> None:
        if x.size == 0:
            return

        m2 = float(np.max(x))
        s2 = float(np.sum(np.exp(x - m2)))

        if self._m is None:
            self._m = m2
            self._s = s2
            return

        if m2 <= self._m:
            self._s = self._s + math.exp(m2 - self._m) * s2
        else:
            self._s = math.exp(self._m - m2) * self._s + s2
            self._m = m2

    def logsumexp(self) -> float:
        if self._m is None:
            raise ValueError("No values were provided to LogSumExpAccumulator.")
        return self._m + math.log(self._s)


class RunningStats:
    def __init__(self) -> None:
        self._count: int = 0
        self._mean: float = 0.0
        self._min: float | None = None

    def update(self, values: NDArray[np.float64]) -> None:
        if values.size == 0:
            return

        for value in values:
            value_f = float(value)
            self._count += 1
            delta = value_f - self._mean
            self._mean += delta / self._count
            if self._min is None or value_f < self._min:
                self._min = value_f

    @property
    def count(self) -> int:
        if self._count == 0:
            raise ValueError("RunningStats.count is undefined before any updates.")
        return self._count

    @property
    def mean(self) -> float:
        if self._count == 0:
            raise ValueError("RunningStats.mean is undefined before any updates.")
        return self._mean

    @property
    def min(self) -> float:
        if self._min is None:
            raise ValueError("RunningStats.min is undefined before any updates.")
        return self._min


class ScaledExpSumAccumulator:
    """Accumulate sums of the form Σ a_i * exp(x_i) in a stable scaled representation."""

    def __init__(self) -> None:
        self._m: float | None = None
        self._s: float = 0.0

    def update(self, x: NDArray[np.float64], a: NDArray[np.float64]) -> None:
        if x.size == 0:
            return
        if x.shape != a.shape:
            raise ValueError("x and a must have identical shapes.")
        if not np.all(np.isfinite(x)):
            raise ValueError("x must contain only finite values.")
        if not np.all(np.isfinite(a)):
            raise ValueError("a must contain only finite values.")
        if np.any(a < 0.0):
            raise ValueError("a must be non-negative.")

        mask = a > 0.0
        if not np.any(mask):
            return

        x_pos = x[mask]
        a_pos = a[mask]

        m2 = float(np.max(x_pos))
        s2 = float(np.sum(a_pos * np.exp(x_pos - m2)))

        if self._m is None:
            self._m = m2
            self._s = s2
            return

        if m2 <= self._m:
            self._s = self._s + math.exp(m2 - self._m) * s2
        else:
            self._s = math.exp(self._m - m2) * self._s + s2
            self._m = m2

    def is_zero(self) -> bool:
        return self._m is None or self._s == 0.0

    def value_log(self) -> float:
        if self.is_zero():
            return float("-inf")
        assert self._m is not None
        return self._m + math.log(self._s)


def log_partition_function(delta_e_eV: NDArray[np.float64], temperature_K: float) -> float:
    """
    Compute log(partition function) using a numerically stable log-sum-exp evaluation.

    The partition function is defined for delta energies as:
        Z = sum_i exp(-delta_e_i / (kB * T))
    and this function returns log(Z).
    """
    if temperature_K <= 0:
        raise ValueError("temperature_K must be > 0.")

    if delta_e_eV.size == 0:
        raise ValueError("delta_e_eV must be non-empty.")

    if not np.all(np.isfinite(delta_e_eV)):
        raise ValueError("delta_e_eV must contain only finite values.")

    x = -delta_e_eV / (K_B_EV_PER_K * temperature_K)
    m = float(np.max(x))
    log_z = m + float(np.log(np.sum(np.exp(x - m))))
    return log_z


def free_energy_from_logZ(logZ: float, temperature_K: float) -> float:
    if temperature_K <= 0:
        raise ValueError("temperature_K must be > 0.")
    return -K_B_EV_PER_K * temperature_K * logZ


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

    if not energies:
        raise ValueError("energies must be non-empty.")

    beta = 1.0 / (K_B_EV_PER_K * T)
    emin = min(energies)

    delta_e = np.asarray([e - emin for e in energies], dtype=float)
    log_z = log_partition_function(delta_e_eV=delta_e, temperature_K=T)

    max_log_float = math.log(float(np.finfo(float).max))
    if log_z <= max_log_float:
        partition_function = math.exp(log_z)
    else:
        partition_function = float("inf")
        LOGGER.warning(
            "Partition function overflow for T=%s K (logZ=%s); storing partition_function=inf.",
            T,
            log_z,
        )

    energies_np = np.asarray(energies, dtype=float)
    mixing_energy_avg_eV = float(np.mean(energies_np))

    free_energy_mix_eV = emin + free_energy_from_logZ(logZ=log_z, temperature_K=T)

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


def boltzmann_diagnostics_from_energies(energies: list[float], T: float) -> ThermoDiagnostics:
    if T <= 0:
        raise ValueError("Temperature T must be > 0.")
    if not energies:
        raise ValueError("energies must be non-empty.")

    energies_np = np.asarray(energies, dtype=float)
    if not np.all(np.isfinite(energies_np)):
        raise ValueError("energies must contain only finite values.")

    beta = 1.0 / (K_B_EV_PER_K * T)
    emin = float(np.min(energies_np))
    delta = energies_np - emin
    x = -beta * delta

    log_z_shifted = log_partition_function(delta_e_eV=delta, temperature_K=T)
    log_z_absolute = log_z_shifted - beta * emin

    lse_w2 = LogSumExpAccumulator()
    lse_w2.update(2.0 * x)
    log_sum_w2 = lse_w2.logsumexp()

    p_min = math.exp(-log_z_shifted)
    effective_sample_size = math.exp(2.0 * log_z_shifted - log_sum_w2)

    if np.any(delta > 0.0):
        m = float(np.max(x))
        exp_shift = np.exp(x - m)
        sum_delta_exp = float(np.sum(delta * exp_shift))
        sum_delta2_exp = float(np.sum((delta ** 2) * exp_shift))

        expected_delta = 0.0 if sum_delta_exp == 0.0 else math.exp(m + math.log(sum_delta_exp) - log_z_shifted)
        expected_delta2 = (
            0.0
            if sum_delta2_exp == 0.0
            else math.exp(m + math.log(sum_delta2_exp) - log_z_shifted)
        )
    else:
        expected_delta = 0.0
        expected_delta2 = 0.0

    variance_e = max(expected_delta2 - expected_delta ** 2, 0.0)
    std_e = math.sqrt(variance_e)

    notes: list[str] = []
    max_log_float = math.log(float(np.finfo(float).max))
    if log_z_shifted > max_log_float:
        notes.append("partition_function_overflow")

    return ThermoDiagnostics(
        temperature_K=T,
        num_configurations=int(energies_np.size),
        expected_energy_eV=emin + expected_delta,
        energy_variance_eV2=variance_e,
        energy_std_eV=std_e,
        p_min=p_min,
        effective_sample_size=effective_sample_size,
        logZ_shifted=log_z_shifted,
        logZ_absolute=log_z_absolute,
        notes=notes,
    )


def diagnostics_from_csv_streaming(
    csv_path: Path,
    temperature_K: float,
    energy_column: str = "energy_eV",
    chunksize: int = 200_000,
) -> ThermoDiagnostics:
    import pandas as pd

    csv_path = Path(csv_path)
    if temperature_K <= 0:
        raise ValueError("temperature_K must be > 0.")
    if chunksize <= 0:
        raise ValueError("chunksize must be > 0.")
    if not csv_path.exists():
        raise ValueError(f"CSV not found: {csv_path}")

    count = 0
    emin: float | None = None
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        if energy_column not in chunk.columns:
            raise ValueError(f"CSV '{csv_path}' is missing required column '{energy_column}'.")

        energies = pd.to_numeric(chunk[energy_column], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(energies)):
            raise ValueError(
                f"CSV '{csv_path}' column '{energy_column}' must contain only finite values."
            )
        if energies.size == 0:
            continue

        chunk_min = float(np.min(energies))
        emin = chunk_min if emin is None else min(emin, chunk_min)
        count += int(energies.size)

    if count == 0 or emin is None:
        raise ValueError("CSV must contain at least one energy value.")

    beta = 1.0 / (K_B_EV_PER_K * temperature_K)
    lse_z = LogSumExpAccumulator()
    lse_w2 = LogSumExpAccumulator()
    sum_delta_exp = ScaledExpSumAccumulator()
    sum_delta2_exp = ScaledExpSumAccumulator()

    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        energies = pd.to_numeric(chunk[energy_column], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(energies)):
            raise ValueError(
                f"CSV '{csv_path}' column '{energy_column}' must contain only finite values."
            )

        delta = energies - emin
        x = -beta * delta

        lse_z.update(x)
        lse_w2.update(2.0 * x)
        sum_delta_exp.update(x, delta)
        sum_delta2_exp.update(x, delta ** 2)

    log_z_shifted = lse_z.logsumexp()
    log_z_absolute = log_z_shifted - beta * emin
    log_sum_w2 = lse_w2.logsumexp()

    p_min = math.exp(-log_z_shifted)
    effective_sample_size = math.exp(2.0 * log_z_shifted - log_sum_w2)

    expected_delta = 0.0
    if not sum_delta_exp.is_zero():
        expected_delta = math.exp(sum_delta_exp.value_log() - log_z_shifted)

    expected_delta2 = 0.0
    if not sum_delta2_exp.is_zero():
        expected_delta2 = math.exp(sum_delta2_exp.value_log() - log_z_shifted)

    variance_e = max(expected_delta2 - expected_delta ** 2, 0.0)
    std_e = math.sqrt(variance_e)

    notes: list[str] = []
    max_log_float = math.log(float(np.finfo(float).max))
    if log_z_shifted > max_log_float:
        notes.append("partition_function_overflow")

    return ThermoDiagnostics(
        temperature_K=temperature_K,
        num_configurations=count,
        expected_energy_eV=emin + expected_delta,
        energy_variance_eV2=variance_e,
        energy_std_eV=std_e,
        p_min=p_min,
        effective_sample_size=effective_sample_size,
        logZ_shifted=log_z_shifted,
        logZ_absolute=log_z_absolute,
        notes=notes,
    )


def thermo_from_csv_streaming(
    csv_path: Path,
    temperature_K: float,
    energy_column: str = "mixing_energy_eV",
    chunksize: int = 200_000,
) -> ThermoResult:
    import pandas as pd

    csv_path = Path(csv_path)
    if temperature_K <= 0:
        raise ValueError("temperature_K must be > 0.")
    if chunksize <= 0:
        raise ValueError("chunksize must be > 0.")
    if not csv_path.exists():
        raise ValueError(f"CSV not found: {csv_path}")

    stats = RunningStats()
    lse = LogSumExpAccumulator()

    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        if energy_column not in chunk.columns:
            raise ValueError(f"CSV '{csv_path}' is missing required column '{energy_column}'.")

        delta_e = pd.to_numeric(chunk[energy_column], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(delta_e)):
            raise ValueError(
                f"CSV '{csv_path}' column '{energy_column}' must contain only finite values."
            )

        stats.update(delta_e)
        x = -delta_e / (K_B_EV_PER_K * temperature_K)
        lse.update(x)

    num_configurations = stats.count
    mixing_energy_min_eV = stats.min
    mixing_energy_avg_eV = stats.mean
    log_z_absolute = lse.logsumexp()
    log_z_shifted = log_z_absolute + (mixing_energy_min_eV / (K_B_EV_PER_K * temperature_K))
    free_energy_mix_eV = -K_B_EV_PER_K * temperature_K * log_z_absolute

    max_log_float = math.log(float(np.finfo(float).max))
    if log_z_shifted <= max_log_float:
        partition_function = math.exp(log_z_shifted)
    else:
        partition_function = float("inf")
        LOGGER.warning(
            "Partition function overflow for T=%s K (logZ=%s); storing partition_function=inf.",
            temperature_K,
            log_z_shifted,
        )

    return ThermoResult(
        temperature_K=temperature_K,
        num_configurations=num_configurations,
        mixing_energy_min_eV=mixing_energy_min_eV,
        mixing_energy_avg_eV=mixing_energy_avg_eV,
        partition_function=partition_function,
        free_energy_mix_eV=free_energy_mix_eV,
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
