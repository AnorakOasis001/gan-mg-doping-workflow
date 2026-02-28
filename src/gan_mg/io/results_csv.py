from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from gan_mg.science.constants import K_B_EV_PER_K
from gan_mg.science.streaming import LogSumExpAccumulator, RunningStats, ScaledExpSumAccumulator
from gan_mg.science.thermo import LOGGER, ThermoDiagnostics, ThermoResult

REQUIRED_RESULTS_COLUMNS = ("structure_id", "mechanism", "energy_eV")

if TYPE_CHECKING:
    import pandas as pd


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
    energy_column: str = "energy_eV",
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
