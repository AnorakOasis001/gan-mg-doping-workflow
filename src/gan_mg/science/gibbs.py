from __future__ import annotations

import csv
import logging
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from gan_mg.science.constants import K_B_EV_PER_K
from gan_mg.science.thermo import log_partition_function

LOGGER = logging.getLogger(__name__)

REQUIRED_MIXING_COLUMNS = (
    "mechanism_code",
    "x_mg_cation",
    "doping_level_percent",
    "dE_eV",
    "energy_mixing_eV",
)

GIBBS_SUMMARY_COLUMNS = (
    "mechanism_code",
    "x_mg_cation",
    "doping_level_percent",
    "T_K",
    "num_configurations",
    "energy_mixing_min_eV",
    "logZ",
    "free_energy_mixing_eV",
    "mixing_energy_avg_eV",
    "dE_min_eV",
    "dE_max_eV",
    "weight_max",
    "n_mismatch_abs_max",
    "n_mismatch_abs_mean",
    "energy_mixing_min_eV_per_atom",
    "free_energy_mixing_eV_per_atom",
    "mixing_energy_avg_eV_per_atom",
    "energy_mixing_min_eV_per_cation",
    "free_energy_mixing_eV_per_cation",
    "mixing_energy_avg_eV_per_cation",
)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(row: dict[str, Any], key: str) -> float:
    return float(str(row[key]))


def _optional_float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None or str(value).strip() == "":
        return None
    return float(str(value))


def validate_mixing_input_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("per_structure_mixing.csv must contain at least one row")

    for i, row in enumerate(rows, start=2):
        for key in REQUIRED_MIXING_COLUMNS:
            if key not in row:
                raise ValueError(f"row {i} is missing required column '{key}'")
        for key in ("x_mg_cation", "doping_level_percent", "dE_eV", "energy_mixing_eV"):
            if not math.isfinite(float(row[key])):
                raise ValueError(f"row {i} has non-finite '{key}'")


def build_gibbs_summary_rows(
    per_structure_mixing_rows: list[dict[str, Any]],
    temperatures_K: list[float],
) -> list[dict[str, Any]]:
    if not temperatures_K:
        raise ValueError("at least one temperature must be provided")
    for t in temperatures_K:
        if t <= 0:
            raise ValueError("temperatures must be > 0 K")

    validate_mixing_input_rows(per_structure_mixing_rows)

    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    mismatch_values: list[float] = []
    for row in per_structure_mixing_rows:
        groups[(str(row["mechanism_code"]), float(row["x_mg_cation"]))].append(row)
        mismatch = _optional_float(row, "n_mismatch")
        if mismatch is not None:
            mismatch_values.append(abs(mismatch))

    if mismatch_values:
        mismatch_nonzero = [v for v in mismatch_values if v > 0.0]
        if mismatch_nonzero:
            LOGGER.warning(
                "n_mismatch is nonzero in %d/%d per_structure_mixing rows (max abs %.6f)",
                len(mismatch_nonzero),
                len(mismatch_values),
                max(mismatch_values),
            )

    out_rows: list[dict[str, Any]] = []
    for mechanism_code, x_mg in sorted(groups):
        grouped_rows = groups[(mechanism_code, x_mg)]
        representative = grouped_rows[0]
        energies = np.asarray([_to_float(r, "energy_mixing_eV") for r in grouped_rows], dtype=float)
        d_e = np.asarray([_to_float(r, "dE_eV") for r in grouped_rows], dtype=float)
        e_min = float(np.min(energies))
        d_e_min = float(np.min(d_e))
        d_e_max = float(np.max(d_e))

        site_count = _optional_float(representative, "site_count_total")
        mg_count = _optional_float(representative, "mg_count")
        ga_count = _optional_float(representative, "ga_count")
        cation_count = None if mg_count is None or ga_count is None else mg_count + ga_count

        mismatch_group = [
            abs(v)
            for r in grouped_rows
            if (v := _optional_float(r, "n_mismatch")) is not None
        ]

        for t in sorted(float(v) for v in temperatures_K):
            log_z = log_partition_function(delta_e_eV=d_e, temperature_K=t)
            beta = 1.0 / (K_B_EV_PER_K * t)
            x = -beta * d_e
            m = float(np.max(x))
            w_unnorm = np.exp(x - m)
            w = w_unnorm / np.sum(w_unnorm)

            weighted_delta = float(np.sum(w * d_e))
            g_mix = e_min - K_B_EV_PER_K * t * log_z
            e_mix_avg = e_min + weighted_delta

            row_out: dict[str, Any] = {
                "mechanism_code": mechanism_code,
                "x_mg_cation": x_mg,
                "doping_level_percent": _to_float(representative, "doping_level_percent"),
                "T_K": t,
                "num_configurations": int(d_e.size),
                "energy_mixing_min_eV": e_min,
                "logZ": log_z,
                "free_energy_mixing_eV": g_mix,
                "mixing_energy_avg_eV": e_mix_avg,
                "dE_min_eV": d_e_min,
                "dE_max_eV": d_e_max,
                "weight_max": float(np.max(w)),
                "n_mismatch_abs_max": max(mismatch_group) if mismatch_group else 0.0,
                "n_mismatch_abs_mean": statistics.fmean(mismatch_group) if mismatch_group else 0.0,
                "energy_mixing_min_eV_per_atom": float("nan"),
                "free_energy_mixing_eV_per_atom": float("nan"),
                "mixing_energy_avg_eV_per_atom": float("nan"),
                "energy_mixing_min_eV_per_cation": float("nan"),
                "free_energy_mixing_eV_per_cation": float("nan"),
                "mixing_energy_avg_eV_per_cation": float("nan"),
            }

            if site_count is not None and site_count > 0:
                row_out["energy_mixing_min_eV_per_atom"] = e_min / site_count
                row_out["free_energy_mixing_eV_per_atom"] = g_mix / site_count
                row_out["mixing_energy_avg_eV_per_atom"] = e_mix_avg / site_count

            if cation_count is not None and cation_count > 0:
                row_out["energy_mixing_min_eV_per_cation"] = e_min / cation_count
                row_out["free_energy_mixing_eV_per_cation"] = g_mix / cation_count
                row_out["mixing_energy_avg_eV_per_cation"] = e_mix_avg / cation_count

            out_rows.append(row_out)

    out_rows.sort(key=lambda r: (str(r["mechanism_code"]), float(r["x_mg_cation"]), float(r["T_K"])))
    return out_rows


def validate_gibbs_summary_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("gibbs_summary.csv must contain at least one row")

    for i, row in enumerate(rows, start=2):
        for key in GIBBS_SUMMARY_COLUMNS:
            if key not in row:
                raise ValueError(f"row {i} is missing required column '{key}'")

        if int(row["num_configurations"]) < 1:
            raise ValueError(f"row {i} has invalid num_configurations")

        for key in (
            "x_mg_cation",
            "doping_level_percent",
            "T_K",
            "energy_mixing_min_eV",
            "logZ",
            "free_energy_mixing_eV",
            "mixing_energy_avg_eV",
            "dE_min_eV",
            "dE_max_eV",
            "weight_max",
        ):
            if not math.isfinite(float(row[key])):
                raise ValueError(f"row {i} has non-finite '{key}'")


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in columns})


def derive_gibbs_summary_dataset(run_dir: Path, temperatures_K: list[float]) -> tuple[Path, Path]:
    run_path = Path(run_dir)
    mixing_path = run_path / "derived" / "per_structure_mixing.csv"
    if not mixing_path.exists():
        raise FileNotFoundError(
            f"per_structure_mixing.csv not found: {mixing_path}. Run `ganmg mix --run-id <id>` first."
        )

    rows = _load_rows(mixing_path)
    summary_rows = build_gibbs_summary_rows(rows, temperatures_K)
    validate_gibbs_summary_rows(summary_rows)

    gibbs_path = run_path / "derived" / "gibbs_summary.csv"
    all_mech_path = run_path / "derived" / "all_mechanisms_gibbs_summary.csv"

    _write_csv(gibbs_path, summary_rows, GIBBS_SUMMARY_COLUMNS)
    _write_csv(all_mech_path, summary_rows, GIBBS_SUMMARY_COLUMNS)
    return gibbs_path, all_mech_path
