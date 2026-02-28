from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

PHASE_MAP_COLUMNS = (
    "x_mg_cation",
    "doping_level_percent",
    "T_K",
    "delta_free_energy_mean_eV",
    "delta_free_energy_low_eV",
    "delta_free_energy_high_eV",
    "preferred_mechanism",
    "robust",
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(PHASE_MAP_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in PHASE_MAP_COLUMNS})


def _pick_value(row: dict[str, str], candidates: tuple[str, ...]) -> float:
    for key in candidates:
        if key in row and str(row[key]).strip() != "":
            return float(row[key])
    raise ValueError(f"Missing required value; expected one of: {', '.join(candidates)}")


def build_phase_map_rows(crossover_uncertainty_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []

    for row in crossover_uncertainty_rows:
        preferred = str(row.get("preferred", row.get("preferred_mechanism", ""))).strip().lower()
        if preferred not in {"vn", "mgi", "uncertain"}:
            raise ValueError(f"invalid preferred mechanism: {preferred}")

        robust_raw = str(row.get("robust", "")).strip().lower()
        robust = robust_raw in {"true", "1", "yes"}

        out_rows.append(
            {
                "x_mg_cation": float(row["x_mg_cation"]),
                "doping_level_percent": float(row["doping_level_percent"]),
                "T_K": float(row["T_K"]),
                "delta_free_energy_mean_eV": _pick_value(row, ("delta_free_energy_eV", "delta_G_mean_eV", "delta_mean", "delta")),
                "delta_free_energy_low_eV": _pick_value(row, ("delta_low", "delta_free_energy_low_eV", "delta_G_ci_low_eV")),
                "delta_free_energy_high_eV": _pick_value(row, ("delta_high", "delta_free_energy_high_eV", "delta_G_ci_high_eV")),
                "preferred_mechanism": preferred,
                "robust": robust,
            }
        )

    out_rows.sort(key=lambda r: (float(r["x_mg_cation"]), float(r["T_K"])))
    return out_rows


def validate_phase_map_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("phase_map.csv must contain at least one row")

    for i, row in enumerate(rows, start=2):
        for key in PHASE_MAP_COLUMNS:
            if key not in row:
                raise ValueError(f"row {i} missing required '{key}'")

        for key in (
            "x_mg_cation",
            "doping_level_percent",
            "T_K",
            "delta_free_energy_mean_eV",
            "delta_free_energy_low_eV",
            "delta_free_energy_high_eV",
        ):
            if not math.isfinite(float(row[key])):
                raise ValueError(f"row {i} has non-finite '{key}'")

        low = float(row["delta_free_energy_low_eV"])
        mean = float(row["delta_free_energy_mean_eV"])
        high = float(row["delta_free_energy_high_eV"])
        if not (low <= mean <= high):
            raise ValueError(f"row {i} has invalid delta CI ordering")

        preferred = str(row["preferred_mechanism"]).strip().lower()
        if preferred not in {"vn", "mgi", "uncertain"}:
            raise ValueError(f"row {i} has invalid preferred_mechanism")


def derive_phase_map_dataset(run_dir: Path) -> Path:
    run_path = Path(run_dir)
    crossover_path = run_path / "derived" / "crossover_uncertainty.csv"
    if not crossover_path.exists():
        raise FileNotFoundError(
            f"crossover_uncertainty.csv not found: {crossover_path}. Run `ganmg uncertainty --run-id <id> ...` first."
        )

    rows = _read_rows(crossover_path)
    phase_map_rows = build_phase_map_rows(rows)
    validate_phase_map_rows(phase_map_rows)

    out_csv = run_path / "derived" / "phase_map.csv"
    _write_rows(out_csv, phase_map_rows)
    return out_csv
