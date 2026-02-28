from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

PHASE_BOUNDARY_COLUMNS = (
    "T_K",
    "x_boundary",
    "doping_level_percent",
    "mech_left",
    "mech_right",
    "robust",
    "delta_G_at_boundary_eV",
)




def _coerce_float(value: Any) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    return float(str(value))


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _build_robust_lookup(df: pd.DataFrame) -> dict[tuple[float, float], bool]:
    return {
        (_coerce_float(row.T_K), _coerce_float(row.x_mg_cation)): _coerce_bool(row.robust)
        for row in df.itertuples(index=False)
    }


def derive_phase_boundary_dataset(run_dir: Path) -> Path:
    run_path = Path(run_dir)
    derived_dir = run_path / "derived"
    phase_map_csv = derived_dir / "phase_map.csv"
    if not phase_map_csv.exists():
        raise FileNotFoundError(f"Missing required input: {phase_map_csv}")

    phase_map_df: pd.DataFrame = pd.read_csv(phase_map_csv)
    required = {"T_K", "x_mg_cation", "winner_mechanism"}
    missing = sorted(required.difference(set(phase_map_df.columns)))
    if missing:
        raise ValueError(f"phase_map.csv missing required columns: {', '.join(missing)}")

    has_doping = "doping_level_percent" in phase_map_df.columns
    has_robust = "robust" in phase_map_df.columns
    has_delta_g = "delta_G_eV" in phase_map_df.columns

    robust_lookup: dict[tuple[float, float], bool] = {}
    crossover_uncertainty_csv = derived_dir / "crossover_uncertainty.csv"
    if not has_robust and crossover_uncertainty_csv.exists():
        crossover_df: pd.DataFrame = pd.read_csv(crossover_uncertainty_csv)
        if {"T_K", "x_mg_cation", "robust"}.issubset(crossover_df.columns):
            robust_lookup = _build_robust_lookup(crossover_df)

    boundary_rows: list[dict[str, float | str | bool]] = []
    grouped = phase_map_df.groupby("T_K", sort=True)
    for temp, group in grouped:
        sorted_group = group.sort_values("x_mg_cation", ascending=True, kind="mergesort").reset_index(drop=True)

        for i in range(len(sorted_group.index) - 1):
            left = sorted_group.iloc[i]
            right = sorted_group.iloc[i + 1]
            mech_left = str(left["winner_mechanism"])
            mech_right = str(right["winner_mechanism"])
            if mech_left == mech_right:
                continue

            x_left = _coerce_float(left["x_mg_cation"])
            x_right = _coerce_float(right["x_mg_cation"])
            x_boundary = 0.5 * (x_left + x_right)

            if has_doping:
                doping_level_percent = 0.5 * (
                    _coerce_float(left["doping_level_percent"]) + _coerce_float(right["doping_level_percent"])
                )
            else:
                doping_level_percent = x_boundary * 100.0

            if has_robust:
                robust = _coerce_bool(left.get("robust")) and _coerce_bool(right.get("robust"))
            elif robust_lookup:
                robust = robust_lookup.get((_coerce_float(temp), x_left), False) and robust_lookup.get(
                    (_coerce_float(temp), x_right),
                    False,
                )
            else:
                robust = False

            if has_delta_g:
                delta_g_at_boundary_eV = min(_coerce_float(left["delta_G_eV"]), _coerce_float(right["delta_G_eV"]))
            else:
                delta_g_at_boundary_eV = float("nan")

            boundary_rows.append(
                {
                    "T_K": _coerce_float(temp),
                    "x_boundary": x_boundary,
                    "doping_level_percent": doping_level_percent,
                    "mech_left": mech_left,
                    "mech_right": mech_right,
                    "robust": robust,
                    "delta_G_at_boundary_eV": delta_g_at_boundary_eV,
                }
            )

    out_df = pd.DataFrame(boundary_rows, columns=PHASE_BOUNDARY_COLUMNS)
    out_df = out_df.sort_values(["T_K", "x_boundary"], ascending=[True, True], kind="mergesort")
    out_csv = derived_dir / "phase_boundary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    return out_csv
