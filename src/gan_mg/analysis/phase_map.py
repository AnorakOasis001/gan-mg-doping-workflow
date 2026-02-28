from __future__ import annotations

from pathlib import Path

import pandas as pd

PHASE_MAP_COLUMNS = (
    "T_K",
    "x_mg_cation",
    "doping_level_percent",
    "winner_mechanism",
    "delta_G_eV",
    "delta_G_ci_low_eV",
    "delta_G_ci_high_eV",
    "robust",
)

EPS = 1e-12


_SORT_COLUMNS = ["T_K", "x_mg_cation", "doping_level_percent"]
_REQUIRED_PHASE_MAP_COLUMNS = ["T_K", "x_mg_cation", "doping_level_percent", "winner_mechanism", "delta_G_eV"]


def _sorted_output(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(_SORT_COLUMNS, ascending=True, kind="mergesort").reset_index(drop=True)


def _from_crossover_uncertainty(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame(
        {
            "T_K": df["T_K"].astype(float),
            "x_mg_cation": df["x_mg_cation"].astype(float),
            "doping_level_percent": df["doping_level_percent"].astype(float),
            "winner_mechanism": df["preferred"].astype(str),
            "delta_G_eV": df["delta_G_mean_eV"].astype(float).abs(),
            "delta_G_ci_low_eV": df["delta_G_ci_low_eV"].astype(float),
            "delta_G_ci_high_eV": df["delta_G_ci_high_eV"].astype(float),
            "robust": df["robust"].astype(bool),
        }
    )
    return _sorted_output(out)


def _from_mechanism_crossover(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame(
        {
            "T_K": df["T_K"].astype(float),
            "x_mg_cation": df["x_mg_cation"].astype(float),
            "doping_level_percent": df["doping_level_percent"].astype(float),
            "winner_mechanism": df["preferred_mechanism"].astype(str),
            "delta_G_eV": df["delta_free_energy_eV"].astype(float).abs(),
        }
    )
    return _sorted_output(out)


def _from_gibbs_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    work = pd.DataFrame(
        {
            "T_K": df["T_K"].astype(float),
            "x_mg_cation": df["x_mg_cation"].astype(float),
            "doping_level_percent": df["doping_level_percent"].astype(float),
            "mechanism_code": df["mechanism_code"].astype(str),
            "free_energy_mixing_eV": df["free_energy_mixing_eV"].astype(float),
        }
    )

    rows: list[dict[str, float | str]] = []
    grouped = work.groupby(_SORT_COLUMNS, sort=True, as_index=False)
    for _, group_df in grouped:
        sorted_group = group_df.sort_values(
            ["free_energy_mixing_eV", "mechanism_code"],
            ascending=[True, True],
            kind="mergesort",
        ).reset_index(drop=True)

        best = float(sorted_group.loc[0, "free_energy_mixing_eV"])
        winner = str(sorted_group.loc[0, "mechanism_code"])

        if len(sorted_group.index) >= 2:
            second = float(sorted_group.loc[1, "free_energy_mixing_eV"])
            delta = max(second - best, 0.0)
        else:
            delta = 0.0

        if delta < EPS:
            winner = "tie"

        rows.append(
            {
                "T_K": float(sorted_group.loc[0, "T_K"]),
                "x_mg_cation": float(sorted_group.loc[0, "x_mg_cation"]),
                "doping_level_percent": float(sorted_group.loc[0, "doping_level_percent"]),
                "winner_mechanism": winner,
                "delta_G_eV": delta,
            }
        )

    out = pd.DataFrame(rows, columns=_REQUIRED_PHASE_MAP_COLUMNS)
    return _sorted_output(out)


def derive_phase_map_dataset(run_dir: Path) -> Path:
    run_path = Path(run_dir)
    derived_dir = run_path / "derived"

    crossover_uncertainty_csv = derived_dir / "crossover_uncertainty.csv"
    mechanism_crossover_csv = derived_dir / "mechanism_crossover.csv"
    gibbs_summary_csv = derived_dir / "gibbs_summary.csv"

    if crossover_uncertainty_csv.exists():
        out_df = _from_crossover_uncertainty(crossover_uncertainty_csv)
    elif mechanism_crossover_csv.exists():
        out_df = _from_mechanism_crossover(mechanism_crossover_csv)
    elif gibbs_summary_csv.exists():
        out_df = _from_gibbs_summary(gibbs_summary_csv)
    else:
        raise FileNotFoundError(
            "No phase-map input found. Expected one of: "
            f"{crossover_uncertainty_csv}, {mechanism_crossover_csv}, {gibbs_summary_csv}."
        )

    out_csv = derived_dir / "phase_map.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    return out_csv
