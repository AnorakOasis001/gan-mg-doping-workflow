from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from gan_mg._mpl import ensure_agg_backend


CROSSOVER_COLUMNS = (
    "x_mg_cation",
    "doping_level_percent",
    "T_K",
    "delta_free_energy_eV",
    "delta_free_energy_eV_per_cation",
    "preferred_mechanism",
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CROSSOVER_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CROSSOVER_COLUMNS})


def build_mechanism_crossover_rows(all_mechanisms_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, float], dict[str, dict[str, str]]] = {}
    has_per_cation = bool(all_mechanisms_rows and "free_energy_mixing_eV_per_cation" in all_mechanisms_rows[0])

    for row in all_mechanisms_rows:
        mechanism = str(row["mechanism_code"]).strip().lower()
        if mechanism not in {"vn", "mgi"}:
            continue
        key = (float(row["x_mg_cation"]), float(row["T_K"]))
        grouped.setdefault(key, {})[mechanism] = row

    out: list[dict[str, Any]] = []
    for x_mg, t_k in sorted(grouped.keys()):
        by_mechanism = grouped[(x_mg, t_k)]
        if "vn" not in by_mechanism or "mgi" not in by_mechanism:
            continue

        vn_row = by_mechanism["vn"]
        mgi_row = by_mechanism["mgi"]
        delta_g = float(vn_row["free_energy_mixing_eV"]) - float(mgi_row["free_energy_mixing_eV"])

        row_out: dict[str, Any] = {
            "x_mg_cation": x_mg,
            "doping_level_percent": float(vn_row["doping_level_percent"]),
            "T_K": t_k,
            "delta_free_energy_eV": delta_g,
            "preferred_mechanism": "vn" if delta_g < 0 else "mgi",
            "delta_free_energy_eV_per_cation": "",
        }

        if has_per_cation:
            row_out["delta_free_energy_eV_per_cation"] = (
                float(vn_row["free_energy_mixing_eV_per_cation"])
                - float(mgi_row["free_energy_mixing_eV_per_cation"])
            )

        out.append(row_out)

    return sorted(out, key=lambda r: (float(r["x_mg_cation"]), float(r["T_K"])))


def derive_mechanism_crossover_dataset(run_dir: Path) -> tuple[Path, Path]:
    run_path = Path(run_dir)
    all_mech_path = run_path / "derived" / "all_mechanisms_gibbs_summary.csv"
    if not all_mech_path.exists():
        raise FileNotFoundError(
            f"all_mechanisms_gibbs_summary.csv not found: {all_mech_path}. Run `ganmg gibbs --run-id <id>` first."
        )

    rows = _read_rows(all_mech_path)
    crossover_rows = build_mechanism_crossover_rows(rows)
    if not crossover_rows:
        raise ValueError("No crossover rows were generated; ensure both vn and mgi mechanisms are present")

    crossover_csv = run_path / "derived" / "mechanism_crossover.csv"
    _write_rows(crossover_csv, crossover_rows)

    crossover_plot = run_path / "figures" / "crossover_map.png"
    plot_mechanism_crossover_map(crossover_csv, crossover_plot)
    return crossover_csv, crossover_plot


def plot_mechanism_crossover_map(crossover_csv: Path, out_png: Path) -> None:
    ensure_agg_backend()
    import matplotlib.pyplot as plt
    import numpy as np

    rows = _read_rows(crossover_csv)
    if not rows:
        raise ValueError(f"No rows found in {crossover_csv}")

    x_values = sorted({float(r["doping_level_percent"]) for r in rows})
    t_values = sorted({float(r["T_K"]) for r in rows})
    delta_by_key = {
        (float(r["doping_level_percent"]), float(r["T_K"])): float(r["delta_free_energy_eV"])
        for r in rows
    }

    grid = np.full((len(t_values), len(x_values)), np.nan, dtype=float)
    for iy, t_k in enumerate(t_values):
        for ix, dop in enumerate(x_values):
            value = delta_by_key.get((dop, t_k))
            if value is not None:
                grid[iy, ix] = value

    X, Y = np.meshgrid(np.asarray(x_values, dtype=float), np.asarray(t_values, dtype=float))

    plt.figure(figsize=(7.5, 5.5))
    contourf = plt.contourf(X, Y, grid, levels=31, cmap="coolwarm")
    plt.colorbar(contourf, label="ΔG = Gmix(vn) - Gmix(mgi) [eV]")

    finite = np.isfinite(grid)
    if np.any(finite) and np.nanmin(grid) <= 0.0 <= np.nanmax(grid):
        plt.contour(X, Y, grid, levels=[0.0], colors="black", linewidths=1.8)

    plt.xlabel("Mg doping (%)")
    plt.ylabel("Temperature (K)")
    plt.title("Mechanism crossover map (vn vs mgi)")
    plt.grid(alpha=0.2)

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()
