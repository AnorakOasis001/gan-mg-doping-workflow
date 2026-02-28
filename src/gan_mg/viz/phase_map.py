from __future__ import annotations

import csv
from pathlib import Path

from gan_mg._mpl import ensure_agg


def _read_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def plot_phase_map_preference(phase_map_csv: Path, out_png: Path) -> None:
    ensure_agg()
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    rows = _read_rows(phase_map_csv)
    if not rows:
        raise ValueError(f"No rows found in {phase_map_csv}")

    dop_values = sorted({float(r["doping_level_percent"]) for r in rows})
    t_values = sorted({float(r["T_K"]) for r in rows})

    code_by_key: dict[tuple[float, float], int] = {}
    for row in rows:
        robust = str(row["robust"]).strip().lower() in {"true", "1", "yes"}
        preferred = str(row["preferred_mechanism"]).strip().lower()
        if not robust or preferred == "uncertain":
            code = 0
        elif preferred == "vn":
            code = 1
        else:
            code = 2
        code_by_key[(float(row["doping_level_percent"]), float(row["T_K"]))] = code

    grid = np.full((len(t_values), len(dop_values)), np.nan, dtype=float)
    for iy, t_k in enumerate(t_values):
        for ix, dop in enumerate(dop_values):
            value = code_by_key.get((dop, t_k))
            if value is not None:
                grid[iy, ix] = float(value)

    cmap = ListedColormap(["#bdbdbd", "#4c78a8", "#f58518"])

    extent: tuple[float, float, float, float] = (
        min(dop_values),
        max(dop_values),
        min(t_values),
        max(t_values),
    )

    plt.figure(figsize=(7.2, 5.5))
    plt.imshow(
        grid,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0,
        vmax=2,
        extent=extent,
    )
    plt.xlabel("Mg doping (%)")
    plt.ylabel("Temperature (K)")
    plt.title("Phase map: mechanism preference (robust regions)")
    plt.grid(alpha=0.15)

    legend_handles = [
        Patch(facecolor="#4c78a8", edgecolor="none", label="vn preferred (robust)"),
        Patch(facecolor="#f58518", edgecolor="none", label="mgi preferred (robust)"),
        Patch(facecolor="#bdbdbd", edgecolor="none", label="uncertain"),
    ]
    plt.legend(handles=legend_handles, fontsize=8, loc="best")

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()


def plot_phase_map_delta_g(phase_map_csv: Path, out_png: Path) -> None:
    ensure_agg()
    import matplotlib.pyplot as plt
    import numpy as np

    rows = _read_rows(phase_map_csv)
    if not rows:
        raise ValueError(f"No rows found in {phase_map_csv}")

    dop_values = sorted({float(r["doping_level_percent"]) for r in rows})
    t_values = sorted({float(r["T_K"]) for r in rows})

    delta_by_key = {
        (float(r["doping_level_percent"]), float(r["T_K"])): float(r["delta_free_energy_mean_eV"])
        for r in rows
    }

    grid = np.full((len(t_values), len(dop_values)), np.nan, dtype=float)
    for iy, t_k in enumerate(t_values):
        for ix, dop in enumerate(dop_values):
            value = delta_by_key.get((dop, t_k))
            if value is not None:
                grid[iy, ix] = value

    X, Y = np.meshgrid(np.asarray(dop_values, dtype=float), np.asarray(t_values, dtype=float))

    plt.figure(figsize=(7.2, 5.5))
    contourf = plt.contourf(X, Y, grid, levels=31, cmap="coolwarm")
    plt.colorbar(contourf, label="ΔG = Gmix(vn) - Gmix(mgi) [eV]")

    finite = np.isfinite(grid)
    if np.any(finite) and np.nanmin(grid) <= 0.0 <= np.nanmax(grid):
        plt.contour(X, Y, grid, levels=[0.0], colors="black", linewidths=1.5)

    plt.xlabel("Mg doping (%)")
    plt.ylabel("Temperature (K)")
    plt.title("Phase map: ΔG mean heatmap")
    plt.grid(alpha=0.2)

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()
