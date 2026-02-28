from __future__ import annotations

import csv
from pathlib import Path

from gan_mg._mpl import ensure_agg


def plot_phase_map(
    phase_map_csv: Path,
    out_png: Path,
    boundary_csv: Path | None = None,
) -> None:
    ensure_agg()
    import matplotlib.pyplot as plt

    with Path(phase_map_csv).open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No rows found in {phase_map_csv}")

    mechanisms = sorted({str(row["winner_mechanism"]) for row in rows})
    cmap = plt.get_cmap("tab10")
    color_by_mechanism = {
        mechanism: cmap(index % cmap.N)
        for index, mechanism in enumerate(mechanisms)
    }

    x = [float(row["x_mg_cation"]) for row in rows]
    y = [float(row["T_K"]) for row in rows]
    c = [color_by_mechanism[str(row["winner_mechanism"])] for row in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, c=c, s=28, alpha=0.9)

    if boundary_csv is not None and Path(boundary_csv).exists():
        with Path(boundary_csv).open("r", encoding="utf-8", newline="") as f:
            boundary_rows = list(csv.DictReader(f))

        if boundary_rows:
            bx = [float(row["x_boundary"]) for row in boundary_rows]
            by = [float(row["T_K"]) for row in boundary_rows]
            robust_mask = [str(row.get("robust", "")).strip().lower() == "true" for row in boundary_rows]

            ax.plot(bx, by, color="black", linewidth=1.0, alpha=0.75, zorder=3)

            robust_x = [xv for xv, is_robust in zip(bx, robust_mask) if is_robust]
            robust_y = [yv for yv, is_robust in zip(by, robust_mask) if is_robust]
            nonrobust_x = [xv for xv, is_robust in zip(bx, robust_mask) if not is_robust]
            nonrobust_y = [yv for yv, is_robust in zip(by, robust_mask) if not is_robust]

            if robust_x:
                ax.scatter(robust_x, robust_y, marker="o", s=34, c="black", label="boundary (robust)", zorder=4)
            if nonrobust_x:
                ax.scatter(
                    nonrobust_x,
                    nonrobust_y,
                    marker="o",
                    s=34,
                    facecolors="none",
                    edgecolors="black",
                    label="boundary (non-robust)",
                    zorder=4,
                )

    for mechanism in mechanisms:
        ax.scatter([], [], c=[color_by_mechanism[mechanism]], s=28, label=mechanism)

    ax.set_xlabel("Mg molar fraction (cation)")
    ax.set_ylabel("Temperature (K)")
    ax.set_title("Mechanism Phase Map")
    ax.legend(title="winner_mechanism", fontsize=8)

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
