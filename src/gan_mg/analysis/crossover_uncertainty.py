from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from gan_mg._mpl import ensure_agg

CROSSOVER_UNCERTAINTY_COLUMNS = (
    "x_mg_cation",
    "doping_level_percent",
    "T_K",
    "delta_G_mean_eV",
    "delta_G_ci_low_eV",
    "delta_G_ci_high_eV",
    "preferred",
    "robust",
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CROSSOVER_UNCERTAINTY_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CROSSOVER_UNCERTAINTY_COLUMNS})


def build_crossover_uncertainty_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, float], dict[str, dict[str, str]]] = {}
    for row in rows:
        mechanism = str(row["mechanism_code"]).strip().lower()
        if mechanism not in {"vn", "mgi"}:
            continue
        key = (float(row["x_mg_cation"]), float(row["T_K"]))
        grouped.setdefault(key, {})[mechanism] = row

    out: list[dict[str, Any]] = []
    for x_mg, t_k in sorted(grouped.keys()):
        pair = grouped[(x_mg, t_k)]
        if "vn" not in pair or "mgi" not in pair:
            continue

        vn = pair["vn"]
        mgi = pair["mgi"]

        delta_mean = float(vn["free_energy_mixing_eV"]) - float(mgi["free_energy_mixing_eV"])
        delta_low = float(vn["free_energy_ci_low_eV"]) - float(mgi["free_energy_ci_high_eV"])
        delta_high = float(vn["free_energy_ci_high_eV"]) - float(mgi["free_energy_ci_low_eV"])

        if delta_high < 0.0:
            preferred = "vn"
            robust = True
        elif delta_low > 0.0:
            preferred = "mgi"
            robust = True
        else:
            preferred = "uncertain"
            robust = False

        out.append(
            {
                "x_mg_cation": x_mg,
                "doping_level_percent": float(vn["doping_level_percent"]),
                "T_K": t_k,
                "delta_G_mean_eV": delta_mean,
                "delta_G_ci_low_eV": delta_low,
                "delta_G_ci_high_eV": delta_high,
                "preferred": preferred,
                "robust": robust,
            }
        )

    return sorted(out, key=lambda r: (float(r["x_mg_cation"]), float(r["T_K"])))


def derive_crossover_uncertainty_dataset(run_dir: Path) -> Path:
    run_path = Path(run_dir)
    uncertainty_csv = run_path / "derived" / "gibbs_uncertainty.csv"
    if not uncertainty_csv.exists():
        raise FileNotFoundError(
            f"gibbs_uncertainty.csv not found: {uncertainty_csv}. Run `ganmg uncertainty --run-id <id>` first."
        )

    rows = _read_rows(uncertainty_csv)
    out_rows = build_crossover_uncertainty_rows(rows)
    if not out_rows:
        raise ValueError("No crossover uncertainty rows were generated; ensure both vn and mgi mechanisms are present")

    out_csv = run_path / "derived" / "crossover_uncertainty.csv"
    _write_rows(out_csv, out_rows)

    out_png = run_path / "figures" / "crossover_map_uncertainty.png"
    plot_crossover_uncertainty_map(out_csv, out_png)
    return out_csv


def plot_crossover_uncertainty_map(crossover_csv: Path, out_png: Path) -> None:
    ensure_agg()
    import matplotlib.pyplot as plt

    rows = _read_rows(crossover_csv)
    if not rows:
        raise ValueError(f"No rows found in {crossover_csv}")

    robust = [r for r in rows if str(r["robust"]).lower() == "true"]
    uncertain = [r for r in rows if str(r["robust"]).lower() != "true"]

    plt.figure(figsize=(7.0, 5.0))
    if robust:
        x = [float(r["doping_level_percent"]) for r in robust]
        y = [float(r["T_K"]) for r in robust]
        colors = ["tab:blue" if r["preferred"] == "vn" else "tab:orange" for r in robust]
        plt.scatter(x, y, c=colors, marker="s", s=75, label="robust")

    if uncertain:
        x = [float(r["doping_level_percent"]) for r in uncertain]
        y = [float(r["T_K"]) for r in uncertain]
        plt.scatter(x, y, c="gray", marker="x", s=60, label="uncertain")

    plt.xlabel("Mg doping (%)")
    plt.ylabel("Temperature (K)")
    plt.title("Crossover robustness map (vn vs mgi)")
    plt.grid(alpha=0.2)
    plt.legend(fontsize=8)

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()
