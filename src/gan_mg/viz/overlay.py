from __future__ import annotations

import csv
from pathlib import Path

from gan_mg._mpl import ensure_agg


def _read_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def plot_overlay_dgmix_vs_doping_multi_t(gibbs_summary_csv: Path, out_png: Path) -> None:
    ensure_agg()
    import matplotlib.pyplot as plt

    rows = _read_rows(gibbs_summary_csv)
    if not rows:
        raise ValueError(f"No rows found in {gibbs_summary_csv}")

    use_per_cation = "free_energy_mixing_eV_per_cation" in rows[0]
    y_col = "free_energy_mixing_eV_per_cation" if use_per_cation else "free_energy_mixing_eV"

    grouped: dict[tuple[str, float], list[dict[str, str]]] = {}
    for row in rows:
        key = (str(row["mechanism_code"]), float(row["T_K"]))
        grouped.setdefault(key, []).append(row)

    plt.figure(figsize=(8, 5))
    for (mechanism, temp), group_rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        group_rows_sorted = sorted(group_rows, key=lambda r: float(r["doping_level_percent"]))
        x = [float(r["doping_level_percent"]) for r in group_rows_sorted]
        y = [float(r[y_col]) for r in group_rows_sorted]
        plt.plot(x, y, marker="o", linewidth=1.8, label=f"{mechanism}, T={temp:g}K")

    plt.xlabel("Mg doping (%)")
    ylabel = "ΔG_mix (eV/cation)" if y_col.endswith("per_cation") else "ΔG_mix (eV)"
    plt.ylabel(ylabel)
    plt.title("Overlay: ΔG_mix vs Mg doping (%)")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()


def plot_athermal_emin_vs_doping(summary_csv: Path, out_png: Path) -> None:
    ensure_agg()
    import matplotlib.pyplot as plt

    rows = _read_rows(summary_csv)
    if not rows:
        raise ValueError(f"No rows found in {summary_csv}")

    y_col = (
        "energy_mixing_min_eV_per_cation"
        if "energy_mixing_min_eV_per_cation" in rows[0]
        else "energy_mixing_min_eV"
    )

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        mechanism = str(row["mechanism_code"])
        grouped.setdefault(mechanism, []).append(row)

    plt.figure(figsize=(8, 5))
    for mechanism, group_rows in sorted(grouped.items(), key=lambda item: item[0]):
        group_rows_sorted = sorted(group_rows, key=lambda r: float(r["doping_level_percent"]))
        x = [float(r["doping_level_percent"]) for r in group_rows_sorted]
        y = [float(r[y_col]) for r in group_rows_sorted]
        plt.plot(x, y, marker="o", linewidth=1.8, label=mechanism)

    plt.xlabel("Mg doping (%)")
    ylabel = "E_mix,min (eV/cation)" if y_col.endswith("per_cation") else "E_mix,min (eV)"
    plt.ylabel(ylabel)
    plt.title("Athermal minimum mixing energy vs Mg doping")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()


def plot_overlay_dgmix_vs_doping_multi_t_ci(gibbs_uncertainty_csv: Path, out_png: Path) -> None:
    ensure_agg()
    import matplotlib.pyplot as plt

    rows = _read_rows(gibbs_uncertainty_csv)
    if not rows:
        raise ValueError(f"No rows found in {gibbs_uncertainty_csv}")

    grouped: dict[tuple[str, float], list[dict[str, str]]] = {}
    for row in rows:
        key = (str(row["mechanism_code"]), float(row["T_K"]))
        grouped.setdefault(key, []).append(row)

    plt.figure(figsize=(8, 5))
    for (mechanism, temp), group_rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        group_rows_sorted = sorted(group_rows, key=lambda r: float(r["doping_level_percent"]))
        x = [float(r["doping_level_percent"]) for r in group_rows_sorted]
        y = [float(r["free_energy_mixing_eV"]) for r in group_rows_sorted]
        y_low = [float(r["free_energy_ci_low_eV"]) for r in group_rows_sorted]
        y_high = [float(r["free_energy_ci_high_eV"]) for r in group_rows_sorted]

        (line,) = plt.plot(x, y, marker="o", linewidth=1.8, label=f"{mechanism}, T={temp:g}K")
        plt.fill_between(x, y_low, y_high, alpha=0.2, color=line.get_color())

    plt.xlabel("Mg doping (%)")
    plt.ylabel("ΔG_mix (eV)")
    plt.title("Overlay: ΔG_mix vs Mg doping (%) with 95% bootstrap CI")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()
