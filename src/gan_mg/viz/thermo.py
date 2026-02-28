from __future__ import annotations

from pathlib import Path

from gan_mg.analysis.thermo import ThermoSweepRow


def plot_thermo_vs_T(rows: list[ThermoSweepRow], out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")  # headless backend (CI-safe)
    import matplotlib.pyplot as plt

    temperature_K = [r["temperature_K"] for r in rows]
    free_energy_mix_eV = [r["free_energy_mix_eV"] for r in rows]
    mixing_energy_avg_eV = [r["mixing_energy_avg_eV"] for r in rows]

    plt.figure()
    plt.plot(temperature_K, free_energy_mix_eV, label="free_energy_mix_eV")
    plt.plot(temperature_K, mixing_energy_avg_eV, label="mixing_energy_avg_eV")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Energy (eV)")
    plt.legend()

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
