from __future__ import annotations

from pathlib import Path

from gan_mg.analysis.thermo import plot_thermo_vs_T, sweep_thermo_from_csv


DEFAULT_THERMO_T_MIN = 300.0
DEFAULT_THERMO_T_MAX = 1500.0
DEFAULT_THERMO_NT = 25


def regenerate_thermo_figure(
    run_dir: Path,
    energy_col: str = "energy_eV",
    t_min: float = DEFAULT_THERMO_T_MIN,
    t_max: float = DEFAULT_THERMO_T_MAX,
    n_t: int = DEFAULT_THERMO_NT,
) -> Path:
    """Regenerate the canonical thermodynamic figure for a run.

    Reads `<run_dir>/inputs/results.csv` and writes
    `<run_dir>/figures/thermo_vs_T.png`.
    """
    if n_t < 2:
        raise ValueError("n_t must be >= 2")

    run_dir = Path(run_dir)
    csv_path = run_dir / "inputs" / "results.csv"

    t_values = [t_min + i * (t_max - t_min) / (n_t - 1) for i in range(n_t)]
    rows = sweep_thermo_from_csv(csv_path, t_values, energy_col=energy_col)

    out_png = run_dir / "figures" / "thermo_vs_T.png"
    plot_thermo_vs_T(rows, out_png)
    return out_png
