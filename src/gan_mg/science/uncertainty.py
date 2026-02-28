from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from gan_mg.science.constants import K_B_EV_PER_K

UNCERTAINTY_COLUMNS = (
    "mechanism_code",
    "x_mg_cation",
    "doping_level_percent",
    "T_K",
    "num_configurations",
    "free_energy_mixing_eV",
    "free_energy_ci_low_eV",
    "free_energy_ci_high_eV",
    "mixing_energy_avg_eV",
    "mixing_energy_ci_low_eV",
    "mixing_energy_ci_high_eV",
    "weight_max",
    "ess",
    "top_5_weight_sum",
    "n_mismatch_mean",
)


def _stable_logsumexp(x: np.ndarray) -> float:

def compute_weights(dE: np.ndarray, T: float, kB: float, top_k: int = 5) -> tuple[np.ndarray, float, float, float]:
    if dE.size == 0:
        raise ValueError("dE must be non-empty")
    if T <= 0:
        raise ValueError("temperature must be > 0")

    x = -dE / (kB * T)
    m = float(np.max(x))
    w_unnorm = np.exp(x - m)
    w = w_unnorm / np.sum(w_unnorm)

    weight_max = float(np.max(w))
    ess = float(1.0 / np.sum(w**2))
    k = max(1, min(top_k, int(w.size)))
    top_k_sum = float(np.sort(w)[-k:].sum())
    return w, weight_max, ess, top_k_sum


def bootstrap_gibbs_for_group(
    dE: np.ndarray,
    Emin: float,
    T: float,
    kB: float,
    B: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float, float, float]:
    if B < 1:
        raise ValueError("B must be >= 1")

    x = -dE / (kB * T)
    logz = _stable_logsumexp(x)
    g_mean = float(Emin - kB * T * logz)

    w, _, _, _ = compute_weights(dE=dE, T=T, kB=kB)
    emix_mean = float(Emin + np.sum(w * dE))

    n = int(dE.size)
    boot_g = np.empty(B, dtype=float)
    boot_e = np.empty(B, dtype=float)

    for b in range(B):
        sample = dE[rng.integers(0, n, size=n)]
        logz_b = _stable_logsumexp(-sample / (kB * T))
        boot_g[b] = Emin - kB * T * logz_b

        w_b, _, _, _ = compute_weights(dE=sample, T=T, kB=kB)
        boot_e[b] = Emin + float(np.sum(w_b * sample))

    g_low, g_high = np.quantile(boot_g, [0.025, 0.975])
    e_low, e_high = np.quantile(boot_e, [0.025, 0.975])
    return g_mean, float(g_low), float(g_high), emix_mean, float(e_low), float(e_high)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(UNCERTAINTY_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in UNCERTAINTY_COLUMNS})


def validate_uncertainty_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("gibbs_uncertainty.csv must contain at least one row")

    for i, row in enumerate(rows, start=2):
        for key in (
            "mechanism_code",
            "x_mg_cation",
            "doping_level_percent",
            "T_K",
            "num_configurations",
            "free_energy_mixing_eV",
            "free_energy_ci_low_eV",
            "free_energy_ci_high_eV",
            "mixing_energy_avg_eV",
            "mixing_energy_ci_low_eV",
            "mixing_energy_ci_high_eV",
            "weight_max",
            "ess",
        ):
            if key not in row:
                raise ValueError(f"row {i} missing required '{key}'")

        n = int(row["num_configurations"])
        if n < 1:
            raise ValueError(f"row {i} has invalid num_configurations")

        for key in (
            "x_mg_cation",
            "doping_level_percent",
            "T_K",
            "free_energy_mixing_eV",
            "free_energy_ci_low_eV",
            "free_energy_ci_high_eV",
            "mixing_energy_avg_eV",
            "mixing_energy_ci_low_eV",
            "mixing_energy_ci_high_eV",
            "weight_max",
            "ess",
        ):
            if not math.isfinite(float(row[key])):
                raise ValueError(f"row {i} has non-finite '{key}'")

        g_low = float(row["free_energy_ci_low_eV"])
        g_mean = float(row["free_energy_mixing_eV"])
        g_high = float(row["free_energy_ci_high_eV"])
        if not (g_low <= g_mean <= g_high):
            raise ValueError(f"row {i} has invalid free-energy CI ordering")

        e_low = float(row["mixing_energy_ci_low_eV"])
        e_mean = float(row["mixing_energy_avg_eV"])
        e_high = float(row["mixing_energy_ci_high_eV"])
        if not (e_low <= e_mean <= e_high):
            raise ValueError(f"row {i} has invalid mixing-energy CI ordering")

        ess = float(row["ess"])
        if ess < 1.0 or ess > float(n):
            raise ValueError(f"row {i} has ess outside [1, N]")


def derive_gibbs_uncertainty_dataset(
    run_dir: Path,
    temperatures_K: list[float],
    n_bootstrap: int = 200,
    seed: int = 0,
) -> Path:
    if not temperatures_K:
        raise ValueError("at least one temperature must be provided")

    run_path = Path(run_dir)
    mixing_path = run_path / "derived" / "per_structure_mixing.csv"
    if not mixing_path.exists():
        raise FileNotFoundError(
            f"per_structure_mixing.csv not found: {mixing_path}. Run `ganmg mix --run-id <id>` first."
        )

    rows = _load_rows(mixing_path)
    if not rows:
        raise ValueError("per_structure_mixing.csv must contain at least one row")

    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["mechanism_code"]), float(row["x_mg_cation"]))].append(row)

    rng = np.random.default_rng(seed)
    out_rows: list[dict[str, Any]] = []

    for mechanism_code, x_mg in sorted(groups):
        group_rows = groups[(mechanism_code, x_mg)]
        representative = group_rows[0]
        dE = np.asarray([float(row["dE_eV"]) for row in group_rows], dtype=float)
        energies = np.asarray([float(row["energy_mixing_eV"]) for row in group_rows], dtype=float)
        e_min = float(np.min(energies))

        n_mismatch_mean: float | None = None
        if all("n_mismatch" in row and str(row["n_mismatch"]).strip() != "" for row in group_rows):
            n_mismatch_mean = float(np.mean([float(row["n_mismatch"]) for row in group_rows]))

        for t_k in sorted(float(t) for t in temperatures_K):
            g_mean, g_low, g_high, e_mean, e_low, e_high = bootstrap_gibbs_for_group(
                dE=dE,
                Emin=e_min,
                T=t_k,
                kB=K_B_EV_PER_K,
                B=n_bootstrap,
                rng=rng,
            )
            _, weight_max, ess, top5 = compute_weights(dE=dE, T=t_k, kB=K_B_EV_PER_K)

            row_out: dict[str, Any] = {
                "mechanism_code": mechanism_code,
                "x_mg_cation": float(x_mg),
                "doping_level_percent": float(representative["doping_level_percent"]),
                "T_K": float(t_k),
                "num_configurations": int(dE.size),
                "free_energy_mixing_eV": g_mean,
                "free_energy_ci_low_eV": g_low,
                "free_energy_ci_high_eV": g_high,
                "mixing_energy_avg_eV": e_mean,
                "mixing_energy_ci_low_eV": e_low,
                "mixing_energy_ci_high_eV": e_high,
                "weight_max": weight_max,
                "ess": ess,
                "top_5_weight_sum": top5,
            }
            if n_mismatch_mean is not None:
                row_out["n_mismatch_mean"] = n_mismatch_mean
            out_rows.append(row_out)

    out_rows.sort(key=lambda r: (str(r["mechanism_code"]), float(r["x_mg_cation"]), float(r["T_K"])))
    validate_uncertainty_rows(out_rows)

    out_csv = run_path / "derived" / "gibbs_uncertainty.csv"
    _write_rows(out_csv, out_rows)
    return out_csv
