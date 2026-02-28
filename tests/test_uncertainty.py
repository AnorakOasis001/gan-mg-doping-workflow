from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gan_mg.analysis.crossover_uncertainty import build_crossover_uncertainty_rows
from gan_mg.science.uncertainty import bootstrap_gibbs_for_group, compute_weights

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_cli(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
    return subprocess.run(
        [sys.executable, "-m", "gan_mg.cli", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )


def _write_per_structure_mixing_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "structure_id",
        "mechanism_code",
        "mechanism_label",
        "energy_mixing_eV",
        "dE_eV",
        "x_mg_cation",
        "doping_level_percent",
        "site_count_total",
        "mg_count",
        "ga_count",
        "n_mismatch",
    ]

    rows = [
        {"structure_id": "vn-25-a", "mechanism_code": "vn", "mechanism_label": "vn", "energy_mixing_eV": -1.00, "dE_eV": 0.0, "x_mg_cation": 0.25, "doping_level_percent": 25.0, "site_count_total": 12, "mg_count": 2, "ga_count": 4, "n_mismatch": 0.0},
        {"structure_id": "vn-25-b", "mechanism_code": "vn", "mechanism_label": "vn", "energy_mixing_eV": -1.00, "dE_eV": 0.0, "x_mg_cation": 0.25, "doping_level_percent": 25.0, "site_count_total": 12, "mg_count": 2, "ga_count": 4, "n_mismatch": 0.0},
        {"structure_id": "vn-25-c", "mechanism_code": "vn", "mechanism_label": "vn", "energy_mixing_eV": -1.00, "dE_eV": 0.0, "x_mg_cation": 0.25, "doping_level_percent": 25.0, "site_count_total": 12, "mg_count": 2, "ga_count": 4, "n_mismatch": 0.0},
        {"structure_id": "mgi-25-a", "mechanism_code": "mgi", "mechanism_label": "mgi", "energy_mixing_eV": -0.80, "dE_eV": 0.0, "x_mg_cation": 0.25, "doping_level_percent": 25.0, "site_count_total": 12, "mg_count": 2, "ga_count": 4, "n_mismatch": 0.0},
        {"structure_id": "mgi-25-b", "mechanism_code": "mgi", "mechanism_label": "mgi", "energy_mixing_eV": -0.80, "dE_eV": 0.0, "x_mg_cation": 0.25, "doping_level_percent": 25.0, "site_count_total": 12, "mg_count": 2, "ga_count": 4, "n_mismatch": 0.0},
        {"structure_id": "mgi-25-c", "mechanism_code": "mgi", "mechanism_label": "mgi", "energy_mixing_eV": -0.80, "dE_eV": 0.0, "x_mg_cation": 0.25, "doping_level_percent": 25.0, "site_count_total": 12, "mg_count": 2, "ga_count": 4, "n_mismatch": 0.0},
        {"structure_id": "vn-50-a", "mechanism_code": "vn", "mechanism_label": "vn", "energy_mixing_eV": -0.70, "dE_eV": 0.0, "x_mg_cation": 0.50, "doping_level_percent": 50.0, "site_count_total": 12, "mg_count": 2, "ga_count": 4, "n_mismatch": 1.0},
        {"structure_id": "vn-50-b", "mechanism_code": "vn", "mechanism_label": "vn", "energy_mixing_eV": -0.68, "dE_eV": 0.02, "x_mg_cation": 0.50, "doping_level_percent": 50.0, "site_count_total": 12, "mg_count": 2, "ga_count": 4, "n_mismatch": 1.0},
        {"structure_id": "vn-50-c", "mechanism_code": "vn", "mechanism_label": "vn", "energy_mixing_eV": -0.66, "dE_eV": 0.04, "x_mg_cation": 0.50, "doping_level_percent": 50.0, "site_count_total": 12, "mg_count": 2, "ga_count": 4, "n_mismatch": 1.0},
        {"structure_id": "mgi-50-a", "mechanism_code": "mgi", "mechanism_label": "mgi", "energy_mixing_eV": -0.69, "dE_eV": 0.0, "x_mg_cation": 0.50, "doping_level_percent": 50.0, "site_count_total": 12, "mg_count": 2, "ga_count": 4, "n_mismatch": 1.0},
        {"structure_id": "mgi-50-b", "mechanism_code": "mgi", "mechanism_label": "mgi", "energy_mixing_eV": -0.67, "dE_eV": 0.02, "x_mg_cation": 0.50, "doping_level_percent": 50.0, "site_count_total": 12, "mg_count": 2, "ga_count": 4, "n_mismatch": 1.0},
        {"structure_id": "mgi-50-c", "mechanism_code": "mgi", "mechanism_label": "mgi", "energy_mixing_eV": -0.65, "dE_eV": 0.04, "x_mg_cation": 0.50, "doping_level_percent": 50.0, "site_count_total": 12, "mg_count": 2, "ga_count": 4, "n_mismatch": 1.0},
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_bootstrap_bounds_and_weights() -> None:
    d_e = np.array([0.0, 0.05, 0.2], dtype=float)
    rng = np.random.default_rng(0)
    g_mean, g_low, g_high, e_mean, e_low, e_high = bootstrap_gibbs_for_group(
        dE=d_e,
        Emin=-1.0,
        T=300.0,
        kB=8.617333262145e-5,
        B=50,
        rng=rng,
    )
    assert g_low <= g_mean <= g_high
    assert e_low <= e_mean <= e_high

    _, weight_max, ess, top_5 = compute_weights(d_e, T=300.0, kB=8.617333262145e-5)
    assert 0.0 <= weight_max <= 1.0
    assert 1.0 <= ess <= 3.0
    assert 0.0 <= top_5 <= 1.0


def test_build_crossover_uncertainty_rows_classification() -> None:
    rows = [
        {"mechanism_code": "vn", "x_mg_cation": "0.25", "doping_level_percent": "25", "T_K": "300", "free_energy_mixing_eV": "-1.0", "free_energy_ci_low_eV": "-1.0", "free_energy_ci_high_eV": "-1.0"},
        {"mechanism_code": "mgi", "x_mg_cation": "0.25", "doping_level_percent": "25", "T_K": "300", "free_energy_mixing_eV": "-0.8", "free_energy_ci_low_eV": "-0.8", "free_energy_ci_high_eV": "-0.8"},
        {"mechanism_code": "vn", "x_mg_cation": "0.5", "doping_level_percent": "50", "T_K": "300", "free_energy_mixing_eV": "-0.7", "free_energy_ci_low_eV": "-0.75", "free_energy_ci_high_eV": "-0.65"},
        {"mechanism_code": "mgi", "x_mg_cation": "0.5", "doping_level_percent": "50", "T_K": "300", "free_energy_mixing_eV": "-0.69", "free_energy_ci_low_eV": "-0.74", "free_energy_ci_high_eV": "-0.64"},
    ]

    out = build_crossover_uncertainty_rows(rows)
    assert len(out) == 2
    assert out[0]["preferred"] == "vn"
    assert out[0]["robust"] is True
    assert out[1]["preferred"] == "uncertain"
    assert out[1]["robust"] is False


def test_cli_uncertainty_outputs_and_crossover(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")

    run_dir = tmp_path / "runs"
    run_id = "uncertainty-test"
    run_path = run_dir / run_id
    run_path.mkdir(parents=True, exist_ok=True)

    _write_per_structure_mixing_csv(run_path / "derived" / "per_structure_mixing.csv")

    _run_cli(
        "uncertainty",
        "--run-dir",
        str(run_dir),
        "--run-id",
        run_id,
        "--temps",
        "1,300",
        "--n-bootstrap",
        "50",
        "--seed",
        "0",
        cwd=tmp_path,
    )

    uncertainty_csv = run_path / "derived" / "gibbs_uncertainty.csv"
    overlay_ci_png = run_path / "figures" / "overlay_dGmix_vs_doping_multiT_ci.png"
    crossover_csv = run_path / "derived" / "crossover_uncertainty.csv"

    assert uncertainty_csv.exists() and uncertainty_csv.stat().st_size > 0
    assert overlay_ci_png.exists() and overlay_ci_png.stat().st_size > 0
    assert crossover_csv.exists() and crossover_csv.stat().st_size > 0

    with uncertainty_csv.open("r", encoding="utf-8", newline="") as f:
        out_rows = list(csv.DictReader(f))

    assert len(out_rows) == 2 * 2 * 2
    for row in out_rows:
        g_low = float(row["free_energy_ci_low_eV"])
        g_mean = float(row["free_energy_mixing_eV"])
        g_high = float(row["free_energy_ci_high_eV"])
        assert g_low <= g_mean <= g_high

        e_low = float(row["mixing_energy_ci_low_eV"])
        e_mean = float(row["mixing_energy_avg_eV"])
        e_high = float(row["mixing_energy_ci_high_eV"])
        assert e_low <= e_mean <= e_high

        n = int(row["num_configurations"])
        ess = float(row["ess"])
        assert 1.0 <= ess <= n

    with crossover_csv.open("r", encoding="utf-8", newline="") as f:
        crossover_rows = list(csv.DictReader(f))

    assert crossover_rows
    assert any(row["robust"].lower() == "true" and row["preferred"] == "vn" for row in crossover_rows)
