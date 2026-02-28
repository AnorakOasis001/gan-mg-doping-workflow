from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest

from gan_mg.analysis.phase_map import PHASE_MAP_COLUMNS, build_phase_map_rows

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


def _write_crossover_uncertainty_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "x_mg_cation",
        "doping_level_percent",
        "T_K",
        "delta_G_mean_eV",
        "delta_G_ci_low_eV",
        "delta_G_ci_high_eV",
        "preferred",
        "robust",
    ]
    rows = [
        {"x_mg_cation": 0.50, "doping_level_percent": 50.0, "T_K": 600.0, "delta_G_mean_eV": 0.03, "delta_G_ci_low_eV": 0.01, "delta_G_ci_high_eV": 0.05, "preferred": "mgi", "robust": True},
        {"x_mg_cation": 0.25, "doping_level_percent": 25.0, "T_K": 300.0, "delta_G_mean_eV": -0.10, "delta_G_ci_low_eV": -0.12, "delta_G_ci_high_eV": -0.08, "preferred": "vn", "robust": True},
        {"x_mg_cation": 0.25, "doping_level_percent": 25.0, "T_K": 600.0, "delta_G_mean_eV": 0.00, "delta_G_ci_low_eV": -0.02, "delta_G_ci_high_eV": 0.02, "preferred": "uncertain", "robust": False},
        {"x_mg_cation": 0.50, "doping_level_percent": 50.0, "T_K": 300.0, "delta_G_mean_eV": -0.01, "delta_G_ci_low_eV": -0.03, "delta_G_ci_high_eV": 0.01, "preferred": "uncertain", "robust": False},
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_build_phase_map_rows_sort_and_columns() -> None:
    rows = [
        {"x_mg_cation": "0.5", "doping_level_percent": "50", "T_K": "600", "delta_G_mean_eV": "0.03", "delta_G_ci_low_eV": "0.01", "delta_G_ci_high_eV": "0.05", "preferred": "mgi", "robust": "true"},
        {"x_mg_cation": "0.25", "doping_level_percent": "25", "T_K": "300", "delta_G_mean_eV": "-0.10", "delta_G_ci_low_eV": "-0.12", "delta_G_ci_high_eV": "-0.08", "preferred": "vn", "robust": "true"},
    ]
    out = build_phase_map_rows(rows)
    assert list(out[0].keys()) == list(PHASE_MAP_COLUMNS)
    assert out[0]["x_mg_cation"] == 0.25
    assert out[1]["x_mg_cation"] == 0.5


def test_cli_phase_map_generates_dataset_and_plot(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")

    run_dir = tmp_path / "runs"
    run_id = "phase-map-test"
    run_path = run_dir / run_id
    _write_crossover_uncertainty_csv(run_path / "derived" / "crossover_uncertainty.csv")

    _run_cli("phase-map", "--run-dir", str(run_dir), "--run-id", run_id, cwd=tmp_path)

    phase_csv = run_path / "derived" / "phase_map.csv"
    pref_png = run_path / "figures" / "phase_map_preference.png"

    assert phase_csv.exists() and phase_csv.stat().st_size > 0
    assert pref_png.exists() and pref_png.stat().st_size > 0

    with phase_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows
    for col in PHASE_MAP_COLUMNS:
        assert col in rows[0]

    key_pairs = [(float(r["x_mg_cation"]), float(r["T_K"])) for r in rows]
    assert key_pairs == sorted(key_pairs)
