from __future__ import annotations

import csv
from pathlib import Path

from gan_mg.analysis.crossover import build_mechanism_crossover_rows, derive_mechanism_crossover_dataset


def _write_all_mechanisms(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "mechanism_code",
        "x_mg_cation",
        "doping_level_percent",
        "T_K",
        "free_energy_mixing_eV",
        "free_energy_mixing_eV_per_cation",
    ]
    rows = [
        {"mechanism_code": "vn", "x_mg_cation": 0.25, "doping_level_percent": 25.0, "T_K": 300.0, "free_energy_mixing_eV": -1.20, "free_energy_mixing_eV_per_cation": -0.20},
        {"mechanism_code": "mgi", "x_mg_cation": 0.25, "doping_level_percent": 25.0, "T_K": 300.0, "free_energy_mixing_eV": -1.10, "free_energy_mixing_eV_per_cation": -0.18},
        {"mechanism_code": "vn", "x_mg_cation": 0.25, "doping_level_percent": 25.0, "T_K": 600.0, "free_energy_mixing_eV": -1.00, "free_energy_mixing_eV_per_cation": -0.16},
        {"mechanism_code": "mgi", "x_mg_cation": 0.25, "doping_level_percent": 25.0, "T_K": 600.0, "free_energy_mixing_eV": -1.05, "free_energy_mixing_eV_per_cation": -0.17},
        {"mechanism_code": "vn", "x_mg_cation": 0.50, "doping_level_percent": 50.0, "T_K": 300.0, "free_energy_mixing_eV": -0.70, "free_energy_mixing_eV_per_cation": -0.11},
        {"mechanism_code": "mgi", "x_mg_cation": 0.50, "doping_level_percent": 50.0, "T_K": 300.0, "free_energy_mixing_eV": -0.80, "free_energy_mixing_eV_per_cation": -0.13},
        {"mechanism_code": "vn", "x_mg_cation": 0.50, "doping_level_percent": 50.0, "T_K": 600.0, "free_energy_mixing_eV": -0.72, "free_energy_mixing_eV_per_cation": -0.12},
        {"mechanism_code": "mgi", "x_mg_cation": 0.50, "doping_level_percent": 50.0, "T_K": 600.0, "free_energy_mixing_eV": -0.70, "free_energy_mixing_eV_per_cation": -0.11},
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_build_mechanism_crossover_rows_sign_and_preference() -> None:
    input_rows = [
        {"mechanism_code": "vn", "x_mg_cation": "0.25", "doping_level_percent": "25", "T_K": "300", "free_energy_mixing_eV": "-1.2", "free_energy_mixing_eV_per_cation": "-0.2"},
        {"mechanism_code": "mgi", "x_mg_cation": "0.25", "doping_level_percent": "25", "T_K": "300", "free_energy_mixing_eV": "-1.1", "free_energy_mixing_eV_per_cation": "-0.18"},
        {"mechanism_code": "vn", "x_mg_cation": "0.25", "doping_level_percent": "25", "T_K": "600", "free_energy_mixing_eV": "-1.0", "free_energy_mixing_eV_per_cation": "-0.16"},
        {"mechanism_code": "mgi", "x_mg_cation": "0.25", "doping_level_percent": "25", "T_K": "600", "free_energy_mixing_eV": "-1.05", "free_energy_mixing_eV_per_cation": "-0.17"},
    ]

    rows = build_mechanism_crossover_rows(input_rows)
    assert len(rows) == 2

    cold = rows[0]
    hot = rows[1]

    assert float(cold["delta_free_energy_eV"]) < 0
    assert cold["preferred_mechanism"] == "vn"

    assert float(hot["delta_free_energy_eV"]) > 0
    assert hot["preferred_mechanism"] == "mgi"


def test_derive_mechanism_crossover_dataset_writes_csv_and_plot(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "cross"
    _write_all_mechanisms(run_dir / "derived" / "all_mechanisms_gibbs_summary.csv")

    out_csv, out_png = derive_mechanism_crossover_dataset(run_dir)

    assert out_csv.exists()
    assert out_png.exists() and out_png.stat().st_size > 0

    with out_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 4
    assert {row["preferred_mechanism"] for row in rows} == {"vn", "mgi"}
