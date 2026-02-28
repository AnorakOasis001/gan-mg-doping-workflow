from __future__ import annotations

import csv
from pathlib import Path

import pytest

from gan_mg.analysis.phase_map import EPS, derive_phase_map_dataset


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str | float | bool]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_derive_phase_map_from_crossover_uncertainty(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    crossover_uncertainty = run_dir / "derived" / "crossover_uncertainty.csv"
    _write_csv(
        crossover_uncertainty,
        [
            "x_mg_cation",
            "doping_level_percent",
            "T_K",
            "delta_G_mean_eV",
            "delta_G_ci_low_eV",
            "delta_G_ci_high_eV",
            "preferred",
            "robust",
        ],
        [
            {
                "x_mg_cation": 0.50,
                "doping_level_percent": 50.0,
                "T_K": 350.0,
                "delta_G_mean_eV": -0.12,
                "delta_G_ci_low_eV": -0.20,
                "delta_G_ci_high_eV": -0.05,
                "preferred": "vn",
                "robust": True,
            },
            {
                "x_mg_cation": 0.25,
                "doping_level_percent": 25.0,
                "T_K": 300.0,
                "delta_G_mean_eV": 0.04,
                "delta_G_ci_low_eV": -0.01,
                "delta_G_ci_high_eV": 0.09,
                "preferred": "mgi",
                "robust": False,
            },
        ],
    )

    out_path = derive_phase_map_dataset(run_dir)
    assert out_path.exists()

    with out_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows
    assert list(rows[0].keys()) == [
        "T_K",
        "x_mg_cation",
        "doping_level_percent",
        "winner_mechanism",
        "delta_G_eV",
        "delta_G_ci_low_eV",
        "delta_G_ci_high_eV",
        "robust",
    ]

    assert [float(r["T_K"]) for r in rows] == [300.0, 350.0]
    assert [r["winner_mechanism"] for r in rows] == ["mgi", "vn"]
    assert [float(r["delta_G_eV"]) for r in rows] == [0.04, 0.12]


def test_derive_phase_map_falls_back_to_mechanism_crossover(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    mechanism_crossover = run_dir / "derived" / "mechanism_crossover.csv"
    _write_csv(
        mechanism_crossover,
        [
            "x_mg_cation",
            "doping_level_percent",
            "T_K",
            "delta_free_energy_eV",
            "delta_free_energy_eV_per_cation",
            "preferred_mechanism",
        ],
        [
            {
                "x_mg_cation": 0.4,
                "doping_level_percent": 40.0,
                "T_K": 500.0,
                "delta_free_energy_eV": -0.02,
                "delta_free_energy_eV_per_cation": "",
                "preferred_mechanism": "vn",
            }
        ],
    )

    out_path = derive_phase_map_dataset(run_dir)
    with out_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0]["winner_mechanism"] == "vn"
    assert float(rows[0]["delta_G_eV"]) == 0.02
    assert "delta_G_ci_low_eV" not in rows[0]
    assert "delta_G_ci_high_eV" not in rows[0]
    assert "robust" not in rows[0]


def test_derive_phase_map_falls_back_to_gibbs_summary_with_tie(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    gibbs_summary = run_dir / "derived" / "gibbs_summary.csv"
    _write_csv(
        gibbs_summary,
        [
            "mechanism_code",
            "x_mg_cation",
            "doping_level_percent",
            "T_K",
            "free_energy_mixing_eV",
        ],
        [
            {
                "mechanism_code": "mgi",
                "x_mg_cation": 0.25,
                "doping_level_percent": 25.0,
                "T_K": 300.0,
                "free_energy_mixing_eV": -0.80,
            },
            {
                "mechanism_code": "vn",
                "x_mg_cation": 0.25,
                "doping_level_percent": 25.0,
                "T_K": 300.0,
                "free_energy_mixing_eV": -0.90,
            },
            {
                "mechanism_code": "mgi",
                "x_mg_cation": 0.50,
                "doping_level_percent": 50.0,
                "T_K": 350.0,
                "free_energy_mixing_eV": -0.60,
            },
            {
                "mechanism_code": "vn",
                "x_mg_cation": 0.50,
                "doping_level_percent": 50.0,
                "T_K": 350.0,
                "free_energy_mixing_eV": -0.60 + (EPS / 2.0),
            },
        ],
    )

    out_path = derive_phase_map_dataset(run_dir)
    with out_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2

    first = rows[0]
    assert first["winner_mechanism"] == "vn"
    assert float(first["delta_G_eV"]) == pytest.approx(0.1)

    second = rows[1]
    assert second["winner_mechanism"] == "tie"
    assert float(second["delta_G_eV"]) < EPS
