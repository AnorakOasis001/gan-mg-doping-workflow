from __future__ import annotations

import csv
from pathlib import Path

import pytest

from gan_mg.analysis.phase_boundary import derive_phase_boundary_dataset
from gan_mg.viz.phase_map import plot_phase_map


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_derive_phase_boundary_dataset_single_flip(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    phase_map_csv = run_dir / "derived" / "phase_map.csv"
    _write_csv(
        phase_map_csv,
        ["T_K", "x_mg_cation", "doping_level_percent", "winner_mechanism", "delta_G_eV", "robust"],
        [
            {"T_K": 1000.0, "x_mg_cation": 0.0, "doping_level_percent": 0.0, "winner_mechanism": "vn", "delta_G_eV": 0.2, "robust": True},
            {"T_K": 1000.0, "x_mg_cation": 0.1, "doping_level_percent": 10.0, "winner_mechanism": "vn", "delta_G_eV": 0.1, "robust": True},
            {"T_K": 1000.0, "x_mg_cation": 0.2, "doping_level_percent": 20.0, "winner_mechanism": "mgi", "delta_G_eV": 0.3, "robust": True},
            {"T_K": 1000.0, "x_mg_cation": 0.3, "doping_level_percent": 30.0, "winner_mechanism": "mgi", "delta_G_eV": 0.4, "robust": True},
        ],
    )

    out_csv = derive_phase_boundary_dataset(run_dir)

    with out_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    row = rows[0]
    assert float(row["x_boundary"]) == pytest.approx(0.15)
    assert row["mech_left"] == "vn"
    assert row["mech_right"] == "mgi"
    assert row["robust"] == "True"


def test_derive_phase_boundary_dataset_multiple_flips(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    phase_map_csv = run_dir / "derived" / "phase_map.csv"
    _write_csv(
        phase_map_csv,
        ["T_K", "x_mg_cation", "doping_level_percent", "winner_mechanism", "delta_G_eV", "robust"],
        [
            {"T_K": 1000.0, "x_mg_cation": 0.0, "doping_level_percent": 0.0, "winner_mechanism": "vn", "delta_G_eV": 0.2, "robust": True},
            {"T_K": 1000.0, "x_mg_cation": 0.1, "doping_level_percent": 10.0, "winner_mechanism": "mgi", "delta_G_eV": 0.1, "robust": True},
            {"T_K": 1000.0, "x_mg_cation": 0.2, "doping_level_percent": 20.0, "winner_mechanism": "vn", "delta_G_eV": 0.3, "robust": True},
        ],
    )

    out_csv = derive_phase_boundary_dataset(run_dir)

    with out_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert [float(row["x_boundary"]) for row in rows] == pytest.approx([0.05, 0.15])


def test_plot_phase_map_with_boundary_overlay_writes_png(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    run_dir = tmp_path / "run"
    phase_map_csv = run_dir / "derived" / "phase_map.csv"
    boundary_csv = run_dir / "derived" / "phase_boundary.csv"

    _write_csv(
        phase_map_csv,
        ["T_K", "x_mg_cation", "doping_level_percent", "winner_mechanism", "delta_G_eV", "robust"],
        [
            {"T_K": 1000.0, "x_mg_cation": 0.1, "doping_level_percent": 10.0, "winner_mechanism": "vn", "delta_G_eV": 0.2, "robust": True},
            {"T_K": 1000.0, "x_mg_cation": 0.2, "doping_level_percent": 20.0, "winner_mechanism": "mgi", "delta_G_eV": 0.3, "robust": True},
        ],
    )
    _write_csv(
        boundary_csv,
        ["T_K", "x_boundary", "doping_level_percent", "mech_left", "mech_right", "robust", "delta_G_at_boundary_eV"],
        [
            {"T_K": 1000.0, "x_boundary": 0.15, "doping_level_percent": 15.0, "mech_left": "vn", "mech_right": "mgi", "robust": True, "delta_G_at_boundary_eV": 0.2}
        ],
    )

    out_png = run_dir / "figures" / "phase_map.png"
    plot_phase_map(phase_map_csv=phase_map_csv, out_png=out_png, boundary_csv=boundary_csv)

    assert out_png.exists()
