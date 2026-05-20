from __future__ import annotations

import csv
from pathlib import Path

import pytest

from gan_mg.science.per_structure import (
    build_per_structure_rows,
    canonicalize_mechanism,
    count_composition_from_structure,
)


def _write_extxyz(path: Path, symbols: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{len(symbols)}\n")
        f.write("energy=-10.5\n")
        for idx, symbol in enumerate(symbols):
            f.write(f"{symbol} {idx}.0 0.0 0.0\n")


def _write_results_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_extxyz_composition_counting(tmp_path: Path) -> None:
    extxyz = tmp_path / "cfg.extxyz"
    _write_extxyz(extxyz, ["Mg", "Ga", "Ga", "N", "N", "N"])

    mg_count, ga_count, n_count, total = count_composition_from_structure(extxyz)

    assert mg_count == 1
    assert ga_count == 2
    assert n_count == 3
    assert total == 6


def test_mechanism_canonicalization_variants() -> None:
    assert canonicalize_mechanism("MgGa+VN") == "vn"
    assert canonicalize_mechanism("vn") == "vn"
    assert canonicalize_mechanism("Mgi+2MgGa") == "mgi"
    assert canonicalize_mechanism("mGi pathway") == "mgi"
    assert canonicalize_mechanism("something_else") == "unknown"


def test_build_per_structure_rows_accepts_legacy_schema(tmp_path: Path) -> None:
    run_path = tmp_path / "legacy-run"
    _write_results_csv(
        run_path / "inputs" / "results.csv",
        ["structure_id", "mechanism", "energy_eV", "mg_count", "ga_count", "n_count"],
        [
            {
                "structure_id": "s001",
                "mechanism": "MgGa+VN",
                "energy_eV": -1.10,
                "mg_count": 1,
                "ga_count": 3,
                "n_count": 4,
            }
        ],
    )

    rows = build_per_structure_rows(run_path)

    assert rows[0]["structure_id"] == "s001"
    assert rows[0]["mechanism_code"] == "vn"
    assert rows[0]["mechanism_label"] == "MgGa+VN"
    assert rows[0]["energy_total_eV"] == -1.10


def test_build_per_structure_rows_accepts_archer2_schema(tmp_path: Path) -> None:
    run_path = tmp_path / "archer2-run"
    (run_path / "structures").mkdir(parents=True)
    _write_extxyz(run_path / "structures" / "relaxed-archer2-001.extxyz", ["Mg", "Ga", "Ga", "N", "N", "N"])
    _write_results_csv(
        run_path / "inputs" / "results.csv",
        [
            "structure_id",
            "mechanism_code",
            "doping_index",
            "x_mg_cation",
            "sample_num",
            "n_atoms",
            "relaxed_energy_eV",
            "energy_per_atom_eV",
            "relaxation_time_s",
            "input_structure_name",
            "relaxed_structure_name",
            "calculator_name",
            "model_name",
            "workflow_stage",
            "source_file",
        ],
        [
            {
                "structure_id": "archer2_001",
                "mechanism_code": "vn",
                "doping_index": 1,
                "x_mg_cation": 1 / 3,
                "sample_num": 7,
                "n_atoms": 6,
                "relaxed_energy_eV": -123.456,
                "energy_per_atom_eV": -20.576,
                "relaxation_time_s": 42.0,
                "input_structure_name": "input.extxyz",
                "relaxed_structure_name": "relaxed-archer2-001.extxyz",
                "calculator_name": "mace",
                "model_name": "research-model",
                "workflow_stage": "relaxed",
                "source_file": "ARCHER2/results.csv",
            }
        ],
    )

    rows = build_per_structure_rows(run_path)

    assert rows[0]["structure_id"] == "archer2_001"
    assert rows[0]["mechanism_code"] == "vn"
    assert rows[0]["mechanism_label"] == "vn"
    assert rows[0]["energy_total_eV"] == -123.456
    assert rows[0]["mg_count"] == 1
    assert rows[0]["ga_count"] == 2
    assert rows[0]["n_count"] == 3
    assert rows[0]["relaxed_structure_ref"].endswith("relaxed-archer2-001.extxyz")


def test_build_per_structure_rows_prefers_explicit_aliases_and_falls_back_per_row(tmp_path: Path) -> None:
    run_path = tmp_path / "mixed-run"
    _write_results_csv(
        run_path / "inputs" / "results.csv",
        [
            "structure_id",
            "mechanism",
            "mechanism_code",
            "energy_eV",
            "relaxed_energy_eV",
            "mg_count",
            "ga_count",
            "n_count",
        ],
        [
            {
                "structure_id": "s001",
                "mechanism": "MgGa+VN",
                "mechanism_code": "mgi",
                "energy_eV": -99.0,
                "relaxed_energy_eV": -1.23,
                "mg_count": 1,
                "ga_count": 2,
                "n_count": 3,
            },
            {
                "structure_id": "s002",
                "mechanism": "MgGa+VN",
                "mechanism_code": "",
                "energy_eV": -2.34,
                "relaxed_energy_eV": "",
                "mg_count": 1,
                "ga_count": 3,
                "n_count": 4,
            },
        ],
    )

    rows = build_per_structure_rows(run_path)

    assert rows[0]["structure_id"] == "s001"
    assert rows[0]["mechanism_code"] == "mgi"
    assert rows[0]["mechanism_label"] == "mgi"
    assert rows[0]["energy_total_eV"] == -1.23
    assert rows[1]["structure_id"] == "s002"
    assert rows[1]["mechanism_code"] == "vn"
    assert rows[1]["mechanism_label"] == "MgGa+VN"
    assert rows[1]["energy_total_eV"] == -2.34


def test_build_per_structure_rows_reports_accepted_mechanism_aliases(tmp_path: Path) -> None:
    run_path = tmp_path / "missing-mechanism-run"
    _write_results_csv(
        run_path / "inputs" / "results.csv",
        ["structure_id", "relaxed_energy_eV", "mg_count", "ga_count", "n_count"],
        [
            {
                "structure_id": "s001",
                "relaxed_energy_eV": -1.0,
                "mg_count": 1,
                "ga_count": 2,
                "n_count": 3,
            }
        ],
    )

    with pytest.raises(ValueError, match="row 2 missing mechanism/mechanism_code"):
        build_per_structure_rows(run_path)


def test_build_per_structure_rows_reports_accepted_energy_aliases(tmp_path: Path) -> None:
    run_path = tmp_path / "missing-energy-run"
    _write_results_csv(
        run_path / "inputs" / "results.csv",
        ["structure_id", "mechanism_code", "mg_count", "ga_count", "n_count"],
        [
            {
                "structure_id": "s001",
                "mechanism_code": "vn",
                "mg_count": 1,
                "ga_count": 2,
                "n_count": 3,
            }
        ],
    )

    with pytest.raises(ValueError, match="row 2 missing relaxed_energy_eV/energy_eV"):
        build_per_structure_rows(run_path)


def test_cli_derive_creates_per_structure_csv(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"

    run_id = "derive-test"
    run_dir = tmp_path / "runs"
    run_path = run_dir / run_id
    (run_path / "inputs").mkdir(parents=True)
    (run_path / "structures").mkdir(parents=True)

    results_path = run_path / "inputs" / "results.csv"
    with results_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["structure_id", "mechanism", "energy_eV"])
        writer.writeheader()
        writer.writerow({"structure_id": "s002", "mechanism": "Mgi+2MgGa", "energy_eV": -1.20})
        writer.writerow({"structure_id": "s001", "mechanism": "MgGa+VN", "energy_eV": -1.10})

    _write_extxyz(run_path / "structures" / "s001.extxyz", ["Mg", "Ga", "N", "N"])
    _write_extxyz(run_path / "structures" / "s002.extxyz", ["Mg", "Mg", "Ga", "N", "N", "N"])

    subprocess.run(
        [
            sys.executable,
            "-m",
            "gan_mg.cli",
            "derive",
            "--run-dir",
            str(run_dir),
            "--run-id",
            run_id,
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    out_csv = run_path / "derived" / "per_structure.csv"
    assert out_csv.exists()

    with out_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert [row["structure_id"] for row in rows] == ["s001", "s002"]
    assert set(rows[0].keys()) == {
        "structure_id",
        "mechanism_code",
        "mechanism_label",
        "energy_total_eV",
        "mg_count",
        "ga_count",
        "n_count",
        "site_count_total",
        "x_mg_cation",
        "doping_level_percent",
        "relaxed_structure_ref",
    }


def test_cli_derive_rejects_raw_dataset_with_derived_thermo_fields(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"

    run_id = "derive-invalid-raw-boundary"
    run_dir = tmp_path / "runs"
    run_path = run_dir / run_id
    (run_path / "inputs").mkdir(parents=True)

    with (run_path / "inputs" / "results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["structure_id", "mechanism", "energy_eV", "free_energy_mix_eV"])
        writer.writeheader()
        writer.writerow(
            {"structure_id": "s001", "mechanism": "MgGa+VN", "energy_eV": -1.10, "free_energy_mix_eV": -0.4}
        )

    proc = subprocess.run(
        [sys.executable, "-m", "gan_mg.cli", "derive", "--run-dir", str(run_dir), "--run-id", run_id],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )

    assert proc.returncode != 0
    assert "boundary violation" in (proc.stderr + proc.stdout)
