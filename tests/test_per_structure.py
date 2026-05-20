from __future__ import annotations

import csv
from pathlib import Path

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


def test_build_per_structure_rows_supports_archer2_alias_columns(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "real-test"
    (run_dir / "inputs").mkdir(parents=True)
    (run_dir / "structures").mkdir(parents=True)
    _write_extxyz(run_dir / "structures" / "s001.extxyz", ["Mg", "Ga", "N", "N"])

    with (run_dir / "inputs" / "results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["structure_id", "mechanism_code", "relaxed_energy_eV"],
        )
        writer.writeheader()
        writer.writerow({"structure_id": "s001", "mechanism_code": "MgGa+VN", "relaxed_energy_eV": -1.23})

    rows = build_per_structure_rows(run_dir)
    assert len(rows) == 1
    assert rows[0]["mechanism_label"] == "MgGa+VN"
    assert rows[0]["mechanism_code"] == "vn"
    assert rows[0]["energy_total_eV"] == -1.23


def test_build_per_structure_rows_supports_mixed_alias_columns(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "mixed-test"
    (run_dir / "inputs").mkdir(parents=True)
    (run_dir / "structures").mkdir(parents=True)
    _write_extxyz(run_dir / "structures" / "s001.extxyz", ["Mg", "Ga", "N", "N"])
    _write_extxyz(run_dir / "structures" / "s002.extxyz", ["Mg", "Mg", "Ga", "N", "N", "N"])

    with (run_dir / "inputs" / "results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "structure_id",
                "mechanism",
                "energy_eV",
                "mechanism_code",
                "relaxed_energy_eV",
            ],
        )
        writer.writeheader()
        writer.writerow({"structure_id": "s001", "mechanism": "Mgi+2MgGa", "energy_eV": -1.1})
        writer.writerow({"structure_id": "s002", "mechanism_code": "MgGa+VN", "relaxed_energy_eV": -1.2})

    rows = build_per_structure_rows(run_dir)
    assert [row["structure_id"] for row in rows] == ["s001", "s002"]
    assert rows[0]["mechanism_code"] == "mgi"
    assert rows[0]["energy_total_eV"] == -1.1
    assert rows[1]["mechanism_code"] == "vn"
    assert rows[1]["energy_total_eV"] == -1.2


def test_build_per_structure_rows_accepts_direct_x_mg_cation(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "real-thermo"
    (run_dir / "inputs").mkdir(parents=True)

    with (run_dir / "inputs" / "results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["structure_id", "mechanism_code", "relaxed_energy_eV", "x_mg_cation", "n_atoms"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "structure_id": "mgi_x017_cfg0001",
                "mechanism_code": "mgi",
                "relaxed_energy_eV": -100.0,
                "x_mg_cation": 0.17,
                "n_atoms": 64,
            }
        )

    rows = build_per_structure_rows(run_dir)
    assert len(rows) == 1
    assert rows[0]["x_mg_cation"] == 0.17
    assert rows[0]["doping_level_percent"] == 17.0
    assert rows[0]["mg_count"] == 0
    assert rows[0]["ga_count"] == 0
    assert rows[0]["n_count"] == 0


def test_build_per_structure_rows_prefers_direct_x_mg_cation_in_mixed_schema(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "mixed-schema"
    (run_dir / "inputs").mkdir(parents=True)

    with (run_dir / "inputs" / "results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "structure_id",
                "mechanism",
                "energy_eV",
                "x_mg_cation",
                "mg_count",
                "ga_count",
                "n_count",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "structure_id": "s001",
                "mechanism": "MgGa+VN",
                "energy_eV": -1.0,
                "x_mg_cation": 0.2,
            }
        )
        writer.writerow(
            {
                "structure_id": "s002",
                "mechanism": "Mgi+2MgGa",
                "energy_eV": -1.2,
                "mg_count": 2,
                "ga_count": 4,
                "n_count": 6,
            }
        )

    rows = build_per_structure_rows(run_dir)
    by_id = {row["structure_id"]: row for row in rows}
    assert by_id["s001"]["x_mg_cation"] == 0.2
    assert by_id["s001"]["mg_count"] == 0
    assert by_id["s002"]["x_mg_cation"] == 2 / 6
    assert by_id["s002"]["mg_count"] == 2


def test_build_per_structure_rows_ignores_invalid_counts_when_x_mg_present(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "real-schema"
    (run_dir / "inputs").mkdir(parents=True)

    with (run_dir / "inputs" / "results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["structure_id", "mechanism_code", "relaxed_energy_eV", "x_mg_cation", "mg_count", "n_atoms"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "structure_id": "mgi_x017_cfg0001",
                "mechanism_code": "mgi",
                "relaxed_energy_eV": -100.0,
                "x_mg_cation": 0.17,
                "mg_count": "not_an_int",
                "n_atoms": 64,
            }
        )

    rows = build_per_structure_rows(run_dir)
    assert len(rows) == 1
    assert rows[0]["x_mg_cation"] == 0.17
    assert rows[0]["mg_count"] == 0
    assert rows[0]["ga_count"] == 0
    assert rows[0]["n_count"] == 0


def test_build_per_structure_rows_has_clear_composition_error(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "missing-composition"
    (run_dir / "inputs").mkdir(parents=True)

    with (run_dir / "inputs" / "results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["structure_id", "mechanism", "energy_eV"])
        writer.writeheader()
        writer.writerow({"structure_id": "s001", "mechanism": "MgGa+VN", "energy_eV": -1.0})

    import pytest

    with pytest.raises(ValueError, match="Unable to determine composition"):
        build_per_structure_rows(run_dir)


def test_build_per_structure_rows_infers_from_filename_tokens(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "filename-infer"
    (run_dir / "inputs").mkdir(parents=True)
    with (run_dir / "inputs" / "results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "structure_id",
                "mechanism_code",
                "relaxed_energy_eV",
                "input_structure_name",
                "n_atoms",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "structure_id": "mgi_x017_cfg0001",
                "mechanism_code": "mgi",
                "relaxed_energy_eV": -100.0,
                "input_structure_name": "GaN_MgSub2_MgInt1_Sample1_20250718_170425.cif",
                "n_atoms": 361,
            }
        )
    rows = build_per_structure_rows(run_dir)
    assert rows[0]["mg_count"] == 3
    assert rows[0]["ga_count"] == 178
    assert rows[0]["n_count"] == 180


def test_build_per_structure_rows_filename_parse_error_is_deterministic(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "filename-bad"
    (run_dir / "inputs").mkdir(parents=True)
    with (run_dir / "inputs" / "results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["structure_id", "mechanism_code", "relaxed_energy_eV", "input_structure_name", "n_atoms"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "structure_id": "mgi_x017_cfg0001",
                "mechanism_code": "mgi",
                "relaxed_energy_eV": -100.0,
                "input_structure_name": "GaN_NoTokens_Sample1.cif",
                "n_atoms": 361,
            }
        )
    import pytest

    with pytest.raises(ValueError, match="Unable to determine composition for structure_id='mgi_x017_cfg0001'"):
        build_per_structure_rows(run_dir)
