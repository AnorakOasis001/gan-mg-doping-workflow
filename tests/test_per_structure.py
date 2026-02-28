from __future__ import annotations

import csv
from pathlib import Path

from gan_mg.science.per_structure import (
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
