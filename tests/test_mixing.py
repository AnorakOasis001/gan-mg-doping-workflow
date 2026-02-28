from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from gan_mg.science.reference import load_reference_config

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


def _write_per_structure_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "structure_id": "s2",
                "mechanism_code": "vn",
                "mechanism_label": "MgGa+VN",
                "energy_total_eV": -29.0,
                "mg_count": 3,
                "ga_count": 3,
                "n_count": 5,
                "site_count_total": 11,
                "x_mg_cation": 0.5,
                "doping_level_percent": 50.0,
                "relaxed_structure_ref": "",
            }
        )
        writer.writerow(
            {
                "structure_id": "s1",
                "mechanism_code": "vn",
                "mechanism_label": "MgGa+VN",
                "energy_total_eV": -30.0,
                "mg_count": 3,
                "ga_count": 3,
                "n_count": 5,
                "site_count_total": 11,
                "x_mg_cation": 0.5,
                "doping_level_percent": 50.0,
                "relaxed_structure_ref": "",
            }
        )
        writer.writerow(
            {
                "structure_id": "s3",
                "mechanism_code": "mgi",
                "mechanism_label": "Mgi+2MgGa",
                "energy_total_eV": -31.0,
                "mg_count": 3,
                "ga_count": 6,
                "n_count": 8,
                "site_count_total": 17,
                "x_mg_cation": 1.0 / 3.0,
                "doping_level_percent": 100.0 / 3.0,
                "relaxed_structure_ref": "",
            }
        )


def test_cli_mix_computes_expected_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs"
    run_id = "mix-test"
    run_path = run_dir / run_id
    (run_path / "inputs").mkdir(parents=True)

    _write_per_structure_csv(run_path / "derived" / "per_structure.csv")

    reference = {
        "model": "gan_mg3n2",
        "energies": {
            "E_GaN_fu": -5.0,
            "E_Mg3N2_fu": -12.0,
        },
    }
    (run_path / "inputs" / "reference.json").write_text(json.dumps(reference), encoding="utf-8")

    _run_cli("mix", "--run-dir", str(run_dir), "--run-id", run_id, cwd=tmp_path)

    mixing_path = run_path / "derived" / "per_structure_mixing.csv"
    summary_path = run_path / "derived" / "mixing_athermal_summary.csv"
    assert mixing_path.exists()
    assert summary_path.exists()

    with mixing_path.open("r", encoding="utf-8", newline="") as f:
        mixing_rows = list(csv.DictReader(f))

    # deterministic sort by mechanism_code, x_mg_cation, structure_id
    assert [r["structure_id"] for r in mixing_rows] == ["s3", "s1", "s2"]

    # s1 and s2 share the same (mechanism_code, x_mg_cation) and E_ref = 3*(-5) + 1*(-12) = -27 eV
    row_s1 = next(r for r in mixing_rows if r["structure_id"] == "s1")
    row_s2 = next(r for r in mixing_rows if r["structure_id"] == "s2")
    assert float(row_s1["energy_reference_eV"]) == -27.0
    assert float(row_s1["energy_mixing_eV"]) == -3.0
    assert float(row_s2["energy_mixing_eV"]) == -2.0
    assert float(row_s1["dE_eV"]) == 0.0
    assert float(row_s2["dE_eV"]) == 1.0

    # n_count mismatch diagnostic is explicit for VN-like defect row
    assert float(row_s1["n_mismatch"]) == -0.0
    assert float(row_s2["n_mismatch"]) == -0.0

    group_values: dict[tuple[str, str], list[float]] = {}
    for row in mixing_rows:
        key = (row["mechanism_code"], row["x_mg_cation"])
        group_values.setdefault(key, []).append(float(row["dE_eV"]))
    for values in group_values.values():
        assert min(values) == 0.0

    with summary_path.open("r", encoding="utf-8", newline="") as f:
        summary_rows = list(csv.DictReader(f))

    unique_groups = {(r["mechanism_code"], r["x_mg_cation"]) for r in mixing_rows}
    assert len(summary_rows) == len(unique_groups)


def test_mix_requires_reference_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs"
    run_id = "mix-missing-ref"
    run_path = run_dir / run_id
    _write_per_structure_csv(run_path / "derived" / "per_structure.csv")

    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"

    completed = subprocess.run(
        [sys.executable, "-m", "gan_mg.cli", "mix", "--run-dir", str(run_dir), "--run-id", run_id],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "Reference config not found" in completed.stderr or "Reference config not found" in completed.stdout


def test_reference_loader_reports_missing_required_keys(tmp_path: Path) -> None:
    cfg = tmp_path / "reference.json"
    cfg.write_text(json.dumps({"model": "gan_mg3n2", "energies": {"E_GaN_fu": -5.0}}), encoding="utf-8")

    with pytest.raises(ValueError, match=r"model=gan_mg3n2 is missing required energies: E_Mg3N2_fu"):
        load_reference_config(cfg)


def test_reference_loader_reports_missing_mu_keys(tmp_path: Path) -> None:
    cfg = tmp_path / "reference.json"
    cfg.write_text(json.dumps({"model": "chemical_potentials", "energies": {"mu_Ga": -1.0, "mu_N": -2.0}}), encoding="utf-8")

    with pytest.raises(ValueError, match=r"model=chemical_potentials is missing required energies: mu_Mg"):
        load_reference_config(cfg)
