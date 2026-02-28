from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

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
        "energy_total_eV",
        "energy_reference_eV",
        "energy_mixing_eV",
        "energy_mixing_eV_per_atom",
        "energy_mixing_eV_per_cation",
        "dE_eV",
        "dE_eV_per_atom",
        "dE_eV_per_cation",
        "mg_count",
        "ga_count",
        "n_count",
        "site_count_total",
        "x_mg_cation",
        "doping_level_percent",
        "n_mismatch",
        "relaxed_structure_ref",
    ]

    def make_row(mech: str, x: float, dop: float, sid: str, e_mix: float, d_e: float) -> dict[str, object]:
        mg = 2
        ga = 4
        site_total = 12
        cation = mg + ga
        return {
            "structure_id": sid,
            "mechanism_code": mech,
            "mechanism_label": mech,
            "energy_total_eV": e_mix,
            "energy_reference_eV": 0.0,
            "energy_mixing_eV": e_mix,
            "energy_mixing_eV_per_atom": e_mix / site_total,
            "energy_mixing_eV_per_cation": e_mix / cation,
            "dE_eV": d_e,
            "dE_eV_per_atom": d_e / site_total,
            "dE_eV_per_cation": d_e / cation,
            "mg_count": mg,
            "ga_count": ga,
            "n_count": 6,
            "site_count_total": site_total,
            "x_mg_cation": x,
            "doping_level_percent": dop,
            "n_mismatch": 0.0,
            "relaxed_structure_ref": "",
        }

    rows = [
        make_row("vn", 0.25, 25.0, "vn-25-a", -1.20, 0.0),
        make_row("vn", 0.25, 25.0, "vn-25-b", -1.15, 0.05),
        make_row("vn", 0.25, 25.0, "vn-25-c", -1.00, 0.20),
        make_row("vn", 0.50, 50.0, "vn-50-a", -0.80, 0.0),
        make_row("vn", 0.50, 50.0, "vn-50-b", -0.70, 0.10),
        make_row("vn", 0.50, 50.0, "vn-50-c", -0.60, 0.20),
        make_row("mgi", 0.25, 25.0, "mgi-25-a", -1.10, 0.0),
        make_row("mgi", 0.25, 25.0, "mgi-25-b", -1.03, 0.07),
        make_row("mgi", 0.25, 25.0, "mgi-25-c", -0.90, 0.20),
        make_row("mgi", 0.50, 50.0, "mgi-50-a", -0.90, 0.0),
        make_row("mgi", 0.50, 50.0, "mgi-50-b", -0.88, 0.02),
        make_row("mgi", 0.50, 50.0, "mgi-50-c", -0.84, 0.06),
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_cli_gibbs_generates_summary_and_plots(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")

    run_dir = tmp_path / "runs"
    run_id = "gibbs-test"
    run_path = run_dir / run_id
    _write_per_structure_mixing_csv(run_path / "derived" / "per_structure_mixing.csv")

    _run_cli("gibbs", "--run-dir", str(run_dir), "--run-id", run_id, "--temps", "1,300", cwd=tmp_path)

    gibbs_path = run_path / "derived" / "gibbs_summary.csv"
    all_mech_path = run_path / "derived" / "all_mechanisms_gibbs_summary.csv"
    overlay_png = run_path / "figures" / "overlay_dGmix_vs_doping_multiT.png"
    crossover_csv = run_path / "derived" / "mechanism_crossover.csv"
    crossover_png = run_path / "figures" / "crossover_map.png"

    assert gibbs_path.exists() and gibbs_path.stat().st_size > 0
    assert all_mech_path.exists() and all_mech_path.stat().st_size > 0

    with gibbs_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2 * 2 * 2

    for row in rows:
        assert float(row["dE_min_eV"]) == 0.0

    cold_rows = [r for r in rows if float(r["T_K"]) == 1.0]
    assert cold_rows
    for row in cold_rows:
        assert float(row["free_energy_mixing_eV"]) == pytest.approx(float(row["energy_mixing_min_eV"]), abs=2e-4)

    assert overlay_png.exists() and overlay_png.stat().st_size > 0
    assert crossover_csv.exists() and crossover_csv.stat().st_size > 0
    assert crossover_png.exists() and crossover_png.stat().st_size > 0


def test_cli_gibbs_requires_mixing_input(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs"
    run_id = "missing-mix"
    (run_dir / run_id).mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "gan_mg.cli",
            "gibbs",
            "--run-dir",
            str(run_dir),
            "--run-id",
            run_id,
            "--temps",
            "300",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "ganmg mix --run-id" in (completed.stderr + completed.stdout)


def test_cli_reproduce_overlay_writes_manifest(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")

    run_dir = tmp_path / "runs"
    run_id = "repro"
    run_path = run_dir / run_id
    _write_per_structure_mixing_csv(run_path / "derived" / "per_structure_mixing.csv")
    (run_path / "derived" / "per_structure.csv").write_text("structure_id\nexample\n", encoding="utf-8")
    (run_path / "inputs").mkdir(parents=True, exist_ok=True)
    (run_path / "inputs" / "results.csv").write_text("structure_id,energy_eV\nexample,-1.0\n", encoding="utf-8")

    reference_path = run_path / "inputs" / "reference.json"
    reference_path.write_text(
        '{"model":"gan_mg3n2","energies":{"E_GaN_fu":-10.0,"E_Mg3N2_fu":-5.0}}',
        encoding="utf-8",
    )

    _run_cli(
        "reproduce",
        "overlay",
        "--run-dir",
        str(run_dir),
        "--run-id",
        run_id,
        "--reference",
        str(reference_path),
        "--temps",
        "300,600",
        cwd=tmp_path,
    )

    manifest_path = run_path / "derived" / "repro_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == run_id
    assert manifest["reference"]["model"] == "gan_mg3n2"
    assert manifest["temperatures_K"] == [300.0, 600.0]
    assert sorted(manifest["inputs"].keys()) == [
        "derived/per_structure.csv",
        "derived/per_structure_mixing.csv",
        "inputs/reference.json",
        "inputs/results.csv",
    ]
    assert manifest["outputs"]
