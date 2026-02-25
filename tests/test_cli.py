import csv
import json
import os
import subprocess
import sys

import pytest
from pathlib import Path


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


@pytest.mark.parametrize('model', ['demo', 'toy'])
def test_cli_generate_analyze_sweep_end_to_end(tmp_path: Path, model: str) -> None:
    pytest.importorskip("pandas")
    run_dir = tmp_path / "runs"
    run_id = "pytest-run"

    _run_cli("generate", "--run-dir", str(run_dir), "--run-id", run_id, "--n", "6", "--seed", "11", "--model", model, cwd=tmp_path)

    run_path = run_dir / run_id
    meta_path = run_path / "run.json"
    demo_csv = run_path / "inputs" / "results.csv"
    assert meta_path.exists()
    assert demo_csv.exists()

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["run_id"] == run_id
    assert meta["model"] == model

    _run_cli("analyze", "--run-dir", str(run_dir), "--run-id", run_id, "--T", "298.15", cwd=tmp_path)
    thermo_files = list((run_path / "outputs").glob("thermo_T*.txt"))
    assert len(thermo_files) == 1
    thermo_text = thermo_files[0].read_text(encoding="utf-8")
    assert "temperature_K = 298.15" in thermo_text
    assert "free_energy_mix_eV =" in thermo_text

    _run_cli("sweep", "--run-dir", str(run_dir), "--run-id", run_id, "--nT", "7", cwd=tmp_path)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert "reproducibility_hash" in meta

    sweep_csv = run_path / "outputs" / "thermo_vs_T.csv"
    assert sweep_csv.exists()

    with sweep_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 7
    expected_headers = {
        "temperature_K",
        "num_configurations",
        "mixing_energy_min_eV",
        "mixing_energy_avg_eV",
        "partition_function",
        "free_energy_mix_eV",
    }
    assert set(rows[0].keys()) == expected_headers


def test_cli_sweep_reproducibility_hash_tracks_inputs_and_temperature_grid(tmp_path: Path) -> None:
    pytest.importorskip("pandas")
    run_dir = tmp_path / "runs"
    run_id_1 = "repro-1"
    run_id_2 = "repro-2"

    _run_cli("generate", "--run-dir", str(run_dir), "--run-id", run_id_1, "--n", "6", "--seed", "11", cwd=tmp_path)
    _run_cli("generate", "--run-dir", str(run_dir), "--run-id", run_id_2, "--n", "6", "--seed", "11", cwd=tmp_path)

    _run_cli("sweep", "--run-dir", str(run_dir), "--run-id", run_id_1, "--nT", "7", cwd=tmp_path)
    _run_cli("sweep", "--run-dir", str(run_dir), "--run-id", run_id_2, "--nT", "7", cwd=tmp_path)

    meta_1 = json.loads((run_dir / run_id_1 / "run.json").read_text(encoding="utf-8"))
    meta_2 = json.loads((run_dir / run_id_2 / "run.json").read_text(encoding="utf-8"))
    assert meta_1["reproducibility_hash"] == meta_2["reproducibility_hash"]

    _run_cli("sweep", "--run-dir", str(run_dir), "--run-id", run_id_2, "--nT", "8", cwd=tmp_path)
    meta_2_updated = json.loads((run_dir / run_id_2 / "run.json").read_text(encoding="utf-8"))
    assert meta_1["reproducibility_hash"] != meta_2_updated["reproducibility_hash"]


def test_cli_doctor_reports_logging_level(tmp_path: Path) -> None:
    default = _run_cli("doctor", "--run-dir", "runs", cwd=tmp_path)
    assert "logging_level     : INFO" in default.stdout

    verbose = _run_cli("--verbose", "doctor", "--run-dir", "runs", cwd=tmp_path)
    assert "logging_level     : DEBUG" in verbose.stdout

    quiet = _run_cli("--quiet", "doctor", "--run-dir", "runs", cwd=tmp_path)
    assert quiet.stdout == ""


def test_cli_profile_outputs_runtime_and_config_count(tmp_path: Path) -> None:
    pytest.importorskip("pandas")
    run_dir = tmp_path / "runs"
    run_id = "profile-run"

    generate = _run_cli(
        "--profile",
        "generate",
        "--run-dir",
        str(run_dir),
        "--run-id",
        run_id,
        "--n",
        "6",
        "--seed",
        "11",
        cwd=tmp_path,
    )
    assert "[profile] generate runtime_s=" in generate.stdout
    assert "num_configurations=6" in generate.stdout

    analyze = _run_cli(
        "--profile",
        "analyze",
        "--run-dir",
        str(run_dir),
        "--run-id",
        run_id,
        "--T",
        "298.15",
        cwd=tmp_path,
    )
    assert "[profile] analyze runtime_s=" in analyze.stdout
    assert "num_configurations=6" in analyze.stdout

    sweep = _run_cli(
        "--profile",
        "sweep",
        "--run-dir",
        str(run_dir),
        "--run-id",
        run_id,
        "--nT",
        "7",
        cwd=tmp_path,
    )
    assert "[profile] sweep runtime_s=" in sweep.stdout
    assert "num_configurations=6" in sweep.stdout


def test_cli_verbose_and_quiet_flags_conflict(tmp_path: Path) -> None:
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"

    completed = subprocess.run(
        [sys.executable, "-m", "gan_mg.cli", "--verbose", "--quiet", "doctor"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "--verbose and --quiet cannot be used together" in completed.stderr


def test_cli_analyze_fails_with_informative_validation_error(tmp_path: Path) -> None:
    pytest.importorskip("pandas")
    bad_csv = tmp_path / "results.csv"
    bad_csv.write_text("structure_id,mechanism\ndemo_0001,MgGa+VN\n", encoding="utf-8")

    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"

    completed = subprocess.run(
        [sys.executable, "-m", "gan_mg.cli", "analyze", "--csv", str(bad_csv)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "Input validation error:" in completed.stderr
    assert "missing required columns" in completed.stderr
