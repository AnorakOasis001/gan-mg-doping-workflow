import csv
import json
import os
import subprocess
import sys
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


def test_cli_generate_analyze_sweep_end_to_end(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs"
    run_id = "pytest-run"

    _run_cli("generate", "--run-dir", str(run_dir), "--run-id", run_id, "--n", "6", "--seed", "11", cwd=tmp_path)

    run_path = run_dir / run_id
    meta_path = run_path / "run.json"
    demo_csv = run_path / "inputs" / "results.csv"
    assert meta_path.exists()
    assert demo_csv.exists()

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["run_id"] == run_id

    _run_cli("analyze", "--run-dir", str(run_dir), "--run-id", run_id, "--T", "298.15", cwd=tmp_path)
    thermo_files = list((run_path / "outputs").glob("thermo_T*.txt"))
    assert len(thermo_files) == 1
    thermo_text = thermo_files[0].read_text(encoding="utf-8")
    assert "temperature_K = 298.15" in thermo_text
    assert "free_energy_mix_eV =" in thermo_text

    _run_cli("sweep", "--run-dir", str(run_dir), "--run-id", run_id, "--nT", "7", cwd=tmp_path)

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


def test_cli_doctor_reports_logging_level(tmp_path: Path) -> None:
    default = _run_cli("doctor", "--run-dir", "runs", cwd=tmp_path)
    assert "logging_level     : INFO" in default.stdout

    verbose = _run_cli("--verbose", "doctor", "--run-dir", "runs", cwd=tmp_path)
    assert "logging_level     : DEBUG" in verbose.stdout

    quiet = _run_cli("--quiet", "doctor", "--run-dir", "runs", cwd=tmp_path)
    assert quiet.stdout == ""


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
