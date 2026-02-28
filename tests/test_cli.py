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

    metrics_path = run_path / "outputs" / "metrics.json"
    assert metrics_path.exists()
    analyze_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert analyze_metrics["temperature_K"] == pytest.approx(298.15)
    assert analyze_metrics["num_configurations"] == 6
    assert isinstance(analyze_metrics["mixing_energy_min_eV"], float)
    assert isinstance(analyze_metrics["mixing_energy_avg_eV"], float)
    assert isinstance(analyze_metrics["partition_function"], float)
    assert isinstance(analyze_metrics["free_energy_mix_eV"], float)
    assert isinstance(analyze_metrics["reproducibility_hash"], str)
    assert isinstance(analyze_metrics["created_at"], str)

    _run_cli("sweep", "--run-dir", str(run_dir), "--run-id", run_id, "--nT", "7", cwd=tmp_path)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert "reproducibility_hash" in meta

    sweep_csv = run_path / "outputs" / "thermo_vs_T.csv"
    assert sweep_csv.exists()

    sweep_metrics_path = run_path / "outputs" / "metrics_sweep.json"
    assert sweep_metrics_path.exists()
    sweep_metrics = json.loads(sweep_metrics_path.read_text(encoding="utf-8"))
    assert len(sweep_metrics["temperature_grid_K"]) == 7
    assert sweep_metrics["num_temperatures"] == 7
    assert isinstance(sweep_metrics["reproducibility_hash"], str)
    assert isinstance(sweep_metrics["created_at"], str)

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
    analyze_metrics = json.loads((run_dir / run_id / "outputs" / "metrics.json").read_text(encoding="utf-8"))
    assert "timings" in analyze_metrics
    assert analyze_metrics["timings"]["runtime_s"] > 0

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
    sweep_metrics = json.loads((run_dir / run_id / "outputs" / "metrics_sweep.json").read_text(encoding="utf-8"))
    assert "timings" in sweep_metrics
    assert sweep_metrics["timings"]["runtime_s"] > 0


def test_metrics_contains_provenance_block(tmp_path: Path) -> None:
    pytest.importorskip("pandas")
    run_dir = tmp_path / "runs"
    run_id = "prov-metrics"

    _run_cli("generate", "--run-dir", str(run_dir), "--run-id", run_id, "--n", "4", "--seed", "13", cwd=tmp_path)
    _run_cli("analyze", "--run-dir", str(run_dir), "--run-id", run_id, "--T", "500", cwd=tmp_path)

    metrics_path = run_dir / run_id / "outputs" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    provenance = metrics["provenance"]

    assert "provenance" in metrics
    assert "schema_version" in provenance
    assert "python_version" in provenance
    assert "platform" in provenance
    assert "cli_args" in provenance
    assert "input_hash" in provenance
    assert provenance["git_commit"] is None or isinstance(provenance["git_commit"], str)


def test_diagnostics_contains_provenance_block(tmp_path: Path) -> None:
    pytest.importorskip("pandas")
    run_dir = tmp_path / "runs"
    run_id = "prov-diagnostics"

    _run_cli("generate", "--run-dir", str(run_dir), "--run-id", run_id, "--n", "4", "--seed", "13", cwd=tmp_path)
    _run_cli(
        "analyze",
        "--run-dir",
        str(run_dir),
        "--run-id",
        run_id,
        "--T",
        "500",
        "--diagnostics",
        cwd=tmp_path,
    )

    diagnostics_path = run_dir / run_id / "outputs" / "diagnostics_T500.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    provenance = diagnostics["provenance"]

    assert "provenance" in diagnostics
    assert "schema_version" in provenance
    assert "python_version" in provenance
    assert "platform" in provenance
    assert "cli_args" in provenance
    assert "input_hash" in provenance
    assert provenance["git_commit"] is None or isinstance(provenance["git_commit"], str)


def test_cli_runs_show_prints_metadata_and_latest_metrics(tmp_path: Path) -> None:
    pytest.importorskip("pandas")
    run_dir = tmp_path / "runs"
    run_id = "show-run"

    _run_cli("generate", "--run-dir", str(run_dir), "--run-id", run_id, "--n", "4", "--seed", "13", cwd=tmp_path)
    _run_cli("analyze", "--run-dir", str(run_dir), "--run-id", run_id, "--T", "500", cwd=tmp_path)
    _run_cli("sweep", "--run-dir", str(run_dir), "--run-id", run_id, "--nT", "4", cwd=tmp_path)

    shown = _run_cli("runs", "--run-dir", str(run_dir), "show", "--run-id", run_id, cwd=tmp_path)

    assert f"run_id            : {run_id}" in shown.stdout
    assert "latest_metrics    :" in shown.stdout
    assert "metrics.json      :" in shown.stdout
    assert "metrics_sweep.json:" in shown.stdout
    assert '"temperature_K": 500.0' in shown.stdout
    assert '"reproducibility_hash":' in shown.stdout
    assert '"temperature_grid_K": [' in shown.stdout


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


def test_cli_phase_map_writes_dataset(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs"
    run_id = "phase-map-dataset"
    input_csv = run_dir / run_id / "derived" / "crossover_uncertainty.csv"
    input_csv.parent.mkdir(parents=True, exist_ok=True)
    input_csv.write_text(
        "x_mg_cation,doping_level_percent,T_K,delta_G_mean_eV,delta_G_ci_low_eV,delta_G_ci_high_eV,preferred,robust\n"
        "0.25,25.0,300.0,0.04,-0.01,0.09,mgi,False\n",
        encoding="utf-8",
    )

    _run_cli("phase-map", "--run-dir", str(run_dir), "--run-id", run_id, cwd=tmp_path)

    assert (run_dir / run_id / "derived" / "phase_map.csv").exists()


def test_cli_phase_map_plot_writes_png(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    run_dir = tmp_path / "runs"
    run_id = "phase-map-plot"
    input_csv = run_dir / run_id / "derived" / "crossover_uncertainty.csv"
    input_csv.parent.mkdir(parents=True, exist_ok=True)
    input_csv.write_text(
        "x_mg_cation,doping_level_percent,T_K,delta_G_mean_eV,delta_G_ci_low_eV,delta_G_ci_high_eV,preferred,robust\n"
        "0.25,25.0,300.0,0.04,-0.01,0.09,mgi,False\n"
        "0.50,50.0,350.0,0.10,0.05,0.15,vn,True\n",
        encoding="utf-8",
    )

    _run_cli("phase-map", "--run-dir", str(run_dir), "--run-id", run_id, "--plot", cwd=tmp_path)

    assert (run_dir / run_id / "figures" / "phase_map.png").exists()


def test_cli_import_csv_into_existing_run_writes_metadata(tmp_path: Path) -> None:
    pytest.importorskip("pandas")

    existing_csv = (
        "structure_id,mechanism,energy_eV\n"
        "demo_0001,baseline,-1.500\n"
    )
    imported_csv = (
        "structure_id,mechanism,energy_eV\n"
        "ext_0001,HPC,-1.234\n"
        "ext_0002,HPC,-1.111\n"
    )

    run_root = tmp_path / "runs"
    run_id = "imported-run"
    run_path = run_root / run_id
    (run_path / "inputs").mkdir(parents=True)
    (run_path / "inputs" / "results.csv").write_text(existing_csv, encoding="utf-8")

    source_csv = tmp_path / "external_results.csv"
    source_csv.write_text(imported_csv, encoding="utf-8")

    _run_cli(
        "import",
        "--run-id",
        run_id,
        "--run-dir",
        str(run_root),
        "--results",
        str(source_csv),
        cwd=tmp_path,
    )

    canonical_csv = run_path / "inputs" / "results.csv"
    imported_target = run_path / "inputs" / "imported_results.csv"
    metadata_path = run_path / "inputs" / "import.json"
    run_meta_path = run_path / "run.json"

    assert canonical_csv.read_text(encoding="utf-8") == existing_csv
    assert imported_target.read_text(encoding="utf-8") == imported_csv
    assert metadata_path.exists()
    assert run_meta_path.exists()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["source_path"] == str(source_csv.resolve())
    assert metadata["format"] == ".csv"
    assert metadata["results_csv"] == str(imported_target)
    assert isinstance(metadata["sha256"], str)
    assert len(metadata["sha256"]) == 64
    assert "imported_at" in metadata

    run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
    assert run_meta["command"] == "import"
    assert run_meta["run_id"] == run_id
    assert run_meta["inputs_csv"] == str(imported_target)


def test_cli_import_fails_when_results_file_is_missing(tmp_path: Path) -> None:
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"

    missing = tmp_path / "no_such_file.csv"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "gan_mg.cli",
            "import",
            "--run-id",
            "missing-import",
            "--run-dir",
            str(tmp_path / "runs"),
            "--results",
            str(missing),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "Input validation error:" in completed.stderr
    assert "Import path not found" in completed.stderr


def test_cli_import_fails_on_invalid_csv_schema(tmp_path: Path) -> None:
    pytest.importorskip("pandas")
    bad_csv = tmp_path / "external_results.csv"
    bad_csv.write_text(
        "structure_id,mechanism\n"
        "ext_0001,HPC\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "gan_mg.cli",
            "import",
            "--run-id",
            "bad-import",
            "--run-dir",
            str(tmp_path / "runs"),
            "--results",
            str(bad_csv),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "Input validation error:" in completed.stderr
    assert "missing required columns" in completed.stderr


def test_cli_plot_thermo_creates_figure_when_plot_extra_available(tmp_path: Path) -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("matplotlib")

    run_dir = tmp_path / "runs"
    run_id = "plot-run"

    _run_cli("generate", "--run-dir", str(run_dir), "--run-id", run_id, "--n", "6", "--seed", "11", cwd=tmp_path)
    _run_cli("plot", "--run-dir", str(run_dir), "--run-id", run_id, "--kind", "thermo", cwd=tmp_path)

    figure_path = run_dir / run_id / "figures" / "thermo_vs_T.png"
    assert figure_path.exists()
    assert figure_path.stat().st_size > 0


def test_cli_bench_thermo_runs_with_small_defaults(tmp_path: Path) -> None:
    out_path = tmp_path / "outputs" / "bench.json"

    completed = _run_cli(
        "bench",
        "thermo",
        "--n",
        "16",
        "--nT",
        "4",
        "--out",
        str(out_path),
        cwd=tmp_path,
    )

    assert "Benchmark timing summary" in completed.stdout
    assert out_path.exists()

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["command"] == "bench thermo"
    assert payload["params"]["n"] == 16
    assert payload["params"]["nT"] == 4
    assert payload["timings"]["sweep_runtime_s"] >= 0.0
    assert payload["timings"]["time_per_temperature_ms"] >= 0.0


def test_cli_analyze_writes_optional_diagnostics_json(tmp_path: Path) -> None:
    pytest.importorskip("pandas")
    run_dir = tmp_path / "runs"
    run_id = "diag-run"

    _run_cli("generate", "--run-dir", str(run_dir), "--run-id", run_id, "--n", "6", "--seed", "11", cwd=tmp_path)
    _run_cli(
        "analyze",
        "--run-dir",
        str(run_dir),
        "--run-id",
        run_id,
        "--T",
        "750",
        "--diagnostics",
        cwd=tmp_path,
    )

    diagnostics_path = run_dir / run_id / "outputs" / "diagnostics_T750.json"
    assert diagnostics_path.exists()

    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    expected_keys = {
        "temperature_K",
        "num_configurations",
        "expected_energy_eV",
        "energy_variance_eV2",
        "energy_std_eV",
        "p_min",
        "effective_sample_size",
        "logZ_shifted",
        "logZ_absolute",
        "notes",
        "provenance",
    }
    assert set(diagnostics.keys()) == expected_keys
    assert diagnostics["num_configurations"] == 6


def test_cli_help_lists_common_examples(tmp_path: Path) -> None:
    completed = _run_cli("--help", cwd=tmp_path)

    assert "Minimal end-to-end run" in completed.stdout
    assert "ganmg --config configs/run_config.toml" in completed.stdout
    assert "--validate-output" in completed.stdout


def test_cli_analyze_validate_output_flag_succeeds_on_minimal_outputs(tmp_path: Path) -> None:
    pytest.importorskip("pandas")
    run_dir = tmp_path / "runs"
    run_id = "validate-flag"

    _run_cli("generate", "--run-dir", str(run_dir), "--run-id", run_id, "--n", "6", "--seed", "11", cwd=tmp_path)
    completed = _run_cli(
        "analyze",
        "--run-dir",
        str(run_dir),
        "--run-id",
        run_id,
        "--T",
        "500",
        "--validate-output",
        cwd=tmp_path,
    )

    assert completed.returncode == 0
    assert (run_dir / run_id / "outputs" / "metrics.json").exists()


def test_cli_analyze_missing_csv_shows_friendly_error(tmp_path: Path) -> None:
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"

    missing = tmp_path / "missing.csv"
    completed = subprocess.run(
        [sys.executable, "-m", "gan_mg.cli", "analyze", "--csv", str(missing)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "Input file not found:" in completed.stderr


def test_cli_config_missing_file_shows_friendly_error(tmp_path: Path) -> None:
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"

    missing = tmp_path / "missing.toml"
    completed = subprocess.run(
        [sys.executable, "-m", "gan_mg.cli", "--config", str(missing)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "Config file not found:" in completed.stderr


def test_cli_repropack_exports_run_artifacts_and_zip(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs"
    run_id = "repropack"
    run_path = run_dir / run_id

    (run_path / "inputs").mkdir(parents=True, exist_ok=True)
    (run_path / "derived").mkdir(parents=True, exist_ok=True)
    (run_path / "figures").mkdir(parents=True, exist_ok=True)

    (run_path / "inputs" / "results.csv").write_text("structure_id,energy_eV\ns1,-1.0\n", encoding="utf-8")
    (run_path / "inputs" / "reference.toml").write_text('model = "gan_mg3n2"\n', encoding="utf-8")
    (run_path / "derived" / "per_structure.csv").write_text("structure_id\ns1\n", encoding="utf-8")
    (run_path / "derived" / "gibbs_summary.csv").write_text("mechanism_code\nvn\n", encoding="utf-8")
    (run_path / "figures" / "overlay.png").write_bytes(b"png")
    (run_path / "derived" / "repro_manifest.json").write_text('{"git_commit":"abc123"}', encoding="utf-8")

    out_dir = tmp_path / "exports"
    _run_cli(
        "repropack",
        "--run-dir",
        str(run_dir),
        "--run-id",
        run_id,
        "--out",
        str(out_dir),
        "--zip",
        cwd=tmp_path,
    )

    packed = out_dir / run_id
    assert (packed / "inputs" / "results.csv").exists()
    assert (packed / "inputs" / "reference.toml").exists()
    assert (packed / "derived" / "per_structure.csv").exists()
    assert (packed / "figures" / "overlay.png").exists()
    assert (packed / "index.md").exists()
    assert "abc123" in (packed / "index.md").read_text(encoding="utf-8")

    assert (out_dir / f"{run_id}.zip").exists()


def test_cli_repropack_requires_core_inputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs"
    run_id = "missing-core"
    run_path = run_dir / run_id
    (run_path / "inputs").mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"

    completed = subprocess.run(
        [sys.executable, "-m", "gan_mg.cli", "repropack", "--run-dir", str(run_dir), "--run-id", run_id],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "Core input missing" in (completed.stderr + completed.stdout)
