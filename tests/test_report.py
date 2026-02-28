import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_cli(*args: str, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
    return subprocess.run(
        [sys.executable, "-m", "gan_mg.cli", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=check,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _seed_run(run_dir: Path) -> None:
    (run_dir / "outputs").mkdir(parents=True)
    (run_dir / "derived").mkdir(parents=True)
    (run_dir / "figures").mkdir(parents=True)
    (run_dir / "inputs").mkdir(parents=True)

    (run_dir / "outputs" / "metrics.json").write_text(
        json.dumps(
            {
                "created_at": "2026-02-01T00:00:00+00:00",
                "reproducibility_hash": "abc123",
                "temperature_K": 900,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "outputs" / "diagnostics_T900.json").write_text('{"ok": true}\n', encoding="utf-8")
    (run_dir / "derived" / "phase_map.csv").write_text(
        "temperature_K,x,winner_mechanism\n900,0.1,mgi\n900,0.2,vn\n900,0.3,mgi\n",
        encoding="utf-8",
    )
    (run_dir / "derived" / "phase_boundary.csv").write_text(
        "temperature_K,boundary\n900,0.4\n900,0.5\n1000,0.6\n",
        encoding="utf-8",
    )
    (run_dir / "derived" / "repro_manifest.json").write_text(
        json.dumps({"created_at": "2026-02-01T00:00:00+00:00", "reproducibility_hash": "abc123"}),
        encoding="utf-8",
    )
    (run_dir / "figures" / "phase_map.png").write_bytes(b"PNG")
    (run_dir / "inputs" / "results.csv").write_text("id,energy\na,-1\n", encoding="utf-8")


def test_report_cli_writes_manifest_and_readme(tmp_path: Path) -> None:
    run_id = "run-report"
    run_dir = tmp_path / "runs" / run_id
    _seed_run(run_dir)

    reports_root = tmp_path / "reports"
    _run_cli("report", "--run-dir", str(tmp_path / "runs"), "--run-id", run_id, "--out", str(reports_root / run_id), cwd=tmp_path)

    report_dir = reports_root / run_id
    readme = report_dir / "README.md"
    manifest_path = report_dir / "manifest.json"
    assert readme.exists()
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == run_id
    included = manifest["included_files"]
    assert included == sorted(included)
    assert "outputs/metrics.json" in included
    assert "derived/repro_manifest.json" in included
    assert "README.md" in included
    assert "manifest.json" not in included

    for rel_path in included:
        assert manifest["sha256"][rel_path] == _sha256(report_dir / rel_path)

    readme_text = readme.read_text(encoding="utf-8")
    assert "Run id" in readme_text
    assert "winner counts: mgi=2, vn=1" in readme_text
    assert "boundaries per temperature: 900K=2, 1000K=1" in readme_text


def test_report_cli_skips_missing_and_force_and_zip(tmp_path: Path) -> None:
    run_id = "run2"
    run_dir = tmp_path / "runs" / run_id
    _seed_run(run_dir)
    (run_dir / "derived" / "phase_map.csv").unlink()

    report_dir = tmp_path / "reports" / run_id

    _run_cli(
        "report",
        "--run-dir",
        str(tmp_path / "runs"),
        "--run-id",
        run_id,
        "--out",
        str(report_dir),
        "--zip",
        "--include-raw",
        cwd=tmp_path,
    )

    manifest = json.loads((report_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "derived/phase_map.csv" not in manifest["included_files"]
    assert "inputs/results.csv" in manifest["included_files"]

    zip_path = tmp_path / "reports" / f"{run_id}.zip"
    assert zip_path.exists()
    with zipfile.ZipFile(zip_path, "r") as archive:
        names = sorted(archive.namelist())
    assert "manifest.json" in names
    assert "README.md" in names

    failed = _run_cli(
        "report",
        "--run-dir",
        str(tmp_path / "runs"),
        "--run-id",
        run_id,
        "--out",
        str(report_dir),
        cwd=tmp_path,
        check=False,
    )
    assert failed.returncode != 0
    assert "--force" in failed.stderr or "--force" in failed.stdout

    _run_cli(
        "report",
        "--run-dir",
        str(tmp_path / "runs"),
        "--run-id",
        run_id,
        "--out",
        str(report_dir),
        "--force",
        cwd=tmp_path,
    )
    assert (report_dir / "manifest.json").exists()
