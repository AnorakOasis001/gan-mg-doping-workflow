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


def test_runs_list_with_runs_dir_and_stable_ordering(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"

    (runs_dir / "runA" / "outputs").mkdir(parents=True)
    (runs_dir / "runA" / "outputs" / "metrics.json").write_text(
        json.dumps({"created_at": "2026-01-01T00:00:00+00:00", "reproducibility_hash": "abc123"}),
        encoding="utf-8",
    )

    (runs_dir / "runB" / "derived").mkdir(parents=True)
    (runs_dir / "runB" / "derived" / "repro_manifest.json").write_text(
        json.dumps({"created_at": "2026-01-02T00:00:00+00:00", "reproducibility_hash": "def456"}),
        encoding="utf-8",
    )
    (runs_dir / "runB" / "derived" / "gibbs_summary.csv").write_text("T,delta_g\n300,0.1\n", encoding="utf-8")

    (runs_dir / "runC").mkdir(parents=True)

    completed = _run_cli("runs", "list", "--runs-dir", str(runs_dir), cwd=tmp_path)

    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert "run_id" in lines[0]
    run_rows = [line for line in lines if line.startswith("run") and not line.startswith("run_id")]
    assert [row.split()[0] for row in run_rows] == ["runA", "runB", "runC"]
    assert any("metrics" in row for row in run_rows if row.startswith("runA"))
    assert any("gibbs" in row for row in run_rows if row.startswith("runB"))


def test_runs_latest_prefers_created_at(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    (runs_dir / "run_old" / "outputs").mkdir(parents=True)
    (runs_dir / "run_new" / "outputs").mkdir(parents=True)

    (runs_dir / "run_old" / "outputs" / "metrics.json").write_text(
        json.dumps({"created_at": "2025-01-01T00:00:00+00:00"}),
        encoding="utf-8",
    )
    (runs_dir / "run_new" / "outputs" / "metrics.json").write_text(
        json.dumps({"created_at": "2025-01-02T00:00:00+00:00"}),
        encoding="utf-8",
    )

    completed = _run_cli("runs", "latest", "--runs-dir", str(runs_dir), cwd=tmp_path)
    assert completed.stdout.strip() == "run_new"


def test_runs_latest_falls_back_to_mtime(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    first = runs_dir / "run_first"
    second = runs_dir / "run_second"
    first.mkdir(parents=True)
    second.mkdir(parents=True)

    os.utime(first, (1_700_000_000, 1_700_000_000))
    os.utime(second, (1_700_000_100, 1_700_000_100))

    completed = _run_cli("runs", "latest", "--runs-dir", str(runs_dir), cwd=tmp_path)
    assert completed.stdout.strip() == "run_second"


def test_runs_json_modes_emit_valid_json(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    (runs_dir / "runX" / "outputs").mkdir(parents=True)
    (runs_dir / "runX" / "outputs" / "metrics.json").write_text(
        json.dumps({"created_at": "2025-01-02T00:00:00+00:00", "reproducibility_hash": "abc"}),
        encoding="utf-8",
    )

    list_completed = _run_cli("runs", "list", "--runs-dir", str(runs_dir), "--json", cwd=tmp_path)
    list_payload = json.loads(list_completed.stdout)
    assert isinstance(list_payload, list)
    assert list_payload[0]["run_id"] == "runX"

    latest_completed = _run_cli("runs", "latest", "--runs-dir", str(runs_dir), "--json", cwd=tmp_path)
    latest_payload = json.loads(latest_completed.stdout)
    assert latest_payload["run_id"] == "runX"
    assert latest_payload["run_path"].endswith("runX")
