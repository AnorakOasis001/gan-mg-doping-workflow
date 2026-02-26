import json
import os
import subprocess
import sys
from pathlib import Path


def test_benchmark_script_smoke(tmp_path: Path) -> None:
    outdir = tmp_path / "bench_out"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path("src").resolve())

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_thermo.py",
            "--rows",
            "200",
            "--temperature",
            "300",
            "--seed",
            "0",
            "--outdir",
            str(outdir),
            "--chunksize",
            "50",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr

    summary_path = outdir / "thermo_benchmarks.jsonl"
    assert summary_path.exists()

    records = [json.loads(line) for line in summary_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(records) == 1

    record = records[0]
    assert record["rows"] == 200
    assert record["parity_passed"] is True
    assert record["time_in_memory_s"] >= 0.0
    assert record["time_streaming_s"] >= 0.0
