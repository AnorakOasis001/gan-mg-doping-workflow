from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _build_env(repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(repo_root / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not existing else f"{src_path}{os.pathsep}{existing}"
    return env


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    fixture_csv = repo_root / "data" / "golden" / "v1" / "inputs" / "realistic_small.csv"
    outputs_root = repo_root / "examples" / "minimal_end_to_end" / "_outputs"
    outputs_root.mkdir(parents=True, exist_ok=True)

    if not fixture_csv.exists():
        print(f"Missing fixture CSV: {fixture_csv}", file=sys.stderr)
        return 1

    command = [
        sys.executable,
        "-m",
        "gan_mg.cli",
        "analyze",
        "--csv",
        str(fixture_csv),
        "--T",
        "300",
        "--diagnostics",
    ]

    subprocess.run(
        command,
        cwd=outputs_root,
        env=_build_env(repo_root),
        check=True,
    )

    metrics_path = outputs_root / "results" / "tables" / "metrics.json"
    diagnostics_path = outputs_root / "results" / "tables" / "diagnostics_T300.json"

    print(f"metrics.json: {metrics_path}")
    if diagnostics_path.exists():
        print(f"diagnostics: {diagnostics_path}")
    else:
        print("diagnostics: not generated")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
