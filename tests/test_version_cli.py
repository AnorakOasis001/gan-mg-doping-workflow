from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from gan_mg.version import __version__


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_ganmg_dashdashversion_prints_package_version(tmp_path: Path) -> None:
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"

    completed = subprocess.run(
        [sys.executable, "-m", "gan_mg.cli", "--version"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert completed.stdout.strip() == __version__
