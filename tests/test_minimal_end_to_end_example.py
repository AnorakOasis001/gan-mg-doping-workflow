from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

from gan_mg.validation import validate_output_file


def test_run_example_exposes_main_function() -> None:
    script_path = Path("examples/minimal_end_to_end/run_example.py")
    assert script_path.exists(), f"Missing example script: {script_path}"

    spec = importlib.util.spec_from_file_location("minimal_end_to_end_run_example", script_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, "main")
    assert callable(module.main)


def test_run_example_outputs_pass_contract_validation(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path("src").resolve())

    subprocess.run(
        [sys.executable, str(Path("examples/minimal_end_to_end/run_example.py").resolve())],
        cwd=tmp_path,
        env=env,
        check=True,
    )

    outputs = Path("examples/minimal_end_to_end/_outputs/results/tables")
    validate_output_file(outputs / "metrics.json")
    validate_output_file(outputs / "diagnostics_T300.json")
