from __future__ import annotations

import importlib.util
from pathlib import Path


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
