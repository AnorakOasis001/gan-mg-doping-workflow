from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    inputs_dir: Path
    outputs_dir: Path
    meta_path: Path


def make_run_id(seed: int | None = None, n: int | None = None) -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    parts = [ts]
    if seed is not None:
        parts.append(f"seed{seed}")
    if n is not None:
        parts.append(f"n{n}")
    return "_".join(parts)


def init_run(run_root: Path, run_id: str) -> RunPaths:
    run_dir = Path(run_root) / run_id
    inputs_dir = run_dir / "inputs"
    outputs_dir = run_dir / "outputs"
    inputs_dir.mkdir(parents=True, exist_ok=False)
    outputs_dir.mkdir(parents=True, exist_ok=False)

    meta_path = run_dir / "run.json"
    return RunPaths(run_dir=run_dir, inputs_dir=inputs_dir, outputs_dir=outputs_dir, meta_path=meta_path)


def write_run_meta(meta_path: Path, meta: dict) -> None:
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")