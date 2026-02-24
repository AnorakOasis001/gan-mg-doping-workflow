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


def list_runs(run_root: Path) -> list[Path]:
    """
    Return run directories under run_root (only directories).
    """
    run_root = Path(run_root)
    if not run_root.exists():
        return []
    return sorted([p for p in run_root.iterdir() if p.is_dir()])


def latest_run_id(run_root: Path) -> str:
    """
    Return the most recently modified run directory name under run_root.
    Raises if none exist.
    """
    runs = list_runs(run_root)
    if not runs:
        raise FileNotFoundError(f"No runs found under: {run_root}")

    latest = max(runs, key=lambda p: p.stat().st_mtime)
    return latest.name