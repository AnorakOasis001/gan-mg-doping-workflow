from __future__ import annotations

import json
import logging
from hashlib import sha256
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence, TypedDict, cast


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    inputs_dir: Path
    outputs_dir: Path
    meta_path: Path


class RunMeta(TypedDict, total=False):
    command: str
    run_id: str
    n: int
    seed: int
    model: str
    inputs_csv: str
    reproducibility_hash: str


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
    logger.debug("Initializing run directory at %s", run_dir)
    inputs_dir.mkdir(parents=True, exist_ok=False)
    outputs_dir.mkdir(parents=True, exist_ok=False)

    meta_path = run_dir / "run.json"
    return RunPaths(run_dir=run_dir, inputs_dir=inputs_dir, outputs_dir=outputs_dir, meta_path=meta_path)


def write_run_meta(meta_path: Path, meta: RunMeta) -> None:
    logger.debug("Writing run metadata to %s", meta_path)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def list_runs(run_root: Path) -> list[Path]:
    """
    Return run directories under run_root (only directories).
    """
    run_root = Path(run_root)
    if not run_root.exists():
        logger.debug("Run root does not exist: %s", run_root)
        return []
    runs = sorted([p for p in run_root.iterdir() if p.is_dir()])
    logger.debug("Discovered %d run(s) under %s", len(runs), run_root)
    return runs


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


def load_run_meta(run_dir: Path) -> RunMeta:
    """Load run.json metadata for a given run directory."""
    meta_path = run_dir / "run.json"
    if not meta_path.exists():
        logger.debug("No run metadata file at %s", meta_path)
        return {}
    logger.debug("Loading run metadata from %s", meta_path)
    raw_meta: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
    return cast(RunMeta, raw_meta)


def compute_reproducibility_hash(
    input_csv: Path,
    temperature_grid: Sequence[float],
    code_version: str,
) -> str:
    """Build a SHA256 fingerprint from input data, temperature grid, and code version."""
    hasher = sha256()
    hasher.update(input_csv.read_bytes())
    hasher.update(json.dumps(list(temperature_grid), sort_keys=True).encode("utf-8"))
    hasher.update(code_version.encode("utf-8"))
    return hasher.hexdigest()
