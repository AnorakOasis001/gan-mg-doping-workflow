from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    path: Path
    created_at: str | None
    reproducibility_hash: str | None
    has_metrics: bool
    has_sweep: bool
    has_gibbs: bool
    has_uncertainty: bool
    has_crossover: bool
    has_phase_map: bool
    num_configurations: int | None
    temperature_K: float | None
    temperature_grid: str | None


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _parse_created_at(value: str | None) -> datetime | None:
    if not value:
        return None
    candidate = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(candidate)
    except ValueError:
        return None


def _temperature_grid_summary(metrics_sweep: dict[str, Any] | None) -> str | None:
    if not metrics_sweep:
        return None
    raw_grid = metrics_sweep.get("temperature_grid_K")
    if not isinstance(raw_grid, list) or not raw_grid:
        return None

    values: list[float] = []
    for value in raw_grid:
        parsed = _as_float(value)
        if parsed is None:
            return None
        values.append(parsed)

    return f"{values[0]:g}..{values[-1]:g} ({len(values)})"


def summarize_run(run_path: Path) -> RunSummary:
    outputs_dir = run_path / "outputs"
    derived_dir = run_path / "derived"

    metrics = _read_json(outputs_dir / "metrics.json")
    metrics_sweep = _read_json(outputs_dir / "metrics_sweep.json")
    repro_manifest = _read_json(derived_dir / "repro_manifest.json")

    created_at: str | None = None
    for payload in (repro_manifest, metrics, metrics_sweep):
        if isinstance(payload, dict):
            raw_created = payload.get("created_at") or payload.get("timestamp")
            if isinstance(raw_created, str):
                created_at = raw_created
                break

    reproducibility_hash: str | None = None
    for payload in (repro_manifest, metrics, metrics_sweep):
        if isinstance(payload, dict):
            raw_hash = payload.get("reproducibility_hash")
            if isinstance(raw_hash, str):
                reproducibility_hash = raw_hash
                break

    raw_num_configurations = None
    if isinstance(metrics, dict):
        raw_num_configurations = metrics.get("num_configurations")
    elif isinstance(metrics_sweep, dict):
        raw_num_configurations = metrics_sweep.get("num_configurations")

    num_configurations = raw_num_configurations if isinstance(raw_num_configurations, int) else None

    temperature_K = _as_float(metrics.get("temperature_K") if metrics else None)

    has_metrics = (outputs_dir / "metrics.json").exists()
    has_sweep = (outputs_dir / "metrics_sweep.json").exists()
    has_gibbs = (derived_dir / "gibbs_summary.csv").exists()
    has_uncertainty = (derived_dir / "gibbs_uncertainty.csv").exists() or (derived_dir / "crossover_uncertainty.csv").exists()
    has_crossover = (derived_dir / "mechanism_crossover.csv").exists()
    has_phase_map = (derived_dir / "phase_map.csv").exists()

    return RunSummary(
        run_id=run_path.name,
        path=run_path,
        created_at=created_at,
        reproducibility_hash=reproducibility_hash,
        has_metrics=has_metrics,
        has_sweep=has_sweep,
        has_gibbs=has_gibbs,
        has_uncertainty=has_uncertainty,
        has_crossover=has_crossover,
        has_phase_map=has_phase_map,
        num_configurations=num_configurations,
        temperature_K=temperature_K,
        temperature_grid=_temperature_grid_summary(metrics_sweep),
    )


def find_runs(repo_root: Path, runs_dir: Path | None = None) -> list[RunSummary]:
    base_dir = runs_dir if runs_dir is not None else (Path(repo_root) / "runs")
    if not base_dir.exists():
        return []

    run_dirs = sorted((p for p in base_dir.iterdir() if p.is_dir()), key=lambda p: p.name)
    return [summarize_run(run_dir) for run_dir in run_dirs]


def latest_run_id(repo_root: Path, runs_dir: Path | None = None) -> str | None:
    summaries = find_runs(repo_root=repo_root, runs_dir=runs_dir)
    if not summaries:
        return None

    def _sort_key(summary: RunSummary) -> tuple[datetime, float, str]:
        parsed_created_at = _parse_created_at(summary.created_at)
        mtime = summary.path.stat().st_mtime
        if parsed_created_at is None:
            parsed_created_at = datetime.fromtimestamp(mtime)
        return parsed_created_at, mtime, summary.run_id

    latest = max(summaries, key=_sort_key)
    return latest.run_id


def run_summary_to_dict(summary: RunSummary) -> dict[str, Any]:
    payload = asdict(summary)
    payload["path"] = str(summary.path)
    return payload


__all__ = ["RunSummary", "find_runs", "latest_run_id", "run_summary_to_dict"]
