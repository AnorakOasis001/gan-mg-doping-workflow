from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPORT_CANDIDATES: tuple[str, ...] = (
    "outputs/metrics.json",
    "outputs/metrics_sweep.json",
    "outputs/thermo_vs_T.csv",
    "derived/gibbs_summary.csv",
    "derived/gibbs_uncertainty.csv",
    "derived/mechanism_crossover.csv",
    "derived/crossover_uncertainty.csv",
    "derived/phase_map.csv",
    "derived/phase_boundary.csv",
    "derived/repro_manifest.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def safe_copy(src: Path, dst: Path) -> bool:
    if not src.exists() or not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _collect_temperature_line(metrics: dict[str, Any] | None, sweep: dict[str, Any] | None) -> str:
    if metrics is not None and isinstance(metrics.get("temperature_K"), (int, float)):
        return f"Single temperature: {float(metrics['temperature_K']):g} K"
    grid = sweep.get("temperature_grid_K") if sweep is not None else None
    if isinstance(grid, list) and grid:
        parsed: list[float] = []
        for value in grid:
            if not isinstance(value, (int, float)):
                parsed = []
                break
            parsed.append(float(value))
        if parsed:
            return f"Temperature grid: {parsed[0]:g}..{parsed[-1]:g} K ({len(parsed)} points)"
    return "Temperature info: unavailable"


def _winner_counts_line(phase_map_path: Path) -> str | None:
    if not phase_map_path.exists():
        return None
    counts: dict[str, int] = {}
    with phase_map_path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        for row in rows:
            winner = (row.get("winner_mechanism") or "unknown").strip() or "unknown"
            counts[winner] = counts.get(winner, 0) + 1
    if not counts:
        return "- `phase_map.csv`: present but contained no rows."
    summary = ", ".join(f"{key}={counts[key]}" for key in sorted(counts))
    return f"- `phase_map.csv` winner counts: {summary}."


def _boundary_counts_line(phase_boundary_path: Path) -> str | None:
    if not phase_boundary_path.exists():
        return None
    counts: dict[str, int] = {}
    with phase_boundary_path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        for row in rows:
            raw_temp = (row.get("temperature_K") or "unknown").strip() or "unknown"
            counts[raw_temp] = counts.get(raw_temp, 0) + 1
    if not counts:
        return "- `phase_boundary.csv`: present but contained no rows."
    def _sort_key(value: str) -> tuple[int, float, str]:
        try:
            return (0, float(value), value)
        except ValueError:
            return (1, 0.0, value)

    parts = [f"{temp}K={counts[temp]}" for temp in sorted(counts, key=_sort_key)]
    return "- `phase_boundary.csv` boundaries per temperature: " + ", ".join(parts) + "."


def _find_raw_input_path(run_dir: Path, repro_payload: dict[str, Any] | None, metrics_payload: dict[str, Any] | None) -> Path | None:
    candidates: list[str] = []
    for payload in (repro_payload, metrics_payload):
        if payload is None:
            continue
        raw_candidate = payload.get("input_csv")
        if isinstance(raw_candidate, str):
            candidates.append(raw_candidate)
        inputs = payload.get("inputs")
        if isinstance(inputs, dict):
            for key in ("inputs/results.csv", "results.csv", "csv"):
                value = inputs.get(key)
                if isinstance(value, str):
                    candidates.append(value)
                elif isinstance(value, dict):
                    nested = value.get("path")
                    if isinstance(nested, str):
                        candidates.append(nested)

    for candidate in candidates:
        path = Path(candidate)
        if not path.is_absolute():
            path = (run_dir / candidate).resolve()
        if path.exists() and path.is_file() and path.suffix.lower() == ".csv":
            return path
    fallback = run_dir / "inputs" / "results.csv"
    if fallback.exists() and fallback.is_file():
        return fallback
    return None


def build_report(run_dir: Path, out_dir: Path, zip_out: bool, force: bool, include_raw: bool) -> tuple[Path, Path | None]:
    run_id = run_dir.name
    report_dir = out_dir
    if report_dir.exists():
        if not force:
            raise ValueError(
                f"Report directory already exists: {report_dir}. Use --force to overwrite."
            )
        shutil.rmtree(report_dir)

    report_dir.mkdir(parents=True, exist_ok=True)

    included_relpaths: list[str] = []

    for rel_path in REPORT_CANDIDATES:
        src = run_dir / rel_path
        dst = report_dir / rel_path
        if safe_copy(src, dst):
            included_relpaths.append(rel_path)

    for src in sorted((run_dir / "outputs").glob("diagnostics_T*.json")):
        rel_path = src.relative_to(run_dir).as_posix()
        if safe_copy(src, report_dir / rel_path):
            included_relpaths.append(rel_path)

    for src in sorted((run_dir / "figures").glob("*.png")):
        rel_path = src.relative_to(run_dir).as_posix()
        if safe_copy(src, report_dir / rel_path):
            included_relpaths.append(rel_path)

    metrics_payload = _read_json(run_dir / "outputs" / "metrics.json")
    sweep_payload = _read_json(run_dir / "outputs" / "metrics_sweep.json")
    repro_payload = _read_json(run_dir / "derived" / "repro_manifest.json")

    if include_raw:
        raw_path = _find_raw_input_path(run_dir, repro_payload, metrics_payload)
        if raw_path is not None:
            raw_name = raw_path.name
            rel_path = f"inputs/{raw_name}"
            if safe_copy(raw_path, report_dir / rel_path):
                included_relpaths.append(rel_path)

    included_relpaths = sorted(set(included_relpaths))

    created_at = None
    for payload in (repro_payload, metrics_payload, sweep_payload):
        if payload is None:
            continue
        value = payload.get("created_at") or payload.get("timestamp_utc")
        if isinstance(value, str):
            created_at = value
            break

    repro_hash = None
    for payload in (repro_payload, metrics_payload, sweep_payload):
        if payload is None:
            continue
        value = payload.get("reproducibility_hash")
        if isinstance(value, str):
            repro_hash = value
            break

    interpretation_lines: list[str] = []
    winner_line = _winner_counts_line(run_dir / "derived" / "phase_map.csv")
    if winner_line is not None:
        interpretation_lines.append(winner_line)
    boundary_line = _boundary_counts_line(run_dir / "derived" / "phase_boundary.csv")
    if boundary_line is not None:
        interpretation_lines.append(boundary_line)
    if not interpretation_lines:
        interpretation_lines.append("- No phase-map or phase-boundary data were available.")

    readme_lines = [
        f"# Report Bundle: {run_id}",
        "",
        f"- Run id: `{run_id}`",
        f"- Created at: `{created_at if created_at else 'unknown'}`",
        f"- Reproducibility hash: `{repro_hash if repro_hash else 'unknown'}`",
        f"- {_collect_temperature_line(metrics_payload, sweep_payload)}",
        "",
        "## Included artifacts",
        "",
    ]
    if included_relpaths:
        readme_lines.extend(f"- `{rel_path}`" for rel_path in included_relpaths)
    else:
        readme_lines.append("- _(none found)_")

    readme_lines.extend(["", "## Interpretation", ""])
    readme_lines.extend(interpretation_lines)
    (report_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    files_for_manifest = sorted(included_relpaths + ["README.md"])
    manifest = {
        "run_id": run_id,
        "generated_at": _iso_utc_now(),
        "source_run_dir": str(run_dir.resolve()),
        "included_files": files_for_manifest,
        "sha256": {rel_path: sha256_file(report_dir / rel_path) for rel_path in files_for_manifest},
    }
    (report_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    zip_path: Path | None = None
    if zip_out:
        zip_path = report_dir.parent / f"{run_id}.zip"
        with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for item in sorted(report_dir.rglob("*")):
                if item.is_dir():
                    continue
                archive.write(item, item.relative_to(report_dir))

    return report_dir, zip_path


__all__ = ["build_report", "safe_copy", "sha256_file"]
