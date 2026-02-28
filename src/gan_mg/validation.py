from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Callable


class ValidationError(ValueError):
    """Raised when a JSON payload violates an output contract."""

    def __init__(self, path: str, message: str) -> None:
        super().__init__(f"{path}: {message}")
        self.path = path
        self.message = message


def _join_path(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def require_key(obj: dict[str, Any], key: str, *, path: str) -> Any:
    if key not in obj:
        raise ValidationError(_join_path(path, key), "missing required key")
    return obj[key]


def require_type(value: Any, expected_type: type[Any] | tuple[type[Any], ...], *, path: str) -> None:
    if isinstance(value, bool):
        raise ValidationError(path, "bool is not an accepted numeric/string/object type here")
    if not isinstance(value, expected_type):
        expected = (
            ", ".join(t.__name__ for t in expected_type)
            if isinstance(expected_type, tuple)
            else expected_type.__name__
        )
        raise ValidationError(path, f"expected {expected}, got {type(value).__name__}")


def require_finite_number(value: Any, *, path: str) -> float:
    require_type(value, (int, float), path=path)
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError(path, "must be a finite number")
    return number


def require_int(value: Any, *, path: str, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValidationError(path, f"expected int, got {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise ValidationError(path, f"must be >= {minimum}")
    return value


def require_string(value: Any, *, path: str) -> str:
    require_type(value, str, path=path)
    assert isinstance(value, str)
    return value


def require_array(value: Any, *, path: str) -> list[Any]:
    require_type(value, list, path=path)
    assert isinstance(value, list)
    return value


def require_object(value: Any, *, path: str) -> dict[str, Any]:
    require_type(value, dict, path=path)
    assert isinstance(value, dict)
    return value


def require_in_range(value: float, *, path: str, minimum: float | None = None, maximum: float | None = None) -> None:
    if minimum is not None and value < minimum:
        raise ValidationError(path, f"must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValidationError(path, f"must be <= {maximum}")


def _validate_thermo_summary(payload: dict[str, Any], *, path: str) -> None:
    temp = require_finite_number(require_key(payload, "temperature_K", path=path), path=_join_path(path, "temperature_K"))
    require_in_range(temp, path=_join_path(path, "temperature_K"), minimum=1e-12)

    require_int(require_key(payload, "num_configurations", path=path), path=_join_path(path, "num_configurations"), minimum=1)

    require_finite_number(require_key(payload, "mixing_energy_min_eV", path=path), path=_join_path(path, "mixing_energy_min_eV"))
    require_finite_number(require_key(payload, "mixing_energy_avg_eV", path=path), path=_join_path(path, "mixing_energy_avg_eV"))

    partition = require_finite_number(require_key(payload, "partition_function", path=path), path=_join_path(path, "partition_function"))
    require_in_range(partition, path=_join_path(path, "partition_function"), minimum=1e-300)

    require_finite_number(require_key(payload, "free_energy_mix_eV", path=path), path=_join_path(path, "free_energy_mix_eV"))


def _validate_provenance(payload: dict[str, Any], *, path: str) -> None:
    require_string(require_key(payload, "schema_version", path=path), path=_join_path(path, "schema_version"))
    git_commit = require_key(payload, "git_commit", path=path)
    if git_commit is not None:
        require_string(git_commit, path=_join_path(path, "git_commit"))
    require_string(require_key(payload, "python_version", path=path), path=_join_path(path, "python_version"))
    require_string(require_key(payload, "platform", path=path), path=_join_path(path, "platform"))
    require_object(require_key(payload, "cli_args", path=path), path=_join_path(path, "cli_args"))
    require_string(require_key(payload, "input_hash", path=path), path=_join_path(path, "input_hash"))


def _validate_metrics(payload: dict[str, Any], *, path: str) -> None:
    _validate_thermo_summary(payload, path=path)
    created_at = payload.get("created_at")
    if created_at is not None:
        require_string(created_at, path=_join_path(path, "created_at"))
    repro = payload.get("reproducibility_hash")
    if repro is not None:
        require_string(repro, path=_join_path(path, "reproducibility_hash"))
    provenance = payload.get("provenance")
    if provenance is not None:
        _validate_provenance(require_object(provenance, path=_join_path(path, "provenance")), path=_join_path(path, "provenance"))


def _validate_diagnostics(payload: dict[str, Any], *, path: str) -> None:
    temp = require_finite_number(require_key(payload, "temperature_K", path=path), path=_join_path(path, "temperature_K"))
    require_in_range(temp, path=_join_path(path, "temperature_K"), minimum=1e-12)
    ncfg = require_int(require_key(payload, "num_configurations", path=path), path=_join_path(path, "num_configurations"), minimum=1)

    require_finite_number(require_key(payload, "expected_energy_eV", path=path), path=_join_path(path, "expected_energy_eV"))
    variance = require_finite_number(require_key(payload, "energy_variance_eV2", path=path), path=_join_path(path, "energy_variance_eV2"))
    require_in_range(variance, path=_join_path(path, "energy_variance_eV2"), minimum=0.0)

    std = require_finite_number(require_key(payload, "energy_std_eV", path=path), path=_join_path(path, "energy_std_eV"))
    require_in_range(std, path=_join_path(path, "energy_std_eV"), minimum=0.0)

    p_min = require_finite_number(require_key(payload, "p_min", path=path), path=_join_path(path, "p_min"))
    require_in_range(p_min, path=_join_path(path, "p_min"), minimum=0.0, maximum=1.0)

    ess = require_finite_number(require_key(payload, "effective_sample_size", path=path), path=_join_path(path, "effective_sample_size"))
    require_in_range(ess, path=_join_path(path, "effective_sample_size"), minimum=0.0, maximum=float(ncfg))

    require_finite_number(require_key(payload, "logZ_shifted", path=path), path=_join_path(path, "logZ_shifted"))
    require_finite_number(require_key(payload, "logZ_absolute", path=path), path=_join_path(path, "logZ_absolute"))

    notes = require_array(require_key(payload, "notes", path=path), path=_join_path(path, "notes"))
    for index, note in enumerate(notes):
        require_string(note, path=f"{_join_path(path, 'notes')}[{index}]")

    provenance = payload.get("provenance")
    if provenance is not None:
        _validate_provenance(require_object(provenance, path=_join_path(path, "provenance")), path=_join_path(path, "provenance"))


def _validate_run_manifest(payload: dict[str, Any], *, path: str) -> None:
    require_string(require_key(payload, "config_sha256", path=path), path=_join_path(path, "config_sha256"))
    require_int(require_key(payload, "config_schema_version", path=path), path=_join_path(path, "config_schema_version"), minimum=1)
    git_commit = require_key(payload, "git_commit", path=path)
    if git_commit is not None:
        require_string(git_commit, path=_join_path(path, "git_commit"))
    require_string(require_key(payload, "input_csv_sha256", path=path), path=_join_path(path, "input_csv_sha256"))
    require_string(require_key(payload, "package_version", path=path), path=_join_path(path, "package_version"))
    require_string(require_key(payload, "platform", path=path), path=_join_path(path, "platform"))
    require_string(require_key(payload, "python_version", path=path), path=_join_path(path, "python_version"))

    outputs = require_array(require_key(payload, "produced_outputs", path=path), path=_join_path(path, "produced_outputs"))
    for index, output in enumerate(outputs):
        require_string(output, path=f"{_join_path(path, 'produced_outputs')}[{index}]")


_VALIDATORS: dict[str, Callable[[dict[str, Any]], None]] = {
    "thermo_summary": lambda p: _validate_thermo_summary(p, path="root"),
    "metrics": lambda p: _validate_metrics(p, path="root"),
    "diagnostics": lambda p: _validate_diagnostics(p, path="root"),
    "run_manifest": lambda p: _validate_run_manifest(p, path="root"),
}


def detect_kind_from_filename(path: Path) -> str:
    file_path = Path(path)
    name = file_path.name.lower()
    if name == "metrics.json":
        return "metrics"
    if name == "run_manifest.json":
        return "run_manifest"
    if re.match(r"diagnostics_t\d+\.json", name):
        return "diagnostics"
    return "thermo_summary"


def validate_output(obj: dict[str, Any], *, kind: str | None = None) -> None:
    output_kind = "thermo_summary" if kind is None else kind
    validator = _VALIDATORS.get(output_kind)
    if validator is None:
        raise ValidationError("root", f"unknown output kind '{output_kind}'")
    validator(obj)


def validate_output_file(path: Path, *, kind: str | None = None) -> None:
    file_path = Path(path)
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValidationError("root", f"expected top-level object, got {type(payload).__name__}")
    validate_output(payload, kind=kind or detect_kind_from_filename(file_path))
