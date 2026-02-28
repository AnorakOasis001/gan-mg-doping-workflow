from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


THERMO_VS_T_FIELDNAMES = [
    "temperature_K",
    "num_configurations",
    "mixing_energy_min_eV",
    "mixing_energy_avg_eV",
    "partition_function",
    "free_energy_mix_eV",
]


def ensure_parent_dir(path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Mapping[str, Any]) -> None:
    path = Path(path)
    ensure_parent_dir(path)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_thermo_vs_T_csv(rows: Sequence[Mapping[str, Any]], out_csv: Path) -> None:
    out_csv = Path(out_csv)
    ensure_parent_dir(out_csv)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=THERMO_VS_T_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            missing = [k for k in THERMO_VS_T_FIELDNAMES if k not in row]
            if missing:
                raise ValueError(f"Missing required thermo_vs_T row keys: {', '.join(missing)}")
            writer.writerow({k: row[k] for k in THERMO_VS_T_FIELDNAMES})
