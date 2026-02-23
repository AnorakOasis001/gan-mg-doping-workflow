import csv
import random
from pathlib import Path


def generate_demo_csv(n: int, seed: int, out_csv: Path) -> Path:
    """
    Demo-only generator.
    Writes a CSV with placeholder energies and simple metadata.

    Later you'll replace this with real structure generation + relaxation outputs.
    """
    random.seed(seed)

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(n):
        rows.append(
            {
                "structure_id": f"demo_{i:04d}",
                "mechanism": random.choice(["MgGa+VN", "Mgi+2MgGa"]),
                "energy_eV": round(random.uniform(-0.8, 0.2), 6),
            }
        )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return out_csv