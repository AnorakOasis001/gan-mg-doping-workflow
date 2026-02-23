import argparse
import csv
import random
from pathlib import Path


def main() -> None:
    """
    Demo-only generator.

    Writes a small CSV with placeholder energies so the analysis stage can run end-to-end
    without Janus-core / MACE. Later, you'll replace this with real structure generation.
    """
    parser = argparse.ArgumentParser(description="Generate a small demo dataset.")
    parser.add_argument("--n", type=int, default=10, help="Number of demo structures.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    args = parser.parse_args()

    random.seed(args.seed)

    out_dir = Path("data") / "sample_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "results.csv"

    rows = []
    for i in range(args.n):
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

    print(f"Wrote {len(rows)} rows -> {out_csv}")


if __name__ == "__main__":
    main()