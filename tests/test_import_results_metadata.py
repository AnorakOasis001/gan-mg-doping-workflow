from __future__ import annotations

import csv
from pathlib import Path

from gan_mg.import_results import import_results_to_run


def test_import_results_preserves_composition_and_structure_name_metadata(tmp_path: Path) -> None:
    source_csv = tmp_path / "source.csv"
    with source_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "structure_id",
                "mechanism",
                "energy_eV",
                "mg_count",
                "ga_count",
                "n_count",
                "input_structure_name",
                "relaxed_structure_name",
                "n_atoms",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "structure_id": "s1",
                "mechanism": "mgi",
                "energy_eV": -1.0,
                "mg_count": 3,
                "ga_count": 178,
                "n_count": 180,
                "input_structure_name": "GaN_MgSub2_MgInt1_Sample1.cif",
                "relaxed_structure_name": "relaxed_s1.cif",
                "n_atoms": 361,
            }
        )

    run_dir = tmp_path / "runs" / "r1"
    import_results_to_run(run_dir, source_csv)
    out_csv = run_dir / "inputs" / "results.csv"
    with out_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["mg_count"] == "3"
    assert rows[0]["input_structure_name"] == "GaN_MgSub2_MgInt1_Sample1.cif"
    assert rows[0]["relaxed_structure_name"] == "relaxed_s1.cif"
    assert rows[0]["n_atoms"] == "361"


def test_import_results_can_register_and_copy_artifacts(tmp_path: Path) -> None:
    source_csv = tmp_path / "source.csv"
    artifacts_src = tmp_path / "structures_source"
    artifacts_src.mkdir()
    src_cif = artifacts_src / "relaxed_s1.cif"
    src_cif.write_text("data_test\n", encoding="utf-8")
    with source_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["structure_id", "mechanism", "energy_eV", "relaxed_structure_name"],
        )
        writer.writeheader()
        writer.writerow({"structure_id": "s1", "mechanism": "mgi", "energy_eV": -1.0, "relaxed_structure_name": "relaxed_s1.cif"})
    run_dir = tmp_path / "runs" / "r2"
    metadata = import_results_to_run(
        run_dir,
        source_csv,
        artifact_roots=[artifacts_src],
        copy_artifacts=True,
    )
    manifest = run_dir / "inputs" / "structures.csv"
    assert manifest.exists()
    with manifest.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["structure_id"] == "s1"
    copied_path = Path(rows[0]["path"])
    assert copied_path.exists()
    assert copied_path.parent == run_dir / "artifacts"
    assert "structures_manifest" in metadata
