from __future__ import annotations

from pathlib import Path

import gan_mg


def _read_project_version() -> str:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    in_project_section = False

    for raw_line in pyproject_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            in_project_section = line == "[project]"
            continue
        if in_project_section and line.startswith("version"):
            _, value = line.split("=", maxsplit=1)
            return value.strip().strip('"').strip("'")

    raise AssertionError("Could not find [project].version in pyproject.toml")


def test_dunder_version_is_non_empty_string() -> None:
    assert isinstance(gan_mg.__version__, str)
    assert gan_mg.__version__.strip() != ""


def test_dunder_version_matches_pyproject() -> None:
    assert gan_mg.__version__ == _read_project_version()
