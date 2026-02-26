# Installation

Python **3.10+** is required.

We recommend a fresh virtual environment.

```bash
python -m pip install -U pip
python -m pip install -e ".[dev]" --no-build-isolation
```

Editable install is required; do not run modules directly from `src/`.

## Optional docs tooling

Documentation dependencies are intentionally separate from runtime dependencies.
Install docs dependencies with `pip install -e ".[docs]".
MkDocs is pinned <2 due to mkdocs-material compatibility.

```bash
python -m pip install -e ".[docs]"
python -m mkdocs serve
python -m mkdocs build --strict
```

## Validate your installation

```bash
ganmg --help
pytest
```
