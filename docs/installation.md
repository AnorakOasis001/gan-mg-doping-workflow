# Installation

Python **3.10+** is required.

We recommend a fresh virtual environment.

```bash
python -m pip install -U pip
python -m pip install -e ".[dev]" --no-build-isolation
```

Editable install is required; do not run modules directly from `src/`.

## Validate your installation

```bash
ganmg --help
pytest
```
