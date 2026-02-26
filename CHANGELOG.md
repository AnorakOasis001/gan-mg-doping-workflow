# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows Semantic Versioning.

## [Unreleased]

### Added
- Release hygiene checks in CI to build wheel/sdist artifacts and verify installation from built wheel.

## [0.1.0] - 2026-02-25

### Added
- Golden fixture corpus and regression tests under `data/golden/v1` and `tests/test_golden_regression.py`.
- Streaming analysis parity coverage between batch and chunked APIs in `tests/test_streaming_parity.py`.
- Provenance and schema tagging in output records via `schema_version` and metadata emitted by analysis APIs/CLI.
- Minimal end-to-end runnable example in `examples/minimal_end_to_end/` with a dedicated test.
- Optional docs toolchain extras and strict MkDocs build workflow.
- Fresh-install CI matrix to validate clean installation/import/test flows on Linux and Windows.
