# Changelog

All notable changes to this project are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning: [semver](https://semver.org/).

## [Unreleased]

### Added

- `examples/` — three self-contained scripts (retention/gauge visualization,
  mode-resolved vs scalar criteria, minimal allocation demo), each generating
  its own synthetic data and figures
- `CONTRIBUTING.md`, issue/PR templates, this changelog
- GitHub Actions CI: pytest matrix, checksum verification, claim gate,
  frozen-zone drift gate

### Changed

- README rewritten library-first (was: reproducibility-package framing);
  the published benchmark is now a separate "Benchmark reproduction" section
- `pyproject.toml`: version 0.2.0, license/urls/authors/keywords/classifiers
  metadata, `examples` extra

## [0.1.0] — 2026-09-09 (tag `science-closed`)

Initial public snapshot of the research artifact:

- `src/calibinfo` — Schur-complement delta-Fisher information, retention
  spectrum, gauge closed form, mode tracking, allocation sensitivity kernels,
  estimators, dataset loaders
- frozen benchmark: OpenIllumination controlled-corruption mode-ranking study,
  calibration-allocation evaluation, DiLiGenT sanity, synthetic validity panels
- known-answer test suite (V/V-B series) and independent reproduction tests
- preregistered allocation evaluation: strong grade on both regimes
  (Δ_random 11/11 objects improved, CI excludes 0)
