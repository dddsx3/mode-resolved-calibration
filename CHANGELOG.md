# Changelog

All notable changes to this project are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning: [semver](https://semver.org/).

## [0.3.0] — 2026-09-10

Math-freeze correctness gates (v1.0 §57, MF-0.1–MF-0.6) and documentation
freeze. No frozen artifact under `results/` was modified; the corrected math
interface is exercised by a preregistered factorial rerun into
`results/openillumination/correctness/`.

### Fixed

- **M0-1** noise-fit coefficient order: `noise_fit`
  (`experiments/openillumination_validation.py`) and `calibrate`
  (`src/calibinfo/estimators/joint_map.py`) consumed the `np.polyfit` return
  `[slope, intercept]` swapped into the model `Var ≈ a + b·I`. Both now expose
  `convention="legacy"|"corrected"` (legacy default preserves frozen-artifact
  bit-reproducibility); known-answer tests added
- **M0-2** empirical mode-projection coordinate: corrected normalized dual
  coordinate `E @ F∞^{1/2} U` (strict `1/ρ_j` theorem) available beside the
  frozen `E @ U` via `predicted_degradation(mode_coordinate=...)`
- README quickstart referenced an undefined `Finf` and misread the
  `retention_spectrum` return (dict, not matrix)
- frozen-artifact hash gate was EOL-broken: the recorded
  `frozen_artifacts_sha256` values are CRLF-basis while `.gitattributes` pins
  LF checkouts, so `openillumination_severity.py` failed its own integrity
  assertion on any conforming clone; the check is now EOL-robust (zero
  tolerance unchanged) with a regression test

### Added

- `delta_f_marginal_factor` / `nuisance_factor` (schur.py): covariance-factor
  marginalization `Σ_c = LLᵀ, C = BL` — the correct route for singular proper
  Gaussian covariance (M0-5); known-answer B=[1,1], Σ_c=diag(1,0) test
  (factor/marginal = 0.5, pseudoinverse-precision shortcut = 0, never equivalent)
- `retention_spectrum` now also returns `modes` (R's eigenvectors paired with
  the ascending `rho`)
- `calibinfo.allocation.rank_invariance.rank_invariance_report` (M0-6): ΔF
  numerical-rank invariance across candidates/budgets, licensing the
  E/A/D-optimal naming for the classical baselines
- `experiments/openillumination_factorial.py` (M0-3): preregistered A/B/C/D
  factorial rerun of the controlled-corruption protocol (arm A = frozen
  benchmark machinery anchor; arm D = paper-facing, fixed a priori), outputs
  under `results/openillumination/correctness/`
- `tests/test_math_freeze_gates.py`: MF-0.1/0.4/0.5/0.6 known-answer gates
  (13 tests, no raw-data dependency)

### Changed

- Docs freeze (v1.0 §46): retention dimension `r = rank(F∞)` (not `q`);
  generalized-eigenvalue phrasing everywhere (ordinary-eigenvalue-ratio claim
  banned); λ⋆ renamed **directional crossover precision** with existence
  condition and within-scene interpretation lock; current mode-aware allocation
  named **adaptive normalized weak-Fisher-mode sensitivity heuristic**;
  profiling-precision vs proper-covariance semantics stated in schur.py and
  README; wording gate enforces the new banned families
- CI03 note: its analysis basis is the whitened coordinate system; the strict
  `R⁻¹` coordinate relation is covered by the MF-0.4 known-answer test

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
