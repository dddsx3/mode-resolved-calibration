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
- Paper-facing caliber sync (math-freeze-v1): README, docs/WORDING.md §1,
  docs/REPRODUCIBILITY.md (N1–N3 annotated, N13 added) and CITATION.cff now
  lead with the corrected-interface arm D numbers; the frozen arm-A record
  (including the retired stratified severity comparison) is kept explicitly
  labeled as provenance

## [Unreleased]

### Fixed

- `stratified_valid_levels` in the MF-0 factorial summary stored a boolean
  instead of the finite-level count (an external audit read it as "only 1 of
  6 levels valid"); all six levels are finite - field corrected, summary
  regenerated from the committed per-arm rows, evidence test pins the count
- `docs/EXPERIMENTS.md` section 6 `P_mode` formula did not match the code
  (`P_mode = 1 - rho_min`, not `1/(1 - rho_mode)`); definition freeze (L8)
  added distinguishing `P_mode` from the unbounded `pred_deg = 1/rho_j`

### Changed

- Caliber downgrade (potential-assessment adjudication): the within-cell
  statistic `R_A` is directional validation only - by construction it is
  rank-equivalent to the mode-index baseline (66/66 cells, deviation exactly
  0.0, both calibers; `experiments/directional_amplitude.py` ->
  `results/magnitude/directional_amplitude_summary.json`). README/WORDING/
  REPRODUCIBILITY/CITATION reframed accordingly; the amplitude layer
  (emp/pred ratio distribution, median 201.1 on the frozen record) is the
  magnitude-level finding
- New math-foundations lock (`tests/test_math_foundations.py`): direction
  parameterization known-answer, per-light Loewner monotonicity, midpoint
  convexity of J_A/J_E/J_D on the PD region (400/400), joint operator
  concavity, low-rank retention identity (spectral deviation <=1e-10, P-3L
  count exact), J_A gradient vs finite differences (<=1e-6, assembled dF)
- README: research-direction section for the certified allocation-analysis
  program with generated showcase figures
  (`scripts/make_direction_figures.py` -> `docs/img/`)

### Added

- P-CERT preregistration (`configs/certified_gaps.yaml`, committed before the
  run) + certificate machinery (`src/calibinfo/allocation/convex.py`) +
  driver (`experiments/certified_gaps.py`): budget-constrained convex program
  `min tr DeltaF(t)^{-1} s.t. sum(t_k-1) <= B` with Frank-Wolfe duality-gap
  global certificates, on the 11 real OpenIllumination objects
  (corrected-interface scenes, frozen allocation rng spec)

### Added

- P-LOWRANK-FULLRES preregistration (`configs/lowrank_fullres.yaml`) +
  driver (`experiments/lowrank_fullres.py`): full-resolution / full-142-light
  certified gap table via the Woodbury low-rank route (batched full-res
  Lambertian PS, no pixel subsampling)

### Added

- P-SUBMOD negative-result pack (`experiments/submodularity_search.py` →
  `results/submodularity/submodularity_search.json`): adversarial submodularity
  search with correct refinement-set semantics, degeneracy guard, and a
  detector self-check. Findings: E-opt (1/lambda_min) has genuine submodularity
  violations (1518 triples over 10 instances, gamma_min = 0.704); A-opt is at
  most marginally non-submodular (gamma_min = 0.99989); D-opt clean;
  adversarial near-collinear family clean. Harness tests pin a reproducible
  counterexample and the detector sensitivity/specificity

### Added

- P-ALLOC2 preregistration (`configs/allocation_mode_tail.yaml`) + driver
  (`experiments/allocation_mode_tail.py`): mode-tail targeted calibration
  intervention - targeted arm = the frozen weak-Fisher-mode heuristic vs
  random_active48, paired-by-seed, endpoint = bottom-5 tracked-mode energy in
  the normalized dual coordinate (M0-2 corrected); outcome-independent
- `experiments/lowrank_fullres.py` v1.1: per-object checkpoint resume +
  multiprocessing driver (cloud migration package mirrors it in
  Multi-Illumination-Inverse-Rendering branch cloud-mode-tail)

### Added (planned, on this branch)

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
