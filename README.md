# mode-resolved-calibration

Reproducibility package for the mode-resolved calibration-confidence continuum: a
framework that scores an estimator's calibratability by the *weakest identifiable
modes* of its Fisher information, rather than by aggregate spectral summaries.

## Overview

For a photometric inverse problem `y = A x + B δc + ε` with nuisance terms `δc` that
absorb the uncalibrated manifold of the scene, the information content along the
calibratable subspace shrinks as the nuisance prior weakens (`Λ → 0`).  We study that
shrinkage through the Schur complement (delta-Fisher information), its normalized
retention spectrum, and the closed-form response of the gauge direction.  The main
empirical result, on the OpenIllumination controlled-corruption benchmark:
mode-resolved coincides with E-optimality on the tracked modes (max diff ≤ 1e-14)
and improves the stratified median over trace and log-determinant (0.536 vs 0.418 /
0.400) — object-cluster bootstrap 95% CI [−0.096, 0.858] (n = 11; not significant at
the object level).

## Mathematical core

1. Model and nuisance priors:

   `y = A x + B δc + ε`,   `δc ~ N(0, Σ_c)`,   `Λ = σ² Σ_c⁻¹`

2. Delta-Fisher (Schur complement, the `calibratable` residual information):

   `ΔF(Λ) = Aᵀ [ I − B (BᵀB + Λ)⁻¹ Bᵀ ] A`

3. Calibration-retention spectrum:

   `R(Λ) = F∞^(−1/2) · ΔF(Λ) · F∞^(−1/2)`,   `0 ≼ R(Λ) ≼ I`

   The eigenvalues `0 ≤ ρ₁ ≤ … ≤ ρ_q ≤ 1` are per-mode retention levels; the weakest
   (bottom) modes are the smallest eigenvalues of `R(Λ)`, and the mode-resolved
   criterion tracks those bottom modes across the hyperparameter path.

4. Gauge spectral response (closed form):

   `aᵀΔF(λ) a = Σᵢ αᵢ² sᵢ² λ / (sᵢ² + λ)`,   `(sᵢ², V) = eig(BᵀB)`,   `α = Vᵀ c̄`

## Key results

On the OpenIllumination controlled-corruption benchmark (11 objects × 6 corruption
levels, 20 seeds per cell, held-out objects):

- median within-cell Spearman $R_A$ = 0.90 (object-cluster bootstrap 95% CI
  [0.90, 0.95], 66/66 cells, 11/11 objects);
- stratified (fixed-level) median Spearman 0.536 (mode-resolved) vs 0.418
  (log-determinant) / 0.400 (trace) — best stratified median across levels
  (per-level reversals disclosed: at L1, trace 0.455 > mode 0.245; at L3,
  logdet 0.700 > mode 0.673).

![fig1](paper/figures/fig1_stratified_median.png)

![fig2](paper/figures/fig2_pooled.png)

See `results/openillumination/` for the frozen per-cell tables and
`tests/test_reproduction.py` for the independent recomputation of all headline numbers.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate          # or: .venv\Scripts\activate
pip install -e .
pytest
bash reproduce_paper.sh
```

The first command fetches dependencies; the second runs the full test suite including
the independent recomputation of the nine headline numbers; the third regenerates
figures and tables from the frozen results.

## Data

Raw benchmarks are not included. Download OpenIllumination (Hugging Face) and
DiLiGenT (official page) separately; metadata and manifests live in
`data/manifests/`. See `docs/DATA.md`.

## Repository layout

- `src/calibinfo/` — library: information (Schur, retention, gauge, mode tracking),
  estimators, datasets, metrics, io
- `experiments/` — one entry-point script per experiment
- `configs/` — frozen per-experiment protocol YAMLs
- `tests/` — unit, known-answer, and reproduction tests
- `results/` — frozen numerical artifacts (per-cell tables, summaries)
- `docs/` — `EXPERIMENTS.md`, `REPRODUCIBILITY.md`, `DATA.md`
- `reproduce_paper.sh` — figures/tables regeneration

## Citation / License

Cite via `CITATION.cff`. Licensed under the terms in `LICENSE`.