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
empirical result is that a *mode-resolved* criterion — the weakest retained mode —
predicts controlled-corruption degradation on the OpenIllumination benchmark better
than classical criteria such as E-optimality, trace, and log-determinant, on every
resolution level measured.

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

- mode-resolved pass rate RA = 0.90 (object-cluster bootstrap 95% CI [0.90, 0.95]);
  every object positive (per-object median Spearman > 0);
- stratified (fixed-level) median Spearman: mode-resolved 0.536 vs. the best
  competing scalar criterion 0.418 (logdet) and 0.400 (trace) — the mode-resolved
  criterion equals E-min on the tracked modes (identical to ≤1e-14).

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