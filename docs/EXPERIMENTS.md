# Experiment protocols

Each section records one experiment as an objective protocol with the fixed fields
Dataset → sampling → corruption → estimator → metric → bootstrap → seed → output.
No analysis history is included; the fields below are exactly what a fresh run needs
to reproduce the frozen artifacts.

---

## 1. Synthetic algebraic identities  (`experiments/numerical_identities.py`)

- **Dataset**: synthetic Gaussian-linear system `y = A x + B δc + ε` built on the fly
- **Sampling**: grid over `q ∈ {1,3,9,36}` with underdetermined (`m<q`), critical (`m=q`)
  and overdetermined (`m=4q`) blocks; full-rank and low-rank `J_φ` maps; homoscedastic
  and diagonal-heteroscedastic `Σ_y`; 5 seeds per case; a photometric structure case
  (`A = D(s)`, `B = D(aH)Y`, strong coupling + scale gauge, `shift_gate=0.1`) is included
- **Corruption**: none (pure algebra; Λ swept including Λ=0)
- **Estimator**: none — computes `ΔF` via Schur dual route (white/non-white) vs marginal
- **Metric**: elementwise relative error of the dual routes, identity-panel tolerances
  (`<1e-10`), rank-definite Λ=0 continuity
- **Bootstrap**: not applicable
- **Seed**: `20260907`
- **Output**: `results/synthetic/ci01_formal_summary.json`

## 2. Gauge spectrum and λ⋆  (`experiments/gauge_spectrum.py`)

- **Dataset**: synthetic bridgeless configurations; geometries
  `{sphere, bumpy, terminator_heavy}`, albedo spreads `{narrow, medium, wide}`,
  light elevations `{25°, 45°, 65°}`, light counts `{1, 3}`
- **Sampling**: full grid across the geometry × albedo × elevation × light-count space
- **Corruption**: none — λ grid spans 25 decades for the closed-form sweep
- **Estimator**: none
- **Metric**: closed-form gauge response vs direct Rayleigh on the same grid
  (max relative error), λ⋆ location and its log-error median
- **Bootstrap**: not applicable
- **Seed**: `20260907`
- **Output**: `results/gauge_spectrum/ci02_formal_summary.json`
- **Headline bound**: closed-vs-direct max = 2.55e-8 (bound 3.9e-8) over the 25 decades

## 3. Monte-Carlo tightness and linearization validity  (`experiments/monte_carlo_validation.py`)

- **Dataset**: synthetic; photometric structures (`P=400`, `N∈{1,3}`, two noise levels,
  one high-condition terminatrix-heavy grid with 10000 trials) and one random-block case
- **Sampling**: 3 scenes per case; joint ensemble draws of `(δc, ε)`
- **Corruption**: none (ensemble noise)
- **Estimator**: linear/gamma-fit estimators under both ensemble arms
- **Metric**: empirical covariance vs `σ²ΔF⁻¹` (variance ratio median/IQR), nominal
  68%/95% coverage, bias ratio and conditional-variance ratio for the fixed-δc arm
- **Bootstrap**: not applicable (Monte-Carlo draws)
- **Seed**: `20260907`; per-case trials 4000 (10000 on the high-condition grid)
- **Output**: `results/monte_carlo/ci03_formal_summary.json`
- **Headline**: median variance ratio ≈ 1.0 (registration 1.0045)

## 4. Nonlinear validity envelope  (`experiments/nonlinear_validity.py`)

- **Dataset**: synthetic scenes re-rendered through an analytic nonlinear SH+ReLU model
  (same geometry family as §2)
- **Sampling**: 2 scenes per geometry; corruption strength grid
  `k ∈ {0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0}`; 2000 trials per scene
- **Corruption**: controlled nonlinear rendering mismatch (arm B)
- **Estimator**: linear-theory estimator applied to the nonlinear renderer output
- **Metric**: theoretical vs empirical degradation ratio `|emp/pred − 1|`
  (weak-mode median); valid-domain gate `<10%`
- **Bootstrap**: not applicable
- **Seed**: `20260907`
- **Output**: `results/nonlinear/ci03nl_nl_formal_summary.json`

## 5. OpenIllumination controlled corruption  (`experiments/openillumination_validation.py`)

- **Dataset**: OpenIllumination (HF `OpenIllumination/OpenIllumination`, CC BY 4.0),
  held-out objects: `obj_03_pumpkin, obj_04_dolphin, obj_07_pumpkin2, obj_09_ball,
  obj_10_pumpkin3, obj_11_pine, obj_13_mushroom, obj_16_friends_cup, obj_17_pumpkin5,
  obj_18_fabric_hat, obj_19_cylinder`
- **Sampling**: 48 of 142 lights and 1200 pixels per object (frozen selection recorded
  in `data/manifests/openillumination_dev_manifest.json`); 20 seeds per cell
- **Corruption**: joint (intensity + direction-position) with
  levels `[0.1, 0.2, 0.35, 0.5, 0.75, 1.0]`
- **Estimator**: joint calibration on the corrupted renderings
- **Metric**: within-cell Spearman of predicted vs empirical degradation, pooled and
  per-level; clusters: (object, level) cells = 66, mode points = 5 tracked modes × 330
- **Bootstrap**: object-cluster bootstrap, **B=10000**, seed **20260908**
- **Seed**: experiment seed `20260907`
- **Output**: `results/openillumination/ci04_formal_summary.json`,
  `results/openillumination/mode_ranking.csv`, `level_severity.csv`

## 6. Fixed-level severity: mode-resolved vs scalar criteria  (`experiments/openillumination_severity.py`)

Criterion definitions used throughout:

- **Mode-resolved (P_mode)**: damage predicted by the bottom tracked modes,
  `P_mode = 1/(1 − ρ_mode)` where `ρ_mode` derives from the inverse of the weakest
  tracked retention eigenvalue; 5 tracked modes along λ.
- **E-min (P_emin)**: operand = the same weakest-mode datum (theoretically equal to
  P_mode on the tracked modes).
- **Trace (P_trace)**: residual trace of the retention spectrum.
- **Log-determinant (P_logdet)**: log deficit of the retention spectrum.
- **Fixed-level severity**: for each corruption level, Spearman between each criterion's
  prediction and the empirical degradation across objects;
  the stratified statistic is the median of the six per-level values.
- **Within-cell Spearman**: within one (object, level) cell, Spearman between the
  theoretical damage `S = 1 − 1/pred_deg` (per tracked mode) and the empirical
  degradation of the same modes.
- **Object-cluster bootstrap**: whole objects are the sampling units (object-internal
  cells are kept together), B=10000, seed 20260908 (for the fixed-level statistics)
- **Seed**: data seed `20260907`; bootstrap seed `20260908`
- **Output**: `results/openillumination/predictor_comparison.csv`,
  `results/openillumination/level_severity.csv`,
  regenerated `results/openillumination/validation_summary.json`

## 7. DiLiGenT external sanity and failure taxonomy  (`experiments/diligent_sanity.py::run_sanity`)

- **Dataset**: DiLiGenT 10 objects (ballPNG, …)
- **Sampling**: 96 lights kept, 1200 masked pixels per object
- **Corruption**: none (real non-Lambertian effects: shadow, specularity)
- **Estimator**: full 96-light calibration for per-object Lambertian residual
- **Metric**: per-object Lambertain residual + outlier fraction; weak-mode floor <<
  median; gauge cross-observability; failure taxonomy: model-mismatch variance share
  vs calibration-propagation variance share (magnitude comparison only)
- **Bootstrap**: not applicable
- **Seed**: `20260907`
- **Output**: `results/diligent/ci05_formal_summary.json`

## 8. Robustness ablations  (`experiments/diligent_sanity.py::run_ablation`)

- **Dataset**: synthetic within-scene
- **Sampling**: fixed geometry (bumpy), 4 scenes, light counts `{1,2,3}`
- **Corruption/probes**: (1) whitening vs raw weighting; (2) Λ mis-specified ×4;
  (3) light-count sweep; (4) noise-model sweep (homoscedastic vs diagonal-heteroscedastic)
- **Estimator**: λI weighting and whitened weighting under each probe
- **Metric**: within-scene retention spectra and λ⋆ shifts (median/IQR)
- **Bootstrap**: not applicable
- **Seed**: `20260907`
- **Output**: `results/diligent_ablation/ci05abl_ablation_summary.json`

## 9. Known-answer preflight  (`tests/test_known_answer_precheck.py`)

Deterministic checks P1–P5 on d=100 diagonals / rank-deficient parallel sums / gauge
identities / identificable-overlap contracts. This is the gate any library change must
re-pass.

---

**Reproduction contract**: every experiment script writes only statistical values and
manifests to `results/*/`; the nine headline numbers are independently recomputed from
the CSVs by `tests/test_reproduction.py`.