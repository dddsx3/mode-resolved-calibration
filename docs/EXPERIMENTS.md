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
- **λ⋆ naming lock**: λ⋆ is the **directional crossover precision** — the
  calibration precision at which a fixed gauge direction's Rayleigh information
  reaches a pre-specified reference floor (existence condition `0 < μ < ‖Aa‖²`);
  within-scene interpretation only (math freeze v1.0 §29)

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
- **Math-interface conventions (math freeze v1.0 M0-1/M0-2)**: the noise fit and
  the empirical mode projection each have a `legacy` (frozen-benchmark
  bit-reproducible, default) and a `corrected` convention
  (`noise_fit_convention` / `mode_coordinate`); the corrected arms are exercised
  by the factorial rerun (§11)

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

## 10. Calibration-allocation evaluation  (`experiments/openillumination_allocation.py`)

Preregistered retrospective actionability evaluation on a fixed held-out cohort.
Config: `configs/openillumination_allocation.yaml` (frozen at tag
`allocation-prereg-frozen`, before any empirical allocation output).

- **Dataset**: the 11 held-out OpenIllumination objects (cohort copied verbatim from
  `configs/openillumination.yaml`; zero selection freedom). Positioning and the n=11
  disclosure are recorded verbatim in the config's `positioning` field.
- **Selection universe**: K = 142 lights per object; the frozen 48-light analysis
  subset is Fisher-active, the remaining 94 lights are inactive placeholders
  (u = 0) with zero effect on prediction or reconstruction.
- **Regimes**: {10×, 100×} — the selected light's Σ_φ diagonal is divided by the
  regime factor (equivalently λ_ℓ × regime).
- **Budgets**: {0.1, 0.2, 0.4, 0.6, 0.8} of K → k = round(b·K) = {14, 28, 57, 85, 114}.
- **Policies (five, one shared sequential frame)**: mode-aware — the **adaptive
  normalized weak-Fisher-mode sensitivity heuristic** (G_ℓ from the per-light
  sensitivity kernel g_ℓj over the bottom-5 tracked modes, recomputed each
  step; a heuristic aggregate, not an exact retention-gradient optimization,
  math freeze v1.0 §33), E-opt (λ_min gain), A-opt (trace(ΔF⁺) decrease,
  positive-subspace pseudo-A as implemented), D-opt (positive-subspace logdet
  gain), random (5 fixed-seed permutations estimating the same-object
  random-policy expectation). Every policy yields one full 142-light
  ordering per (object, level, regime); budgets are prefixes; the selected light's
  precision is updated immediately before the next step; ties break on ascending
  light index.
- **Leakage surface**: the selector reads the whitened per-light blocks only
  (u, M0, Λ₀, F∞, active mask); GT normals and reconstruction errors are not
  representable in the selection state (asserted by `tests/test_allocation_known_answers.py`).
- **Estimator / metric**: unchanged fixed-n̂ whitened per-pixel GLS;
  per-run error = gauge-aligned albedo residual projected on the object's bottom-5
  modes (V_level frozen at base Σ_φ before any allocation).
- **Primary endpoint**: normalized trapezoidal AUC of the budget-curve error;
  lower is better.
- **Statistics**: Δ_random = median over the 11 objects of (U_mode − U_random);
  paired object-cluster bootstrap B=10000, seed 20260910, resamples shared across
  policies; all 11 paired differences + exact sign count reported; Δ_Eopt/Δ_Aopt/
  Δ_Dopt as secondary; frozen result grades strong/neutral/negative per the
  preregistered rule (historical internal labels — see the semantic erratum in
  `docs/WORDING.md`: the award tested beating random, not superiority over the
  classical baselines; public text says "actionable outcome vs random").
- **Provenance**: run manifest, driver, analysis script and the run-layer change
  record live in `results/openillumination/allocation/provenance/`.
- **Output**: `results/openillumination/allocation/` — `per_run_errors.csv`,
  `uos_table.csv`, `selection_orders.json`, `allocation_summary.json`,
  `allocation_deltas.csv`, `allocation_object_pairs.csv`,
  `allocation_policy_pairwise.csv`, `allocation_random48_per_run.csv`,
  `allocation_random48_summary.json`.

Post-hoc analyses (2026-09-10, registered in `docs/WORDING.md` §5; no new
deterministic policy runs):

- **Paired policy comparison** (`provenance/a5_pairwise.py` →
  `allocation_policy_pairwise.csv`, labelled `posthoc_paired_comparison`):
  mode-aware vs each classical baseline on paired differences with the identical
  bootstrap. Classical E/A/D-opt greedy achieve modestly lower AUC than
  mode-aware (all six paired 95% CIs exclude 0; median paired difference
  +0.019 to +0.027).
- **`random_active48` attribution control**
  (`experiments/openillumination_allocation_random48.py` →
  `allocation_random48_summary.json`, labelled `posthoc_attribution_control` —
  the single bounded experiment authorized after science closure): uniform
  permutations over the 48 Fisher-active lights only, same cohort/levels/seeds/
  regimes/budgets and paired raw innovations. Restricting random to the active
  set removes the mode-aware advantage entirely (Δ(mode − random48) = +0.014 at
  10× / +0.019 at 100×, CIs span 0, 3/11 objects), while
  random48 − random_full = −0.198 / −0.342 (CIs exclude 0): the measured benefit
  over full-universe random is an active-set effect, not a mode-ordering effect.

## 11. Math-freeze correctness rerun (MF-0.3 factorial, `experiments/openillumination_factorial.py`)

Preregistered four-arm rerun of the §5 controlled-corruption protocol after the
two math-interface corrections of the math-method freeze v1.0 (M0-1 noise-fit
coefficient order; M0-2 normalized dual-coordinate mode projection). Same
objects, pixel subsets, levels, seeds and statistics across all arms; the arm
labels were fixed before the rerun:

| arm | noise fit (M0-1) | mode coordinate (M0-2) | role |
|-----|------------------|------------------------|------|
| A | legacy | legacy | == frozen benchmark (machinery anchor) |
| B | corrected | legacy | |
| C | legacy | dual | |
| D | corrected | dual | **paper-facing (fixed a priori)** |

- **Sampling**: one shared rng stream; the corrected scenes reuse the legacy
  scenes' frozen (light, pixel) subsets; corruption draws are shared.
- **Machinery check**: arm A vs the frozen §5 summary — max relative difference
  over all pred/emp entries plus the pooled Spearman; tolerance 1e-7
  (expected ~1e-12), hard error on breach.
- **Reported per arm**: R_A (median within-cell Spearman of S = 1 − 1/ρ vs
  empirical degradation), object-cluster bootstrap 95% CI (B=10000, seed
  20260908), positive cells / positive objects, stratified (fixed-level)
  median for P_mode and each scalar predictor, and the paired
  Δ(mode − best scalar) with within-replicate re-selection.
- **Seed**: experiment seed `20260907` (frozen §5 config, unchanged)
- **Output**: `results/openillumination/correctness/` —
  `mf0_factorial_summary.json` (+ `mf0_factorial_rows_<arm>.json`); no frozen
  artifact is modified.

---

**Reproduction contract**: every experiment script writes only statistical values and
manifests to `results/*/`; N1–N9 are independently recomputed from the CSVs by
`tests/test_reproduction.py` (frozen at `science-closed`) and N10–N12 by
`tests/test_manuscript_evidence.py` (frozen at `manuscript-evidence-v1`).