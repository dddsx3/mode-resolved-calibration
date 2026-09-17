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
  within-scene interpretation only (see `docs/methods.md` §4)

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
- **Pipeline conventions (legacy vs corrected)**: the noise fit and
  the empirical mode projection each have a `legacy` (frozen-benchmark
  bit-reproducible, default) and a `corrected` convention
  (`noise_fit_convention` / `mode_coordinate`); the corrected arms are exercised
  by the factorial rerun (§11)

## 6. Fixed-level severity: mode-resolved vs scalar criteria  (`experiments/openillumination_severity.py`)

Criterion definitions used throughout:

- **Mode-resolved (P_mode)**: `P_mode = 1 − ρ_min` (code:
  `1 − 1/max_j pred_deg_j`; identical to `P_emin` on the tracked modes — the
  scalarization identity). ρ_min is the weakest tracked retention eigenvalue.
  **Definition freeze (L8)**: `pred_deg_j = 1/ρ_j` is a *different, unbounded*
  quantity used for within-cell ranking; the two versions coincide in rank but
  not in magnitude — never substitute one for the other in magnitude claims.
  5 tracked modes along λ.
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
  see `docs/methods.md`), E-opt (λ_min gain), A-opt (trace(ΔF⁺) decrease,
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
  Δ_Dopt as secondary. The internal result grades recorded in
  `allocation_summary.json` tested beating random only — not superiority over
  the classical baselines (whose paired comparison is post-hoc; see the
  post-hoc analyses below).
- **Provenance**: run manifest, driver, analysis script and the run-layer change
  record live in `results/openillumination/allocation/provenance/`.
- **Output**: `results/openillumination/allocation/` — `per_run_errors.csv`,
  `uos_table.csv`, `selection_orders.json`, `allocation_summary.json`,
  `allocation_deltas.csv`, `allocation_object_pairs.csv`,
  `allocation_policy_pairwise.csv`, `allocation_random48_per_run.csv`,
  `allocation_random48_summary.json`.

Post-hoc analyses (2026-09-10, registered in `docs/claims.md`; no new
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

## 11. Interface-correctness factorial rerun (`experiments/openillumination_factorial.py`)

Preregistered four-arm rerun of the §5 controlled-corruption protocol after the
two pipeline-interface corrections (the heteroscedastic noise-fit
coefficient order and the normalized dual-coordinate mode projection;
see `docs/methods.md`). Same
objects, pixel subsets, levels, seeds and statistics across all arms; the arm
labels were fixed before the rerun:

| arm | noise fit | mode coordinate | role |
|-----|------------------|------------------------|------|
| A | legacy | legacy | == frozen benchmark (machinery anchor) |
| B | corrected | legacy | |
| C | legacy | dual | |
| D | corrected | dual | **paper-facing (fixed a priori)** |

- **Sampling**: one shared rng stream; the corrected scenes reuse the legacy
  scenes' frozen (light, pixel) subsets; corruption draws are shared.
- **Machinery check**: the legacy arm vs the frozen §5 summary — max relative difference
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

## 12. Certified optimality gaps（P-CERT, `experiments/certified_gaps.py`）

Preregistered protocol (config committed before any run; outcome-independent —
no sign-based gate, every computed row reported).

- **Question**: how much can calibration-precision allocation buy at all?
  Certified answer via the budget-constrained convex program
  `min J_A(t) = tr ΔF(t)^{-1} s.t. Σ_k (t_k − 1) ≤ B, t ∈ [1,κ]^L`.
- **Parameterization (pinned)**: `t` multiplies `Λ0k` — larger t = higher
  precision (machine-locked by `tests/test_math_foundations.py`; the inverted
  reading supports no certificate).
- **Objective (pinned)**: `J_A = tr ΔF⁻¹` — smooth and convex on the feasible
  region (ΔF jointly operator concave + PD; midpoint-verified). `J_E = −λmin`
  is convex too but nonsmooth at eigenvalue crossings and is report-only.
- **Certificates**: Frank–Wolfe duality gaps on the convex program are global
  optimality certificates; with `B = k(κ−1)` the feasible set contains every
  "exactly k refined lights" allocation, so each per-k bound is valid for that
  discrete family.
- **Candidates on the same functional**: J_A-greedy prefix (exact trace
  kernel), 10 seeded random k-subsets of the 48 Fisher-active lights,
  all-refined, none.
- **Grid**: 11 held-out objects × k ∈ {5, 10, 14, 28, 48}; level 0.5; κ = 10;
  corrected noise-fit convention; scene rng identical to the frozen
  allocation experiment.
- **Output**: `results/certification/certified_gaps.json` (rows + per-k
  summary + manifest). No frozen artifact is touched.

## 12a. Full-resolution certification（P-LOWRANK-FULLRES, `experiments/lowrank_fullres.py`）

Preregistered protocol (`configs/lowrank_fullres.yaml`, committed before the
run; v1.1 driver adds per-object checkpoint resume + multiprocessing).
Removes the "only 1200 subsampled pixels" surface of §12:

- **Resolution**: ALL masked pixels per object (P = 3559–10252 across the 11
  objects), ALL 142 lights Fisher-active (no analysis subset).
- **Route**: exact low-rank Woodbury (`src/calibinfo/information/lowrank.py`,
  L5 identity; `route="woodbury"`), O(P(3L)²) — dense is infeasible here.
- **Nominal PS**: batched normal-equation Lambertian alternating least squares
  (semantics of `calibrated_ps`; det-based singularity fallback), corrected
  noise-fit convention.
- **Grid**: 11 objects × k ∈ {5, 14, 48}; level 0.5; κ = 10; FW 40 iters /
  10-eval line search; 6 seeded random k-subsets.
- **Output**: `results/certification/lowrank_fullres.json`.
- **Outcome (2026-09-12)**: dynamic range 27.3–89.4% per object (median
  60.06% at every budget), greedy within 0.002–0.005% of the certified lower
  bound — consistent with the P=1200 subsample table (36–89%, median
  62.87%): subsampling did not distort the certified landscape.

## 13. Mode-tail targeted calibration intervention（P-ALLOC2, `experiments/allocation_mode_tail.py`）

Preregistered protocol (config committed before the run; outcome-independent).
The question only a mode-resolved analysis naturally poses: if the theory says
certain calibration components hurt the weakest tracked modes most, does a
targeted intervention on those components reduce the TARGETED modes' empirical
error, versus the same budget on random Fisher-active lights?

- **Arms (v1.1, three-arm revision)**: targeted = the frozen adaptive
  normalized weak-Fisher-mode sensitivity heuristic (per-object full
  142-light ordering, budgets are prefixes); scalar_targeted = the J_A
  steepest-descent greedy prefix (exact gradient, same scalar-OED objective
  as P-CERT — the distinguishing control); random_active48 = uniform
  permutations over the 48 Fisher-active analysis lights (6 permutations
  averaged per seed). All three arms share ONE residual pipeline function
  (`arm_energy`) — the structural guard that fixes the v1.0 bug below.
- **Grid**: 11 objects × levels {0.2, 0.5, 1.0} × regimes {10, 100} ×
  budgets k ∈ {5, 10, 14, 28, 48} × 10 seeds; scene rng identical to the
  frozen allocation; corrected interface.
- **Endpoint**: per-run energy of the bottom-5 tracked modes in the
  normalized dual coordinate `W = F∞^{1/2} V_bottom5` (corrected pipeline) of the
  gauge-aligned residual, paired by seed across arms.
- **Pre-registered criteria (config, committed before the run)**: J1 — does
  targeted reduce the energy vs random_active48; J2 — does scalar_targeted
  as well; J3 (distinguishing test) — is targeted's reduction strictly
  larger than scalar_targeted's. Yes ⇒ the mode-resolved language has
  incremental interventional value; no ⇒ honest negative result.
- **Output**: `results/mode_tail/allocation_mode_tail.json`.
- **Withdrawn v1.0 outcome**: the originally reported "all 10 (regime,
  budget) cells, 11/11 objects, all CIs excluding 0" significance was an
  artifact — the v1.0 random arm omitted the gauge-alignment step before
  projection (asymmetric residual calibers; ~28× energy inflation). See
  the `erratum` block in the result JSON.
- **Outcome (v1.1, 2026-09-12, three-arm shared-pipeline rerun)**: with the
  gauge bug fixed and all arms funneled through one pipeline, the
  pre-registered comparisons are reported regardless of direction
  (Δ = targeted − comparator in bottom-5 dual energy; positive = targeted
  worse; the k=48 cells are degenerate — every ordering selects all 48
  active lights, Δ ≡ 0):
  J1 — targeted vs random_active48 is **significantly worse in 5/8
  informative cells** (CIs entirely positive) and spans 0 in the other 3;
  in no cell is targeted better. J2 — scalar_targeted is significantly
  worse in 2/8 cells and indistinguishable in the rest. J3 — targeted shows
  no increment over the scalar arm (1/8 cells significantly different, in
  the worse direction). Honest negative result: after the caliber repair,
  the mode-targeted intervention chain shows no mode-specific advantage —
  consistent with the certified picture that policy differences live far
  inside the certified epsilon (§12). The pipeline regression guard now
  runs in CI on a synthetic scene (`tests/test_alloc2_arms_consistency.py`).

## 14. Linearization validity radius（P-RADIUS, `experiments/linearization_radius.py`）

**Erratum (v1, superseded)**: v1 injected the corruption **estimator-side**
(`estimate_albedo` fitted with the corrupted d2/g). At large levels the
intensity channel `g = exp(logs·σ·ℓ)` reaches `e^±16` per light, a single
extreme light dominates the GLS denominator `Σ w·m²`, and E(ℓ) collapsed by
~6 orders of magnitude (log-log slope −1.9 instead of +2) — v1 measured
estimator numerical collapse, not linearization failure. v1 outputs and its
"dev crosses 2× the reference energy" radius semantics are withdrawn.

- **Dataset**: the 11-object held-out OpenIllumination cohort
  (`configs/linearization_radius.yaml`), P = 1200 subsampled pixels, 48
  Fisher-active lights, corrected noise-fit convention, scene rng identical
  to the frozen benchmark.
- **Corruption**: joint (direction rotation + intensity scaling), levels
  ℓ ∈ {0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0} in the
  physical units of `CorruptionGenerator("joint", ℓ)`; 20 seeds per level;
  seed space disjoint from the scene rng.
- **Metric domain (v2)**: observation-side injection — the corrupted
  realization (d2, g) generates the true observation
  `I'(k,p) = ρ̂_p·max(n̂_p·d2_k, 0)·g_k` (full nonlinear forward, backface
  flips included); the analyst-side estimator is the **nominal-geometry**
  whitened per-pixel GLS (denominator fixed to the nominal ŝ², level
  independent — numerical collapse removed from the metric domain).
- **Endpoint**: gauge-aligned residual e = ρ̃ − ρ̂; bottom-5 dual-coordinate
  energy (W = F∞^{1/2} V_bottom5) averaged over seeds → E(ℓ);
  dev(ℓ) = E(ℓ)/E(ℓ_min).
- **Metric**: excess factor q(ℓ) = dev(ℓ)/(ℓ/ℓ_min)². First-order theory
  holds while q ≈ 1 (log-log slope +2); the radius is the first level where
  q crosses 2× / 10×. Reported as `radius_2x` / `radius_10x` with
  `median_over_crossed_subset`, `n_crossed`, `n_total` (the median is over
  the crossed subset only — objects that never cross are counted in
  `n_total`, not silently dropped).
- **Bootstrap/seed**: none beyond the 20 corruption seeds per (object,
  level); descriptive protocol, no sign-based gate.
- **Output**: `results/magnitude/linearization_radius.json`.
- **Outcome (v2, 2026-09-13)**: the first-order quadratic law is recovered
  on all 11 objects — small-level (ℓ ≤ 0.5) log-log slope +2.02 to +2.19
  (v1 measured −1.9 with a 6-order collapse). The linearized prediction
  tracks the empirical second moment within a factor 2 up to a median level
  ℓ = 1.0 (`radius_2x`: 11/11 objects crossed, per-object 0.5–1.5) and
  within a factor 10 up to a median ℓ = 1.5 (`radius_10x`: 11/11,
  per-object 1.5–2.0). Beyond those levels the full nonlinear forward
  (backface flips, exponential gains) diverges from the nominal-Jacobian
  prediction — that is the honest linearization validity radius.

## 15. Certificate concentration（P-CONC, `experiments/certificate_concentration.py`）

- **Dataset**: the same 11-object cohort; full masked pixel set per object,
  142 lights, corrected convention.
- **Perturbation**: R = 10 Gaussian noise realizations of the calibration
  source (noise model a + b·I, fitted level; seed 20260916 + obj_idx);
  per realization a full photometric-stereo refit (`calibrated_ps_fullres`)
  rebuilds the allocation problem.
- **Endpoint**: certified dynamic range (J_none − J_all)/|J_none| at the
  Frank–Wolfe lower bound (budget k = 14, κ = 10, 30 FW iterations), per
  realization; plus the Lipschitz transfer bound
  |J_A(A) − J_A(B)| ≤ ‖A−B‖·‖A⁻¹‖·‖B⁻¹‖·P verified per pair.
- **Metric**: `range_rel_spread` = IQR/median over realizations — random
  dispersion **within** the noisy-calibration family; and (v2)
  `clean_vs_mean_ratio` = range_clean/range_mean per object + summary —
  the **systematic** offset between the noiseless calibration source and
  noisy-realization sources, which the IQR does not capture. The two must
  never be conflated: rel_spread is only meaningful within one calibration
  family.
- **Bootstrap/seed**: none; deterministic given the seeds; no sign-based
  gate.
- **Output**: `results/certification/certificate_concentration.json`.
- **Outcome (v2, 2026-09-13)**: `rel_spread_median` = 1.08% (within-family
  IQR dispersion). `clean_vs_mean_ratio`: median 1.42 (min 1.17, max 3.08) —
  the systematic offset between the noiseless and noisy calibration sources
  is +16.8% to +208.2% (median +42.2%) across the 11 objects, an order of
  magnitude larger than the within-family IQR dispersion: the two quantities
  answer different questions and are reported separately. The v2 fields are
  computed post-hoc from the committed v1 `range_clean`/`range_mean` (no
  metric recomputed; `revision` block in the artifact documents this).


## 16. α-approximate submodularity bound(P-ALPHA-BOUND, `experiments/alpha_bound.py`)

> **Status: DONE (2026-09-13)** — the assessment report's H5 open item
> ("也把 α-近似次模界写进论文,而不是把次模当负定理复述") is delivered:
> the submodularity framing under N-1 is superseded by the α-approximate
> statement below; this section replaces the "do not write submodularity
> into the write-up" TODO. Methods stated in `docs/methods.md` §9 (L9–L12),
> bound in `results/submodularity/alpha_bound.json`, gated by
> `tests/test_alpha_bound.py`.

- **Parameterization**: A-opt selection gain G(S) = F(∅) − F(S) with
  F(S) = tr M(S)^{-1}, M(S) = ΔF(t_S): t_k = κ on k ∈ S, else 1
  (refined-set semantics, identical to P-CERT / P-SUBMOD). W_k =
  u_k[K_k(1) − K_k(κ)]u_k^T ⪰ 0 by Loewner monotonicity of
  K_k(t) = (M0_k + tΛ0_k)^{-1} in t.
- **Bound**: γ ≥ 1/(1+α), α = max_x λmax(ΔF(1)^{-1} W_x); α is a function
  of the nominal design only (A, B, Λ0, κ) — computable a priori, no ground
  truth, no measurements, no exhaustive search. Derivation: exact additive
  decomposition + Woodbury marginal gain + two-sided eigenvalue sandwich +
  Loewner monotonicity (methods.md §9 L9–L12).
- **Toy verification (CI-safe, no raw data)**: 20 P-SUBMOD instances
  (10 random / 10 adversarial, L=5, P=40, κ=10), exhaustive triples:
  γ_measured ≥ 1/(1+α) on all 20 (random α ∈ [0.059, 0.119], adversarial
  α ∈ [0.025, 0.090], measured γ ∈ [0.9999, 1.0000]); two-sided sandwich on
  3200 (S,x) pairs with zero violations (min val/lb = 1.0021, min ub/val =
  1.0368).
- **Real objects**: same 11-object cohort, corrected convention, level ∈
  {0.1, 0.5}; α ranges 0.188–0.515 at level 0.5 (γ lower bound 0.66–0.84),
  worst object `obj_10_pumpkin3` (0.635 @ level 0.1). Overall phrasing:
  "A-optimal light-refinement selection is ≥ 0.635-supermodular on every
  held-out object at the probed levels."
- **Output**: `results/submodularity/alpha_bound.json`.
- **Honest framing**: the bound is ~1.6× looser than the *measured*
  γ_min = 0.99989 (N-1/P-SUBMOD). Its value is a-priori computability (no
  search), the explicit α form, and the two α→0 limits (Λ0 → ∞ or → 0
  ⇒ exact submodularity; α peaks at intermediate precision). This is the
  Chamon & Ribeiro (NeurIPS 2017) approximate-supermodularity framework
  instantiated with calibration precision as the design variable; the repo
  does **not** claim to be first to give an approximate-submodularity bound.
- **Outcome**: cf. `docs/claims.md` M9.


## 17. Goal-oriented calibration value(P-GOAL-ORIENTED, `experiments/goal_orientation.py`)

> **Status: DONE (2026-09-13)** — T10/N4 of the CalibrationValue strategy:
> the finite-nonlinearity route's recommended replacement. `J_H(t) =
> tr(H ΔF(t)^{-1} H^T)` via the low-rank push-through route
> (`woodbury_quad_risk`, math-gated in `tests/test_goal_oriented.py`:
> H=I parity < 1e-10, dense parity < 1e-10, gradient FD ≤ 1e-6).
> Methods: `docs/methods.md` §10.

- **Tasks** (on the per-pixel log-albedo-direction parameterization, P
  pixels): `all` (H = I, A-opt reference), `mean` (H = 1/√P·1ᵀ,
  mean-albedo task), `contrast` (bright-quartile mean minus dim-quartile
  mean, unit norm; quartiles by per-pixel mean intensity). κ = 10,
  levels {0.1, 0.5, 2.0, 8.0}, the same 11-object cohort / corrected
  convention / P-CERT scene assembly as P-ALPHA-BOUND.
- **Per-light block value**: gain_k(H) = J_H(t=1) − J_H(t_k=κ, rest 1)
  over the 48 Fisher-active lights; rankings compared via Spearman +
  top-3 overlap.
- **Findings (descriptive, no sign gate)**: (i) value is task-dependent —
  V_all @ 0.5 median 0.628667 (= the P-CERT 62.87% headline, route
  cross-check) vs V_mean 0.89759; at level 0.1: 0.078212 vs 0.879347;
  (ii) rankings are task-dependent — Spearman(mean, contrast) median
  0.936387 (min 0.585541), top-3 disjoint in 11/44 object×level cells.
- **Mechanism (v1.1)**: the ρ-weighted mean functional is *exactly*
  gauge-aligned with the light-intensity nuisance (per light
  `B_phi[:,0] = ŝ·ρ` ⟹ `A_k a = B_k c̄`, `c̄_k = e_1/‖ρ‖`; alignment
  residual at machine precision). For exactly gauge-aligned functionals
  the two-term law `J_a(t) ≈ 1/(c̄ᵀΛ(t)c̄) + 1/‖Aa‖²` gives
  `V_a = (1−1/κ)/(1+qr)`; agreement median |ΔV| 0.000821, max 0.03854
  (level 0.1; 0.008095 at level 0.5); `V_rho_mean` median rises
  0.878324 → 0.899972 along the level grid. The uniform-mean task is not
  exactly aligned (texture residual 0.313745–34.581472) and inherits the
  mechanism qualitatively (`gauge_mechanism_rho_mean` field).
- **Output**: `results/goal_oriented/goal_orientation.json`.
- **Honest scope**: H acts on the per-pixel scalar parameterization; the
  photometric-stereo normals-vs-albedo pair needs the joint 4P (log ρ, n)
  parameterization — flagged as an extension in methods.md §10, not
  claimed here.
- **Outcome**: cf. `docs/claims.md` M11.


## 18. Active-set ablation（P-ACTIVE-SET-ABLATION, `experiments/active_set_ablation.py`）

> **Status: DONE (2026-09-14)** — E4/M1: the "informed beats random" advantage
> formalized as a six-policy ablation on the certified J_A functional, plus
> the realized-value arm's honest finding (the E(k) hump).

- **Six policies, one table** (11 objects × budgets {5,10,14,28,48}, level
  0.5, κ=10, the P-CERT state; existing policies read from
  `certified_gaps.json`, the other three computed on the rebuilt state):
  random_universe (any of 142; inactive lights are exact no-ops) /
  random_active48 / classical A-opt greedy / mode_tail (mode-aware prefix) /
  continuous (FW) / rounded (top-k of fw_t).
- **Decomposition (J_A units, 55 rows)**: active-set dilution
  +0.011..+0.298 (median 0.063, 55/55 ≥ 0); mode-ordering |gap| ≤ 0.004
  (median 5e-5); rounding loss ≤ 0.0005; certified gap ≤ 0.0008 — **on
  the J_A functional the entire advantage over universe-random is the
  active-set effect**; ordering within the active set, rounding, and the
  certified gap are orders smaller. **预算轴口径（P-ABLATION-FEASIBLE）**:
  引用的 ~1370× 用含 k=48 的全网格,而 k=48 = |active| 处 ordering 子项
  定义性零(11/11 行)会稀释比值——剔除后 ~1001×;两者都是数量级差,
  结论不变（`active_set_ablation_feasible.json` 登记两种口径与逐预算
  退化标记）。**泛函限定（可行性修正同步加入）**:在
  realized 法向角误差端点上,排序承载 active-set 内增益的 40–45%
  （可行区间 dAUC 比 0.27–0.51,B9）——两个泛函不共享同一比值,本条
  声明只对 J_A 成立。
- **Realized-value arm (the reframed M1-3)**: the weak-mode endpoint is
  NON-MONOTONE in refinement coverage — the E(k) hump along the a_opt
  ordering (legacy frame, frozen seed spec, **bit-exact frozen anchor**:
  obj_10 k=14 → 1.4648, k=48 → 0.1889; obj_03: 0.59 → 1.21 → 0.20).
  Mechanism: partial refinement removes mostly gauge-parallel error
  (unaligned mae improves, 0.12→0.06 / 0.18→0.14) while redistributing
  into the gauge-aligned weak modes (dual energy worsens); the stable
  100-seed oracle ranking (split-half 0.82/0.88) ANTI-correlates with
  single-light J_A gains (Spearman −0.36 / −0.82; top-5 overlap 2/5, 1/5).
  J_A gains predict total information, not weak-mode energy — the
  decision-level validity envelope.
- **Convention note**: the frozen AUC benchmark (allocation_summary /
  random48_summary) is `legacy`; this ablation's J_A table and the
  corrected-frame readouts are `corrected`. The legacy k-curve uses the
  frozen seed spec and reproduces the frozen per-run numbers bit-exactly,
  anchoring both layers; the AUC reference numbers are copied with source
  labels (no recomputation).
- **Output**: `results/openillumination/active_set_ablation.json`
  (decomposition aggregates, 55 policy rows, 2 oracle rows with mechanism
  anchors and both k-curves, AUC reference layer).
- **Outcome**: cf. `docs/claims.md` C8.


## 19. Model-validity map（P-VALIDITY-MAP, `experiments/validity_map.py`）

> **Status: DONE (2026-09-14)** — E2/M2: "Fisher 线性化在哪个 regime 准",
> 不证明它永远准。**只读 join** 两个已提交产物(corrected arm-D 的逐 cell
> pred_deg/emp_deg x linearization-radius 的曲率指标),无实验重算。

- **四端点(逐 cell,66 = 11 物体 x 6 level)**:Spearman 排序相关;
  **Kendall 符号一致率**(预测 vs 经验模式排序的逐对符号一致,
  M2-1 的新端点,直接从已提交行补算);幅值比中位数 emp/pred;
  跨模式 log-ratio IQR。
- **二维图**(`docs/img/validity_map.png`):横轴 level,纵轴
  excess-over-first-order-scaling q(ℓ)(对数轴,红线 = 一阶边界 q=1),
  四面板着色。
- **核心发现(B8)**:排序/符号效度对曲率稳健——四个曲率箱
  (q<0.75 / 0.75-1.0 / 1.0-1.5 / >=1.5)的 median Spearman 全部 >= 0.8、
  Kendall 全部 >= 0.7;幅值比随曲率单调恶化(median-of-medians
  7.7 -> 10.4 -> 72.0 -> 106.5)。**线性化失效打幅值,不打方向**:
  排序类问题在全网格有效,幅值类问题只在次线性 regime 有效。
- **Output**: `results/magnitude/validity_map.json`(cells + by-excess-bin
  + by-level + manifest 含来源产物 sha256)+ `docs/img/validity_map.png`。
- **Outcome**: cf. `docs/claims.md` B8.


## 20. Decision quality: predicted vs realized（P-DECISION-QUALITY, `experiments/decision_quality.py`）

> **Status: DONE (2026-09-14)** — E3/M3, the impact pillar. Range fixed by
> the acceptance report: NO real recalibration (OpenIllumination is a
> calibration dataset); a controlled corruption-injection
> predicted-vs-realized closed loop on the **single** residual pipeline.

- **Arm（M3-1）**: `arm_metrics` — the same corruption + fixed-n̂ 白化 GLS
  as `arm_energy`（dual 读出逐值一致，测试钉死）, extended with a
  one-step alternating normal refit（`calibrated_ps` 自己的 n-更新:
  掩码内无权重 LSQ + 归一化,einsum 批量向量化;与逐像素循环参照差
  ≤1e-12）→ **法线角误差（度）**;plus gauge 对齐 albedo MSE 与未对齐
  MAE。无第二条残差管线。
- **网格**: 11 物体 × 4 level {0.2,0.5,1.0,2.0} × 2 regime {10,100} ×
  5 预算 {14,28,57,85,114} × 7 selection units（mode_aware, a_opt +
  5 random perms,冻结 rng 规范）× 10 seeds/level;corrected 口径;
  predicted 侧 = 同分配在同一状态上的 J_A。逐对象多进程（workers=2;
  a_opt 排序 ~230s/次,冻结贪心语义不可改）。
- **核心结果（B9,v1.1 双基线）**: 法线角误差端点——dAUC vs
  **universe-random**:a_opt −3.90 [−6.52, −2.45] @10、−5.03
  [−7.92, −3.51] @100;mode_aware −3.68/−4.84;e_opt −3.88/−5.14;
  d_opt −3.61/−4.75（全部 CI 不含 0）。dAUC vs **active48-restricted
  random**（诚实检验,冻结全网格）:**−0.17 ~ −0.55**,8 个 policy×regime
  全部 CI 不含 0——但该全网格被预算饱和**稀释 2.0–2.8 倍**:k=57/85/114
  **超过 active 集 48 盏**,双方前缀选同一批灯、端点逐位相同（逐预算
  88/88 cell）,3/5 预算区间恒零。**可行区间（k ≤ 48,fraction
  [0.1,0.2]）修正值:−0.38 ~ −1.40,全部 CI 不含 0;与可行区间
  universe-random 优势之比 0.27–0.51(≈2–3.7 倍)——不是"一个数量级"**。
  数度级头条主要是 active-set 效应（C8）,active-set 内优势真实且约为
  头条的 1/2.5。诊断与复算:`results/openillumination/
  decision_quality_feasible.json`（P-DQ-FEASIBLE:退化标记 + 可行/全
  网格双列）。Spearman(predicted J_A, realized)
  **88/88 为正**（median +0.917, min +0.681）;informed 族内
  （n=4）池化符号一致 **678/987 = 0.687** [0.657, 0.716],中位
  Spearman +0.800。gauge 对齐端点仍呈再分配（MSE −0.575 / dual
  −0.525）。**验证**: 2200 未变单元 rows / 8800 值与 v1 产物（git 历史读取）
  **逐位一致**（免费全量对拍:同种子同代码路径;`full_comparison` 字段
  登记 8800/8800）;新单元（e/d-opt, A48 randoms）1760 行全有限。
- **诚实披露（v1.1 已内嵌对照）**: informed 族内判别（n=4）池化符号
  一致率 **678/987 = 0.687** [CI 0.657, 0.716]——排序力非纯构造主导,
  但远弱于含 random 的 88/88 头条（random 单元在两轴上天然处于差角）。
  active-set-restricted 随机对照已在本网格内（3U+2A48）,双基线 dAUC
  见上——v1 报告中的 "网格无 A48 臂" 警告已由 v1.1 关闭。
- **量级说明**: dual 端点以原始能量单位报告(物体间量级 1e2–1e5),
  与 mse_aligned(1e-3)差约 7 个量级——只在同一端点内比较,勿跨端点
  比较效应大小;运行代价 v1.0 4.2 h、v1.1 12.4 h(均 workers=2,本机;
  逐物体独立播种,**同机** determinism 与并行度解耦——跨机不逐位,
  见下方复现段)。
- **Output**: `results/openillumination/decision_quality.json`（v1.1:
  3960 rows + 88 AUC cells + 48 bootstrap keys + within-cell Spearman）+
  `docs/img/decision_quality.png`（predicted-vs-realized 散点 + dAUC
  forest）。
- **Outcome**: cf. `docs/claims.md` B9.
- **跨环境复现（P2-1,云跑对拍）**: `results/openillumination/provenance/
  dq_cross_env_repro.json` + 归档云产物
  `decision_quality_cloud42bc299.json`（`experiments/verify_dq_cross_env.py`
  可复跑）。云（32 核/11 workers/Linux/numpy 2.4.6,全程 36.4 min）vs
  本地（2 workers/AMD/numpy 2.4.1,12.4 h）:**统计同一**——792 个
  cell-unit 中 791 个信息侧一致（≤1e-9 rel）,唯一分歧 = obj_19_cylinder
  @ level 1.0/regime 10 的 **e_opt 贪心近平局翻转**（两个候选灯的 J_A
  差 ~1e-6,贪心任选其一均有效）;全部 16 个 ang bootstrap CI 同号且
  显著,informed 统计 **678/987 逐位相同**。informed-only 的下限在此
  披露:per-cell Spearman **min = −1.000**(n=4 族内至少一个 cell 排序
  完全反向——这正是排序力主张落在 0.687 而非 88/88 的直接证据)。
  **不声称跨机逐位一致**——
  末位 ULP 的 BLAS 差异会翻转近平局,这是期望行为;同机 worker-不变性
  由 resume-vs-clean 位级测试单独证明（`dq_resume_vs_clean.json`,见下）。
- **平坦性实证（唯一分歧的科学读法）**: 那次 e_opt 翻转发生在两个
  候选灯 **J_A 差 ~1e-6** 上——比认证间隙中位 **7.8e-5(C2)** 小两个
  数量级。即:连 1e-6 的扰动都会改变"选哪盏灯",说明该选择落在证书
  保证的次优性之内,换一盏灯统计上无差别。这把"跨环境不逐位"从瑕疵
  变为**对"landscape 在最优附近平坦"的独立实证**,并与"策略间差异
  在证书精度内不可分辨"(B4/C8)闭合——一次回答"为何不逐位一致"与
  "证书有什么用"两个问题。
- **同机 worker/崩溃不变性（P1-d 入库证据）**: `results/openillumination/
  provenance/dq_resume_vs_clean.json`
  (`experiments/verify_dq_resume_equivalence.py` 可复跑)——2 物体缩减
  网格:run A 在第一个 checkpoint 落盘后 SIGKILL;run B 同目录重启
  (obj_03 从 checkpoint 恢复、obj_19 新算);run C 干净跑。**B vs C
  位级一致**(rows 32/32 + bootstrap/spearman/informed/auc 全同),且
  **workers=4 vs workers=1 也位级一致**(同网格、逐对象播种——worker
  维度从论证升级为实证);两个产物的文件哈希仅差 manifest 元数据
  (elapsed_s)——这正是 verdict bit-identical 的语义。
  此前该证据仅存于 42bc299 的 commit message(验收 P1-d:最强声明
  不得只靠散文支撑)。
- **v1→v1.1 重用等价(审计记录)**: `results/openillumination/provenance/dq_v1_reuse_equivalence.json`
  (`experiments/verify_dq_v1_reuse_equivalence.py`)——**代码路径同一性
  (未变单元的 selection/J_A/arm 代码 v1→v1.1 逐字节相同)+ obj_03 上的
  18 点抽样验证**(pred_J_A 12 + 实测端点 6,含 v1 dual 聚合顺序),
  18/18 位级一致。**非全量验证**——全量等价由"逻辑论证(种子携带
  object+level+seed 索引)+ 本抽样 + v1.1 全量重跑对未变单元的隐式复现"
  三层构成;证据 JSON 的 v1_artifact_sha256 锚定被验证的 git blob。


## 21. Σ_φ parameter-family sensitivity（P-SIGMA-FAMILY, `experiments/corruption_family_sensitivity.py` + `linearization_radius_family.py` + `decision_quality_family.py`）

**Question.** Which frozen conclusions are properties of the physics and
which are artifacts of the single-line corruption parameterization
(`CorruptionGenerator` joint level: σ_logI = σ_dir_deg = level, forcing a
logI/dir variance ratio of ~3283 at the 0.5 operating point)?

**Family.** Three axes, preregistered in
`configs/corruption_family_sensitivity.yaml` before the run: independent
channel ratio (σ_dir ∈ {0.1…25}° at σ_logI = 0.5 and σ_logI ∈ {0.05…1.0}
at σ_dir = 0.5°/1°), per-light lognormal heterogeneity (het ∈ {0.25, 0.5,
1.0}, seed 20260915), channel coupling (ρ_c ∈ {−0.5, +0.5}; corrected
PSD form: off-diagonals ρ·σI·σR/√2, Corr(logI, angle amplitude) = ρ
exactly, PSD for all |ρ| ≤ 1). 45 unique grid points × 11 objects = 495
rows; tier-2 worst-case combos and a σ_dir × het 4×4 complete tier-3.

**Anchors (S0, commit ec2058f).** The degenerate family reproduces the
frozen machinery bit-exactly: all 264 `channel_decomposition.json` rows
(J_A_1, J_A_kappa, D — float ==) through the family path; the anchor /
intensity-only / direction-only points of the new artifact equal the
frozen joint/intensity/direction @ level 0.5 rows exactly; the anchor's
two-term-law quantities equal `goal_orientation.json`'s frozen gauge
mechanism; the C-arm joint control equals `linearization_radius.json`
per (object, level) — every anchor is runtime- or test-asserted.

**Findings (preregistered outcome rules, all numbers reported).**
- **S2-1 / claim C7 (D, intensity dominance): ROBUST.** Direction share
  of the budget value at the operating point: 0.07% (median). r*
  (σ_dir where the share first reaches 2%, log-interpolated): > 25° on
  **all 11 objects** — censored high; the share curve peaks at ~0.2%
  near σ_dir = 1° and declines beyond. Direction-only D ≤ 1.68% over
  the whole family sweep (the frozen max, at the same 0.2° hump peak).
  Reverse control (fixed σ_dir = 1°, σ_logI swept): intensity share ≥
  0.863 everywhere. The 3283 ratio was never the mechanism — the
  direction channel is intrinsically weakly coupled to the budget value.
- **S2-2 / two-term law (A): PARAMETERIZATION-SPECIFIC — and the
  failure region is exactly the direction-only region.** Median |dV| =
  0.0076 across all 495 rows (within the preregistered 0.01 robustness
  bar), but max |dV| = 0.464. The tag decomposition locates the failure:
  wherever the **intensity channel carries the nuisance** the law stays
  at its frozen accuracy (anchor 0.0081, intensity_only 0.0081, dir_sweep
  0.020, logI_sweep 0.040, reverse_control 0.048 — the last is the worst
  intensity-carrying cell, within the frozen envelope's 0.039 × O(1)),
  while the **direction-only** shapes blow past it (dir_only max 0.464,
  median 0.040; tier-2 combos mixing heavy direction + heterogeneity max
  0.319). Read together with S2-1: **D proves the direction channel is
  not the operating region, and A's law fails precisely there** — the
  two-term law is valid in the region that carries the budget value and
  degrades only in the degenerate direction-dominated limit that D
  already excludes. The identity part (gauge alignment A·a = B·c̄) is
  Σ-free: residual < 1e-9 per object.
- **S2-3 / ceiling (B): guard clean.** Max violation −0.0028 over all
  495 rows (theorem M10/M10′ — zero Σ-dependence; stated in methods §7
  as holding across the family, run as a guard, not billed as evidence).
- **S2-4 C / linearization radius (B6): CHANNEL-DEPENDENT.** Joint
  control: radius_2x/10x medians 1.0/1.5 (bit-exact frozen rerun);
  intensity_only: identical (1.0/1.5); direction_only: **never crosses**
  on [0.05, 8] (0/11 objects — bounded rotations stay within first-order
  scaling); joint_het (het = 1.0): radius shrinks to 0.2/0.35 (the
  heavy-tailed per-light variance brings second-order effects in 5×
  earlier). The amplitude-side envelope is a property of the corruption
  shape, not of the geometry alone. **The het=1.0 shrinkage is this
  round's genuinely new physical finding**: per-light calibration
  heterogeneity (a 5× spread in effective corruption scale across
  lights) makes the nominal linearization break ~5× earlier than any
  uniform-scale corruption of the same total variance — uniform-σ
  level grids systematically overestimate the linearization radius
  whenever the true per-light scales are heterogeneous.
- **S2-4 E / informed-only ordering (B9): see the E arm artifact**
  (`results/openillumination/decision_quality_family.json`, §21b) —
  reduced design (level 0.5, regime 10, budgets {14, 28} which carry
  100% of the dp ≠ 0 pairs, 4 informed policies, 10 seeds, 3 arms).

**21b. E arm — informed-only ordering across the family (S2-4 E).**
Reduced design, preregistered in `configs/decision_quality_family.yaml`
(commit 5b31b73) BEFORE the run: level 0.5, regime 10, budgets {14, 28}
— chosen on POWER, not outcome: in the frozen artifact the predicted
differences dp ≠ 0 live exclusively at k = 14 (62/66 pairs) and k = 28
(63/66); at k ≥ 57 all four informed orderings coincide — the two
budgets carry 100% of the distinguishable pairs. 4 informed policies,
10 seeds (frozen seed spec, level-index 1), 3 arms: anchor (0.5, 0.5°)
control, dir_heavy (0.5, 5°), het (0.5, 0.5°, het = 1.0). Statistics =
the frozen E statistic (pooled informed pairwise sign agreement on
ang_mean_deg + binomial CI), computed by the same code path as the
frozen pipeline (the anchor arm's 88 rows are BIT-IDENTICAL to the
frozen decision_quality.json subset — runtime-asserted).

Findings (`results/openillumination/decision_quality_family.json`):

| arm | agree/total | rate [95% CI] | per-cell informed Spearman (median) |
|---|---|---|---|
| anchor (control) | 88/125 | 0.704 [0.616, 0.782] | +0.789 |
| dir_heavy (σ_dir = 5°) | 104/117 | 0.889 [0.817, 0.939] | +1.000 |
| het (het = 1.0) | 42/119 | 0.353 [0.268, 0.446] | −0.778 |

Outcome (preregistered rule): **parameterization-specific** — the het arm
(0.353 < 0.60) fails the robustness bar. A follow-up 2x2 arm (het_mismatched, prereg
93f45a1) was OOM-blocked on the memory-starved host and superseded by a
zero-cost cross diagnosis on the committed rows
(`experiments/family_e_diag.py` ->
`results/openillumination/corruption_family_e_diag.json`,
P-SIGMA-FAMILY-E-DIAG):

| quantity | value | meaning |
|---|---|---|
| S_pred | 0.949 | anchor-pred vs het-pred policy rankings (the predictor barely moves) |
| S_real | −0.325 | anchor-realized vs het-realized rankings (the truth re-orders) |
| fit_anc / fit_het | 0.789 / −0.778 | within-arm pred-vs-realized validity |
| cross_ah / cross_ha | −0.738 / 0.738 | anchor-pred→het-truth; het-pred→anchor-truth |

The symmetric cross failure (fit_het ≈ cross_ah, fit_anc ≈ cross_ha)
excludes the sigma-misspecification explanation: a wrong-σ predictor
would be the thing that moved, but the predictor's ordering changes by
only 0.05 while the realized benefits fully re-order. **The failure is
truth-end decoupling — the criterion's policy ordering is insensitive
to per-light heterogeneity, the realized-benefit ordering is highly
sensitive to it, and the per-light J_A ordering loses decision validity
under heterogeneity regardless of σ specification.** Two readings, both carried by
the artifact: (i) the frozen 0.687 informed-ordering validity is NOT a
pipeline invariant — it moves by ±2σ across corruption *shapes*;
(ii) the direction of the movement is diagnostic. Making the corruption
direction-heavier (variance 100× the anchor in the direction channel)
makes the predicted orderings MORE faithful (0.889, per-cell Spearman
median +1.0) — direction perturbation hits the normal endpoint in a way
J_A ranks correctly; the intensity channel that dominates the budget
value is the one the angular endpoint ranks worst. Per-light
heterogeneity (het = 1.0) breaks the ordering *below chance* (0.353,
median Spearman −0.778, worst cells 0/5): the uniform-σ J_A model
mis-ranks which lights matter when the corruption scale varies 5×
across lights (e^(±1)·0.5). Frozen references: all-cells 678/987 =
0.687 [0.657, 0.716]; this subset 88/125 = 0.704.

**Bug found and fixed by the first run (documented in-repo).** The v1
rank-one coupling `ρ·σI·σR·e eᵀ` modified the diagonal and drove Σ_22 <
0 at the anchor channel ratio — the family constructor itself emitted a
non-PSD covariance, which crashed the woodbury Schur gate at t = κ
(obj_11_pine). Corrected to the PSD form above (commit 909bd2a); no
artifact had been written before the crash; preregistered rules
unchanged. Regression: `tests/test_corruption_family_degeneracy.py::
test_rho_c_psd_and_exact_correlation`.


## 22. Ball-anchor: Σ_φ anchored to a real sphere calibration（P-BALL-ANCHOR, `experiments/ball_anchor.py`）

**Question (the "physical anchor" gap).** Every quantitative conclusion
lives on Σ_φ = diag(level², radians(level)², radians(level)²) with a
synthetic level knob. What is level=0.5 in a real system?

**Procedure.** Standard sphere photometric-stereo calibration on
DiLiGenT ballPNG (96 real images of a known sphere, GT normals):
alternating-LSQ rank-1 factorization — per-pixel albedo LSQ over all
lights ↔ per-light 3-parameter LSQ of the bilinear product — on
unsaturated (327/15791 px dropped), lit pixels (GT-direction lit mask,
the only place GT enters before the comparison; the estimator itself
never sees GT). Convergence 192 iterations.

**Measured anchor.**

| channel | error | value |
|---|---|---|
| direction | RMS over 96 lights | **2.96°** (median 2.96, max 4.47) |
| intensity | std of log e/e_GT (after log-domain scalar rescale) | **0.0159** (1.6%) |

The real procedure's direction variance is **~10.6× its intensity
variance** (radians vs logI) — the OPPOSITE of the joint
parameterization's forced 3283. Real sphere calibrations are
direction-error dominated.

**Direction share at the measured anchor** (frozen family machinery on
the OpenIllumination cohort, same functional as §21): median **36.0%**,
per-object 4.8% (obj_04_dolphin) to 71.4% (obj_11_pine). Nearest frozen
S1 grid point: (σ_logI 0.05, σ_dir 1.0°) at log-distance 0.69 — between
the reverse-control and dir-sweep cells, in the direction-coupled
region.

**Outcome (preregistered): D-flipped.** At the measured operating point
of a standard sphere calibration the direction channel carries a third
of the budget value. Read as a **channel-conditionality criterion**
(positive contribution): the share of the calibration-budget value
carried by each nuisance channel is a function of the procedure's error
profile — measurable per procedure by a sphere calibration, not a
universal constant. C7's intensity dominance is the criterion's value
in the intensity-dominated region; the sphere anchor instantiates the
direction-coupled region. Both regions are now empirically grounded
(synthetic family sweep + real measured anchor).

**Relation to the family grid (why C9 and C10 do not contradict).**
The S1 sweep pinned σ_logI ∈ {0.05 … 1.0} — i.e. 5–100% intensity
error, physically implausible magnitudes — so intensity variance
dominated every swept point and the direction share stayed small there
(C9's "robust"). The measured anchor sits at σ_logI = 0.0159, **3.1×
below the sweep's smallest non-degenerate grid value** (0.05), at
log10-distance 0.686 from the nearest grid point (the reverse-control
cell 0.05/1.0°). C9 sampled the intensity-dominated region and found
invariance there; C10 measured the real procedure's location outside
it. Together: "intensity dominance" was never robust — it was the
sweep's fixed σ_logI holding it up, exactly the single-knob artifact
the family analysis was designed to expose.

**Scope of the anchor (do not over-read).**
- One object (ballPNG), one procedure (alternating-LSQ sphere
  calibration): it instantiates *a* standard procedure, not all real
  calibrations.
- The measured 2.96° RMS direction error is consistent with the
  published observation that DiLiGenT's own light directions carry
  degree-level inaccuracies (the benchmark's authors note GT light
  directions are approximate); the anchor measures the *procedure +
  data* error, not an implementation defect.
- The flip is **object-conditional as well as channel-conditional**:
  direction_share spans 4.8% (obj_04_dolphin) to 71.4% (obj_11_pine) —
  a 15× object-to-object spread that the median 36% summarizes but
  does not replace; per-object values are in the artifact.


## 23. DiLiGenT queue: second-dataset transfer（P-DILIGENT-QUEUE, `experiments/diligent_queue.py`）

**Question (the second-dataset gap).** Do the three core findings
transfer from OpenIllumination (11 objects, 142 lights) to DiLiGenT
(10 objects, 96 lights — an independent published benchmark with its
own capture rig)? Adapter: `diligent_oi_adapter.py` + `n_lights_total`
(96-light draw space = a NEW queue by construction, not bit-comparable
to the frozen OI artifacts).

**Findings** (`results/diligent/diligent_queue.json`, preregistered
outcome rule, 433 s):

**Loader correction (2026-09-16, defect report).** The v1 OI-format
adapter dropped the per-light intensity normalization (divided by 255
only), biasing the DiLiGenT cohort's nominal reconstruction vs GT by
15.6–26.3° (2.56–6.34° when normalized like `diligent.py:43`). The
drift endpoints were structurally blind to it (two-sided same-source
bias cancels in nominal-vs-corrupted differences); only an absolute
check or a cross-loader gate can catch it — both now exist
(`tests/test_diligent_loader_consistency.py`;
`results/diligent/provenance/loader_normalization_fix.json`). All
numbers below are POST-correction; the pre-correction readings
(direction max 51.2%, pot1/pot2 ≈86%) are withdrawn.

1. **Linearization radius: TRANSFERS EXACTLY (hardened).** radius_2x
   median 1.0 (per-object 0.75–1.5), radius_10x median 1.5 with
   **10/10 objects at exactly 1.5** — the same values as
   OpenIllumination, unchanged by the loader correction. The
   amplitude-side envelope is a property of the corruption geometry,
   replicated across two acquisition systems.
2. **Channel decomposition: intensity dominance TRANSFERS.** With the
   corrected loader the DiLiGenT direction channel is tiny (max D
   0.20% vs OI's 1.68%) and the intensity channel reproduces joint to
   <0.05 pp at every level — the same channel split as OI. The
   pre-correction "direction channel is far larger on DiLiGenT"
   (51.2%) was the normalization artifact.
3. **Ball-anchor direction share: object-dependent, and the cohorts
   differ.** At the measured anchor (σ_logI 0.0159, σ_dir 2.96°):
   DiLiGenT median ≈ 0 (cat 22.1%, pot1 29.5%, the rest ≈0; harvest
   and reading carry finite small denominators, annotated degenerate)
   vs the OI cohort's median 36.0% (4.8–71.4%). The OI 36% flip does
   not reproduce on DiLiGenT at the same error profile — the
   criterion's object axis survives, its dataset-channel-split axis is
   retracted.

**Outcome (preregistered): transfer-partial** — radius confirmed,
anchor share not confirmed as a dataset-level flip. Post-correction the
reading is: the criterion's axes are (i) the procedure's error profile
(measured, C10) and (ii) the object / scene (OI 36% median vs DiLiGenT
≈0 at the same profile, with cat/pot1 as the DQ high objects) — but
NOT a dataset-level channel-split difference, which the loader
artifact had falsely suggested. A recommendation is impossible without
measuring the profile AND the object — which is what the criterion
prescribes.


## 24. Baselines: literature-style selection on the same lights（P-BASELINE, `experiments/baseline_comparison.py`）

**Question (the baseline gap).** A literature-style baseline table is expected.
The implementable same-lights baseline is the classic
well-conditioned-configuration idea (Drbohlav & Chantler, ICCV 2005),
operationalized as greedy farthest-point direction sampling (DC05):
seed = the max-z light, then add the light minimizing the max
dot-product with the selected set — pure geometry, no corruption
model, no prediction model.

**Frame.** Identical to the frozen E-arm wherever shared (level 0.5,
regime 10, budgets {14, 28}, 10 seeds, ang_mean_deg endpoint). OI
cohort: 4 informed + DC05 + 3 random, with a HARD BIT-ANCHOR — the
informed rows are bit-identical to the frozen E-arm anchor subset
(88/88, shared orderings + seeds). DiLiGenT: a_opt + DC05 + 3 random.

**v1.1 (acceptance-corrected: active-set control).** The v1 DC05
candidate pool spanned ALL lights — 36–86% of its budget fell on
non-illuminating lights (per-object overlap 2–9 of 14 at k=14), so
v1's "DC05 ≈ universe-random" merely replicated C8. v1.1 adds
dc05_active (same geometry rule, pool restricted to the illuminating
lights), randomA48 controls (permute the active set only), and records
the overlap per object. Reading basis: dc05_active vs randomA48
(equal effective budget).

| cell | dc05_active dev | informed dev (vs randomA48) | reading |
|---|---|---|---|
| OI k=14 | −0.438° | −0.704° | geometry-insufficient |
| OI k=28 | −0.615° | −1.204° | geometry-insufficient |
| DQ k=14 | −0.336° | −0.274° | geometry-informative (both beat random) |
| DQ k=28 | −0.335° | −0.369° | geometry-insufficient |

**Loader correction (2026-09-16).** The v1.1 DiLiGenT numbers were
computed through the adapter's missing per-light intensity normalization
(15.6–26.3° nominal-vs-GT bias). Corrected, the DQ column changes
qualitatively: **both geometry-only and the informed family beat
active-restricted random at both budgets**, and the pre-correction
"informed loses to random" claim is withdrawn. What remains true from
the v1.1 reading: on OI every informed policy beats geometry clearly.

**OpenIllumination: geometry helps but does not replace the model.**
dc05_active (5.235°/4.005°) genuinely improves on randomA48
(5.78–5.99°/4.49–5.25°) — direction spread has real value at equal
effective budget — but every informed policy remains clearly better
(e_opt 4.520°/2.894°). The v1 all-pool DC05's apparent chance-level
performance was the visibility confound (C8), now self-documenting in
the artifact's overlap fields.

**DiLiGenT (corrected): both approaches carry value; the model's
increment is marginal.** With the fixed loader, dc05_active
(−0.336/−0.335) and the informed family (−0.274/−0.369) both beat
active-restricted random at both budgets; at k=14 geometry is nominally
better (within noise) and at k=28 the informed family is marginally
better (6.88° vs 6.68° medians). The honest one-line summary across
cohorts: **geometry-only selection carries real value on BOTH
acquisition systems** (a positive, transferable result), while the
calibrated model's ADDITIONAL margin over geometry is large on OI and
marginal on DiLiGenT. The pre-correction "the model's ordering does
not transfer" reading was a loader artifact and is withdrawn.

**Comparability (Gardi 2022 / ReLeaPS 2023).** They optimize light
POSITIONS / next-light selection sequences under their own objectives;
we allocate per-light calibration PRECISION under a Fisher-risk
objective. Objectives differ — numbers are not directly comparable;
DC05 is the one implementable same-lights geometry baseline. The
statement is registered in the artifact's `baseline_definition`.


## 25. Anchor-share mechanism: why the direction channel differs by cohort（P-ANCHOR-MECHANISM, `experiments/anchor_mechanism.py`）

**Question (the one C11 left open).** At the SAME measured anchor
(σ_logI 0.0159, σ_dir 2.96°) the direction share is OI median 36.0%
(4.8–71.4%) but DiLiGenT median ≈0 (cat 22.1%, pot1 29.5% the only high
objects). What scene property explains the gap?

**Method.** Six scene statistics computed from the NOMINAL scene (none
from the D functional — no circularity), correlated against the
per-object share from the two committed artifacts (21 objects):
Finf level and spread, light-direction spread and isotropy, and two
mechanistic candidates — `dir_int_weak_ratio` (direction-nuisance vs
intensity-nuisance whitened energy in the bottom-5 weak-mode subspace)
and `gauge_cos` (coupling between each light's whitened intensity
column and its direction-nuisance subspace).

**Result: an OI-internal mechanism; the cross-cohort claim is NOT
supported** (revised after the acceptance review, section 3.3).

| statistic | pooled (all 21) | **pooled (valid, n=14)** | OI (valid) | DiLiGenT (valid) |
|---|---|---|---|---|
| **dir_int_weak_ratio** | −0.899 | **−0.688** | **−0.927 (n=11)** | not computable (3 objects) |
| finf_spread | −0.868 | −0.596 | −0.618 | — |
| light_spread_deg | +0.519 | — | — | — |
| light_isotropy | +0.673 | — | — | — |
| gauge_cos | +0.538 | — | — | — |

Seven of the ten DiLiGenT shares are 0/0-shaped (|share| < 0.01 — the
artifact flags them `degenerate_share`, the same semantics as
`diligent_queue.json`'s `degenerate_objects`). Counting those as
ordinary ranks lifts the pooled correlation from **−0.688 to −0.899**,
i.e. across the preregistered 0.7 threshold; the v1 rule is therefore
retained as reference only. On the valid samples the mechanism holds
**within OpenIllumination** (n=11, ρ=−0.927, every share at
interpretable magnitude) and the DiLiGenT side cannot be tested for
want of valid objects.

The statistic itself (corrected description, acceptance section 3.2) is
the mass the bottom-5 weak-mode subspace places on the direction
parameter axes vs the intensity axis — a weak-subspace loading measure,
not a per-light nuisance-energy ratio.


---

**Reproduction contract**: every experiment script writes only statistical values and
manifests to `results/*/`; N1–N9 are independently recomputed from the CSVs by
`tests/test_reproduction.py` (the frozen benchmark gates) and N10–N12 by
`tests/test_benchmark_evidence.py`.