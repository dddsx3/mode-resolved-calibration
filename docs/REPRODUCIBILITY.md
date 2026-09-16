# Reproducibility guide

Every quantitative statement in this repository is reproducible from
committed artifacts and pinned commands. This page is the index: for each
headline result it gives the command, the evidence file, and the test that
re-derives the value. The binding table (claim → evidence file → field) is
`docs/claims.md`.

## 1. Integrity model

- `checksums.sha256` pins the SHA-256 of **every committed file** (LF basis;
  `.gitattributes` enforces LF). CI runs `sha256sum -c checksums.sha256`, so
  any modification of any committed artifact is detected.
- Frozen evidence files (the benchmark results under `results/`) are treated
  as immutable data: tests re-derive their headline values from the files,
  and the pipelines that produced them can be re-run to compare byte-for-byte.
- Raw datasets are external (see `docs/DATA.md`); the repository carries only
  manifests and derived evidence.

### 1.1 Provenance note: git SHAs recorded in result manifests

Twenty-five distinct `git_sha` values are recorded across the committed result
manifests. They are not all the current HEAD: results are produced from an
earlier working-tree state, and two of the recorded commits were replaced
during the 2026-09-09 repository reorganization. The complete ledger:

| git_sha (12-char) | Results | Reachable from HEAD | Note |
|---|---|---|---|
| `bd67897193c9` | the seven frozen science-closed manifests (`diligent/ci05`, `diligent_ablation/ci05abl`, `gauge_spectrum/ci02`, `monte_carlo/ci03`, `nonlinear/ci03nl`, `openillumination/ci04`, `synthetic/ci01`) | no | commit replaced in the 2026-09-09 reorganization; artifacts pinned by `checksums.sha256` |
| `fa5440410dcb` | `certification/certificate_concentration.json`, `certification/lowrank_fullres.json` | no | commit replaced in the 2026-09-09 reorganization; both carry an in-file provenance note (`provenance_unknown` + `sha_unreachable_reason`) |
| `da28fef4d11e` | `certification/certified_gaps.json` | yes (ancestor of HEAD) | earlier certified-gap run |
| `b070f7e831f0` | `openillumination/validation_summary.json` | yes (ancestor of HEAD) | frozen benchmark record |
| `f54d6308b3c8` | `mode_tail/allocation_mode_tail.json` | yes (ancestor of HEAD) | corrected three-arm rerun |
| `785c7485f37e` | `magnitude/linearization_radius.json` | yes (ancestor of HEAD) | v2 metric-domain rerun |
| `21f1ef9e3919` | `submodularity/alpha_bound.json` | yes (ancestor of HEAD) | P-ALPHA-BOUND v3 run: proof-limits adversarial-search extension (2026-09-13) |
| `f289a2d2a5b1` | `goal_oriented/goal_orientation.json` | yes (ancestor of HEAD) | P-GOAL-ORIENTED v1.1 run: rho_mean gauge-mechanism extension (2026-09-13) |
| `7f7094b20c7a` | `openillumination/active_set_ablation.json` | yes (ancestor of HEAD) | P-ACTIVE-SET-ABLATION v1 run (2026-09-14) |
| `d5765bf1ef8c` | `magnitude/validity_map.json` | yes (ancestor of HEAD) | P-VALIDITY-MAP v1.0.1 run: resolvable source-path labels (2026-09-14) |
| `1b6cf9afb80c` | `openillumination/provenance/dq_v1_reuse_equivalence.json` | yes (ancestor of HEAD) | P-REUSE-EQUIV v2: 18-point spot check + FULL 2200-row/8800-value comparison vs the v1 git blob, all bit-exact (2026-09-14) |
| `1b6cf9afb80c` | `openillumination/provenance/dq_cross_env_repro.json` | yes (ancestor of HEAD) | P-CROSS-ENV v1.1: headline reorder, 1/792 diverged, statistically identical (2026-09-14) |
| `2a365b6a3846` | `openillumination/provenance/dq_resume_vs_clean.json` | yes (ancestor of HEAD) | P-RESUME-EQUIV v2: crash/resume AND workers 1v4, both bit-identical (2026-09-14) |
| `42bc299fb744` | `openillumination/decision_quality.json` | yes (ancestor of HEAD) | P-DECISION-QUALITY v1.1 run: 4 informed + 3U/2A48 dual baselines (2026-09-14) |
| `ddfccf1dc7e5` | `magnitude/calibration_value_ceiling.json` | yes (ancestor of HEAD) | P-CEILING v1 run (2026-09-13) |
| `e67a8bf51c8e` | `certification/certified_gaps_levels.json` | yes (ancestor of HEAD) | P-CERT-LEVELS v1 run (2026-09-13) |
| `f878dc41cca1` | `openillumination/channel_decomposition.json` | yes (ancestor of HEAD) | P-CHANNEL-DECOMP v1 run (2026-09-13) |
| `909bd2a5eee1` | `openillumination/corruption_family_sensitivity.json` | yes (ancestor of HEAD) | P-SIGMA-FAMILY predictor-side run (S1 grid + S2-1/2/3, 2026-09-15): preregistration commit 6cf3485; launch state 909bd2a = prereg + the PSD-coupling fix (the v1 rank-one bug crashed the first run before any artifact was written) |
| `70def9f393ad` | `magnitude/linearization_radius_family.json` | yes (ancestor of HEAD) | P-SIGMA-FAMILY C-arm run (S2-4 C, 2026-09-15): joint control bit-identical to the frozen P-RADIUS v2 rows |
| `5b31b734138b` | `openillumination/decision_quality_family.json` | yes (ancestor of HEAD) | P-SIGMA-FAMILY E-arm run (S2-4 E, 2026-09-15): anchor control BIT-IDENTICAL to the frozen decision_quality.json subset (88/125 = 0.704 reproduced exactly); reduced power-maximal design (budgets 14/28 carry 100% of dp!=0 pairs) |
| `20d5fbe585e1` | `openillumination/corruption_family_e_diag.json` | yes (ancestor of HEAD) | P-SIGMA-FAMILY-E-DIAG zero-cost cross diagnosis on the committed E-arm rows (2026-09-15): S_pred 0.949 / S_real -0.325, symmetric cross fits -- sigma misspecification excluded (supersedes the OOM-blocked het_mismatched 2x2 prereg 93f45a1) |
| `c1ceba833dc8` | `openillumination/ball_anchor.json` | yes (ancestor of HEAD) | P-BALL-ANCHOR run (2026-09-15): measured sphere-calibration anchor (2.96 deg / 0.0159), D-flipped at 36% median direction share |
| `95f15ad647c9` | `diligent/diligent_queue.json` | yes (ancestor of HEAD) | P-DILIGENT-QUEUE run (2026-09-15): second-dataset transfer, radius exact / channel split object-conditional |
| `951503954d45` | `baseline/baseline_comparison.json` | yes (ancestor of HEAD) | P-BASELINE v1.1 run (2026-09-16): active-set-controlled (dc05_active + randomA48 + overlap), OI geometry-insufficient / DQ geometry-informative via the informed policy's failure; 88 OI rows bit-anchored |
| `951503954d45` | `openillumination/decision_quality_feasible.json` | yes (ancestor of HEAD) | P-DQ-FEASIBLE zero-cost diagnostic (2026-09-16): budget-grid saturation (k=57/85/114 > |active|=48) + feasible-interval dAUC |
| `665d01ed88a9` | `openillumination/active_set_ablation_feasible.json` | yes (ancestor of HEAD) | P-ABLATION-FEASIBLE zero-cost diagnostic (2026-09-16): budget-axis degeneracy marking for the C8 ablation (k=48 = |active| kills the ordering sub-term only) + the two calibers (1364x / 1001x) |
| `29f9b9d239dc` | `diligent/diligent_queue.json`, `diligent/provenance/loader_normalization_fix.json`, `baseline/baseline_comparison.json` | yes (ancestor of HEAD) | P-DILIGENT-LOADER-FIX reruns (2026-09-16): DiLiGenT loader intensity normalization restored; queue + DQ-half baseline recomputed, OI rows unchanged |

The two unreachable SHAs are the ones the repo reorganization replaced. All
seventeen artifacts remain byte-for-byte as committed and are pinned by
`checksums.sha256`; a `git log --all` cannot reach the two orphaned commits,
which is why `git_sha` is documented here rather than re-derived at runtime.
The invalid v1 `linearization_radius.json` was replaced by the
metric-domain-corrected v2 rerun (reachable `785c7485f37e`).

## 2. Environment

```bash
pip install -e .[dev]          # numpy, scipy, pyyaml, pytest (+ Pillow for data tests)
python -m pytest               # full suite; data-dependent tests skip when raw data is absent
sha256sum -c checksums.sha256  # artifact integrity
```

Bit-exactness environment note: the frozen artifacts were produced under
**numpy 2.4.1** on Windows (the production environment). ULP-level
differences in `eigh`/`inv` accumulate through the scene assembly, so
bit-exact reproduction of the end-to-end anchors requires the same
numpy/BLAS build (verified empirically by the acceptance reviewer: under
numpy 2.5.3 even the UNCHANGED frozen code path drifts to ~3e-11 relative
at level 4.0 — an environment effect, not a code regression; see
`tests/test_sigma_source_exchange.py` for the CI-safe, environment-
independent part of the anchors). `pyproject.toml` pins `numpy>=1.24`
(no upper bound) deliberately; record your numpy version when producing
new artifacts.

Determinism: all experiments are seeded (`numpy.random.default_rng` with
committed seeds), all reductions are deterministic (eigh/cholesky/solve —
no iterative randomness), and every run writes a manifest (git SHA, config
hash, seeds, conventions).

## 3. Two pipeline versions (legacy vs corrected)

The real-data pipeline has two interface versions, both committed:

- `legacy` — bit-reproduces the original frozen benchmark artifacts
  (`results/openillumination/ci04_formal_summary.json` and friends).
- `corrected` — the documented heteroscedastic noise-fit coefficient order
  and the normalized dual-coordinate mode projection. **Reported numbers
  use this version.**

The preregistered A/B/C/D factorial
(`experiments/openillumination_factorial.py`,
`results/openillumination/correctness/mf0_factorial_summary.json`) verifies
that arm A (legacy/legacy) reproduces the frozen benchmark bit-close (max
relative difference 0.0 on all 66×5 pred/emp entries and the pooled
Spearman), and re-measures the interface corrections on the identical
protocol (same objects, pixel subsets, levels, seeds).

## 4. Headline results and how to reproduce them

| Result | Value | Command | Evidence | Test |
|---|---|---|---|---|
| Directional validation of the retention ordering (real objects) | R_A = 0.90, CI [0.7, 0.95], 65/66 cells | `python experiments/openillumination_factorial.py` (arm D rows committed) | `results/openillumination/correctness/mf0_factorial_summary.json` (`variants.D`) | `tests/test_reproduction.py::test_N1_mode_resolved_pass_rate` |
| Directional-validation construction disclosure | rank-equivalent to the mode-index baseline, deviation exactly 0.0 (66/66 cells) | `python experiments/directional_amplitude.py` | `results/magnitude/directional_amplitude_summary.json` | `tests/test_math_gates.py` |
| Certified dynamic range (P = 1200 subsample) | 36–89% per object, median 62.87% | `python experiments/certified_gaps.py` | `results/certification/certified_gaps.json` | `tests/test_certified_gaps_evidence.py` |
| Certified greedy optimality | 0.011–0.028% above the convex lower bound | same | same | same |
| Full-resolution confirmation (all masked pixels, all 142 lights) | 27.3–89.4%, median 60.06%; greedy 0.002–0.005% | `python experiments/lowrank_fullres.py` | `results/certification/lowrank_fullres.json` | `tests/test_m2_m3_evidence.py::test_fullres_structure` |
| Mode-tail targeted intervention (three-arm, corrected calibers) | honest null: targeted worse than random in 5/8 informative cells, never better; no increment over the scalar arm | `python experiments/allocation_mode_tail.py` | `results/mode_tail/allocation_mode_tail.json` | `tests/test_m2_m3_evidence.py::test_alloc2_structure` |
| Linearization validity radius (P-RADIUS v2) | median radius_2x = 1.0, radius_10x = 1.5 (11/11 objects); small-level slope +2.02–2.19 | `python experiments/linearization_radius.py` | `results/magnitude/linearization_radius.json` | `tests/test_pradius_pconc_artifacts.py` |
| Certificate concentration (P-CONC) | rel_spread_median 1.08%; clean_vs_mean_ratio median 1.42 (1.17–3.08) | `python experiments/certificate_concentration.py` | `results/certification/certificate_concentration.json` | `tests/test_pradius_pconc_artifacts.py` |
| Submodularity negative result | E-opt: 1518 violating triples over 10 instances, γ_min = 0.704; A-opt marginal; D-opt clean | `python experiments/submodularity_search.py --out /tmp/submod.json` (recomputation; the committed artifact stays untouched) | `results/submodularity/submodularity_search.json` | `tests/test_submodularity_harness.py` |
| Amplitude validity envelope | emp/pred median 201.1 (original pipeline), 1.0045 synthetic MC | `python experiments/directional_amplitude.py` | `results/magnitude/directional_amplitude_summary.json` | `tests/test_math_gates.py` |
| Certified allocation rank invariance | 594 frozen orderings, rank 1200 = rank(F∞) everywhere | `python experiments/allocation_rank_check.py` | `results/openillumination/correctness/allocation_rank_check.json` | `tests/test_math_gates.py` |
| Monte-Carlo variance identity (synthetic) | emp/analytic covariance ratio 1.0045 | `python scripts/run_experiments.py --experiment monte_carlo --config configs/monte_carlo.yaml` | `results/monte_carlo/ci03_formal_summary.json` | `tests/test_reproduction.py::test_N7_covariance_ratio` |
| Gauge closed form vs direct | max rel 2.55e-8 over 25 decades | `python scripts/run_experiments.py --experiment gauge_spectrum --config configs/gauge_spectrum.yaml` | `results/gauge_spectrum/ci02_formal_summary.json` | `tests/test_reproduction.py::test_N6_gauge_closed_form` |
| Nonlinear validity envelope | weak-mode median deviation < 10% gate | `python scripts/run_experiments.py --experiment nonlinear --config configs/nonlinear.yaml` | `results/nonlinear/ci03nl_nl_formal_summary.json` | collected by `tests/test_reproduction.py` |
| DiLiGenT sanity taxonomy | weak-mode taxonomy median ≈ 273× | `python scripts/run_experiments.py --experiment diligent --config configs/diligent.yaml` | `results/diligent/ci05_formal_summary.json` | `tests/test_reproduction.py::test_N8_taxonomy` |
| Σ_φ parameter-family sensitivity, predictor side (P-SIGMA-FAMILY) | D robust: direction share 0.07% at the operating point, r* > 25° (11/11 censored); two-term law parameterization-specific (median \|dV\| 0.0076, max 0.464); ceiling guard −0.0028 over 495 rows | `python experiments/corruption_family_sensitivity.py` | `results/openillumination/corruption_family_sensitivity.json` | `tests/test_corruption_family_sensitivity.py` |
| Σ_φ family, linearization radius (C arm) | channel-dependent: joint/intensity_only 1.0/1.5 (bit-exact frozen control); direction_only never crosses on [0.05, 8] (0/11); joint_het 0.2/0.35 | `python experiments/linearization_radius_family.py` | `results/magnitude/linearization_radius_family.json` | `tests/test_corruption_family_sensitivity.py` |
| Ball-anchor: Σ_φ from a real sphere calibration (P-BALL-ANCHOR) | measured anchor sig_dir 2.96 deg / sig_logI 0.0159; direction share at anchor median 36.0% -> D-flipped (preregistered rule) | `python experiments/ball_anchor.py` | `results/openillumination/ball_anchor.json` | `tests/test_ball_anchor.py` |
| DiLiGenT queue: second-dataset transfer (P-DILIGENT-QUEUE) | radius_2x median 1.0 (all objects [0.75,1.5]); direction max D 51.2%; ball-anchor share splits by object (pot1/pot2 86%) -> transfer-partial | `python experiments/diligent_queue.py` | `results/diligent/diligent_queue.json` | `tests/test_diligent_queue.py` |
| Baseline: literature-style selection on the same lights (P-BASELINE v1.1, active-set-controlled) | OI geometry-insufficient (dc05_active -0.438/-0.615 vs randomA48; every informed policy better); DiLiGenT geometry-informative via the informed policy losing to random (+0.242/+0.474) | `python experiments/baseline_comparison.py` | `results/baseline/baseline_comparison.json` | `tests/test_baseline_comparison.py` |
| Feasible-budget DQ diagnostic (P-DQ-FEASIBLE) | degenerate budgets k=57/85/114 flagged (88/88 bit-identical cells); feasible dAUC −0.38..−1.40 vs active48-random, dilution 2.0–2.8×, ratio to universe-random 0.27–0.51 | `python experiments/decision_quality_feasible.py` | `results/openillumination/decision_quality_feasible.json` | `pytest tests/test_dq_feasible.py` |
| Ablation budget-axis degeneracy (P-ABLATION-FEASIBLE) | k=48 = |active|: ordering sub-term definitionally zero (11/11 rows); C8 ratio caliber 1364x (full) / 1001x (excl. k=48); dilution term valid at k=48 | `python experiments/active_set_ablation_feasible.py` | `results/openillumination/active_set_ablation_feasible.json` | `pytest tests/test_ablation_feasible.py` |

Full per-experiment protocols (dataset → sampling → corruption → estimator →
metric → seed → output) are in `docs/EXPERIMENTS.md`; raw-data sourcing in
`docs/DATA.md`.

## 5. Frozen benchmark artifacts

The OpenIllumination controlled-corruption benchmark and the calibration-
allocation evaluation were executed once, preregistered, and committed:

- `results/openillumination/` — mode ranking, per-level severity, factorial
  correctness rerun, amplitude/directional analyses;
- `results/openillumination/allocation/` — the 29,700-reconstruction
  allocation evaluation (`uos_table.csv`, paired statistics, provenance with
  the exact cloud driver and statistics scripts);
- `results/{synthetic,gauge_spectrum,monte_carlo,nonlinear,diligent,diligent_ablation}/`
  — the synthetic validity panels and external sanity panels;
- `results/certification|mode_tail|submodularity|magnitude/` — the certified
  allocation analyses (this repository's active research direction).

Modifying any committed artifact breaks `sha256sum -c checksums.sha256`;
re-deriving a value differs from editing one — the tests exist to keep those
separate.

## 6. Cloud execution

Heavy runs (full-resolution certification) have a self-contained migration
package on the `cloud-mode-tail` branch of the companion repository
(`Multi-Illumination-Inverse-Rendering`): vendored driver + data subset +
a resumable launcher for a 32-core machine (~1.5–2.5 h). The default local
path runs the same computation on a workstation (~4–8 h sequential, less
with `--workers`).
