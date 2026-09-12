# Claims & evidence registry

Every quantitative claim in this repository's documentation is bound to a
committed evidence file under `results/`. This table is the single index of
those bindings: a reader (or reviewer, or downstream user) can go from any
number in the docs to the file and field that produces it, and re-derive it
with the listed command.

The enforcement is mechanical: `tests/test_claims_gate.py` blocks
overclaim phrasings, and its `test_readme_numeric_claims_traceable`
requires every number printed
in the README to appear verbatim under `results/**`.

## Library / methods claims

| # | Claim | Value / bound | Evidence | Re-derive |
|---|---|---|---|---|
| M1 | Dual-route equivalence (Schur profiling vs Gaussian marginalization) | elementwise rel < 1e-10 | unit assertion | `pytest tests/test_information_modules.py` |
| M2 | Operator concavity of t ↦ ΔF(t) | midpoint residual ~1e-15 | `results/`-independent | `pytest tests/test_math_foundations.py` |
| M3 | Convexity of J_A, J_E, J_D on the PD feasible region | 0/400 midpoint violations each | unit assertion | `pytest tests/test_math_foundations.py` |
| M4 | Exact low-rank retention identity R = I − VVᵀ | spectral deviation ≤ 1e-10; #ρ≡1 = P − 3L | unit assertion | `pytest tests/test_lowrank_identity.py` |
| M5 | J_A gradient exactness | FD agreement ≤ 1e-6 | unit assertion | `pytest tests/test_math_foundations.py` |
| M6 | Convex-program certificates are global | FW gap ≥ 0; no feasible point beats the bound | `results/certification/certified_gaps.json` | `pytest tests/test_convex_certificates.py` |
| M7 | Singular-covariance semantics | factor/marginal = 0.5, pseudoinverse-precision shortcut = 0 (never equivalent) | unit assertion | `pytest tests/test_math_gates.py` |
| M8 | Retention-covariance theorem | F∞^{1/2} Cov F∞^{1/2}/σ² = R⁻¹ (matched GLS) | unit assertion | `pytest tests/test_math_gates.py` |

## Real-data benchmark claims (11 held-out OpenIllumination objects)

| # | Claim | Value | Evidence | Re-derive |
|---|---|---|---|---|
| B1 | Directional validation of the retention ordering | within-cell Spearman R_A = 0.90, CI [0.7, 0.95], 65/66 cells, 11/11 objects; exactly rank-equivalent to the mode-index baseline (66/66 cells, deviation 0.0) | `results/magnitude/directional_amplitude_summary.json` | `pytest tests/test_reproduction.py::test_N1_mode_resolved_pass_rate` |
| B2 | Amplitude validity envelope | emp/pred ratio median 201.1, 5–95% [7.3, 1525.8] (original pipeline); synthetic matched MC 1.0045 | `results/magnitude/directional_amplitude_summary.json` | `python experiments/directional_amplitude.py` |
| B3 | Stratified severity branch retired | arm-D stratified −0.495 vs frozen +0.536; flip isolated to the noise-fit correction | `results/openillumination/correctness/mf0_factorial_summary.json` | `pytest tests/test_factorial_evidence.py` |
| B4 | Policy-comparison nulls are structural | certified epsilon vs worst policy deviation, per k (median certified dynamic range 0.6287 vs random-mean-minus-lower-bound): k=5 → 125×, k=10 → 98×, k=14 → 127×, k=28 → 247×; at k=48 the random mean attains the lower bound (gap 0, ratio undefined). A magnitude gap per-k, not a pinned constant; the envelope's absolute scale is set by the Finf heterogeneity tail | `results/certification/certified_gaps.json` (`by_k`) | `pytest tests/test_certified_gaps_evidence.py` |
| B5 | Allocation benefit is an active-set effect | Δ(mode − random_active48) = +0.014/+0.019, CIs span 0, 3/11 improved | `results/openillumination/allocation/allocation_random48_summary.json` | `pytest tests/test_benchmark_evidence.py::test_N11_random48_control_ci_spans_zero` |
| B6 | Linearization validity radius (v2: observation-side injection + nominal-geometry estimator) | radius_2x median 1.0 (11/11 crossed, per-object 0.5–1.5); radius_10x median 1.5 (per-object 1.5–2.0); small-level log-log slope +2.02–2.19 (v1: −1.9, withdrawn as estimator-collapse artifact) | `results/magnitude/linearization_radius.json` | `pytest tests/test_pradius_pconc_artifacts.py` |
| B7 | Certificate concentration: two distinct uncertainty quantities | rel_spread_median 1.08% (within-family IQR); clean_vs_mean_ratio median 1.42, min 1.17, max 3.08 (systematic noiseless-vs-noisy calibration-source offset: +16.8% to +208.2%, median +42.2%). Reported separately; rel_spread is not the total bound accuracy | `results/certification/certificate_concentration.json` | `pytest tests/test_pradius_pconc_artifacts.py` |

## Certified-allocation claims

| # | Claim | Value | Evidence | Re-derive |
|---|---|---|---|---|
| C1 | Certified dynamic range, P=1200 subsample | 36–89% per object (median 62.87%) | `results/certification/certified_gaps.json` (`display`) | `pytest tests/test_certified_gaps_evidence.py` |
| C2 | Greedy is certified essentially optimal | 0.011–0.028% above the convex lower bound at every budget | `results/certification/certified_gaps.json` | `pytest tests/test_certified_gaps_evidence.py` |
| C3 | Full-resolution confirmation (all masked pixels P = 3559–10252, all 142 lights) | dynamic range 27.3–89.4% (median 60.06%); greedy within 0.002–0.005% | `results/certification/lowrank_fullres.json` (`display`) | `pytest tests/test_m2_m3_evidence.py::test_fullres_structure` |
| C4 | **Superseded** — the v1.0 "targeted intervention is significant" claim was a gauge-alignment asymmetry artifact; the corrected three-arm rerun (same file, `aggregated`) finds no mode-specific advantage: targeted is never better than random and shows no increment over the scalar arm | `results/mode_tail/allocation_mode_tail.json` (`erratum`, `aggregated`) | `results/mode_tail/allocation_mode_tail.json` | `pytest tests/test_m2_m3_evidence.py::test_alloc2_structure` |
| C5 | Allocation rank invariance | all 594 frozen orderings keep numerical rank 1200 = rank(F∞) at every budget prefix | `results/openillumination/correctness/allocation_rank_check.json` | `pytest tests/test_math_gates.py` |

## Registered negative results

| # | Finding | Evidence |
|---|---|---|
| N-1 | E-optimal gains violate submodularity (1518 triples over 10 instances, γ_min = 0.704); A-optimal at most marginal (γ_min ≥ 0.99989); D-optimal clean | `results/submodularity/submodularity_search.json` |
| N-2 | The stratified (cross-object) severity comparison is retired: the corrected-interface factorial flips its sign, and P_mode ≡ P_emin is a scalarization identity, not an independent predictor | `results/openillumination/correctness/mf0_factorial_summary.json` |
| N-3 | Three dynamic-range numbers, one law, different Finf tails: the assessment report's 4.02% is a synthetic-scene instance value (uniform Finf — the flat end of the law); real objects measure 36–89% at P=1200 (median 62.87%) and 27.3–89.4% full-resolution (median 60.06%). The certified dynamic range grows monotonically with Finf heterogeneity (max/min): uniform-Finf synthetic scenes sit near 4%, real objects with heavy Finf tails sit at 27–89% — the three values are the same mechanism sampled at different Finf tails, not a contradiction | `results/certification/provenance/reconciliation.json`, `results/certification/certified_gaps.json`, `results/certification/lowrank_fullres.json` |

## Two pipeline versions

Two versions exist for the real-data pipeline: `legacy` (bit-reproduces the
original frozen benchmark artifacts) and `corrected` (the mathematically
documented noise-fit coefficient order and the normalized dual-coordinate
mode projection). Reported numbers use `corrected`; `legacy` outputs are kept for
reference. The preregistered A/B/C/D factorial that motivates
this is documented in `docs/EXPERIMENTS.md` and its evidence gated by
`tests/test_factorial_evidence.py`.
