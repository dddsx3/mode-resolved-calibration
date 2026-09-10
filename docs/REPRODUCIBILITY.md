# Reproducibility map

Each manuscript headline number is traceable along the chain
paper item → script → committed source data.  All distributions below are committed
under `results/`; expect reproducers to re-run the scripts and to re-check the values
with `pytest tests/test_reproduction.py`.

| # | Paper item | Script | Source data |
|---|---|---|---|
| N1 | **Frozen arm-A record**: RA = 0.90 (95% CI [0.90, 0.95], 66 cells, 11 objects). Paper-facing corrected-interface headline is arm D (RA = 0.90, 95% CI [0.70, 0.95]) — see N13 | `experiments/openillumination_validation.py`, `experiments/openillumination_severity.py` | `results/openillumination/mode_ranking.csv` |
| N2 | **Retired (frozen arm-A record)**: stratified median mode/E-min 0.536, logdet 0.418, trace 0.400 — does not survive the corrected interface (arm D −0.495); not paper-facing — see N13 | `experiments/openillumination_severity.py` | `results/openillumination/level_severity.csv` (per-cell columns P_*) |
| N3 | **Retired (frozen arm-A record)**: pooled Spearman 0.871 / 0.871 / 0.859 / 0.733 / 0.867 — descriptive only under the frozen interface; arm D pooled values in the factorial summary — see N13 | `experiments/openillumination_severity.py` | `results/openillumination/level_severity.csv` |
| N4 | P_emin ≡ P_mode, max diff ≤ 1e-14 | `experiments/openillumination_severity.py` | `results/openillumination/level_severity.csv` |
| N5 | Old pooled 0.728, cluster CI [0.705, 0.754] | `experiments/openillumination_validation.py` | `results/openillumination/ci04_formal_summary.json` |
| N6 | Gauge closed form vs direct ≤ 3.9e-8 (25 decades of λ) | `experiments/gauge_spectrum.py` | `results/gauge_spectrum/ci02_formal_summary.json` |
| N7 | Cov(x̂) vs σ²ΔF⁻¹ median ratio 1.0045 | `experiments/monte_carlo_validation.py` | `results/monte_carlo/ci03_formal_summary.json` |
| N8 | DiLiGenT taxonomy 55.9–1131.4×, median 273× | `experiments/diligent_sanity.py` (run_sanity) | `results/diligent/ci05_formal_summary.json` |
| N9 | V1–V6 known-answer suite green | `tests/test_covariance_identity.py`, `test_gauge_closed_form.py`, `test_retention_bounds.py`, `test_parameterization.py`, `test_scale_invariance.py`, `test_mode_tracking.py` | `tests/_reference_impl.py` |
| N10 | Post-hoc paired policy comparison: mode−E/A/D paired medians +0.019…+0.027, all six 95% CIs > 0 (classical slightly lower AUC) | `results/openillumination/allocation/provenance/a5_pairwise.py` | `results/openillumination/allocation/allocation_policy_pairwise.csv` |
| N11 | random_active48 control: Δ(mode − random48) = +0.014 / +0.019, CIs span 0, 3/11 improved | `experiments/openillumination_allocation_random48.py` | `results/openillumination/allocation/allocation_random48_summary.json` |
| N12 | Active-set attribution: random48 − random_full = −0.198 / −0.342, CIs exclude 0, 11/11 improved | `experiments/openillumination_allocation_random48.py` | `results/openillumination/allocation/allocation_random48_summary.json` |
| N13 | MF-0 factorial: arm A reproduces the frozen benchmark bit-close (max rel diff 0.0 on all 66×5 pred/emp entries and the pooled Spearman); paper-facing arm D (corrected/corrected, fixed a priori): RA = 0.90 (95% CI [0.70, 0.95], 65/66 positive cells, 11/11 positive objects); stratified severity association retires (−0.495 vs +0.536, flip isolated to M0-1); allocation rank invariance: all 594 frozen orderings keep rank 1200 = rank(F∞) | `experiments/openillumination_factorial.py`, `experiments/allocation_rank_check.py` | `results/openillumination/correctness/` |

## Recompute procedure

```bash
pip install -e .
pytest                      # N1-N9 in tests/test_reproduction.py (frozen at
                            # science-closed), N10-N12 in
                            # tests/test_manuscript_evidence.py (frozen at
                            # manuscript-evidence-v1)
bash reproduce_paper.sh     # regenerates figures and tables from results/
```

`tests/test_reproduction.py` loads the CSVs/JSONs above and recomputes N1–N9
independently (Spearman, per-level stratified medians, object-cluster bootstrap with
B=10000/seed 20260908, pooled statistics) — it does not import the experiment
pipeline; it is byte-frozen at tag `science-closed`.
`tests/test_manuscript_evidence.py` recomputes N10–N12 the same way (re-deriving
the paired policy comparison from `uos_table.csv` and cross-checking the
post-hoc CSVs/JSONs) and is frozen at tag `manuscript-evidence-v1`:
so it double-checks the frozen numbers rather than re-executing the computation that
produced them.
N13 (the MF-0 factorial and rank-invariance evidence) is guarded by
`tests/test_mf03_factorial_evidence.py` and `tests/test_math_freeze_gates.py`;
its paper-facing caliber is arm D of the factorial (see `docs/WORDING.md` §1/§6
for the two-caliber template rule).

## Interpretation of reported intervals

stratified_median_mode_ci95 is the marginal object-cluster bootstrap CI of the mode
predictor's stratified association. It is not a confidence interval for superiority
over another predictor. Comparisons against trace/logdet must report
CI(ρ_mode − ρ_trace) computed within the same bootstrap resamples; comparing two
marginal CIs is not a valid inference procedure.

## Official data inventory (factual record, 2026-09-09)

The official OpenIllumination OLAT manifest (`data_olat.json`, the only OLAT source of
the authors' download script) contains exactly 20 objects (obj_01–obj_20): 11 form the
frozen held-out cohort of this repository, 8 were consumed by the development manifest,
and obj_20_greenhead lacks the required thumbnail layer — no further OLAT-eligible
objects exist upstream. The "64 objects" figure refers to the official `data.json`
lighting_patterns layer: its 13 lighting patterns are simultaneous groups of the 142
LEDs, so the physical direction of every single LED is present in the data, but the
per-LED separable observation structure required by the per-light calibration model is
not. A 30-object independent cohort is therefore not constructible from the official
distribution; the preregistered allocation evaluation is run on the fixed held-out
cohort instead (`configs/openillumination_allocation.yaml`).

## Determinism record (reruns of 2026-09-09)

Every rerun below was executed in a sandbox output root and diffed against the frozen
artifact. No numeric value differs beyond 1e-8 on any of the 824+ compared leaves, and
the key sets match exactly.

- `synthetic` (runner → `experiments/numerical_identities.py`): identical to
  `results/synthetic/ci01_formal_summary.json`
- `gauge_spectrum`: identical to `results/gauge_spectrum/ci02_formal_summary.json`
- `monte_carlo`: identical to `results/monte_carlo/ci03_formal_summary.json`
- `nonlinear`: identical to `results/nonlinear/ci03nl_nl_formal_summary.json`
- `openillumination_severity` (prediction-side recompute, gate rel ≤ 1e-9; measured
  4.4e-12): regenerates `mode_ranking.csv`, `level_severity.csv`,
  `predictor_comparison.csv` bit-identically (byte-equal), and produces
  `validation_summary.json` under `results/openillumination/`
- `diligent` (runner → `experiments/diligent_sanity.py::run_sanity`): identical to
  `results/diligent/ci05_formal_summary.json`
- `diligent_ablation` (→ `run_ablation`): identical to
  `results/diligent_ablation/ci05abl_ablation_summary.json`
- cleanroom: a fresh virtualenv (`pip install -e .[dev,reproduce]`) runs the full
  test suite green — 85 passed / 5 skipped on machines without the raw datasets
  (the 5 data-dependent tests skip), 90 passed on machines with them

## Allocation evaluation (preregistered, run once)

The allocation evaluation (`experiments/openillumination_allocation.py`, config
`configs/openillumination_allocation.yaml`, frozen at tag
`allocation-prereg-frozen`, audited state tag `allocation-prerun-audited`) was
executed once on the cloud: 29,700 reconstructions, 33/33 cells, 0 failures,
0 exclusions; pre-run seal output in
`results/openillumination/allocation/provenance/A4_run_manifest.json`.

- Reproducing its **statistics** needs no raw data: `uos_table.csv` +
  `selection_orders.json` are committed, and `allocation_deltas.csv` /
  `allocation_object_pairs.csv` / `allocation_summary.json` are re-derived from
  them by the frozen `_analyze` (verified: cross-level means reproduce exactly).
  The post-hoc pairwise CSV is re-derived by `provenance/a5_pairwise.py` from
  the same table.
- The **`random_active48` attribution control**
  (`allocation_random48_per_run.csv`, `allocation_random48_summary.json`) is
  re-run with `python experiments/openillumination_allocation_random48.py
  --data <OLAT root> --meta <meta root>`; it needs the raw 11-object data,
  reuses the paired corruption stream of the benchmark, and anchors its frozen
  deltas against `allocation_summary.json` (assert < 1e-9).
- Reproducing the **29,700-run simulation** itself requires the raw 11-object
  data (see `docs/DATA.md`) and the driver preserved in
  `results/openillumination/allocation/provenance/a4_cloud_driver.py`; completed
  cells are skipped on re-run (resumable).

## Provenance

`results/openillumination/allocation/provenance/` records what code produced the
allocation results and in what environment: the pre-run seal output
(`A4_run_manifest.json`), the exact run driver (`a4_cloud_driver.py`), the frozen
analysis script (`a5_analyze.py`), the post-hoc pairwise script (`a5_pairwise.py`),
and the run-layer change record (`CHANGES_20260909.diff`). See the `README.md`
inside that directory for the full mapping. The raw cloud run logs are preserved
in `provenance/run_logs/` (`log_precheck.log`, `log_a4_run.txt`, `log_stage2.log`,
`log_a5.log`, `log_full_run.log`).

## Run-layer chronology (evidence, 2026-09-09 UTC)

The operator fixed two implementation defects in the cloud run-layer scripts
(a stage-2 cross-level accumulation that overwrote per-level rows instead of
averaging them — i.e. it did not compute the preregistered endpoint, and an
unpacked 6-tuple call) **while the package was being staged**. The five
timestamped facts below — each verifiable in the committed logs — establish that
the corrected stage-2 code was in place before any empirical data was generated
and before any grade or aggregate was computed:

1. `2026-09-09 14:39:50` — original upload of the cloud package files
   (old-side mtimes recorded in `CHANGES_20260909.diff`).
2. `2026-09-09 16:26:45` — operator's corrected `a4_stage2_check.py` saved
   (new-side mtime in `CHANGES_20260909.diff`); corrections to the driver
   followed at 16:32:26.
3. `2026-09-09 16:32:57` — pre-run seal PASS on the audited worktree
   (`provenance/run_logs/log_precheck.log`, `timestamp_utc` field; 85 passed /
   5 skipped, checksum + F5 manifest clean).
4. `2026-09-09 16:33–16:45:56` — the 29,700-reconstruction run itself
   (heartbeats in `provenance/run_logs/log_a4_run.txt`: first 16:32:57 "0/33",
   last 16:45:27 "5/33 running"; `log_full_run.log` mtime 16:45:56).
5. `2026-09-09 16:51:10 / 16:51:30` — stage-2 integrity checks PASS (all
   mechanical checks OK, `log_stage2.log`) and A5 statistics written
   (`allocation_summary.json`, zip entry mtime 16:51:30).

Conclusion: the defective stage-2 version (14:39:50) never executed against the
empirical data — the entire reconstruction run and all integrity checks ran
after the 16:26:45 fix. No frozen number was produced by known-buggy code.

## Math-freeze correctness gate (MF-0, 2026-09-10)

The math-method freeze v1.0 required two correctness gates on the
theory-to-empirical interface before headline numbers are locked, plus four
smaller gates. Status and repo mapping:

- **M0-1 (noise-fit coefficient order)**: `noise_fit`
  (`experiments/openillumination_validation.py`) and `calibrate`
  (`src/calibinfo/estimators/joint_map.py`) consumed the `np.polyfit` return
  order `[slope, intercept]` into the model `Var ≈ a + b·I` swapped (a←slope).
  Both now expose `convention="legacy"|"corrected"`; legacy (default) keeps the
  frozen artifacts bit-reproducible, corrected is the documented model.
  Known-answer tests: `tests/test_math_freeze_gates.py` (MF-0.1).
- **M0-2 (mode-projection coordinate)**: the empirical error projection is now
  available as `E @ F∞^{1/2} U` (normalized dual coordinate of the strict 1/ρ_j
  theorem) beside the frozen `E @ U`; selected by
  `predicted_degradation(mode_coordinate=...)`. Known-answer algebra + MC:
  MF-0.2 within `tests/test_math_freeze_gates.py` (MF-0.4 identity).
- **M0-3 (factorial rerun)**: `experiments/openillumination_factorial.py` reruns
  the controlled-corruption protocol as the preregistered A/B/C/D factorial
  (A = legacy/legacy reproduces the frozen benchmark bit-close and anchors the
  machinery; D = corrected/corrected is the paper-facing arm, fixed a priori).
  Outputs: `results/openillumination/correctness/` — protocol in
  `docs/EXPERIMENTS.md` §11, registered wording in `docs/WORDING.md` §6.
  Outcome (2026-09-10): arm A reproduces the frozen benchmark bit-close (max
  relative difference 0.0 on all 66×5 pred/emp entries and on the pooled
  Spearman); the primary within-cell mode-ranking result is unchanged under
  arm D (R_A = 0.90, CI [0.7, 0.95], 65/66 positive cells, 11/11 positive
  objects), while the fixed-level scalar-severity association flips with the
  corrected noise fit (stratified −0.495 vs +0.536; arm B −0.577, arm C
  +0.509 isolates the flip to M0-1). Arm D is the paper-facing record per the
  preregistered rule.
- **M0-4 (covariance theorem known-answer)**: `F∞^{1/2} Cov(x̂) F∞^{1/2}/σ² =
  R⁻¹` (matched GLS ensemble) + the forbidden `F∞^{-1/2}` form as a
  documentational contrast — MF-0.4 tests. `retention_spectrum` now also
  returns R's eigenvectors (`modes`).
- **M0-5 (singular covariance)**: B=[1,1], Σ_c=diag(1,0), σ=1, A=[1] →
  covariance-factor/marginal routes = 0.5; the unconstrained
  pseudoinverse-precision shortcut = 0 (never equivalent; API/doc gate in
  `schur.py` + wording guideline §3). New routes: `delta_f_marginal_factor`,
  `nuisance_factor`. MF-0.5 tests.
- **EOL pitfall in the frozen-artifact hash gate (found & fixed 2026-09-10)**:
  the `frozen_artifacts_sha256` values recorded in `configs/openillumination.yaml`
  were computed on a CRLF working tree, while `.gitattributes` pins `* text
  eol=lf` — so the bare-sha assertion in `openillumination_severity.py` could
  never pass on a conforming LF checkout (broken on a fresh clone).
  `load_and_assert` now compares EOL-robustly (LF- and CRLF-normalized
  candidates; tampered content matches neither — zero tolerance unchanged) and
  a regression test pins the fix (`test_frozen_artifact_sha_eol_robust`).
  The committed artifacts themselves are untouched and bit-identical to their
  blobs; `checksums.sha256` (LF basis) was never affected.
- **M0-6 (allocation rank invariance)**: ΔF numerical rank across
  candidates/budgets is checked by
  `calibinfo.allocation.rank_invariance.rank_invariance_report`
  (`tests/test_math_freeze_gates.py` MF-0.6). Executed on the frozen states
  (`experiments/allocation_rank_check.py` →
  `results/openillumination/correctness/allocation_rank_check.json`): all 594
  frozen orderings (11 objects × 3 levels × 2 regimes × 9 policies) keep
  numerical rank 1200 = rank(F∞) at every budget prefix — the working subspace
  is common and the E/A/D-optimal naming is licensed (the positive-subspace
  A/D implementation coincides with the full working subspace).

## Checksums

`checksums.sha256` fixes every committed file of this repository. **Basis: LF** —
`.gitattributes` pins `* text eol=lf`, so working-tree bytes equal blob bytes on every
platform. Regenerate after any content change with (2026-09-09):

```bash
git ls-files | grep -v '^checksums.sha256$' | sort | xargs sha256sum > checksums.sha256
```

Verify on any machine with `sha256sum -c checksums.sha256` (all lines must report OK).

## Frozen data contract

- The point-wise CI [0.668, 0.783] embedded in `ci04_formal_summary.json` (flat
  330-point bootstrap) is superseded by the object-cluster recomputation
  [0.705, 0.754] (N5); the embedded value is retained only as part of the frozen run
  record.
- The per-cell tables under `results/openillumination/` are frozen; a mismatch between
  the manifest checksums and the files on disk means the artifacts were modified and
  the numbers must be treated as unverified.
- Raw benchmark downloads are external (see `docs/DATA.md`); the repository never
  re-derives data from different copies silently — object lists and per-file SHA-256
  records are pinned in `data/manifests/`.