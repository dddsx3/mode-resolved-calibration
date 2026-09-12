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

### 1.1 Provenance note: unreachable git SHAs in early manifests

Two `git_sha` values recorded inside committed result manifests are **not
reachable** in the current history (`git cat-file -t` fails): the original
commits were replaced during the 2026-09-09 repository reorganization. This
is a known, documented state — not an omission:

- `fa54404…` — `results/certification/lowrank_fullres.json` and
  `results/certification/certificate_concentration.json` (v1). The former
  carries an explicit `provenance_unknown: true` +
  `sha_unreachable_reason` block; the latter keeps the v1 manifest and its
  v2 fields (`clean_vs_mean_ratio`) were added post-hoc from committed v1
  fields, documented by an in-file `revision` block (the full-resolution
  rerun is a cloud-scale job). The invalid v1
  `results/magnitude/linearization_radius.json` (same SHA) was replaced by
  the metric-domain-corrected v2 rerun.
- `bd67897…` — the seven frozen science-closed manifests (all with
  `git_dirty: true`: run on a dirty working tree):
  `results/synthetic/ci01_formal_manifest.json`,
  `results/gauge_spectrum/ci02_formal_manifest.json`,
  `results/monte_carlo/ci03_formal_manifest.json`,
  `results/nonlinear/ci03nl_nl_formal_manifest.json`,
  `results/openillumination/ci04_formal_manifest.json`,
  `results/diligent/ci05_formal_manifest.json`,
  `results/diligent_ablation/ci05abl_ablation_manifest.json`.
  These are immutable frozen benchmark artifacts and are kept byte-for-byte
  as committed; their integrity is independently pinned by
  `checksums.sha256`, and the producing scripts + configs are committed and
  testable — only the historical commit pointer itself is stale.

## 2. Environment

```bash
pip install -e .[dev]          # numpy, scipy, pyyaml, pytest (+ Pillow for data tests)
python -m pytest               # full suite; data-dependent tests skip when raw data is absent
sha256sum -c checksums.sha256  # artifact integrity
```

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
| Directional validation of the retention ordering (real objects) | R_A = 0.90, CI [0.7, 0.95], 65/66 cells | `python experiments/openillumination_factorial.py` (arm A machinery anchor; arm D rows committed) | `results/openillumination/mode_ranking.csv`, `results/openillumination/correctness/mf0_factorial_rows_D.json` | `tests/test_reproduction.py::test_N1_mode_resolved_pass_rate` |
| Directional-validation construction disclosure | rank-equivalent to the mode-index baseline, deviation exactly 0.0 (66/66 cells) | `python experiments/directional_amplitude.py` | `results/magnitude/directional_amplitude_summary.json` | `tests/test_math_gates.py` |
| Certified dynamic range (P = 1200 subsample) | 36–89% per object, median 62.87% | `python experiments/certified_gaps.py` | `results/certification/certified_gaps.json` | `tests/test_certified_gaps_evidence.py` |
| Certified greedy optimality | 0.011–0.028% above the convex lower bound | same | same | same |
| Full-resolution confirmation (all masked pixels, all 142 lights) | 27.3–89.4%, median 60.06%; greedy 0.002–0.005% | `python experiments/lowrank_fullres.py` | `results/certification/lowrank_fullres.json` | `tests/test_m2_m3_evidence.py::test_fullres_structure` |
| Mode-tail targeted intervention (three-arm, corrected calibers) | honest null: targeted worse than random in 5/8 informative cells, never better; no increment over the scalar arm | `python experiments/allocation_mode_tail.py` | `results/mode_tail/allocation_mode_tail.json` | `tests/test_m2_m3_evidence.py::test_alloc2_structure` |
| Linearization validity radius (P-RADIUS v2) | median radius_2x = 1.0, radius_10x = 1.5 (11/11 objects); small-level slope +2.02–2.19 | `python experiments/linearization_radius.py` | `results/magnitude/linearization_radius.json` | `tests/test_pradius_pconc_artifacts.py` |
| Certificate concentration (P-CONC) | rel_spread_median 1.08%; clean_vs_mean_ratio median 1.42 (1.17–3.08) | `python experiments/certificate_concentration.py` | `results/certification/certificate_concentration.json` | `tests/test_pradius_pconc_artifacts.py` |
| Submodularity negative result | E-opt: 1518 violating triples over 10 instances, γ_min = 0.704; A-opt marginal; D-opt clean | `python experiments/submodularity_search.py` | `results/submodularity/submodularity_search.json` | `tests/test_submodularity_harness.py` |
| Amplitude validity envelope | emp/pred median 201.1 (original pipeline), 1.0045 synthetic MC | `python experiments/directional_amplitude.py` | `results/magnitude/directional_amplitude_summary.json` | `tests/test_math_gates.py` |
| Certified allocation rank invariance | 594 frozen orderings, rank 1200 = rank(F∞) everywhere | `python experiments/allocation_rank_check.py` | `results/openillumination/correctness/allocation_rank_check.json` | `tests/test_math_gates.py` |
| Monte-Carlo variance identity (synthetic) | emp/analytic covariance ratio 1.0045 | `python scripts/run_experiments.py --experiment monte_carlo --config configs/monte_carlo.yaml` | `results/monte_carlo/ci03_formal_summary.json` | `tests/test_reproduction.py::test_N7_covariance_ratio` |
| Gauge closed form vs direct | max rel 2.55e-8 over 25 decades | `python scripts/run_experiments.py --experiment gauge_spectrum --config configs/gauge_spectrum.yaml` | `results/gauge_spectrum/ci02_formal_summary.json` | `tests/test_reproduction.py::test_N6_gauge_closed_form` |
| Nonlinear validity envelope | weak-mode median deviation < 10% gate | `python scripts/run_experiments.py --experiment nonlinear --config configs/nonlinear.yaml` | `results/nonlinear/ci03nl_nl_formal_summary.json` | collected by `tests/test_reproduction.py` |
| DiLiGenT sanity taxonomy | weak-mode taxonomy median ≈ 273× | `python scripts/run_experiments.py --experiment diligent --config configs/diligent.yaml` | `results/diligent/ci05_formal_summary.json` | `tests/test_reproduction.py::test_N8_taxonomy` |

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
