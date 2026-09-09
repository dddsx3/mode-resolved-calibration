# Reproducibility map

Each manuscript headline number is traceable along the chain
paper item → script → committed source data.  All distributions below are committed
under `results/`; expect reproducers to re-run the scripts and to re-check the values
with `pytest tests/test_reproduction.py`.

| # | Paper item | Script | Source data |
|---|---|---|---|
| N1 | RA = 0.90 (95% CI [0.90, 0.95], 66 cells, 11 objects) | `experiments/openillumination_validation.py`, `experiments/openillumination_severity.py` | `results/openillumination/mode_ranking.csv` |
| N2 | Stratified median: mode/E-min 0.536, logdet 0.418, trace 0.400 | `experiments/openillumination_severity.py` | `results/openillumination/level_severity.csv` (per-cell columns P_*) |
| N3 | Pooled Spearman: 0.871 / 0.871 / 0.859 / 0.733 / 0.867 | `experiments/openillumination_severity.py` | `results/openillumination/level_severity.csv` |
| N4 | P_emin ≡ P_mode, max diff ≤ 1e-14 | `experiments/openillumination_severity.py` | `results/openillumination/level_severity.csv` |
| N5 | Old pooled 0.728, cluster CI [0.705, 0.754] | `experiments/openillumination_validation.py` | `results/openillumination/ci04_formal_summary.json` |
| N6 | Gauge closed form vs direct ≤ 3.9e-8 (25 decades of λ) | `experiments/gauge_spectrum.py` | `results/gauge_spectrum/ci02_formal_summary.json` |
| N7 | Cov(x̂) vs σ²ΔF⁻¹ median ratio 1.0045 | `experiments/monte_carlo_validation.py` | `results/monte_carlo/ci03_formal_summary.json` |
| N8 | CI05 taxonomy 55.9–1131.4×, median 273× | `experiments/diligent_sanity.py` (run_sanity) | `results/diligent/ci05_formal_summary.json` |
| N9 | V1–V6 known-answer suite green | `tests/test_covariance_identity.py`, `test_gauge_closed_form.py`, `test_retention_bounds.py`, `test_parameterization.py`, `test_scale_invariance.py`, `test_mode_tracking.py` | `tests/_reference_impl.py` |

## Recompute procedure

```bash
pip install -e .
pytest                      # includes tests/test_reproduction.py (N1-N9 independent recomputation)
bash reproduce_paper.sh     # regenerates figures and tables from results/
```

`tests/test_reproduction.py` loads the CSVs/JSONs above and recomputes N1–N9
independently (Spearman, per-level stratified medians, object-cluster bootstrap with
B=10000/seed 20260908, pooled statistics) — it does not import the experiment pipeline,
so it double-checks the frozen numbers rather than re-executing the computation that
produced them.

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
  `validation_summary.json` / `summary.json` under `results/openillumination/`
- `diligent` (runner → `experiments/diligent_sanity.py::run_sanity`): identical to
  `results/diligent/ci05_formal_summary.json`
- `diligent_ablation` (→ `run_ablation`): identical to
  `results/diligent_ablation/ci05abl_ablation_summary.json`
- cleanroom: a fresh virtualenv (`pip install -e .[dev,reproduce]`) runs the full
  70-test suite green

## Checksums

`checksums.sha256` fixes every committed file of this repository (SHA-256, UTF-8 text
mode). Regenerate after any content change with

```bash
cd "$(dirname "$0")" && git ls-files -z | xargs -0 sha256sum | sort -k2 > checksums.sha256
```

## Frozen data contract

- The per-cell tables under `results/openillumination/` are frozen; a mismatch between
  the manifest checksums and the files on disk means the artifacts were modified and
  the numbers must be treated as unverified.
- Raw benchmark downloads are external (see `docs/DATA.md`); the repository never
  re-derives data from different copies silently — object lists and per-file SHA-256
  records are pinned in `data/manifests/`.