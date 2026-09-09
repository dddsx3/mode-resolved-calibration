# Neutral wording guide

> **Dual role (Repo-First v1.0)**: this file is both (a) the approved wording
> registry for the frozen benchmark evidence and (b) the **claims table for the
> whole repository** — README, examples/, tutorials/, and docs pages may only
> make numeric or comparative claims that are registered here and trace to
> `results/**` or `examples/**` output. The claim gate scans all of them.

This guide keeps every claim in this repository traceable to the committed data.
It is a wording contract, not a governance document: the rules below exist so that
README/docs statements never outrun what the files under `results/` actually contain.

## 1. Approved sentence templates

Claims about the OpenIllumination benchmark may use exactly these forms:

- Overview (main empirical result):
  "mode-resolved coincides with E-optimality on the tracked modes (max diff ≤ 1e-14)
  and improves the stratified median over trace and log-determinant (0.536 vs 0.418 /
  0.400)."
- Per-level honesty clause (required whenever "best stratified median" is claimed):
  "best stratified median across levels (per-level reversals disclosed: at L1,
  trace 0.455 > mode 0.245; at L3, logdet 0.700 > mode 0.673)."
- Headline statistic:
  "median within-cell Spearman $R_A$ = 0.90 (object-cluster bootstrap 95% CI
  [0.90, 0.95], 66/66 cells, 11/11 objects)"
- Stratified-median uncertainty (required at first mention):
  "object-cluster bootstrap 95% CI [−0.096, 0.858] (n = 11; not significant at the
  object level)."
- Structural identity disclosure (always with any E-min/E-optimality comparison):
  "P_emin ≡ P_mode on the tracked modes (max absolute difference ≤ 1e-14)."
- Paired-difference reporting (any comparison of two predictors):
  report the CI of the *within-resample paired difference*; never argue from two
  marginal CIs both crossing zero.

## 2. Approved result-grade wording (allocation, expert-frozen)

Grade A (strong result) — verbatim, registered before use:
  "mode-resolved calibration analysis can support allocation decisions that
  improve reconstruction under a fixed calibration budget."

Grade B (neutral) — verbatim:
  "mode-derived allocation improves over random allocation, while classical
  optimal-design criteria achieve comparable aggregate performance; the mode
  decomposition additionally exposes which vulnerable directions drive the
  decision."

Grade C (negative) — verbatim:
  "mode-resolved vulnerability predicts degradation but did not produce a robust
  downstream allocation advantage under the tested policy."

## 2. Banned sentence families

- "better than classical criteria such as E-optimality … on every resolution level"
- "on every resolution level measured" (any unqualified universal claim)
- "pass rate" (as a name for $R_A$; use "median within-cell Spearman")
- Any numeric claim in README/docs that cannot be traced to a specific field of a
  specific file under `results/`.

## 3. Claim tracing rule

Every numeric claim in `README.md` or `docs/*.md` must be traceable: a reader must be
able to point at a file under `results/` and a field/column within it that produces
the number. `tests/test_wording_gate.py` enforces the banned families above with
zero-tolerance grep; new numeric claims must be registered here (with their source
file/field) before they appear in any markdown.
