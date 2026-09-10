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
- Structural identity clarification (required whenever the identity is used in an
  argument about allocation): the identity is a **scalarization identity**, not a
  policy identity. P_emin and P_mode are the same *number* on the tracked modes
  because the E-min criterion applied to the retention spectrum is constructed
  from the weakest-mode datum. It does **not** follow that an E-optimal *allocation
  policy* coincides with the mode-aware allocation policy — the policies greedily
  select lights by different objective sequences and select different light
  orderings (verified: 22/22 object-regime pairs have level-dependent mode-aware
  orderings). Never cite "P_emin ≡ P_mode" as if it predicted equal allocation
  behaviour.
- Paired-difference reporting (any comparison of two predictors):
  report the CI of the *within-resample paired difference*; never argue from two
  marginal CIs both crossing zero.

## 2. Approved result-grade wording (allocation, expert-frozen)

**Semantic erratum (post-adjudication, 2026-09-10)**: the frozen grade names are
historical internal labels. Grade A ("strong") was awarded by the preregistered
rule for beating **random** allocation stably — it never tested, and must not be
read as, superiority over the classical E/A/D-optimal baselines. The post-hoc
paired comparison (`allocation_policy_pairwise.csv`) shows mode-aware achieving
modestly *lower* AUC than classical E/A/D-opt greedy (paired CIs exclude 0, both
regimes). Public-facing materials must therefore describe the outcome as an
**"actionable outcome vs random"**, not a "strong result"; public text must not
use the word "strong" for the allocation grade (the label survives only inside
frozen artifacts such as `allocation_summary.json`, where it is part of the run
record).

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

## 3. Banned sentence families

- "better than classical criteria such as E-optimality … on every resolution level"
- "on every resolution level measured" (any unqualified universal claim)
- "pass rate" (as a name for $R_A$; use "median within-cell Spearman")
- Any numeric claim in README/docs that cannot be traced to a specific field of a
  specific file under `results/`.

## 4. Claim tracing rule

Every numeric claim in `README.md` or `docs/*.md` must be traceable: a reader must be
able to point at a file under `results/` and a field/column within it that produces
the number. `tests/test_wording_gate.py` enforces the banned families above with
zero-tolerance grep; new numeric claims must be registered here (with their source
file/field) before they appear in any markdown.

## 5. Registered post-hoc wording (allocation, paired policy comparison)

Approved factual sentences for the post-hoc paired comparison
(`results/openillumination/allocation/allocation_policy_pairwise.csv`, derived from
the frozen `uos_table.csv` by `provenance/a5_pairwise.py`; derived 2026-09-10,
labelled `posthoc_paired_comparison`):

- "In post-hoc paired comparisons, the classical E/A/D-optimal greedy baselines
  achieved modestly lower AUC than the mode-aware policy (paired bootstrap 95% CIs
  exclude 0 in both regimes; median paired difference +0.019 to +0.027 AUC)."
- "The mode-aware policy improves over the full-universe random allocation in 11/11
  objects (Δ AUC −0.150 at 10×, −0.279 at 100×); the classical baselines improve
  over random by similar amounts."
- random_active48 control (registered from `allocation_random48_summary.json`,
  `mode_minus_random_active48` / `random_active48_minus_random_full` fields):
  "Restricting the random baseline to the 48 Fisher-active lights (uniform
  permutations, paired innovations) removes the mode-aware advantage entirely:
  Δ(mode − random_active48) is +0.014 at 10× and +0.019 at 100× with paired
  bootstrap 95% CIs spanning 0 ([−0.024, +0.048] and [−0.012, +0.063]; 3/11
  objects improved). The active-set effect itself is large and significant:
  random_active48 − random_full = −0.198 at 10× / −0.342 at 100× (CIs exclude 0).
  The measured benefit of every informed policy over full-universe random is
  therefore an active-set effect, not an ordering effect."
- Attribution caveat (required at first mention of the allocation result): "the
  48 Fisher-active lights are a subset of the 142-light universe; part of any
  policy's advantage over full-universe random allocation reflects selecting any
  active light rather than the specific ordering — bounded by the
  `random_active48` control (`allocation_random48_summary.json`)."
