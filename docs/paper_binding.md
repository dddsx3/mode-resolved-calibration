# Paper ↔ registry binding

The manuscript does not print registry numbers in its text. This file is the
repo-side binding that makes the mapping explicit: for each manuscript
section, it lists the registry row(s) in `docs/claims.md` that carry the
claim. Every row cited below has its own evidence path and re-derive command
in `docs/claims.md`; each cited path resolves in this repository.

The repository itself remains manuscript-agnostic: this table records the
binding, not the manuscript text.

## Mapping by manuscript section

| Manuscript section | Claim | Registry row(s) |
|---|---|---|
| §I Introduction (contributions) | refined precision upper bound; task-weighted risk; low-rank scaling | M10 / M10' / M11 / C14 |
| §II Mode-Resolved Information | effective information; retention operator; matched covariance semantics; unit-retention multiplicity `P − rank(V)` | M1 / M2 / M4 / M8 / M7 |
| §III Task-Aware Value | ceiling for any fixed quadratic task; exact two-term decomposition with remainder; convexity | M10' / M14 / M3 / M5 |
| §III (subsection C) | why the certificate does not rely on diminishing returns: exact counterexample to the candidate bound, fixed-α family, condition-number-aware spectral replacement | M13 / M15 |
| §IV Convex Certified Allocation | convex relaxation; Frank–Wolfe lower bound; discrete suboptimality upper bound | M6 / M10 / B4 / B7 |
| §V Evaluation (TABLE I) | full-resolution calibration value and allocation certificates | C3 (full-resolution headline) / C1 (P=1200 subsample cross-check) |
| §V (corner-endpoint comparison) | allocation-cell relevance, endpoint dependence, active-set vs ordering | B5 / B8 / B9 / C8 / C12 |
| §VI From Information Risk to Reconstruction (A) | matched generation–estimation ensemble | C13 |
| §VI (transfer bridge) | mean-square-error identity (covariance + bias); measurable risk radius and pairwise margin | M16 |
| §VI (joint extension) | intrinsic joint log-albedo/unit-normal; `GL(3)` gauge; retention ceiling and convex certificate on the fixed quotient | M17 / M4 |
| §VII Discussion / limitations | known limitations; open problems | N-1 / N-2 / N-3; C4 (historical withdrawal) |

## Historical rows: current status

| Row | Status | Binding |
|---|---|---|
| M9 | `[refuted]` — the historical alpha-only candidate bound is false by exact counterexamples; the refutations are M13 (physical subclass counterexample) and M15 (spectral replacement bound) | `docs/claims.md` M9 row |
| B1 | superseded — historical R_A 0.90 is retained for traceability; the current caliber is the per-sample projection control (0.55, CI spanning zero) | `docs/claims.md` B1 row |
| B2 | superseded — the amplitude headline moved to the coordinate-matched caliber | `docs/claims.md` B2 row |
| B8 | historical caliber only — the validity map is not a current projection-pipeline guarantee | `docs/claims.md` B8 row |

Notes:

- Do not attach the manuscript's §III counterexample to M9: M9 is the
  refuted historical candidate; what refutes it are M13 (physical subclass
  counterexample) and M15 (spectral replacement bound).
- Acceptance check: pick any manuscript claim, find its row above, then
  follow that row's evidence path and re-derive command in `docs/claims.md`.
