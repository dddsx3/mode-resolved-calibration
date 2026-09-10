# Provenance — allocation experiment (A4/A5)

Purpose: a clean-room user can determine **what code produced what results,
from which commit, in what environment** — without trusting any verbal claim.

## What ran

| Item | Value |
|---|---|
| Code state | tag `allocation-prerun-audited` = `ddd44b53` (all 7 manifest files hash-verified at run start, `A4_run_manifest.json`) |
| Grid | 11 objects × 3 levels {0.2, 0.5, 1.0} × 10 seeds × 2 regimes {10×, 100×} × 9 units × 5 budgets = **29,700 reconstructions** |
| Seeds | bootstrap 20260910; corruption stream per (obj, level, seed); permutations 20260911 — all pinned in `configs/openillumination_allocation.yaml` |
| Result rows | 29,700 (0 missing, 0 duplicate, 0 NaN/Inf, 0 exclusion; failure log empty) |
| Result grade | frozen label **strong** (both regimes) — historical internal label of the preregistered rule: it certifies mode-aware stably beating **random** (11/11 objects improved), NOT superiority over the classical E/A/D-opt baselines (the post-hoc paired comparison `allocation_policy_pairwise.csv` shows classical baselines with modestly lower AUC). Public wording: “actionable outcome vs random” — see the semantic erratum in `docs/WORDING.md` §2 |

## Files in this directory

| File | What it is |
|---|---|
| `A4_run_manifest.json` | Pre-run seal output: HEAD hash, tag, worktree state, F5 manifest match, checksums, pytest summary, timestamp |
| `a4_cloud_driver.py` | The exact driver that ran A4 on the cloud instance (BLAS 1-thread budget, forkserver start, heartbeat/watchdog, per-cell progress; resumable). Grid/seeds/policies identical to the frozen config — run-layer enhancements only |
| `a5_analyze.py` | The exact frozen-statistics script that produced `allocation_summary.json` / `allocation_deltas.csv` |
| `a5_pairwise.py` | Post-hoc paired policy comparison (mode-aware vs E/A/D, same bootstrap machinery) → `allocation_policy_pairwise.csv`; creates no new simulations |
| `run_logs/` | Raw cloud run logs (`log_precheck.log`, `log_a4_run.txt`, `log_stage2.log`, `log_a5.log`, `log_full_run.log`) — primary evidence for the run chronology |
| `CHANGES_20260909.diff` | The complete diff of run-layer changes made by the operator after `allocation-prerun`: BLAS 1-thread budget, forkserver start, heartbeat, and **an implementation bug fix in the stage-2 cross-level accumulation** (the uploaded version overwrote per-level rows instead of averaging them, i.e. it did not compute the preregistered endpoint). Chronology in `docs/REPRODUCIBILITY.md` shows the corrected stage-2 (16:26:45 UTC) pre-dated the run (16:32:57–16:45:56) — the defective version never executed on empirical data |

## Frozen-value declaration

The diff above touches run-layer concerns only (thread budget, monitoring,
process start method, path plumbing). It contains **no changes** to the grid,
seeds, policies, metric, statistics, or any file under the frozen
`results/` directories. Independent verification: `git diff pre-ci06-gates-passed
<result-commit> --name-only` shows no frozen-path file, and recomputing the A5
statistics from `uos_table.csv` reproduces every reported number exactly.

## Reproducing the statistics (not the 29,700-run simulation)

Re-running the full reconstruction grid is not required to verify the claims:
`pip install -e . && pytest` includes the independent recomputation of all
headline numbers from the committed `results/` artifacts, and
`allocation_deltas.csv` / `allocation_object_pairs.csv` can be re-derived from
`uos_table.csv` + `selection_orders.json` by `a5_analyze.py` on any machine.
