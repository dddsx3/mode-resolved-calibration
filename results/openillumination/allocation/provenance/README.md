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
| Result grade | **strong** (both regimes): mode-aware stably beats random (11/11 objects improved) and at least one classical baseline; grade rule and wording per `docs/WORDING.md` |

## Files in this directory

| File | What it is |
|---|---|
| `A4_run_manifest.json` | Pre-run seal output: HEAD hash, tag, worktree state, F5 manifest match, checksums, pytest summary, timestamp |
| `a4_cloud_driver.py` | The exact driver that ran A4 on the cloud instance (BLAS 1-thread budget, forkserver start, heartbeat/watchdog, per-cell progress; resumable). Grid/seeds/policies identical to the frozen config — run-layer enhancements only |
| `a5_analyze.py` | The exact frozen-statistics script that produced `allocation_summary.json` / `allocation_deltas.csv` |
| `CHANGES_20260909.diff` | The complete diff of run-layer changes made by the operator after `allocation-prerun` — audited as grid/seed/statistics-neutral |

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
