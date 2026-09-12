# Provenance — allocation experiment

Purpose: a clean-room user can determine **what code produced what results,
from which commit, in what environment** — without trusting any verbal claim.

## What ran

| Item | Value |
|---|---|
| Code state | commit `ddd44b53` (all 7 manifest files hash-verified at run start; seal in `A4_run_manifest.json`) |
| Grid | 11 objects × 3 levels {0.2, 0.5, 1.0} × 10 seeds × 2 regimes {10×, 100×} × 9 units × 5 budgets = **29,700 reconstructions** |
| Seeds | bootstrap 20260910; corruption stream per (obj, level, seed); permutations 20260911 — all pinned in `configs/openillumination_allocation.yaml` |
| Result rows | 29,700 (0 missing, 0 duplicate, 0 NaN/Inf, 0 exclusion) |
| Result wording | "actionable outcome vs random" — mode-aware allocation stably improved over random (11/11 objects); the classical E/A/D-opt baselines improved similarly (post-hoc paired comparison in `allocation_policy_pairwise.csv`) |

## Files in this directory

| File | What it is |
|---|---|
| `A4_run_manifest.json` | Pre-run seal output: HEAD hash, worktree state, F5 manifest match, checksums, pytest summary, timestamp |
| `a4_cloud_driver.py` | The exact driver that ran the grid on the cloud instance (resumable). Grid/seeds/policies identical to the frozen config |
| `a5_analyze.py` | The exact frozen-statistics script that produced `allocation_summary.json` / `allocation_deltas.csv` |
| `a5_pairwise.py` | Post-hoc paired policy comparison (mode-aware vs E/A/D, same bootstrap machinery) → `allocation_policy_pairwise.csv`; creates no new simulations |

## Reproducing the statistics (not the 29,700-run simulation)

The reported statistics were produced by the committed scripts above on the
frozen grid; re-running the 29,700-run grid is not required to verify the
claims: `pip install -e . && pytest` includes the independent recomputation
of all headline numbers from the committed `results/` artifacts, and
`allocation_deltas.csv` / `allocation_object_pairs.csv` can be re-derived from
`uos_table.csv` + `selection_orders.json` by `a5_analyze.py` on any machine.
