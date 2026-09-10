"""P0-2 post-hoc paired policy comparison (expert adjudication 2026-09-09).

Question the frozen grade does NOT answer: is mode-aware *better than* the
classical E/A/D-optimal greedy baselines?  Uses the identical Uos and the
identical shared-resample bootstrap machinery as a5_analyze.py
(np.random.default_rng(20260910), B=10000, same draw-generation order), but on
paired differences  Delta = U_o(mode_aware) - U_o(policy)  with policy in
{e_opt, a_opt, d_opt}.  Delta > 0 means mode-aware is WORSE (higher AUC).

This is a post-hoc analysis: it creates no new simulations and touches no
frozen artifact; it reads uos_table.csv exactly as A5 did.

Writes: allocation_policy_pairwise.csv  (analysis_status = posthoc_paired_comparison)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

BUDGETS = [0.1, 0.2, 0.4, 0.6, 0.8]
REGIMES = [10, 100]
CLASSICAL = ["e_opt", "a_opt", "d_opt"]
BOOTSTRAP_SEED = 20260910
B = 10000


def trapezoid_auc(Es):
    Es = np.asarray(Es, float)
    bs = np.asarray(BUDGETS, float)
    return float(np.sum((Es[:-1] + Es[1:]) / 2.0 * np.diff(bs)) / (0.8 - 0.1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    args = ap.parse_args()
    out = Path(args.results)
    table = {}
    with open(out / "uos_table.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            table[(r["object"], int(r["regime"]), r["policy"],
                   float(r["budget"]))] = float(r["E_osb"])

    lines = ["regime,comparison,median_delta,ci_lo,ci_hi,worse_of_11,"
             "analysis_status"]
    for regime in REGIMES:
        objs = sorted({o for (o, r, _p, _b) in table if r == regime})
        assert len(objs) == 11
        Uos = {o: {p: trapezoid_auc(
            [table[(o, regime, p, b)] for b in BUDGETS])
            for p in ["mode_aware", "random"] + CLASSICAL} for o in objs}
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        n = len(objs)
        draws = [rng.integers(0, n, n) for _ in range(B)]   # shared resamples
        for p in CLASSICAL:
            d = np.array([Uos[o]["mode_aware"] - Uos[o][p] for o in objs])
            boots = [float(np.median(d[idx])) for idx in draws]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            worse = int((d > 0).sum())
            comp = f"mode_aware_minus_{p}"
            lines.append(f"{regime},{comp},{np.median(d):.6g},{lo:.6g},{hi:.6g},"
                         f"{worse}/11,posthoc_paired_comparison")
            print(f"[pairwise] regime {regime} {comp}: median "
                  f"{np.median(d):+.4g} CI [{lo:+.4g}, {hi:+.4g}] "
                  f"worse {worse}/11")
    (out / "allocation_policy_pairwise.csv").write_bytes(
        ("\n".join(lines) + "\n").encode("utf-8"))
    print("[pairwise] wrote allocation_policy_pairwise.csv")


if __name__ == "__main__":
    main()
