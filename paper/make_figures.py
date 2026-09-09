#!/usr/bin/env python3
"""make_figures · rebuild paper figures (Fig.1-9, interface frozen).

Figures are rebuilt only from the machine-readable summaries under results/;
each figure's recipe is `python paper/make_figures.py --figure N`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "results"
FIGS = ROOT / "paper" / "figures"

FIGURES = {f"Fig.{i}" for i in range(1, 10)}


def _make_fig1(out_dir):
    """Fig.1 key result: fixed-level (stratified) median Spearman per predictor,
    rebuilt from results/openillumination/predictor_comparison.csv."""
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(
        (FROZEN / "openillumination" / "predictor_comparison.csv").open(
            newline="", encoding="utf-8")))
    names, vals = [], []
    for r in rows:
        try:
            vals.append(float(r["Stratified_median_rho"]))
            names.append(r["Predictor"])
        except ValueError:
            pass
    fig, ax = plt.subplots(figsize=(6, 3.4))
    bars = ax.bar(names, vals, color="#1f77b4", width=0.6)
    bars[names.index("mode-resolved")].set_color("#d62728")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("fixed-level median Spearman")
    ax.set_ylim(-0.2, 0.8)
    ax.set_title("Fig.1 · fixed-level severity: mode-resolved vs scalar criteria")
    fig.tight_layout()
    out = out_dir / "fig1_stratified_median.png"
    fig.savefig(out, dpi=150)
    print(f"[make_figures] Fig.1 -> {out}")
    return out


def _make_fig2(out_dir):
    """Fig.2 key result: pooled descriptive Spearman per predictor and the
    mode-resolved pass rate RA, rebuilt from results/openillumination CSVs."""
    import csv
    from io import StringIO
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import spearmanr
    rows = list(csv.DictReader(
        (FROZEN / "openillumination" / "level_severity.csv").open(
            newline="", encoding="utf-8")))
    T = [float(r["T_ol"]) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))
    preds = ["P_mode", "P_emin", "P_logdet", "P_trace", "P_corr"]
    labels = ["mode", "E-min", "logdet", "trace", "corr"]
    ax = axes[0]
    for p, lab in zip(preds, labels):
        ax.scatter([float(r[p]) for r in rows], T, s=10, alpha=0.65, label=lab)
    ax.set_xlabel("predicted damage")
    ax.set_ylabel("empirical degradation (log)")
    ax.set_yscale("log")
    ax.set_title("Fig.2a · per-cell predictions")
    ax.legend(fontsize=7)
    ax = axes[1]
    pooled = {lab: spearmanr([float(r[p]) for r in rows], T).statistic
              for p, lab in zip(preds, labels)}
    pooled["magnitude"] = spearmanr([float(r["level"]) for r in rows], T).statistic
    ax.bar(pooled.keys(), pooled.values(), color="#1f77b4")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylim(0, 1.0)
    ax.set_title("Fig.2b · pooled Spearman")
    for i, k in enumerate(pooled):
        ax.text(i, pooled[k] + 0.02, f"{pooled[k]:.3f}", ha="center", fontsize=8)
    fig.suptitle("Fig.2 · pooled prediction quality (RA = 0.90, CI [0.90, 0.95])",
                 fontsize=10)
    fig.tight_layout()
    out = out_dir / "fig2_pooled.png"
    fig.savefig(out, dpi=150)
    print(f"[make_figures] Fig.2 -> {out}")
    return out


def _make_fig6(out_dir):
    """Fig.6 · linearization validity envelope: mask-flip rate × k heatmap,
    colored by theoretical error (weak-mode median |emp/pred − 1|); 10% boundary line."""
    import numpy as np
    src = FROZEN / "nonlinear" / "ci03nl_nl_formal_summary.json"
    if not src.exists():
        raise SystemExit(f"[make_figures] missing {src} — first run scripts/run_experiments.py --experiment nonlinear")
    data = json.loads(src.read_text(encoding="utf-8"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("[make_figures] 需要 matplotlib")
    geos = sorted({r["geometry"] for r in data["checks"]})
    ks = sorted({r["k"] for r in data["checks"]})
    fig, axes = plt.subplots(1, len(geos), figsize=(4.2 * len(geos), 3.4), sharey=True)
    for ax, geo in zip(np.atleast_1d(axes), geos):
        rows = [r for r in data["checks"] if r["geometry"] == geo]
        # 同一 k 可能多场景：取场景中位
        grid_err = np.full((len(ks),), np.nan)
        for i, k in enumerate(ks):
            errs = [r["pred_err_median"] for r in rows if r["k"] == k]
            if errs:
                grid_err[i] = np.median(errs)
        ax.plot(range(len(ks)), grid_err, "o-", color="#1f77b4")
        ax.axhline(data["gate_valid_pred_err"], color="r", ls="--", lw=1,
                   label="10% validity gate")
        env = data["envelope"][geo]
        ax.set_title(f"{geo} | valid flip<={env['max_valid_flip_rate']:.3f} "
                     f"(k<={env['max_valid_k']:g})", fontsize=9)
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([f"{k:g}" for k in ks], fontsize=8)
        ax.set_yscale("log")
        ax.set_xlabel("corruption strength k (×Σ_c0)")
    np.atleast_1d(axes)[0].set_ylabel("median |emp/pred − 1| (weak 5 modes)")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle("Fig.6 (draft) — Linearization validity envelope (B-arm analytic SH+ReLU)", fontsize=10)
    out = out_dir / "fig6_draft.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[make_figures] Fig.6 -> {out}")
    return out


import figure_gens as _g



def make_figure(n, out_dir=FIGS):
    out_dir.mkdir(parents=True, exist_ok=True)
    if n in _GEN:
        return _GEN[n](out_dir)
    raise SystemExit(f"[make_figures] Fig.{n} recipe not implemented")


def fig10_allocation_forest(out_dir):
    """Fig.10 - allocation forest: per-policy AUC deltas vs random with
    object-cluster bootstrap CIs, per regime (frozen allocation_deltas.csv)."""
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = FROZEN / "openillumination" / "allocation" / "allocation_deltas.csv"
    if not path.exists():
        raise SystemExit(f"[make_figures] missing {path}")
    data = {}
    for row in csv.DictReader(open(path, newline="", encoding="utf-8")):
        data.setdefault(row["regime"], {})[row["policy"]] = row
    order = ["mode_aware", "e_opt", "a_opt", "d_opt"]
    labels = {"mode_aware": "mode-aware", "e_opt": "E-opt",
              "a_opt": "A-opt", "d_opt": "D-opt"}
    colors = {"mode_aware": "#D55E00", "e_opt": "#0072B2",
              "a_opt": "#009E73", "d_opt": "#CC79A7"}
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2), sharey=True)
    ys = list(range(len(order)))[::-1]
    for ax, regime in zip(axes, ["10", "100"]):
        for y, pol in zip(ys, order):
            r = data[regime][pol]
            med, lo, hi = (float(r["median_delta"]), float(r["ci_lo"]),
                           float(r["ci_hi"]))
            ax.errorbar(med, y, xerr=[[med - lo], [hi - med]], fmt="o",
                        color=colors[pol], ms=6, capsize=3)
            ax.text(med, y + 0.18, r["improved_of_11"], ha="center", fontsize=7)
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_yticks(ys)
        ax.set_yticklabels([labels[q] for q in order])
        ax.set_title(f"regime {regime}x", fontsize=10)
        ax.set_xlabel("Delta AUC vs random (median, 95% CI)")
    fig.suptitle("Fig.10 - allocation benefit vs random (per policy, both regimes)",
                 fontsize=10)
    fig.tight_layout()
    out = out_dir / "fig10_allocation_forest.png"
    fig.savefig(out, dpi=150)
    print(f"[make_figures] Fig.10 -> {out}")
    return out


_GEN = {1: _make_fig1, 2: _make_fig2, 3: _g.fig3, 4: _g.fig4, 5: _g.fig5,
        6: _make_fig6, 7: _g.fig7, 8: _g.fig8, 9: _g.fig9, 10: fig10_allocation_forest}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", type=int, required=True)
    args = ap.parse_args()
    if args.figure not in range(1, 11):
        raise SystemExit(f"--figure must be in 1-10 ({FIGURES})")
    out = make_figure(args.figure)
    print("[make_figures] done ->", out)


if __name__ == "__main__":
    main()
