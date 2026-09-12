#!/usr/bin/env python3
"""make_showcase_figures · README showcase figures.

All data comes from committed artifacts under results/ (certified gap,
negative results) or from the committed library machinery applied to the
public OpenIllumination sample (hero figure; needs raw data locally, never
run in CI). No experiment is re-run; figures visualize committed evidence.

Outputs (committed):
  docs/img/hero.png                        three-panel hero figure
  docs/img/retention_lowrank_structure.png upgraded low-rank spectrum
  docs/img/certified_gap.png               certified optimality-gap chart
  docs/img/negative_results.png            registered negative results

Usage: python scripts/make_showcase_figures.py [--hero-only|--artifacts-only]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

REPO = Path(__file__).resolve().parents[1]
import sys

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

OUT = REPO / "docs" / "img"

BLUE = "#3E7CB8"      # matches the overview schematic palette
BLUE_LT = "#A9C6DE"
ORANGE = "#E07B39"
RED = "#C0504D"
GRAY = "#9AA3AB"
INK = "#3A4450"
CMAP_FRAG = LinearSegmentedColormap.from_list(
    "frag", ["#2E5E8C", "#7FA8C9", "#E8D9C4", "#E8A25C", "#C0562B"])


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRAY)
    ax.tick_params(colors=INK, labelsize=8)
    ax.title.set_color(INK)


# ---------------------------------------------------------------- hero (raw data)
def fig_hero():
    from calibinfo.allocation.blocks import LightBlocks
    from calibinfo.allocation.convex import CertificateProblem
    from calibinfo.datasets.openillumination import load_object
    from calibinfo.models.corruption import CorruptionGenerator
    from experiments.allocation_mode_tail import scalar_greedy_prefix
    from experiments.openillumination_validation import NominalScene

    cfg_name = "obj_03_pumpkin"
    obj_idx = 0
    data_root = "D:/data/OpenIllumination"
    data_meta = "D:/data/OpenIllumination_meta"
    kappa, k_budget = 10.0, 14

    obj = load_object(data_root, cfg_name, data_meta=data_meta)
    scen = NominalScene(obj, np.random.default_rng([20260910, obj_idx]),
                        noise_fit_convention="corrected")
    L, P = scen.I.shape
    finf = scen.Finf_diag

    # allocation problem on the analysis scene (same assembly as the
    # committed certification scripts)
    gen = CorruptionGenerator("joint", 0.5)
    lam0 = np.stack([np.linalg.inv(gen.sigma_phi_diag())] * L)
    blocks = LightBlocks((scen.w * scen.s_hat)[:, :, None] * scen.B_phi,
                         np.einsum("kpi,kpj->kij", scen.B_phi,
                                   scen.w[:, :, None] * scen.B_phi),
                         lam0, finf, np.ones(L, bool))
    prob = CertificateProblem(blocks=blocks, kappa=kappa, route="dense")

    def spectrum(t):
        DF = prob.delta_f(t)
        Finv_h = np.diag(1.0 / np.sqrt(finf))
        return np.linalg.eigvalsh(Finv_h @ DF @ Finv_h)

    rho_before = spectrum(np.ones(L))
    chosen = scalar_greedy_prefix(prob, kappa, k_budget)[k_budget]
    t_after = np.ones(L)
    t_after[chosen] = kappa
    rho_after = spectrum(t_after)

    # panel 1: per-pixel Fisher information (1/Finf) on the object surface,
    # nearest-filled and Gaussian-smoothed for display (the analysis
    # subsample is sparse on the mask)
    from scipy import ndimage
    mask = np.asarray(obj["mask"], bool)
    H, W = mask.shape
    frag = 1.0 / np.maximum(finf, 1e-12)
    lo, hi = np.percentile(frag, 5), np.percentile(frag, 95)
    frag_n = np.clip((frag - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    vals = np.full(int(mask.sum()), np.nan)
    vals[np.asarray(scen.pidx)] = frag_n
    bad = np.isnan(vals)
    idx = ndimage.distance_transform_edt(bad, return_distances=False,
                                         return_indices=True)
    filled = ndimage.gaussian_filter(vals[tuple(idx)], 6)
    img = np.full((H, W), np.nan)
    img[mask] = filled

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.4),
                             gridspec_kw=dict(width_ratios=[1.0, 1.35, 1.35]))
    ax = axes[0]
    im = ax.imshow(img, cmap=CMAP_FRAG)
    ax.contour(mask.astype(float), levels=[0.5], colors=INK, linewidths=0.8)
    ax.set_title("where is information thin?", fontsize=11, color=INK, pad=8)
    ax.text(0.5, -0.07, "per-pixel Fisher information, real object "
            f"({cfg_name.split('_', 1)[1]}); smoothed for display",
            transform=ax.transAxes, ha="center", fontsize=8, color=GRAY)
    ax.axis("off")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=7, colors=INK)
    cb.outline.set_edgecolor(GRAY)
    cb.set_label("fragility", fontsize=8, color=INK)

    ax = axes[1]
    x = np.arange(P)
    ax.plot(x, rho_before, lw=1.3, color=BLUE)
    ax.axhline(1.0, color=GRAY, lw=0.8, ls=":")
    ax.axvspan(0, 4.5, color=ORANGE, alpha=0.18, lw=0)
    ax.annotate("weakest tracked modes",
                xy=(2, float(rho_before[2])), xytext=(P * 0.16, 0.30),
                fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="->", lw=0.8, color=INK))
    ax.set_title("the retention spectrum sorts them", fontsize=11,
                 color=INK, pad=8)
    ax.set_xlabel("mode index (ascending)", fontsize=8.5, color=INK)
    ax.set_ylabel(r"retention $\rho_j$", fontsize=8.5, color=INK)
    ax.set_ylim(0, 1.02)
    _style(ax)

    ax = axes[2]
    nz = 40
    ax.plot(np.arange(nz), rho_before[:nz], lw=1.4, color=GRAY,
            label="before")
    ax.plot(np.arange(nz), rho_after[:nz], lw=1.8, color=ORANGE,
            label=f"after refining {k_budget} lights")
    for j in range(5):
        ax.annotate("", xy=(j, float(rho_after[j])),
                    xytext=(j, float(rho_before[j])),
                    arrowprops=dict(arrowstyle="->", lw=1.0, color=RED,
                                    alpha=0.8))
    ax.set_title("a calibrated budget lifts exactly them", fontsize=11,
                 color=INK, pad=8)
    ax.set_xlabel("mode index (bottom slice)", fontsize=8.5, color=INK)
    ax.set_ylabel(r"retention $\rho_j$", fontsize=8.5, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_ylim(0, 1.02)
    _style(ax)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT / "hero.png", dpi=133)
    plt.close(fig)
    print(f"wrote {OUT / 'hero.png'}")


# ------------------------------------------------- low-rank structure (upgrade)
def _blocks(seed, L, P, q=3):
    rng = np.random.default_rng(seed)
    w = rng.uniform(0.5, 2.0, size=(L, P))
    s = rng.uniform(0.2, 1.5, size=(L, P))
    B = rng.normal(0.0, 0.5, size=(L, P, q))
    u = (w * s)[:, :, None] * B
    M0 = np.einsum("kpi,kpj->kij", B, w[:, :, None] * B)
    lam0 = np.stack([np.diag([30.0, 800.0, 800.0])] * L)
    finf = (w * s ** 2).sum(0)
    return u, M0, lam0, finf


def _assemble(u, M0, lam0, finf, t):
    from calibinfo.allocation.blocks import sym_inv
    DF = np.diag(finf)
    for k in range(u.shape[0]):
        K = sym_inv(M0[k] + t[k] * lam0[k])
        DF = DF - (u[k] @ K) @ u[k].T
    return DF


def fig_lowrank_structure():
    """Dense P x P spectrum (solid) vs the exact 3L x 3L route (dashed)."""
    L, P = 6, 400
    u, M0, lam0, finf = _blocks(20260911, L, P)
    t = np.full(L, 3.0)
    Fh_inv = np.diag(1.0 / np.sqrt(finf))
    DF = _assemble(u, M0, lam0, finf, t)
    R_dense = Fh_inv @ DF @ Fh_inv
    rho_dense = np.sort(np.clip(np.linalg.eigvalsh(R_dense), 0.0, 1.0))
    # exact low-rank route: eigenvalues of I - V^T V (3L x 3L)
    from calibinfo.allocation.blocks import sym_inv
    Ks = [sym_inv(M0[k] + t[k] * lam0[k]) for k in range(L)]
    V = np.empty((P, 3 * L))
    for k in range(L):
        ev, W = np.linalg.eigh(0.5 * (Ks[k] + Ks[k].T))
        Khalf = (W * np.sqrt(np.clip(ev, 0.0, None))) @ W.T
        V[:, 3 * k:3 * k + 3] = Fh_inv @ (u[k] @ Khalf)
    mu = np.linalg.eigvalsh(V.T @ V)
    rho_low = np.sort(np.concatenate([np.full(P - 3 * L, 1.0), 1.0 - mu]))
    dev = float(np.max(np.abs(rho_dense - rho_low)))

    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.axvspan(0, P - 3 * L - 0.5, color="#D8E2EC", alpha=0.55, lw=0)
    ax.plot(np.arange(P), rho_dense, lw=1.6, color=BLUE,
            label=f"exact dense spectrum (P×P eigh, P = {P})")
    ax.plot(np.arange(P), rho_low, lw=1.2, ls="--", color=ORANGE,
            dashes=(4, 2),
            label=f"3L×3L route: eig(I − VᵀV), 3L = {3 * L}")
    ax.axhline(1.0, color=GRAY, lw=0.8, ls=":")
    ax.annotate(f"{P - 3 * L} modes pinned at exactly 1.0  (= P − 3L)",
                xy=(P * 0.30, 1.0), xytext=(P * 0.30, 0.925),
                ha="center", fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="->", lw=0.8, color=INK))
    ax.text(0.985, 0.05,
            f"curves coincide to {dev:.1e}\n(3L×3L eigenproblem instead of P×P)",
            transform=ax.transAxes, ha="right", fontsize=8.5, color=INK)
    ax.set_xlabel("sorted mode index", fontsize=9, color=INK)
    ax.set_ylabel(r"retention eigenvalue $\rho_j$", fontsize=9, color=INK)
    ax.set_title("Exact low-rank structure of the retention spectrum "
                 r"($R = I - VV^{\top}$)", fontsize=11, color=INK)
    ax.set_ylim(0.80, 1.02)
    ax.legend(fontsize=8.5, frameon=False, loc="center left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "retention_lowrank_structure.png", dpi=160)
    plt.close(fig)
    print(f"wrote {OUT / 'retention_lowrank_structure.png'}")


# ------------------------------------------------------------- certified gap
def fig_certified_gap():
    g = json.loads((REPO / "results/certification/certified_gaps.json")
                   .read_text(encoding="utf-8"))
    ks = sorted({r["k"] for r in g["rows"]})

    def med(key):
        out = []
        for k in ks:
            vals = [r[key] / r["J_none"] for r in g["rows"] if r["k"] == k]
            out.append(float(np.median(vals)))
        return np.asarray(out)

    none = med("J_none")
    rnd = med("random_mean")
    grd = med("greedy_J_A")
    lb = med("fw_lower_bound")
    gap_greedy = np.asarray([
        float(np.median([(r["greedy_J_A"] - r["fw_lower_bound"]) / r["J_none"]
                         for r in g["rows"] if r["k"] == k])) for k in ks])
    gap_random = np.asarray([
        float(np.median([(r["random_mean"] - r["fw_lower_bound"]) / r["J_none"]
                         for r in g["rows"] if r["k"] == k])) for k in ks])

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9),
                             gridspec_kw=dict(width_ratios=[1.3, 1.0]))
    ax = axes[0]
    ax.plot(ks, none, lw=1.6, color=INK, marker="o", ms=4, label="no allocation")
    ax.plot(ks, rnd, lw=1.6, color=GRAY, marker="s", ms=4,
            label="random subsets")
    ax.plot(ks, grd, lw=1.6, color=BLUE, marker="o", ms=4, label="J_A-greedy")
    ax.plot(ks, lb, lw=1.6, color=RED, ls="--", label="convex lower bound")
    ax.set_xlabel("calibration budget k (lights refined)", fontsize=9,
                  color=INK)
    ax.set_ylabel(r"$J_A$ / no-allocation (median)", fontsize=9, color=INK)
    ax.set_title("the allocation landscape on 11 real objects",
                 fontsize=11, color=INK)
    ax.set_xticks(ks)
    ax.legend(fontsize=8, frameon=False, loc="center left")
    _style(ax)

    ax = axes[1]
    for gaps, col, mk, lbl in ((gap_random, GRAY, "s", "random $-$ bound"),
                               (gap_greedy, ORANGE, "o", "greedy $-$ bound")):
        pos = [(k, v) for k, v in zip(ks, gaps) if v > 0]
        ax.plot([k for k, _ in pos], [v for _, v in pos], lw=1.6, color=col,
                marker=mk, ms=5, label=lbl)
    ax.set_yscale("log")
    ax.set_ylim(3e-5, 2e-2)
    ax.axhline(2.8e-4, color=RED, lw=0.9, ls=":")
    ax.text(ks[0], 3.6e-4, "greedy certified within 0.011–0.028%",
            fontsize=8, color=INK)
    ax.text(28, 4.5e-4, "at k = 48 both attain\nthe bound (gap = 0)",
            fontsize=7.5, color=GRAY, ha="center")
    ax.set_xlabel("calibration budget k", fontsize=9, color=INK)
    ax.set_ylabel(r"gap / $J_A$(none), log scale", fontsize=9, color=INK)
    ax.set_title("...and how certified the optimality is",
                 fontsize=11, color=INK)
    ax.set_xticks(ks)
    ax.legend(fontsize=8, frameon=False, loc="center left")
    _style(ax)

    fig.tight_layout(w_pad=2.2)
    fig.savefig(OUT / "certified_gap.png", dpi=160)
    plt.close(fig)
    print(f"wrote {OUT / 'certified_gap.png'}")


# ------------------------------------------------------------ negative results
def fig_negative_results():
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))

    # (1) submodularity gamma_min per family/objective
    s = json.loads((REPO / "results/submodularity/submodularity_search.json")
                   .read_text(encoding="utf-8"))
    ax = axes[0]
    keys = list(s["families"].keys())
    vals = [s["families"][k]["gamma_min"] for k in keys]
    colors = [RED if v < 0.999 else BLUE_LT for v in vals]
    ax.bar(range(len(keys)), vals, color=colors, width=0.62)
    ax.axhline(1.0, color=INK, lw=0.9)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([k.replace("/", "\n") for k in keys], fontsize=6.5,
                       color=INK, rotation=30)
    ax.set_ylim(0.68, 1.02)
    ax.set_title("submodularity ratio $\\gamma$\nE-opt violates it "
                 f"($\\gamma_\\min$ = {s['families']['random/E']['gamma_min']:.3f})",
                 fontsize=10, color=INK)
    ax.set_ylabel(r"$\gamma_{\min}$ over instances", fontsize=8.5, color=INK)
    _style(ax)

    # (2) mode-tail three-arm CIs (paired bootstrap, 10 informative cells)
    a = json.loads((REPO / "results/mode_tail/allocation_mode_tail.json")
                   .read_text(encoding="utf-8"))["aggregated"]
    cells = [c for c in a.keys()]
    ax = axes[1]
    for i, (arm, col, m) in enumerate((("targeted", BLUE, "o"),
                                       ("scalar_targeted", ORANGE, "s"))):
        for j, cell in enumerate(cells):
            d = a[cell][arm]
            med, ci = d["delta_median"], d["delta_ci95"]
            x = j + (i - 0.5) * 0.18
            spans0 = ci[0] <= 0 <= ci[1]
            ax.plot([x, x], ci, color=col, lw=1.2,
                    alpha=0.45 if spans0 else 1.0)
            ax.plot(x, med, m, color=col, ms=4,
                    mfc="white" if spans0 else col)
    ax.axhline(0.0, color=INK, lw=0.9)
    ax.set_yscale("symlog", linthresh=100)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels(cells, rotation=60, fontsize=6.5, color=INK)
    ax.set_title("mode-tail intervention: paired CIs\n(open marker = CI "
                 "spans 0; positive = targeted worse)", fontsize=10, color=INK)
    ax.set_ylabel(r"$\Delta$ energy vs random (symlog)", fontsize=8.5,
                  color=INK)
    _style(ax)

    # (3) active-set control: CI spans 0 in both regimes
    r48 = json.loads((REPO / "results/openillumination/allocation/"
                      "allocation_random48_summary.json")
                     .read_text(encoding="utf-8"))
    ax = axes[2]
    for j, reg in enumerate(("10", "100")):
        v = r48["regimes"][reg]["mode_minus_random_active48"]
        ax.plot([j, j], v["ci95"], color=BLUE, lw=2.0)
        ax.plot(j, v["median"], "o", color=BLUE, ms=6, mfc="white")
    ax.axhline(0.0, color=INK, lw=0.9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["regime 10x", "regime 100x"], fontsize=8.5,
                       color=INK)
    ax.set_title("active-set control: advantage over random\nvanishes "
                 "(both CIs span 0)", fontsize=10, color=INK)
    ax.set_ylabel(r"$\Delta$ AUC (mode $-$ random_active48)", fontsize=8.5,
                  color=INK)
    _style(ax)

    fig.tight_layout(w_pad=2.2)
    fig.savefig(OUT / "negative_results.png", dpi=150)
    plt.close(fig)
    print(f"wrote {OUT / 'negative_results.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hero-only", action="store_true")
    ap.add_argument("--artifacts-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.artifacts_only:
        try:
            fig_hero()
        except Exception as exc:                       # raw data absent
            print(f"[hero] skipped: {exc}")
    if not args.hero_only:
        fig_lowrank_structure()
        fig_certified_gap()
        fig_negative_results()


if __name__ == "__main__":
    main()
