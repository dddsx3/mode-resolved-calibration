#!/usr/bin/env python3
"""make_direction_figures · README showcase figures for the certification
research direction (M0). Both figures are generated from small synthetic
per-light systems (no raw data) by the same constructions pinned in
tests/test_math_foundations.py — reproducible, honest, no benchmark claims.

Outputs (committed):
  docs/img/retention_lowrank_structure.png   spectrum = exact flat head (P−3L
                                             modes at ρ=1) + computable tail
                                             from a 3L×3L eigenproblem
  docs/img/convexity_midpoint.png            midpoint convexity of J_E over
                                             random directions (all points on
                                             or below the diagonal)

Usage: python scripts/make_direction_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from calibinfo.allocation.blocks import sym_inv

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "img"


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
    DF = np.diag(finf)
    for k in range(u.shape[0]):
        K = sym_inv(M0[k] + t[k] * lam0[k])
        DF = DF - (u[k] @ K) @ u[k].T
    return DF


def fig_lowrank_structure():
    """Spectrum of the retention operator: exact flat head + computable tail."""
    L, P = 6, 400
    u, M0, lam0, finf = _blocks(20260911, L, P)
    t = np.full(L, 3.0)
    Fh_inv = np.diag(1.0 / np.sqrt(finf))
    Ks = [sym_inv(M0[k] + t[k] * lam0[k]) for k in range(L)]
    V = np.empty((P, 3 * L))
    for k in range(L):
        ev, W = np.linalg.eigh(0.5 * (Ks[k] + Ks[k].T))
        Khalf = (W * np.sqrt(np.clip(ev, 0.0, None))) @ W.T
        V[:, 3 * k:3 * k + 3] = Fh_inv @ (u[k] @ Khalf)
    mu = np.linalg.eigvalsh(V.T @ V)
    rho_low = np.sort(np.concatenate([np.full(P - 3 * L, 1.0), 1.0 - mu]))

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.plot(np.arange(P), rho_low, lw=1.2, color="#1f77b4")
    ax.axhline(1.0, color="#888", lw=0.8, ls=":")
    ax.axvspan(0, P - 3 * L - 0.5, color="#1f77b4", alpha=0.08)
    ax.annotate(f"exactly 1.0  ({P - 3 * L} modes = P − 3L)",
                xy=(P * 0.18, 1.0), xytext=(P * 0.18, 0.955),
                ha="center", fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#444"))
    ax.annotate(f"computable tail: eig(I − VᵀV)\nof a {3 * L}×{3 * L} matrix",
                xy=(P - 3 * L + 3, 0.985), xytext=(P * 0.55, 0.86),
                fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#444"))
    ax.set_xlabel("sorted mode index")
    ax.set_ylabel(r"retention eigenvalue $\rho_j$")
    ax.set_title(f"Exact low-rank structure of the retention spectrum\n"
                 f"(synthetic per-light system, P={P}, L={L}; "
                 r"$R = I - VV^{\top}$)")
    ax.set_ylim(0.80, 1.02)
    fig.tight_layout()
    fig.savefig(OUT / "retention_lowrank_structure.png", dpi=160)
    plt.close(fig)


def _J_E(DF):
    return -float(np.linalg.eigvalsh(DF)[0])


def fig_convexity_midpoint():
    """Midpoint convexity of J_E over random directions (400 pairs)."""
    L = 4
    u, M0, lam0, finf = _blocks(20260912, L, 40)
    rng = np.random.default_rng(20260913)
    xs, ys = [], []
    worst = -np.inf
    for _ in range(400):
        t1 = rng.uniform(1.0, 5.0, size=L)
        t2 = rng.uniform(1.0, 5.0, size=L)
        avg = 0.5 * (_J_E(_assemble(u, M0, lam0, finf, t1))
                     + _J_E(_assemble(u, M0, lam0, finf, t2)))
        mid = _J_E(_assemble(u, M0, lam0, finf, 0.5 * (t1 + t2)))
        xs.append(avg)
        ys.append(mid)
        worst = max(worst, mid - avg)
    xs, ys = np.asarray(xs), np.asarray(ys)
    lo, hi = float(min(xs.min(), ys.min())), float(max(xs.max(), ys.max()))

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([lo, hi], [lo, hi], "--", lw=1.0, color="#888",
            label="midpoint = average")
    ax.scatter(xs, ys, s=12, color="#1f77b4", alpha=0.65,
               label="J_E( mid(t₁,t₂) )")
    ax.set_xlabel("½ ( J_E(t₁) + J_E(t₂) )")
    ax.set_ylabel("J_E( ½(t₁+t₂) )")
    ax.set_title("Midpoint convexity of the certificate functional\n"
                 r"$J_E = -\lambda_{\min}(\Delta F(t))$:"
                 " 400/400 random pairs, max excess "
                 f"{max(worst, 0.0):.1e}")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "convexity_midpoint.png", dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_lowrank_structure()
    fig_convexity_midpoint()
    print(f"wrote {OUT / 'retention_lowrank_structure.png'}")
    print(f"wrote {OUT / 'convexity_midpoint.png'}")


if __name__ == "__main__":
    main()
