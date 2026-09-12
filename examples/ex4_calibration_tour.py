"""Example 4 — what is this library for? A two-question guided tour.

Scenario: P = 60 "pixels", L = 10 lights, 3 calibration parameters per light
(one log-intensity + two direction angles). The per-pixel albedo x is the
unknown; each light's calibration is uncertain by Sigma_phi.

Question 1 (diagnosis):  which albedo directions lose their information first
                         as calibration uncertainty grows?
Question 2 (decisions):  if you can fully recalibrate only k of the 10 lights,
                         how much information does that buy — and does it
                         matter which k you pick?

Outputs calibration_tour.png in this directory. No downloads, ~seconds.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from calibinfo.allocation.blocks import LightBlocks
from calibinfo.allocation.convex import CertificateProblem, budget_for_k
from calibinfo.information.retention import retention_spectrum
from calibinfo.allocation.blocks import sym_inv

HERE = Path(__file__).resolve().parent

# --- a small per-light linearized scene --------------------------------------
L, P = 10, 15
rng = np.random.default_rng(20260907)
w = rng.uniform(0.6, 1.6, size=(L, P))          # whitening weights
s = rng.uniform(0.2, 1.4, size=(L, P))          # shading per (light, pixel)
Bphi = rng.normal(0.0, 0.5, size=(L, P, 3))     # nuisance Jacobian per light
u = (w * s)[:, :, None] * Bphi                  # whitened design columns
M0 = np.einsum("kpi,kpj->kij", Bphi, w[:, :, None] * Bphi)
lam0 = np.stack([np.diag([0.5, 2.0, 2.0])] * L)   # base precision
# (tuned to be comparable to M0, so recalibration has real value)
finf = (w * s ** 2).sum(0)                      # perfect-calibration F∞ diag


def delta_f(t):
    """DeltaF for per-light precision multipliers t (t_k scales Lambda0k)."""
    DF = np.diag(finf)
    for k in range(L):
        K = sym_inv(M0[k] + t[k] * lam0[k])
        DF = DF - (u[k] @ K) @ u[k].T
    return DF


# --- Q1: which directions are fragile, and do they recover? ------------------
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9))

grid = np.geomspace(1.0, 100.0, 9)              # precision multiplier sweep
tail = np.zeros((len(grid), 6))
for i, lam in enumerate(grid):
    spec = retention_spectrum(delta_f(np.full(L, lam)), np.diag(finf))
    tail[i, :] = np.sort(spec["rho"])[:6]
for j in range(6):
    ax[0].plot(grid, tail[:, j], "o-", ms=3, lw=1.2,
               label=f"mode {j + 1}" if j < 3 else None)
ax[0].set_xscale("log")
ax[0].set_ylim(0, 1.02)
ax[0].set_xlabel("calibration precision multiplier")
ax[0].set_ylabel(r"retention $\rho_j$")
ax[0].set_title("Q1 — which albedo directions are fragile?\n"
                "bottom 6 retention modes vs calibration precision")
ax[0].legend(fontsize=8, loc="lower right")

# --- Q2: what does a recalibration budget buy, certified? --------------------
blocks = LightBlocks(u, M0, lam0, finf, np.ones(L, bool))
prob = CertificateProblem(blocks=blocks, kappa=100.0)
KAP, K_BUD = 100.0, 4                            # refined lights get 100x precision
J_none = prob.J_A(np.ones(L))
J_all = prob.J_A(np.full(L, KAP))
B = budget_for_k(K_BUD, KAP)
fw = prob.frank_wolfe(B, iters=60)
greedy_t = np.ones(L)
for _ in range(K_BUD):
    g = prob.grad_J_A(greedy_t)
    g_masked = np.where(prob.blocks.active, g, np.inf)
    greedy_t[int(np.argmin(g_masked))] = KAP
J_greedy = prob.J_A(greedy_t)
rand_J = []
for _ in range(30):
    t = np.ones(L)
    t[rng.choice(L, K_BUD, replace=False)] = KAP
    rand_J.append(prob.J_A(t))
lower = fw["J_A"] - fw["gap"]

bars = [
    ("no recalibration", J_none, "#bbbbbb"),
    ("random 4 lights", float(np.mean(rand_J)), "#7fb3d5"),
    ("greedy 4 lights", J_greedy, "#2c7fb8"),
    ("all 10 refined", J_all, "#08519c"),
]
for i, (label, val, color) in enumerate(bars):
    ax[1].barh(i, val, color=color, height=0.62)
    ax[1].text(val + 0.01, i, f"{val:.3f}", va="center", fontsize=8)
ax[1].axvline(lower, color="#d95f02", lw=1.6, ls="--")
ax[1].set_xlim(0, J_none * 1.06)
ax[1].set_yticks(range(len(bars)))
ax[1].set_yticklabels([b[0] for b in bars])
ax[1].set_xlabel(r"$J_A = \mathrm{tr}\ \Delta F^{-1}$  (lower = more information)")
cert_range = (J_none - J_all) / J_none * 100
ax[1].set_title(f"Q2 — what does recalibrating 4 lights buy?\n"
                f"certified dynamic range: {cert_range:.0f}%")
fig.tight_layout()
out = HERE / "calibration_tour.png"
fig.savefig(out, dpi=160)
plt.close(fig)
print(f"wrote {out}")
