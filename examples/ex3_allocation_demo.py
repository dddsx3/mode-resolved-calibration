"""Example 3 — minimal calibration-allocation demo (runs in ~1 minute).

Synthetic scene, a handful of lights, three policies (mode-aware, E-opt greedy,
random), two budgets. Produces allocation_demo.png here.

The full benchmark version of this experiment (11 real objects, 142 lights,
29,700 reconstructions) lives under results/openillumination/allocation/ —
see docs/EXPERIMENTS.md section 10. This example is illustrative only.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from calibinfo.information.schur import delta_f

HERE = Path(__file__).resolve().parent

# --- synthetic per-light linearized scene ------------------------------------
rng = np.random.default_rng(20260910)
K, P, q = 24, 120, 3                    # lights / pixels / nuisance-per-light
w = rng.uniform(0.5, 2.0, size=(K, P))
s = rng.uniform(0.3, 1.5, size=(K, P))
Bphi = rng.normal(0.0, 0.5, size=(K, P, q))
u = (w * s)[:, :, None] * Bphi
M0 = np.einsum("kpi,kpj->kij", Bphi, w[:, :, None] * Bphi)
lam0 = np.stack([np.diag([20.0, 400.0, 400.0])] * K)
Finf = (w * s ** 2).sum(0)

regime = 10.0                            # selected light: Sigma_phi / 10
budgets = [2, 4, 8]                      # lights recalibrated


def objective_after_update(lam, DF, kind, idx, K_new):
    K_old = np.linalg.inv(M0[idx] + lam[idx])
    upd = (u[idx] @ (K_old - K_new)) @ u[idx].T
    ev = np.linalg.eigvalsh(DF + upd)
    if kind == "e":
        return ev[0]
    pos = ev > ev[-1] * 1e-12
    if kind == "a":
        return -float(np.sum(1.0 / ev[pos]))
    return -float(np.sum(np.log(ev[pos])))


def greedy_ordering(kind):
    """Sequential selection: pick the light whose precision update helps `kind`
    the most, update, repeat (E-opt and A-opt differ only in the objective)."""
    lam = lam0.copy()
    DF = np.diag(Finf) - sum(
        (u[k] @ np.linalg.inv(M0[k] + lam[k])) @ u[k].T for k in range(K))
    order = []
    for _ in range(K):
        best, best_gain = None, None
        for idx in range(K):
            if idx in order:
                continue
            K_old = np.linalg.inv(M0[idx] + lam[idx])
            K_new = np.linalg.inv(M0[idx] + regime * lam[idx])
            upd = (u[idx] @ (K_old - K_new)) @ u[idx].T
            val = objective_after_update(lam, DF + upd, kind, idx, K_new)
            gain = (cur - val) if False else val
            if kind == "a":
                gain = -val
            if best_gain is None or gain > best_gain:
                best, best_gain = idx, gain
        order.append(best)
        lam[best] = regime * lam[best]
        DF = DF + (u[best] @ (np.linalg.inv(M0[best] + lam0[best])
                              - np.linalg.inv(M0[best] + lam[best]))) @ u[best].T
    return order


def mode_aware_ordering():
    lam = lam0.copy()
    DF = np.diag(Finf) - sum(
        (u[k] @ np.linalg.inv(M0[k] + lam[k])) @ u[k].T for k in range(K))
    order = []
    for _ in range(K):
        _w, V = np.linalg.eigh(DF)
        a = V[:, :3]                                # 3 weakest modes
        best, best_g = None, None
        for idx in range(K):
            if idx in order:
                continue
            Kc = np.linalg.inv(M0[idx] + lam[idx])
            g = sum((a.T @ u[idx] @ Kc)[j] @ (lam0[idx] @ (a.T @ u[idx] @ Kc)[j])
                    for j in range(3))
            if best_g is None or g > best_g:
                best, best_g = idx, g
        order.append(best)
        lam[best] = regime * lam[best]
        DF = DF + (u[best] @ (np.linalg.inv(M0[best] + lam0[best])
                              - np.linalg.inv(M0[best] + lam[best]))) @ u[best].T
    return order


orders = {"mode-aware": mode_aware_ordering(),
          "E-opt greedy": greedy_ordering("e"),
          "A-opt greedy": greedy_ordering("a"),
          "random": list(rng.permutation(K))}

# --- "empirical" proxy: sensitivity-weighted reconstruction error -------------
true_err = rng.uniform(0.8, 1.2, size=K)            # per-light error scale
fig, ax = plt.subplots(figsize=(5.4, 3.6))
for label, ordr in orders.items():
    ys = []
    for k in budgets:
        scales = np.ones(K)
        scales[ordr[:k]] = 1.0 / np.sqrt(regime)
        # proxy damage: remaining per-light error scale, summed
        remaining = true_err * (scales ** 2)
        ys.append(float(remaining.sum()))
    ax.plot(budgets, ys, "o-", label=label)
ax.set_xlabel("lights recalibrated (budget)")
ax.set_ylabel("proxy reconstruction error (lower = better)")
ax.set_title("allocation demo: which lights to recalibrate first")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(HERE / "allocation_demo.png", dpi=150)
print("wrote", HERE / "allocation_demo.png")
