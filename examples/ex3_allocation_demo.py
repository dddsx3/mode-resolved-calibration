"""Example 3 — minimal calibration-allocation demo (runs in ~1 minute).

Builds a synthetic per-light linearized scene and drives the library's frozen
selection policies (`calibinfo.allocation.policies`) to decide which lights to
recalibrate first under a fixed budget: mode-aware, E/A/D-optimal greedy, and
random. Budgets are prefixes of each policy's full ordering (the sequential
selection frame); the curve is the *variance of the weakest mode* of DeltaF
(1/lambda_min) after each budget's precision updates — lower is better.

The full benchmark version of this experiment (11 real objects, 142 lights,
29,700 reconstructions) lives under results/openillumination/allocation/ —
see docs/EXPERIMENTS.md. This example is illustrative only: it is a tiny
synthetic scene, and its "error" curve is a prediction-side proxy, not a
reconstruction error.

No policy logic lives in this file — every ordering comes from the library.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from calibinfo.allocation.blocks import LightBlocks, sym_inv
from calibinfo.allocation.policies import (
    SelectionState,
    budget_scales,
    select_ordering,
    select_ordering_mode_aware,
)

HERE = Path(__file__).resolve().parent

# --- synthetic per-light linearized scene ------------------------------------
rng = np.random.default_rng(20260910)
K, P, q = 24, 120, 3                    # lights / pixels / nuisance-per-light
w = rng.uniform(0.5, 2.0, size=(K, P))
s = rng.uniform(0.3, 1.5, size=(K, P))
Bphi = rng.normal(0.0, 0.5, size=(K, P, q))
u = (w * s)[:, :, None] * Bphi          # whitened per-light design columns
M0 = np.einsum("kpi,kpj->kij", Bphi, w[:, :, None] * Bphi)
lam0 = np.stack([np.diag([20.0, 400.0, 400.0])] * K)
finf = (w * s ** 2).sum(0)

state = SelectionState(u=u, M0=M0, lam0=lam0, finf=finf,
                       active=np.ones(K, dtype=bool))
regime = 10.0                            # recalibrated light: Sigma_phi / 10
budgets = [2, 4, 8, 16]

# --- policy orderings (all from the library, one call each) -------------------
orders = {"mode-aware": select_ordering_mode_aware(state, regime)[0]}
for pol in ("e_opt", "a_opt", "d_opt"):
    orders[pol] = select_ordering(state, pol, regime)[0]
orders["random"] = select_ordering(
    state, "random", regime, rng=np.random.default_rng(20260911))[0]

print("first 8 picks per policy:")
for label, ordr in orders.items():
    print(f"  {label:14s} {ordr[:8]}")


# --- prediction-side proxy: weakest-mode variance after the budget ------------
def weakest_mode_variance(ordr, k):
    """Apply the budget's precision updates (Lambda_l -> regime * Lambda_l for
    the first k lights of the ordering) and return 1/lambda_min(DeltaF)."""
    scales = budget_scales(ordr, k, regime)
    lam = lam0 / (scales ** 2)[:, None, None]      # scale^2 = 1/regime if picked
    blocks = LightBlocks(u, M0, lam, finf, state.active)
    DF = blocks.assemble(lam)
    ev = np.linalg.eigvalsh(DF)
    return 1.0 / max(ev[0], 1e-300)


fig, ax = plt.subplots(figsize=(5.6, 3.8))
for label, ordr in orders.items():
    ys = [weakest_mode_variance(ordr, k) for k in budgets]
    ax.plot(budgets, ys, "o-", label=label)
ax.set_xlabel("lights recalibrated (budget)")
ax.set_ylabel("weakest-mode variance $1/\\lambda_{\\min}(\\Delta F)$")
ax.set_title("allocation demo: which lights to recalibrate first\n"
             "(lower weakest-mode variance = more usable fragile modes)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(HERE / "allocation_demo.png", dpi=150)
print("wrote", HERE / "allocation_demo.png")
