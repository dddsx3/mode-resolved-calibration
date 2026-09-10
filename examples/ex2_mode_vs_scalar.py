"""Example 2 — mode-resolved vs scalar criteria under controlled nuisance
corruption on a synthetic scene.

Generates its own data; produces mode_vs_scalar.png here. No downloads.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from calibinfo.metrics.spectral_criteria import (
    logdet_deficit,
    trace_ratio,
)

def retention_min(DF, Finf_diag):
    """Smallest eigenvalue of the whitened retention spectrum R."""
    Fh = np.diag(1.0 / np.sqrt(Finf_diag))
    R = Fh @ DF @ Fh
    return float(np.linalg.eigvalsh(0.5*(R+R.T)).min())
from calibinfo.information.schur import delta_f

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(7)

# --- synthetic scene: P pixels, q nuisance dims, m observations ---------------
P, q = 200, 6
m = P + q
A_base = rng.normal(size=(m, P)) * np.sqrt(np.linspace(4, 0.1, P))[None, :]
B = rng.normal(size=(m, q)) * 0.6
Finf_diag = np.diag(A_base.T @ A_base)

# --- "corruption": inflate the nuisance prior Sigma_c by a level factor -------
levels = np.geomspace(0.1, 100, 12)
records = []
for level in levels:
    Sig_c = np.diag(np.geomspace(1.0, 1.0, q)) * level
    Lam = np.linalg.inv(Sig_c)
    DF, _M, _diag = delta_f(A_base, B, Lam)
    r_min = retention_min(DF, Finf_diag)
    mode_crit = 1.0 / max(r_min, 1e-12)
    # scalar criteria on the same DeltaF
    trace_crit = 1.0 / trace_ratio(DF)
    logdet_crit = 1.0 / max(-logdet_deficit(DF), 1e-12)
    records.append((level, mode_crit, trace_crit, logdet_crit))

fig, ax = plt.subplots(figsize=(5.2, 3.6))
arr = np.array(records)
for col, label, color in [(1, "mode-resolved (weakest mode)", "#d62728"),
                          (2, "trace-based", "#1f77b4"),
                          (3, "log-det-based", "#2ca02c")]:
    ax.plot(arr[:, 0], arr[:, col], "o-", color=color, label=label, ms=4)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"nuisance prior scale (calibration uncertainty)")
ax.set_ylabel(r"predicted damage on the fragile subspace")
ax.set_title("mode-resolved vs scalar criteria")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(HERE / "mode_vs_scalar.png", dpi=150)
print("wrote", HERE / "mode_vs_scalar.png")
