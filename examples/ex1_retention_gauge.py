"""Example 1 — retention spectrum and gauge response on a synthetic scene.

Generates its own synthetic data; produces retention_spectrum.png and
gauge_response.png in this directory. No downloads, no benchmark data.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from calibinfo.information.gauge import gauge_response
from calibinfo.information.retention import retention_spectrum
from calibinfo.information.schur import delta_f

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(20260907)

# --- synthetic linearized scene with structured nuisance ---------------------
P, q = 300, 9                       # pixels / nuisance dims
m = P + q                           # observations = pixels + nuisance rows
A = rng.normal(size=(m, P)) * np.sqrt(np.linspace(5, 0.05, P))[None, :]
B = rng.normal(size=(m, q)) * 0.5
Sig_c = np.diag(np.geomspace(10.0, 0.1, q))
Finf_diag = np.diag(A.T @ A)

# --- retention spectrum across the calibration path --------------------------
lams = np.geomspace(1e-3, 1e3, 40)
lams = np.concatenate([[0.0], lams])
bottom = np.zeros((len(lams), 5))
for i, lam in enumerate(lams):
    L = lam * np.eye(q)
    DF, _M, _diag = delta_f(A, B, L)
    spec = retention_spectrum(DF, np.diag(Finf_diag))
    bottom[i, :len(spec["rho"])] = np.sort(spec["rho"])[:5]

# --- gauge response on the nuisance manifold ---------------------------------
cbar = rng.normal(size=q)
gauge = gauge_response(B, cbar, lams[1:])

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for j in range(5):
    axes[0].plot(lams, bottom[:, j], lw=1)
axes[0].set_xscale("symlog", linthresh=1e-3)
axes[0].set_xlabel(r"calibration precision scale $\lambda$")
axes[0].set_ylabel(r"retention eigenvalues $\rho_j(\Lambda)$")
axes[0].set_title("retention spectrum (5 weakest modes)")
axes[1].plot(lams[1:], gauge["exact"], lw=2)
axes[1].plot(lams[1:], gauge["slope"] * lams[1:], "--", lw=1, label=r"slope $\sum\alpha_i^2$")
axes[1].axhline(gauge["saturation"], color="gray", ls=":", label=r"saturation $\|B\bar c\|^2$")
axes[1].set_xscale("log")
axes[1].set_xlabel(r"$\lambda$")
axes[1].set_ylabel(r"$\bar a^\top \Delta F(\lambda)\, \bar a$")
axes[1].set_title("gauge spectral response (closed form)")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(HERE / "retention_gauge.png", dpi=150)
print("wrote", HERE / "retention_gauge.png")
