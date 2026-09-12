"""Overview schematic for the README (pure illustration, no data).

Regenerate with:  python paper/make_schematic.py
Output:           docs/img/benchmark/overview_schematic.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures" / "overview_schematic.png"

BLUE, ORANGE, GREEN, PINK, GRAY = "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#7F7F7F"

fig, ax = plt.subplots(figsize=(12.4, 4.4))
ax.set_xlim(0, 12.4)
ax.set_ylim(0, 4.4)
ax.axis("off")


def box(x, y, w, h, title, sub, edge, face="#F7F9FB"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.06,rounding_size=0.12",
                                linewidth=1.4, edgecolor=edge, facecolor=face))
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="#1A2A3A")
    ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center",
            fontsize=8.6, color="#4A5A6A")


def arrow(x0, y0, x1, y1, color=GRAY):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", linewidth=1.6,
                                color=color, mutation_scale=16))


# --- main pipeline row -------------------------------------------------------
Y, H, W = 2.55, 1.25, 2.05
XS = [0.25, 2.75, 5.25, 7.75, 10.25]
box(XS[0], Y, W, H, "photometric\nobservations", r"$y = Ax + B\,\delta c + \varepsilon$", GRAY)
box(XS[1], Y, W, H, "per-light Fisher\nblocks", r"$A_k,\ B_k,\ \Lambda_k$", GRAY)
box(XS[2], Y, W, H, "Schur complement\n" + r"$\Delta F(\Lambda)$",
    "calibratable information", BLUE)
box(XS[3], Y, W, H, "retention spectrum\n" + r"$R(\Lambda)$",
    r"per-mode $\rho_j \in [0,1]$", BLUE)
box(XS[4], Y, W, H, "bottom-k modes", "fragile directions\nranked per light", ORANGE)

for x0, x1 in zip(XS[:-1], XS[1:]):
    arrow(x0 + W + 0.06, Y + H / 2, x1 - 0.06, Y + H / 2)

# --- outcomes row ------------------------------------------------------------
box(6.6, 0.25, 2.6, 1.15, "mode-resolved\nvulnerability ranking",
    "which modes die first", GREEN)
box(9.65, 0.25, 2.6, 1.15, "calibration allocation",
    "which lights to recalibrate\nunder a fixed budget", PINK)
arrow(XS[4] + W / 2, Y - 0.03, 7.9, 1.45, GREEN)
arrow(XS[4] + W / 2, Y - 0.03, 10.95, 1.45, PINK)

ax.text(6.2, 3.95, "prediction side", fontsize=9, color=GRAY, ha="center")
ax.text(9.9, 3.95, "decision side", fontsize=9, color=GRAY, ha="center")
ax.plot([5.05, 5.05], [0.15, 4.05], color="#D8DEE6", lw=1, ls=":")
ax.text(2.5, 0.28, "one audited implementation:  src/calibinfo   "
        "(schur · retention · gauge · mode tracking · allocation)",
        fontsize=8.4, color=GRAY)

fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("wrote", OUT)
