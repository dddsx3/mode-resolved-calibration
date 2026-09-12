#!/usr/bin/env python3
"""make_certified_value_curve · D(level) curve for the README/claims registry.

Visualizes results/certification/certified_gaps_levels.json: median
dynamic range with an IQR band (25-75%) and min/max envelope, log-x on
`level`. Pure artifact visualization (no experiment re-run, CI-safe it
only reads the committed JSON).

Output: docs/img/certified_value_curve.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "img"
ART = REPO / "results/certification/certified_gaps_levels.json"

BLUE = "#3E7CB8"
BLUE_LT = "#A9C6DE"
INK = "#3A4450"
GRAY = "#9AA3AB"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRAY)
    ax.tick_params(colors=INK, labelsize=8)
    ax.title.set_color(INK)


def main():
    j = json.loads(ART.read_text(encoding="utf-8"))
    rows = j["rows"]
    levels = sorted({r["level"] for r in rows})
    x = np.array(levels, float)

    med, lo25, hi75, lo_min, hi_max = [], [], [], [], []
    for lv in levels:
        D = np.array([r["D"] for r in rows if r["level"] == lv]) * 100.0
        med.append(np.median(D))
        lo25.append(np.percentile(D, 25))
        hi75.append(np.percentile(D, 75))
        lo_min.append(D.min())
        hi_max.append(D.max())
    med, lo25, hi75, lo_min, hi_max = map(np.array,
                                          (med, lo25, hi75, lo_min, hi_max))

    fig, ax = plt.subplots(figsize=(6.2, 3.8), dpi=150)
    # min/max 带 + IQR 带 + 中位线
    ax.fill_between(x, lo_min, hi_max, color=BLUE_LT, alpha=0.30,
                    label="per-object range")
    ax.fill_between(x, lo25, hi75, color=BLUE, alpha=0.35,
                    label="IQR (25–75%)")
    ax.plot(x, med, color="#1F4E79", lw=2.2, marker="o", ms=4,
            label="median")
    ax.axhline(90.0, color=GRAY, ls="--", lw=1.0)
    ax.text(x[-1], 90.6, "ceiling 1 − 1/κ = 90%", color=GRAY,
            fontsize=7, ha="right", va="bottom")
    ax.axvline(0.5, color=GRAY, ls=":", lw=1.0)
    ax.text(0.5, 4.0, "P-CERT level 0.5\nmedian 62.87%", color=GRAY,
            fontsize=7, ha="center", va="bottom")

    ax.set_xscale("log")
    ax.set_xlabel("calibration-uncertainty operating point  level", color=INK, fontsize=9)
    ax.set_ylabel("certified dynamic range  D ( % )", color=INK, fontsize=9)
    ax.set_title("Calibration-budget value as a curve in level", color=INK, fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_xlim(x[0] * 0.9, x[-1] * 1.1)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False,
              fontsize=7)
    ax.grid(True, which="both", ls=":", lw=0.4, color=GRAY, alpha=0.4)
    _style(ax)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True, parents=True)
    path = OUT / "certified_value_curve.png"
    fig.savefig(path, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()