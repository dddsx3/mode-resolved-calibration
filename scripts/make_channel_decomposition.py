#!/usr/bin/env python3
"""make_channel_decomposition · channel decomposition figure for the README.

Visualizes results/openillumination/channel_decomposition.json: median
D(level) per channel (joint / intensity / direction) with the direction
channel isolated on a secondary axis (it is <= 2%, invisible on the main
scale). Pure artifact visualization. Output: docs/img/channel_decomposition.png
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
ART = REPO / "results/openillumination/channel_decomposition.json"

BLUE = "#3E7CB8"
ORANGE = "#E07B39"
GRAY = "#9AA3AB"
INK = "#3A4450"


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

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)

    med = {}
    for c in j["channels"]:
        med[c] = np.array([np.median([r["D"] for r in rows
                                      if r["channel"] == c and r["level"] == lv])
                           * 100.0 for lv in levels])
    ax.plot(x, med["joint"], color=BLUE, lw=2.2, marker="o", ms=4,
            label="joint (intensity + direction)")
    ax.plot(x, med["intensity"], color=ORANGE, lw=1.8, ls="--", marker="s", ms=3.5,
            label="intensity only")
    ax.axhline(0.0, color=GRAY, lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("calibration-uncertainty operating point  level", color=INK, fontsize=9)
    ax.set_ylabel("median certified dynamic range  D ( % )", color=INK, fontsize=9)
    ax.set_title("Which nuisance channel carries the calibration-budget value?",
                 color=INK, fontsize=10)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", frameon=False, fontsize=7)

    # 方向通道:次轴放大(主轴上不可见)
    ax2 = ax.twinx()
    ax2.plot(x, med["direction"], color="#C0504D", lw=1.6, marker="^", ms=3.5,
             label="direction only (right axis)")
    ax2.axhline(2.0, color="#C0504D", ls=":", lw=1.0)
    ax2.text(x[0], 2.15, "2% ceiling", color="#C0504D", fontsize=7, ha="left")
    ax2.set_ylabel("direction-only D ( % )", color="#C0504D", fontsize=8)
    ax2.tick_params(axis="y", colors="#C0504D", labelsize=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color("#C0504D")
    ax2.set_ylim(0, 3.0)
    ax2.legend(loc="lower right", frameon=False, fontsize=7, labelcolor="#C0504D")

    ax.grid(True, which="both", ls=":", lw=0.4, color=GRAY, alpha=0.4)
    _style(ax)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True, parents=True)
    path = OUT / "channel_decomposition.png"
    fig.savefig(path, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()