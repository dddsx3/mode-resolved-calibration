"""Figures for the potential-assessment report. ClawsGO Science Harness."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

plt.rcParams.update({
    "font.family": "Noto Sans CJK SC",
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "figure.dpi": 200,
})
# Okabe-Ito
OK = dict(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00",
          purple="#CC79A7", sky="#56B4E9", yellow="#F0E442", grey="#666666")

R = "repo/results/openillumination"

# =============================================================== FIG 1
mr = pd.read_csv(f"{R}/mode_ranking.csv")
rows = []
for (o, l), g in mr.groupby(["object_id", "level"]):
    g = g.sort_values("mode_id")
    rows.append(dict(
        obj=o, level=l,
        r_pred=spearmanr(g.pred_deg.values, g.emp_deg.values).statistic,
        r_idx=spearmanr(-g.mode_id.values, g.emp_deg.values).statistic))
Rd = pd.DataFrame(rows)

fig, ax = plt.subplots(1, 3, figsize=(9.6, 2.9))

a = ax[0]
for (o, l), g in mr.groupby(["object_id", "level"]):
    g = g.sort_values("mode_id")
    a.plot(g.mode_id, g.pred_deg, color=OK["blue"], alpha=0.18, lw=0.8)
a.set_yscale("log")
a.set_xlabel("模式序号 $j$（按构造升序）")
a.set_ylabel(r"预测退化 $\hat{d}_j=1/\rho_j$")
a.set_title("(a) 预测量在模式序号上\n66/66 cell 严格单调", fontsize=9)
a.set_xticks(range(5))

a = ax[1]
jit = np.random.default_rng(0).normal(0, 0.012, len(Rd))
a.scatter(Rd.r_idx + jit, Rd.r_pred + jit, s=16, color=OK["red"],
          alpha=0.65, edgecolor="none")
a.plot([0.2, 1.05], [0.2, 1.05], color=OK["grey"], lw=0.8, ls="--", zorder=0)
a.set_xlabel(r"平凡基线 $\mathrm{Spearman}(-j,\;d_j^{\mathrm{emp}})$")
a.set_ylabel(r"论文头条 $R_A=\mathrm{Spearman}(\hat{d}_j, d_j^{\mathrm{emp}})$")
a.set_title("(b) 两者逐 cell 完全相等\n" r"$\max|\Delta|=0$，66/66 cell", fontsize=9)
a.text(0.42, 0.32, "全部点落在对角线", fontsize=8, color=OK["grey"])

a = ax[2]
ratio = (mr.emp_deg / mr.pred_deg)
bp = a.boxplot([ratio[mr.mode_id == j] for j in range(5)], widths=0.6,
               patch_artist=True, medianprops=dict(color="black", lw=1.2),
               flierprops=dict(ms=2, mfc=OK["grey"], mec="none", alpha=.5))
for b in bp["boxes"]:
    b.set(facecolor=OK["sky"], alpha=0.55, edgecolor=OK["blue"], lw=0.8)
a.axhline(1.0, color=OK["red"], lw=1.1, ls="--")
a.set_yscale("log")
a.set_xlabel("模式序号 $j$")
a.set_ylabel(r"$d_j^{\mathrm{emp}} / \hat{d}_j$")
a.set_title("(c) 幅值预测偏离 1 达两个数量级\n（红线 = 定理预言的 1）", fontsize=9)
a.set_xticklabels(range(5))
fig.tight_layout()
fig.savefig("fig1_headline.pdf", bbox_inches="tight")
fig.savefig("fig1_headline.png", bbox_inches="tight")
print("fig1 done; median emp/pred =", round(float(ratio.median()), 1))

# =============================================================== FIG 2
E = json.load(open("probe/e11.json"))
cert = pd.DataFrame(E["certified"])
fig, ax = plt.subplots(1, 3, figsize=(9.6, 2.9))

a = ax[0]
a.plot(cert.k, cert.random, "o-", ms=4, color=OK["grey"], label="random（中位）")
a.plot(cert.k, cert["mode"], "s-", ms=4, color=OK["orange"], label="mode-aware 启发式")
a.plot(cert.k, cert.greedy, "^-", ms=4, color=OK["blue"], label="greedy A-opt")
a.plot(cert.k, cert.lb, "-", lw=2.2, color=OK["red"], label="凸松弛认证下界")
a.axhline(E["scale"]["f_all"], color=OK["green"], lw=1, ls=":", label="全部标定（下限）")
a.set_xlabel("标定预算 $k$（精化的灯数）")
a.set_ylabel(r"$\mathrm{tr}\,\Delta F^{-1}$")
a.set_title("(a) 认证最优间隙：所有策略\n都挤在 0.3% 带内", fontsize=9)
a.legend(fontsize=6.5, frameon=False, loc="upper right")

a = ax[1]
w = 0.36
x = np.arange(len(cert))
a.bar(x - w / 2, cert.gap_g, w, color=OK["blue"], label="greedy 距认证最优")
a.bar(x + w / 2, cert.gain_g, w, color=OK["purple"], label="greedy 相对 random 增益")
a.set_xticks(x); a.set_xticklabels(cert.k)
a.set_xlabel("标定预算 $k$")
a.set_ylabel("相对幅度 (%)")
a.set_title("(b) 可优化空间 < 策略间差异\n→ null 结果是结构性的", fontsize=9)
a.legend(fontsize=6.5, frameon=False)

a = ax[2]
o = E["objectives"]
names = ["A_full", "D_opt", "A_weak5", "E_opt"]
labels = [r"$\mathrm{tr}\Delta F^{-1}$" + "\n(全谱)", r"$-\log\det$",
          r"$\sum_{j\leq 5}1/\rho_j$" + "\n(弱模式)", r"$1/\rho_{\min}$"]
vals = [o[n]["range_pct"] for n in names]
cols = [OK["grey"], OK["sky"], OK["orange"], OK["red"]]
a.barh(range(4), vals, color=cols)
a.set_yticks(range(4)); a.set_yticklabels(labels, fontsize=7.5)
for i, v in enumerate(vals):
    a.text(v + 1.5, i, f"{v:.1f}%", va="center", fontsize=8)
a.set_xlim(0, 105)
a.set_xlabel("总动态范围 (%)\n（全不标定 → 全标定）")
a.set_title("(c) 目标泛函决定效应可检出性\n相差 4 倍", fontsize=9)
fig.tight_layout()
fig.savefig("fig2_convex.pdf", bbox_inches="tight")
fig.savefig("fig2_convex.png", bbox_inches="tight")
print("fig2 done")
