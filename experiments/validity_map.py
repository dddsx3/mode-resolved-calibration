"""P-VALIDITY-MAP · E2 model-validity map (M2).

"Fisher 线性化在**哪个 regime** 准"——不是证明它永远准。只读两个已提交
产物,按 (object, level) cell 补算四个效度端点,联合线性化曲率指标出
二维图:

  x 轴  corruption level ℓ(6 档 factorial 网格)
  y 轴  excess_over_scaling q(ℓ)(linearization_radius 的现成曲率指标:
        偏离一阶标度的倍数;>1 = 超线性 = 线性化开始失真)
  四个面板(每 cell 一点,66 cells):
    1. rank correlation   Spearman(pred_deg, emp_deg)
    2. sign agreement     Kendall τ(pred_deg, emp_deg)——排序的逐对符号
                          一致率(concordant pairs / total pairs)
    3. absolute ratio     median_j emp_deg_j / pred_deg_j(幅值失配)
    4. ratio spread       IQR of log10(emp/pred) across the 5 modes
                          (幅值失配的跨模式离散度)

来源(只读,不重算):
  results/openillumination/correctness/mf0_factorial_rows_D.json
    (corrected arm-D 的逐 cell pred_deg/emp_deg;66 cells)
  results/magnitude/linearization_radius.json
    (逐 (object, level) 的 excess_over_scaling;corrected)

输出:results/magnitude/validity_map.json + docs/img/validity_map.png

用法:
  python experiments/validity_map.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

REPO = Path(__file__).resolve().parents[1]

ROWS = REPO / "results/openillumination/correctness/mf0_factorial_rows_D.json"
RADIUS = REPO / "results/magnitude/linearization_radius.json"
OUT = REPO / "results/magnitude/validity_map.json"
IMG = REPO / "docs/img/validity_map.png"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, cwd=str(REPO)).stdout.strip()


def _cell_metrics(pred, emp):
    """四端点 + 逐模式比(全部从 pred_deg/emp_deg 直接算,无重算实验)。"""
    s = float(spearmanr(pred, emp).statistic)
    k = float(kendalltau(pred, emp).statistic)
    ratios = emp / np.maximum(pred, 1e-300)
    logr = np.log10(np.maximum(ratios, 1e-300))
    return dict(
        spearman=round(s, 6),
        kendall_sign_agreement=round(k, 6),
        ratio_median=round(float(np.median(ratios)), 6),
        ratio_iqr_log10=round(float(np.percentile(logr, 75)
                                    - np.percentile(logr, 25)), 6),
        ratio_per_mode=[round(float(x), 6) for x in ratios])


def run(rows_path=ROWS, radius_path=RADIUS, out_path=OUT, img_path=IMG):
    _rows_doc = json.loads(Path(rows_path).read_text(encoding="utf-8"))
    rows = _rows_doc["rows"] if isinstance(_rows_doc, dict) else _rows_doc
    rad = json.loads(Path(radius_path).read_text(encoding="utf-8"))
    excess = {}
    for obj_name, o in rad["objects"].items():
        for r in o["rows"]:
            excess[(obj_name, float(r["level"]))] = float(r["excess_over_scaling"])

    cells = []
    for r in rows:
        pred = np.asarray(r["pred_deg"], float)
        emp = np.asarray(r["emp_deg"], float)
        m = _cell_metrics(pred, emp)
        q = excess[(r["object"], float(r["level"]))]
        cells.append(dict(object=r["object"], level=float(r["level"]),
                          excess_over_scaling=round(q, 6), **m))

    # ---- 汇总(把曲率当连续坐标,按曲率分箱看端点走势) ----
    qs = np.array([c["excess_over_scaling"] for c in cells])
    edges = [0.0, 0.75, 1.0, 1.5, np.inf]      # 一阶内 / 边界 / 超线性 / 强超线性
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [c for c in cells if lo <= c["excess_over_scaling"] < hi]
        if not sel:
            continue
        bins.append(dict(
            excess_range=[round(lo, 3), round(hi, 3)], n_cells=len(sel),
            spearman_median=round(float(np.median(
                [c["spearman"] for c in sel])), 6),
            kendall_median=round(float(np.median(
                [c["kendall_sign_agreement"] for c in sel])), 6),
            ratio_median_of_medians=round(float(np.median(
                [c["ratio_median"] for c in sel])), 6),
            ratio_iqr_median=round(float(np.median(
                [c["ratio_iqr_log10"] for c in sel])), 6)))
    by_level = {}
    for lv in sorted({c["level"] for c in cells}):
        sel = [c for c in cells if c["level"] == lv]
        by_level[str(lv)] = dict(
            n=len(sel),
            spearman_median=round(float(np.median(
                [c["spearman"] for c in sel])), 6),
            kendall_median=round(float(np.median(
                [c["kendall_sign_agreement"] for c in sel])), 6),
            ratio_median=round(float(np.median(
                [c["ratio_median"] for c in sel])), 6))

    summary = dict(
        gate="P-VALIDITY-MAP", analysis_status="validity_map_v1",
        n_cells=len(cells),
        levels=sorted({c["level"] for c in cells}),
        question="in which (level, curvature) regime is the Fisher "
                 "linearization valid: rank, sign, and magnitude endpoints",
        sources=dict(
            cells=str(Path(rows_path).relative_to(REPO)).replace("\\", "/"),
            curvature=str(Path(radius_path).relative_to(REPO)).replace("\\", "/")),
        panels=dict(
            rank_correlation="Spearman(pred_deg, emp_deg) per cell",
            sign_agreement="Kendall tau (pairwise sign agreement of the "
                           "predicted vs empirical mode ordering) per cell",
            absolute_ratio="median_j emp_deg_j/pred_deg_j per cell",
            ratio_spread="IQR of log10(emp/pred) across the 5 modes"),
        by_excess_bin=bins, by_level=by_level, cells=cells,
        manifest=dict(
            git_sha=_git_sha(),
            rows_sha256=_sha(Path(rows_path)),
            radius_sha256=_sha(Path(radius_path))),
        note="read-only join of two committed artifacts (corrected arm-D "
             "cells x linearization-radius curvature); no experiment "
             "re-run; descriptive, no sign gate.")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(json.dumps(summary, ensure_ascii=False,
                                          indent=1).encode("utf-8"))

    # ---- 四面板二维图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    panels = [("spearman", "Rank correlation (Spearman)", "viridis", False),
              ("kendall_sign_agreement", "Sign agreement (Kendall tau)",
               "viridis", False),
              ("ratio_median", "Magnitude ratio emp/pred (median)",
               "cividis", True),
              ("ratio_iqr_log10", "log-ratio spread (IQR, log10)", "magma",
               False)]
    x = np.array([c["level"] for c in cells])
    y = np.array([c["excess_over_scaling"] for c in cells])
    for ax, (key, title, cmap, logc) in zip(axes.flat, panels):
        v = np.array([c[key] for c in cells])
        cv = np.log10(np.maximum(v, 1e-6)) if logc else v
        sc = ax.scatter(x, y, c=cv, s=28, cmap=cmap,
                        edgecolors="k", linewidths=0.3)
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("log10 ratio" if logc else key)
        ax.set_title(title, fontsize=10)
        ax.axhline(1.0, color="r", ls="--", lw=0.8, alpha=0.6)
        ax.set_yscale("log")
        ax.grid(alpha=0.25, lw=0.4)
    for ax in axes[-1]:
        ax.set_xlabel("corruption level ℓ")
    for ax in axes[:, 0]:
        ax.set_ylabel("excess over first-order scaling q(ℓ)")
    fig.suptitle("Model-validity map: where the Fisher linearization holds "
                 "(66 cells; red line = first-order boundary q=1)", fontsize=11)
    fig.tight_layout()
    Path(img_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(img_path, dpi=150)
    print(f"[vmap] {len(cells)} cells; wrote {out_path}")
    print(f"[vmap] wrote {img_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=str(ROWS))
    ap.add_argument("--radius", default=str(RADIUS))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--img", default=str(IMG))
    a = ap.parse_args()
    run(a.rows, a.radius, a.out, a.img)
