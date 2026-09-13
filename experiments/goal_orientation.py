"""P-GOAL-ORIENTED · task-dependent value of calibration refinement (T10/N4).

The goal-oriented risk J_H(t) = tr(H ΔF(t)^{-1} Hᵀ) prices the SAME
calibration state for a specific downstream task H (m×P linear functionals
of the per-pixel parameter). Computed with the low-rank push-through route
(`calibinfo.information.lowrank.woodbury_quad_risk`; math gates in
`tests/test_goal_oriented.py`).

This script tabulates, on the 11 held-out OpenIllumination objects:
  - per-light refinement gains under three tasks
    (all = H:I, mean = 1/√P·1ᵀ, contrast = bright−dim quartile mean),
  - the rank agreement between the mean-task and contrast-task orderings
    of the same 48 Fisher-active lights (Spearman + top-3 overlap),
  - the task-value curves V_H(level) = 1 − J_H(κ·1)/J_H(1).

Output: results/goal_oriented/goal_orientation.json.

Usage:
  python experiments/goal_orientation.py [--config configs/goal_orientation.yaml]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from calibinfo.information.lowrank import woodbury_quad_risk             # noqa: E402
from calibinfo.datasets.openillumination import load_object             # noqa: E402
from experiments.alpha_bound import build_state                         # noqa: E402
from experiments.openillumination_validation import NominalScene        # noqa: E402


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, cwd=str(REPO)).stdout.strip()


def _task_operators(scen, cfg):
    """任务算子 H(全部作用于逐像素 log-albedo 方向参数,R^P)。"""
    P = scen.Finf_diag.shape[0]
    tasks = {}
    if "all" in cfg["tasks"]:
        tasks["all"] = np.eye(P)
    if "mean" in cfg["tasks"]:
        tasks["mean"] = np.ones((1, P)) / np.sqrt(P)
    if "contrast" in cfg["tasks"]:
        cut = float(cfg["contrast_quartile"])
        bright = scen.I.mean(0)                            # (P,) 逐像素平均强度
        n_b = max(1, int(round(P * cut)))
        n_d = max(1, int(round(P * cut)))
        order = np.argsort(bright)
        h = np.zeros((1, P))
        h[0, order[-n_b:]] = 1.0 / n_b                     # 最亮四分位
        h[0, order[:n_d]] -= 1.0 / n_d                     # 最暗四分位
        nrm = float(np.linalg.norm(h))
        tasks["contrast"] = h / nrm if nrm > 0 else h
    return tasks


def _gains(blk, H, kappa, active_idx):
    """J_H(t=1) 基线 + 每灯单灯精化增益 gain_k = J_H(1) − J_H(t_k=κ)。"""
    L = blk.u.shape[0]
    t1 = np.ones(L)
    base = woodbury_quad_risk(blk.finf, blk.u, blk.M0, blk.lam0,
                              blk.active, t1, H)
    gains = {}
    for k in active_idx:
        t = t1.copy()
        t[k] = kappa
        gains[int(k)] = base - woodbury_quad_risk(blk.finf, blk.u, blk.M0,
                                                  blk.lam0, blk.active, t, H)
    return float(base), gains


def _ranking(gains):
    """灯索引按增益降序;返回 (ranked light ids, gain array)。"""
    ks = sorted(gains)
    vals = np.array([gains[k] for k in ks])
    order = np.argsort(-vals, kind="stable")
    return [ks[i] for i in order], vals[order]


def run(config_path=REPO / "configs/goal_orientation.yaml",
        out_path=REPO / "results/goal_oriented/goal_orientation.json"):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    kappa = float(cfg["kappa"])
    levels = [float(x) for x in cfg["levels"]]
    t_start = time.time()

    rows = []
    for obj_idx, obj_name in enumerate(cfg["cohort"]):
        obj = load_object(cfg["data_root"], obj_name, data_meta=cfg.get("data_meta"))
        scen = NominalScene(obj, np.random.default_rng([20260910, obj_idx]),
                            noise_fit_convention=cfg["noise_fit_convention"])
        tasks = _task_operators(scen, cfg)
        for lv in levels:
            blk = build_state(scen, lv, kappa, int(cfg["K_lights"]))
            active_idx = np.flatnonzero(blk.active)
            # 任务价值曲线:均匀精化全部 active 灯
            tK = np.ones(int(cfg["K_lights"]))
            tK[active_idx] = kappa
            row = dict(object=obj_name, level=lv)
            base = {}
            for name, H in tasks.items():
                j1 = woodbury_quad_risk(blk.finf, blk.u, blk.M0, blk.lam0,
                                        blk.active, np.ones(int(cfg["K_lights"])), H)
                jk = woodbury_quad_risk(blk.finf, blk.u, blk.M0, blk.lam0,
                                        blk.active, tK, H)
                row[f"J_{name}_1"] = float(j1)
                row[f"J_{name}_kappa"] = float(jk)
                row[f"V_{name}"] = float(1.0 - jk / j1)
                base[name], gains = _gains(blk, H, kappa, active_idx)
                ranked, vals = _ranking(gains)
                row[f"ranking_{name}"] = ranked
                row[f"gains_{name}_min"] = round(float(vals.min()), 6)
                row[f"gains_{name}_max"] = round(float(vals.max()), 6)
            # mean vs contrast 排序一致性(核心对照)
            if "mean" in tasks and "contrast" in tasks:
                rk_m = np.array(row["ranking_mean"])
                rk_c = np.array(row["ranking_contrast"])
                pos_m = {k: i for i, k in enumerate(rk_m)}
                pos_c = {k: i for i, k in enumerate(rk_c)}
                km = np.array([pos_m[k] for k in sorted(pos_m)])
                kc = np.array([pos_c[k] for k in sorted(pos_c)])
                rho = float(spearmanr(km, kc).statistic)
                row["spearman_mean_vs_contrast"] = round(rho, 6)
                row["top3_overlap_mean_vs_contrast"] = len(
                    set(rk_m[:3].tolist()) & set(rk_c[:3].tolist()))
            if "all" in tasks and "mean" in tasks:
                rk_a = np.array(row["ranking_all"])
                rk_m = np.array(row["ranking_mean"])
                pos_a = {k: i for i, k in enumerate(rk_a)}
                pos_m = {k: i for i, k in enumerate(rk_m)}
                ka = np.array([pos_a[k] for k in sorted(pos_a)])
                km2 = np.array([pos_m[k] for k in sorted(pos_m)])
                row["spearman_all_vs_mean"] = round(
                    float(spearmanr(ka, km2).statistic), 6)
            rows.append(row)
        print(f"[goal] {obj_name}: " + "; ".join(
            f"lv={r['level']} rho(m,c)={r.get('spearman_mean_vs_contrast')}"
            for r in rows if r["object"] == obj_name), flush=True)

    # ---- 汇总 ----
    mc = [r["spearman_mean_vs_contrast"] for r in rows
          if "spearman_mean_vs_contrast" in r]
    t3 = [r["top3_overlap_mean_vs_contrast"] for r in rows
          if "top3_overlap_mean_vs_contrast" in r]
    by_task = {}
    for name in cfg["tasks"]:
        vs = [r[f"V_{name}"] for r in rows]
        by_task[name] = dict(min=round(min(vs), 6), max=round(max(vs), 6),
                             median=round(float(np.median(vs)), 6))
    by_level = {}
    for lv in levels:
        rs = [r for r in rows if r["level"] == lv]
        by_level[str(lv)] = {
            name: round(float(np.median([r[f"V_{name}"] for r in rs])), 6)
            for name in cfg["tasks"]}
    summary = dict(
        gate=cfg["gate"], analysis_status=cfg["analysis_status"],
        kappa=kappa, levels=levels,
        noise_fit_convention=cfg["noise_fit_convention"],
        n_objects=len(cfg["cohort"]), n_active_lights=int(len(active_idx)),
        tasks=list(cfg["tasks"]),
        headline=dict(
            n_cells=len(mc),
            spearman_mean_vs_contrast=dict(
                min=round(min(mc), 6), median=round(float(np.median(mc)), 6),
                max=round(max(mc), 6)),
            top3_overlap_mean_vs_contrast=dict(
                min=int(min(t3)), max=int(max(t3)),
                n_cells_disjoint=int(sum(1 for x in t3 if x == 0))),
            value_curves=by_task,
            value_curves_by_level=by_level),
        rows=rows,
        manifest=dict(
            config_sha256=_sha(Path(config_path)),
            git_sha=_git_sha(),
            scene_rng_spec=cfg["scene_rng_spec"],
            elapsed_s=round(time.time() - t_start, 1)),
        note="goal-oriented risk J_H = tr(H DeltaF^-1 H^T) on the per-pixel "
             "log-albedo-direction parameterization; per-light gains are "
             "single-light refinement values J_H(1)-J_H(t_k=kappa); the "
             "mean-vs-contrast comparison is the task-dependence deliverable. "
             "PS normals-vs-albedo needs the joint 4P parameterization "
             "(flagged as extension, not claimed).")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(json.dumps(summary, ensure_ascii=False,
                                          indent=1).encode("utf-8"))
    print(f"[goal] spearman(mean,contrast): min {min(mc):.3f} "
          f"median {float(np.median(mc)):.3f}; top-3 disjoint "
          f"{sum(1 for x in t3 if x == 0)}/{len(t3)} cells")
    print(f"[goal] wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/goal_orientation.yaml"))
    ap.add_argument("--out",
                    default=str(REPO / "results/goal_oriented/goal_orientation.json"))
    args = ap.parse_args()
    run(args.config, args.out)
