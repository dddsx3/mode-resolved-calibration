"""P-ALLOC2 · mode-tail targeted calibration intervention（plan v2 M3）。

预注册协议（configs/allocation_mode_tail.yaml，run 前提交；结果无关条款）。

问题（mode-resolved 分析天然提出的唯一新问题）：理论指出哪些标定分量最伤
最弱 tracked modes——对**这些分量**做定向干预后，被靶向 modes 的经验误差
是否确实优先下降（相比同预算撒在随机 Fisher-active 灯上）？

沿袭：与冻结 allocation 完全同构的场景装配/腐蚀机械/预算网格（P=1200、
48 active/142、corrected interface）；配对 raw innovations 用不相交种子空间。
端点：gauge 对齐残差在 bottom-5 tracked modes 的 **dual 坐标**（M0-2 修正）
能量，按 seed 配对比较 targeted vs random_active48。

输出：results/mode_tail/allocation_mode_tail.json
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

REPO = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from calibinfo.allocation.corruption import (                          # noqa: E402
    apply_scaled_corruption, raw_innovations)
from calibinfo.allocation.policies import (                            # noqa: E402
    SelectionState, budget_scales, select_ordering_mode_aware)
from calibinfo.datasets.openillumination import load_object            # noqa: E402
from calibinfo.models.corruption import CorruptionGenerator            # noqa: E402
from experiments.openillumination_validation import NominalScene       # noqa: E402

N_RANDOM_PERMS = 6


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, cwd=str(REPO)).stdout.strip()


def run(config_path=REPO / "configs/allocation_mode_tail.yaml",
        out_path=REPO / "results/mode_tail/allocation_mode_tail.json"):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    levels = [float(lv) for lv in cfg["levels"]]
    regimes = [int(rg) for rg in cfg["regimes"]]
    budgets_k = list(cfg["budgets_k"])
    seeds_per_level = int(cfg["seeds_per_level"])
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []            # 每 (obj, level, regime, k, seed) 一行：dual 坐标能量×5
    orderings_meta = []
    t_start = time.time()
    for obj_idx, obj_name in enumerate(cfg["cohort"]):
        obj = load_object(cfg["data_root"], obj_name, data_meta=cfg.get("data_meta"))
        scen = NominalScene(obj, np.random.default_rng([20260910, obj_idx]),
                            noise_fit_convention=cfg["noise_fit_convention"])
        K = 142
        u_act = (scen.w * scen.s_hat)[:, :, None] * scen.B_phi
        M0_act = np.einsum("kpi,kpj->kij", scen.B_phi,
                           scen.w[:, :, None] * scen.B_phi)
        finf = scen.Finf_diag
        u142 = np.zeros((K, finf.shape[0], 3))
        M0142 = np.zeros((K, 3, 3))
        lam0142 = np.stack([np.eye(3)] * K)
        u142[scen.sel] = u_act
        M0142[scen.sel] = M0_act
        active142 = np.zeros(K, bool)
        active142[scen.sel] = True
        active = [int(i) for i in scen.sel]

        for lv_i, level in enumerate(levels):
            gen = CorruptionGenerator("joint", level)
            sig_logI, sig_rad = gen.sig_logI, np.radians(gen.sig_deg)
            lam0142[scen.sel] = np.linalg.inv(gen.sigma_phi_diag())
            state = SelectionState(u=u142, M0=M0142, lam0=lam0142.copy(),
                                   finf=finf, active=active142)
            _deg, W_dual = scen.predicted_degradation(
                gen.sigma_phi_diag(), mode_coordinate="dual")
            for regime in regimes:
                # targeted 臂：冻结 mode-aware 启发式（自适应弱模式敏感性）
                t_order, _steps = select_ordering_mode_aware(state, float(regime))
                orderings_meta.append(
                    dict(object=obj_name, level=level, regime=regime,
                         targeted_first8=[int(x) for x in t_order[:8]]))
                for s_i in range(seeds_per_level):
                    rng = np.random.default_rng(
                        [20260916, 7777, obj_idx, lv_i, s_i])
                    raw = raw_innovations(rng, len(scen.sel))
                    for k in budgets_k:
                        energies = {}
                        for policy in ("targeted", "random_active48"):
                            if policy == "targeted":
                                # t_order 是 142 灯全序；分析灯取其 slot（同冻结
                                # allocation 的 slot[scen.sel] 口径）
                                scales48 = budget_scales(t_order, k, regime)[scen.sel]
                            else:
                                # random_active48：6 条均匀排列的平均（方差缩减；
                                # 同冻结 random48 的多排列口径）
                                acc = np.zeros(5)
                                for _p in range(N_RANDOM_PERMS):
                                    ordr48 = rng.permutation(48)  # 位置索引口径
                                    scales48 = budget_scales(ordr48, k, regime)
                                    d2, g = apply_scaled_corruption(
                                        scen.dirs, sig_logI, sig_rad, scales48,
                                        raw)
                                    rho_t = scen.estimate_albedo(d2, g)
                                    e = rho_t - scen.rho
                                    sg = (e * scen.rho).sum()                                         / (scen.rho ** 2).sum()
                                    acc += (e @ W_dual) ** 2
                                energies[policy] = (acc / N_RANDOM_PERMS).tolist()
                                continue
                            d2, g = apply_scaled_corruption(
                                scen.dirs, sig_logI, sig_rad, scales48, raw)
                            rho_t = scen.estimate_albedo(d2, g)
                            e = rho_t - scen.rho
                            sg = (e * scen.rho).sum() / (scen.rho ** 2).sum()
                            e = e - sg * scen.rho
                            proj = e @ W_dual                    # (5,) dual 坐标
                            energies[policy] = (proj ** 2).tolist()
                        rows.append(dict(
                            object=obj_name, level=level, regime=regime, k=k,
                            seed=s_i,
                            targeted=[float(x) for x in energies["targeted"]],
                            random_active48=[float(x)
                                             for x in energies["random_active48"]]))
        print(f"[alloc2] {obj_name} done elapsed={time.time() - t_start:.0f}s",
              flush=True)

    # ---- 聚合：配对 Δ（targeted − random48），object 级 bootstrap ----
    from calibinfo.metrics.cluster_bootstrap import cluster_bootstrap
    agg = {}
    for regime in regimes:
        for k in budgets_k:
            rs = [r for r in rows if r["regime"] == regime and r["k"] == k]
            per_obj = {}
            for r in rs:
                e_t = float(np.sum(r["targeted"]))
                e_r = float(np.sum(r["random_active48"]))
                per_obj.setdefault(r["object"], []).append(e_t - e_r)
            objs = sorted(per_obj)
            deltas = {o: float(np.mean(per_obj[o])) for o in objs}

            def _stat(sub):
                return float(np.median([deltas[o] for o, _ in sub]))

            _pt, ci, _b = cluster_bootstrap([(o, None) for o in objs], _stat,
                                            10000, 20260916)
            per_mode = np.zeros(5)
            for r in rs:
                per_mode += np.asarray(r["targeted"]) - np.asarray(r["random_active48"])
            per_mode /= len(rs)
            agg[f"{regime}/{k}"] = dict(
                delta_median=float(np.median(list(deltas.values()))),
                delta_ci95=[float(ci[0]), float(ci[1])],
                improved_negative=int(sum(1 for o in objs if deltas[o] < 0)),
                per_object={o: deltas[o] for o in objs},
                per_mode_delta_mean=[float(x) for x in per_mode],
                n_rows=len(rs))

    display = {key: dict(delta_median=round(v["delta_median"], 6),
                         ci95=[round(v["delta_ci95"][0], 6),
                               round(v["delta_ci95"][1], 6)])
               for key, v in agg.items()}

    summary = dict(
        gate="P-ALLOC2 v1: mode-tail targeted calibration intervention",
        analysis_status="allocation_mode_tail_v1",
        endpoint=cfg["endpoint"]["primary"],
        noise_fit_convention=cfg["noise_fit_convention"],
        levels=levels, regimes=regimes, budgets_k=budgets_k,
        seeds_per_level=seeds_per_level,
        rows=rows, aggregated=agg, display=display,
        orderings_meta=orderings_meta[:20],
        manifest=dict(config_sha256=_sha(Path(config_path)),
                      git_sha=_git_sha(),
                      bootstrap=dict(B=10000, seed=20260916),
                      elapsed_s=round(time.time() - t_start, 1)),
        note="Paired-by-seed comparison; outcome-independent reporting. "
             "Delta < 0 = targeted arm has LOWER targeted-mode energy "
             "(improvement). The targeted policy is the frozen "
             "weak-Fisher-mode heuristic, not a new algorithm.")
    out.write_bytes(json.dumps(summary, ensure_ascii=False, indent=1)
                    .encode("utf-8"))
    for key, v in agg.items():
        print(f"[alloc2] {key}: d median {v['delta_median']:+.4e} "
              f"CI [{v['delta_ci95'][0]:+.4e}, {v['delta_ci95'][1]:+.4e}] "
              f"improved {v['improved_negative']}/11")
    print(f"[alloc2] wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/allocation_mode_tail.yaml"))
    ap.add_argument("--out", default=str(REPO / "results/mode_tail/allocation_mode_tail.json"))
    args = ap.parse_args()
    run(args.config, args.out)
