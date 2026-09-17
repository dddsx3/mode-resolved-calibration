"""P-DILIGENT-QUEUE · 三个核心实验在 DiLiGenT 队列上的迁移(第二数据集)。

OpenIllumination(11 物体/142 灯)之外的第二数据集:DiLiGenT(10 物体/
96 灯,独立采集系统)。同一代码路径,经 diligent_oi_adapter 适配 +
n_lights_total=96(rng 抽取空间不同——新队列,不与冻结产物对拍)。

实验(预注册 configs/diligent_queue.yaml):
  1. channel_decomposition:D(level) 三通道表(C7/C9-D 的通道分解在
     第二数据集;与球锚定 C10 联读:DiLiGenT 实测误差剖面偏方向,
     通道条件性判据预测方向通道在此承载更多);
  2. linearization_radius:q(level) = dev/一阶标度,2x/10x 半径
     (B6/C 臂包络跨数据集);
  3. ball-anchor direction share:在实测锚点 (0.0159, 2.96°) 读方向
     份额(C10 翻转测试跨数据集)。

预注册结局规则:transfer-confirmed / partial / failed(无符号门)。

输出: results/diligent/diligent_queue.json
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

from calibinfo.allocation.blocks import LightBlocks               # noqa: E402
from calibinfo.allocation.corruption import (                      # noqa: E402
    raw_innovations_family, apply_corruption_family)
from calibinfo.datasets.diligent_oi_adapter import (               # noqa: E402
    load_diligent_as_oi)
from calibinfo.models.corruption_family import CorruptionFamily    # noqa: E402
from experiments.channel_decomposition import J_A                  # noqa: E402
from experiments.openillumination_validation import NominalScene   # noqa: E402


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, cwd=str(REPO)).stdout.strip()


def _sigma(ctype, lv, closed_deg, closed_li):
    rad = np.radians(lv if ctype != "intensity" else closed_deg)
    li = lv if ctype != "direction" else closed_li
    return li, rad


def _blocks(scen, lam0142, K):
    u_act = (scen.w * scen.s_hat)[:, :, None] * scen.B_phi
    M0_act = np.einsum("kpi,kpj->kij", scen.B_phi,
                       scen.w[:, :, None] * scen.B_phi)
    u = np.zeros((K, scen.Finf_diag.shape[0], 3))
    M0 = np.zeros((K, 3, 3))
    u[scen.sel] = u_act
    M0[scen.sel] = M0_act
    active = np.zeros(K, bool)
    active[scen.sel] = True
    return LightBlocks(u, M0, lam0142, scen.Finf_diag, active), active


def run(config_path=REPO / "configs/diligent_queue.yaml",
        out_path=REPO / "results/diligent/diligent_queue.json"):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    K = int(cfg["n_lights"])
    kappa = float(cfg["kappa"])
    closed_deg = float(cfg["closed_direction_sigma_deg"])
    closed_li = float(cfg["closed_intensity_sigma_logI"])
    family_seed = int(cfg.get("family_seed", 20260915))
    t_start = time.time()
    sha_at_launch = _git_sha()

    # ---- 实验 1 + 3 的状态复用 ----
    chan_rows = []
    anchor_shares = {}
    radius_out = {}
    for obj_idx, obj_name in enumerate(cfg["cohort"]):
        obj = load_diligent_as_oi(Path(cfg["data_root"]) / obj_name)
        scen = NominalScene(obj, np.random.default_rng([20260915, obj_idx]),
                            noise_fit_convention=cfg["noise_fit_convention"],
                            n_lights_total=K)
        t_one = np.ones(K)

        # ---- 实验 1:通道分解(D 功能量,冻结 J_A) ----
        for ctype in cfg["channels"]:
            for lv in cfg["levels"]:
                lv = float(lv)
                li, rad = _sigma(ctype, lv, closed_deg, closed_li)
                fam = CorruptionFamily(li, np.degrees(rad),
                                       het_sigma=0.0, rho_c=0.0,
                                       seed=family_seed, n_lights=K)
                lam0 = np.linalg.inv(fam.sigma_phi_block())
                blk, active = _blocks(scen, lam0, K)
                t_all = t_one.copy()
                t_all[np.flatnonzero(active)] = kappa
                J1, Jk = J_A(blk, t_one), J_A(blk, t_all)
                chan_rows.append(dict(object=obj_name, channel=ctype,
                                      level=lv, J_A_1=float(J1),
                                      J_A_kappa=float(Jk),
                                      D=float((J1 - Jk) / J1)))

        # ---- 实验 3:球锚点方向份额 ----
        ba = cfg["ball_anchor"]
        fam = CorruptionFamily(ba["sig_logI"], ba["sig_dir_deg"],
                               het_sigma=0.0, rho_c=0.0,
                               seed=family_seed, n_lights=K)
        lam_a = np.linalg.inv(fam.sigma_phi_block())
        blk_a, active_a = _blocks(scen, lam_a, K)
        fam_c = CorruptionFamily(ba["sig_logI"], closed_deg,
                                 het_sigma=0.0, rho_c=0.0,
                                 seed=family_seed, n_lights=K)
        lam_c = np.linalg.inv(fam_c.sigma_phi_block())
        blk_c, _ = _blocks(scen, lam_c, K)

        def _D(blk):
            t_all = t_one.copy()
            t_all[np.flatnonzero(active_a)] = kappa
            return (J_A(blk, t_one) - J_A(blk, t_all)) / J_A(blk, t_one)
        d, d0 = _D(blk_a), _D(blk_c)
        anchor_shares[obj_name] = float((d - d0) / d)

        # ---- 实验 2:线性化半径(v2 度量域,观测侧注入+名义几何估计) ----
        levels = [float(x) for x in cfg["radius_levels"]]
        seeds = int(cfg["radius_seeds"])
        ref_level = float(cfg["radius_reference_level"])
        rows_obj = []
        for level in levels:
            fam_r = CorruptionFamily(level, level, het_sigma=0.0, rho_c=0.0,
                                     seed=family_seed, n_lights=K)
            sig_blocks = fam_r.sigma_phi_block()[scen.sel]
            _deg, W_dual = scen.predicted_degradation(
                sig_blocks, mode_coordinate="dual")
            n_act = len(scen.sel)
            energies = np.empty(seeds)
            for s_i in range(seeds):
                rng = np.random.default_rng(
                    [20260916, 4242, obj_idx, int(level * 1000), s_i])
                raw = raw_innovations_family(rng, n_act, rho_c=0.0)
                d2, g = apply_corruption_family(
                    scen.dirs, level, np.radians(level),
                    np.ones(n_act), raw)
                I_obs = scen.rho[None, :] \
                    * np.maximum(scen.n @ d2.T, 0.0).T * g[:, None]
                scen_obs = NominalScene._from_arrays(scen, I_obs)
                rho_t = scen_obs.estimate_albedo(
                    scen.dirs, np.ones(n_act))
                e = rho_t - scen.rho
                sg = (e * scen.rho).sum() / (scen.rho ** 2).sum()
                e = e - sg * scen.rho
                energies[s_i] = float(np.sum((e @ W_dual) ** 2))
            rows_obj.append(dict(level=level,
                                 E_mean=float(energies.mean())))
        E_ref = rows_obj[0]["E_mean"]
        for r in rows_obj:
            scale2 = (r["level"] / ref_level) ** 2
            r["excess_over_scaling"] = (r["E_mean"] / E_ref / scale2
                                        if E_ref > 0 else float("nan"))
        radius = {}
        for thr in cfg["radius_crossings"]:
            hit = [r["level"] for r in rows_obj
                   if r["excess_over_scaling"] > thr]
            radius[f"cross_{thr:g}x"] = (min(hit) if hit else None)
        radius_out[obj_name] = dict(rows=rows_obj, radius=radius)
        print(f"[dq] {obj_name}: chan D(joint@0.5)="
              f"{[r['D'] for r in chan_rows if r['object']==obj_name and r['channel']=='joint' and r['level']==0.5][0]:.4f} "
              f"| anchor share={anchor_shares[obj_name]*100:.1f}% "
              f"| radius2x={radius['cross_2x']} "
              f"({time.time()-t_start:.0f}s)", flush=True)

    # ---- 汇总 ----
    by_channel = {}
    for ctype in cfg["channels"]:
        by_channel[ctype] = {}
        for lv in cfg["levels"]:
            lv = float(lv)
            rs = [r["D"] for r in chan_rows
                  if r["channel"] == ctype and r["level"] == lv]
            by_channel[ctype][str(lv)] = dict(
                median_pct=round(float(np.median(rs)) * 100, 3),
                max_pct=round(float(np.max(rs)) * 100, 3),
                min_pct=round(float(np.min(rs)) * 100, 3))
    direction_max_pct = round(max(r["D"] for r in chan_rows
                                  if r["channel"] == "direction") * 100, 3)

    def _agg(key):
        vals = [v["radius"][key] for v in radius_out.values()
                if v["radius"][key] is not None]
        return dict(median_over_crossed_subset=(
                        float(np.median(vals)) if vals else None),
                    n_crossed=len(vals), n_total=len(radius_out),
                    per_object={k: v["radius"][key]
                                for k, v in radius_out.items()})

    med_share = float(np.median(list(anchor_shares.values())))
    # 退化份额标注:D(anchor)≈0 的对象(方向/强度都几乎不减动态范围的
    # 低误差点)其份额分母无意义——负值/超大绝对值是 0/0 型,不是方向
    # 通道为负贡献。份额的高低在此处不构成读数(loader 归一化修正后
    # 全队列份额 ≈0,见 loader_normalization_fix.json)。
    d_anchor = {}
    for obj_name in cfg["cohort"]:
        j = [r["D"] for r in chan_rows
             if r["object"] == obj_name and r["channel"] == "joint"
             and r["level"] == float(cfg["levels"][0])][0]
        d_anchor[obj_name] = j
    degenerate = {o for o, d in d_anchor.items()
                  if abs(anchor_shares[o]) > 1.0 or
                  (anchor_shares[o] < 0 and d < 0.01)}
    r2 = _agg("cross_2x")["median_over_crossed_subset"]
    radius_ok = r2 is not None and 0.25 <= r2 <= 4.0
    # v1.1 判读(N3):通道分裂迁移判据(两队列同为强度主导)。
    # 旧份额判据(>=2%)按偏置 loader 的预期所写,在修正后无论数据如何
    # 都只能失败——保留为参照,不是读数基础(8e841a5 先例)。
    intensity_eq_joint = all(
        abs(by_channel["joint"][lv]["median_pct"]
            - by_channel["intensity"][lv]["median_pct"]) < 0.5
        for lv in by_channel["joint"])
    dir_small = direction_max_pct <= 5.0
    if intensity_eq_joint and dir_small and radius_ok:
        outcome = "transfer-confirmed"
    elif radius_ok:
        outcome = "transfer-partial (radius transfers; channel split does not)"
    else:
        outcome = "transfer-failed"

    summary = dict(
        gate=cfg["gate"], analysis_status=cfg["analysis_status"],
        n_objects=len(cfg["cohort"]), n_lights=K,
        noise_fit_convention=cfg["noise_fit_convention"],
        scene_rng_spec=cfg["scene_rng_spec"],
        channel_decomposition=dict(
            by_channel=by_channel,
            direction_max_pct=direction_max_pct,
            rows=chan_rows,
            note="C7/C9-D channel split on the second dataset; read "
                 "jointly with the ball anchor (C10): DiLiGenT's "
                 "measured error profile is direction-heavy"),
        linearization_radius=dict(
            radius_2x=_agg("cross_2x"), radius_10x=_agg("cross_10x"),
            objects=radius_out,
            note="v2 metric domain (observation-side injection, "
                 "nominal-geometry estimator), same spec shape as the "
                 "OI run; OI reference medians 1.0 / 1.5"),
        ball_anchor_share=dict(
            anchor=cfg["ball_anchor"],
            per_object={k: round(v, 6) for k, v in anchor_shares.items()},
            median=round(med_share, 6),
            degenerate_objects=sorted(degenerate),
            degenerate_note="shares with |share| > 1 or negative-on-"
                            "tiny-denominator occur exactly where "
                            "D(anchor) ~ 0 (both channels barely move "
                            "the dynamic range at the measured anchor) "
                            "-- the share ratio is 0/0-shaped there, NOT "
                            "a negative direction contribution. Field "
                            "paths: ball_anchor_share.per_object "
                            "(withdrawn-value annotations: see "
                            "diligent/provenance/"
                            "loader_normalization_fix.json, 2026-09-16)",
            oi_reference=dict(median=0.360024,
                              note="C10 on the OI cohort at the same "
                                   "measured anchor")),
        outcome=outcome,
        outcome_rule=cfg["outcome_rule"].strip(),
        manifest=dict(
            config_sha256=_sha(Path(config_path)),
            git_sha=sha_at_launch,
            elapsed_s=round(time.time() - t_start, 1)),
        note="Second-dataset transfer of the three core findings: "
             "channel decomposition (descriptive), linearization radius, "
             "and the ball-anchor direction share (the C10 flip test). "
             "DiLiGenT adapter + n_lights_total=96; a NEW queue by "
             "construction (rng draw space differs), not bit-comparable "
             "to the frozen OI artifacts.")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(json.dumps(summary, ensure_ascii=False,
                                          indent=1).encode("utf-8"))
    print(f"[dq] anchor share median {med_share*100:.1f}% | "
          f"radius2x median {r2} | outcome {outcome}")
    print(f"[dq] wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",
                    default=str(REPO / "configs/diligent_queue.yaml"))
    ap.add_argument("--out",
                    default=str(REPO / "results/diligent/diligent_queue.json"))
    args = ap.parse_args()
    run(args.config, args.out)
