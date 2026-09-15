"""P-BALL-ANCHOR · 把 Σ_φ 锚到真实球标定流程的实测误差(TCI 缺口 1)。

仓库所有定量结论长在合成旋钮 level 上(0.025..8, 37 倍跨度)。本实验在
DiLiGenT ballPNG(96 张真实采集的球图像 + GT 法向)上跑**标准球标定**,
与 GT 灯方向/强度比 → 得到逐灯、分通道的真实标定误差分布,再把
Σ_φ 锚到这份实测分布,并在冻结的 OpenIllumination 族机制上读出
实测锚点的方向份额(D 判读)。

两结局均预注册(configs/ball_anchor.yaml):
  D-confirmed(中位方向份额 < 2%):C7/C9-D 从参数化稳健升级为实证锚定;
  D-flipped(≥ 2%):实用建议改写,有效性边界叙事承接。

算法(标准交替 LSQ,预注册):
  (a) 逐像素 albedo:rho(p) = Σ_k I_k(p) e_k (n_p·l_k) / Σ_k (e_k(n_p·l_k))²
  (b) 逐灯秩一:y_k = argmin_{y∈R³} Σ_p (I_k(p)/rho(p) − n_p·y)²
      → e_k = ‖y_k‖, l_k = y_k/‖y_k‖
  (a)↔(b) 交替至 rho 收敛(< 1e-10)。只用非饱和、受光像素;
  lit 掩码用 GT 方向(固定口径;估计本身不见 GT)。

输出: results/openillumination/ball_anchor.json
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


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, cwd=str(REPO)).stdout.strip()


def load_ball(d):
    """球数据(原始灰度,不除 GT 强度——那会抹掉要测的强度误差)。"""
    import scipy.io as sio
    from PIL import Image
    d = Path(d)
    dirs_gt = np.loadtxt(d / "light_directions.txt")          # (96,3)
    ints_gt = np.loadtxt(d / "light_intensities.txt")[:, 0]   # R 通道
    N_gt = sio.loadmat(d / "Normal_gt.mat")["Normal_gt"]
    mask = np.array(Image.open(d / "mask.png").convert("L")) > 128
    imgs = np.stack([np.array(Image.open(str(d / f"{i:03d}.png"))
                              .convert("L")).astype(float)
                     for i in range(1, 97)])                  # (96,H,W)
    return dirs_gt, ints_gt, N_gt, mask, imgs


def sphere_calibrate(I, N, lit_mask, tol, max_iters):
    """交替 LSQ 球标定。I (K,P) 原始灰度;N (P,3) 单位法向;
    lit_mask (K,P) bool。返回 (e (K,), l (K,3), rho (P,), n_iters)。"""
    K, P = I.shape
    # 初始 l:每灯图像的亮度加权平均法向(球标定经典初值),归一化
    w = np.clip(I, 0, None) * lit_mask
    l0 = np.zeros((K, 3))
    for k in range(K):
        wn = w[k] @ N                                          # (3,)
        l0[k] = wn / max(np.linalg.norm(wn), 1e-12)
    l = l0
    e = np.ones(K)
    rho_prev = None
    for it in range(max_iters):
        ndotl = np.einsum("pt,kt->kp", N, l)                   # (K,P)
        use = lit_mask & (ndotl > 0.05) & (I > 0)
        # (a) 逐像素 albedo(全灯联合 LSQ)
        a = np.sum(np.where(use, e[:, None] * ndotl * I, 0.0), axis=0)
        b = np.sum(np.where(use, (e[:, None] * ndotl) ** 2, 0.0), axis=0)
        rho = np.where(b > 1e-12, a / np.maximum(b, 1e-12), 0.0)
        ok = rho > 1e-8
        # (b) 逐灯秩一 LSQ:y_k = argmin || I_k/rho − n y ||²(受光像素)
        y = np.zeros((K, 3))
        for k in range(K):
            m = use[k] & ok
            if m.sum() < 3:
                y[k] = l[k] * e[k]
                continue
            A = N[m]                                           # (m,3)
            rhs = I[k, m] / rho[m]
            y[k] = np.linalg.lstsq(A, rhs, rcond=None)[0]
        e = np.linalg.norm(y, axis=1)
        e = np.where(e > 1e-12, e, 1e-12)
        l = y / e[:, None]
        if rho_prev is not None and np.max(np.abs(
                rho - rho_prev) / np.maximum(np.abs(rho_prev), 1e-12)) < tol:
            return e, l, rho, it + 1
        rho_prev = rho
    return e, l, rho, max_iters


def direction_share(sig_logI, sig_dir_deg, cfg, scen_cache, Dcache):
    """冻结族机制上的方向份额:share = [D(sI,sd) − D(sI,closed)]/D(sI,sd)。

    scen_cache: {obj_name: (u142, M0142, finf, active_idx, rho)}(P-CERT 装配)。
    """
    from calibinfo.allocation.blocks import LightBlocks
    from calibinfo.models.corruption_family import CorruptionFamily
    from experiments.channel_decomposition import J_A
    K = int(cfg["K_lights"])
    kappa = float(cfg["kappa"])
    closed_dir = float(cfg["closed_direction_sigma_deg"])
    shares = {}
    for obj_name, st in scen_cache.items():
        def D_of(li, sd):
            fam = CorruptionFamily(li, sd, het_sigma=0.0, rho_c=0.0,
                                   seed=int(cfg["family_seed"]),
                                   n_lights=K)
            lam0142 = np.linalg.inv(fam.sigma_phi_block())
            blk = LightBlocks(st["u142"], st["M0142"], lam0142,
                              st["finf"], st["active142"])
            t_one = np.ones(K)
            t_all = t_one.copy()
            t_all[st["active_idx"]] = kappa
            return (J_A(blk, t_one) - J_A(blk, t_all)) / J_A(blk, t_one)
        d = D_of(sig_logI, sig_dir_deg)
        d0 = D_of(sig_logI, closed_dir)
        shares[obj_name] = float((d - d0) / d)
    return shares


def run(config_path=REPO / "configs/ball_anchor.yaml",
        out_path=REPO / "results/openillumination/ball_anchor.json"):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    t_start = time.time()
    sha_at_launch = _git_sha()

    # ---- 1) 球标定 ----
    dirs_gt, ints_gt, N_gt, mask, imgs = load_ball(
        Path(cfg["data_root"]) / cfg["object"])
    N = N_gt[mask]                                             # (P,3)
    P = N.shape[0]
    sat = float(cfg["saturation_threshold"])
    sat_pix = (imgs >= sat).any(axis=0)                        # (H,W)
    keep2d = mask & ~sat_pix
    keep = keep2d[mask]                                        # (P,)
    N = N[keep]
    I = imgs[:, keep2d]                                        # (K,P)
    # lit 掩码(GT 方向,固定口径;估计不见 GT)
    ndotl_gt = N @ dirs_gt.T                                   # (P,K)
    lit = (ndotl_gt.T > float(cfg["lit_mask_gt_dot"]))         # (K,P)
    print(f"[ball] pixels kept {keep.sum()}/{mask.sum()} "
          f"(dropped {(sat_pix & mask).sum()} saturated)", flush=True)

    e, l, rho, n_iters = sphere_calibrate(
        I, N, lit, float(cfg["rho_convergence_tol"]),
        int(cfg["rho_max_iters"]))
    print(f"[ball] calibrator converged in {n_iters} iters", flush=True)

    # ---- 2) 与 GT 比对 ----
    # 方向误差(度)
    cosang = np.clip(np.einsum("kt,kt->k", l, dirs_gt), -1.0, 1.0)
    dir_err_deg = np.degrees(np.arccos(cosang))                # (K,)
    # 强度:相对量。GT 的 R 通道 ints_gt(与灰度口径的比例失配由
    # log 域最优标量吸收,只留逐灯偏差)
    log_e = np.log(e)
    log_g = np.log(np.maximum(ints_gt, 1e-12))
    c = np.mean(log_e - log_g)                                 # 最优标量(log域)
    logI_dev = (log_e - log_g) - c                             # (K,)
    rms_dir = float(np.sqrt(np.mean(dir_err_deg ** 2)))
    rms_int = float(np.sqrt(np.mean(np.exp(logI_dev) - 1.0) ** 2))

    # 硬守卫(不过 = 实现错,不是流程差)
    if rms_dir > float(cfg["max_direction_error_deg_rms"]):
        raise RuntimeError(f"sphere calibration failed sanity: RMS dir "
                           f"error {rms_dir:.3f} deg > "
                           f"{cfg['max_direction_error_deg_rms']}")
    if rms_int > float(cfg["max_intensity_error_rel_rms"]):
        raise RuntimeError(f"sphere calibration failed sanity: RMS rel "
                           f"intensity error {rms_int:.3f} > "
                           f"{cfg['max_intensity_error_rel_rms']}")

    # ---- 3) 实测 Σ_φ 锚 ----
    sig_logI = float(np.std(logI_dev))
    sig_dir_deg = rms_dir
    # 逐灯异质性锚:逐灯 |logI_dev| 与 dir_err 的分布宽度
    het_anchor = dict(
        logI_dev_abs_p16_p84=[float(np.percentile(np.abs(logI_dev), 16)),
                               float(np.percentile(np.abs(logI_dev), 84))],
        dir_err_deg_p16_p84=[float(np.percentile(dir_err_deg, 16)),
                              float(np.percentile(dir_err_deg, 84))])

    # ---- 4) 族落位 + D 判读 ----
    from calibinfo.datasets.openillumination import load_object
    from experiments.openillumination_validation import NominalScene
    K = int(cfg["K_lights"])
    scen_cache = {}
    for obj_idx, obj_name in enumerate(cfg["oi_cohort"]):
        obj = load_object(cfg["oi_data_root"], obj_name,
                          data_meta=cfg.get("oi_data_meta"))
        scen = NominalScene(obj, np.random.default_rng([20260910, obj_idx]),
                            noise_fit_convention=cfg["noise_fit_convention"])
        u_act = (scen.w * scen.s_hat)[:, :, None] * scen.B_phi
        M0_act = np.einsum("kpi,kpj->kij", scen.B_phi,
                           scen.w[:, :, None] * scen.B_phi)
        u142 = np.zeros((K, scen.Finf_diag.shape[0], 3))
        M0142 = np.zeros((K, 3, 3))
        u142[scen.sel] = u_act
        M0142[scen.sel] = M0_act
        active142 = np.zeros(K, bool)
        active142[scen.sel] = True
        scen_cache[obj_name] = dict(u142=u142, M0142=M0142,
                                    finf=scen.Finf_diag,
                                    active142=active142,
                                    active_idx=np.flatnonzero(active142))
    shares = direction_share(sig_logI, sig_dir_deg, cfg, scen_cache, None)
    med_share = float(np.median(list(shares.values())))
    outcome = ("D-confirmed" if med_share < 0.02 else "D-flipped")

    # 族网格落位(log 距离最近的 S1 扫描点)
    fam_art = json.loads((REPO / "results/openillumination/"
                          "corruption_family_sensitivity.json")
                         .read_text(encoding="utf-8"))
    grid = fam_art["grid"]
    d_li = np.log10([max(g["sig_logI"], 1e-6) / max(sig_logI, 1e-6)
                     for g in grid])
    d_sd = np.log10([max(g["sig_dir_deg"], 1e-3) / max(sig_dir_deg, 1e-3)
                     for g in grid])
    near = int(np.argmin(np.hypot(d_li, d_sd)))
    nearest_point = dict(grid[near])
    nearest_log_dist = float(np.hypot(d_li[near], d_sd[near]))

    summary = dict(
        gate=cfg["gate"], analysis_status=cfg["analysis_status"],
        object=cfg["object"], n_lights=int(cfg["n_lights"]),
        n_pixels_used=int(keep.sum()),
        n_pixels_dropped_saturated=int((sat_pix & mask).sum()),
        calibrator=dict(iterations=int(n_iters),
                        convergence_tol=float(cfg["rho_convergence_tol"])),
        per_light=dict(
            direction_error_deg=[round(float(x), 4) for x in dir_err_deg],
            logI_deviation=[round(float(x), 6) for x in logI_dev]),
        measured_anchor=dict(
            sig_logI=sig_logI, sig_dir_deg=sig_dir_deg,
            note="std(log e/e_GT) after log-domain scalar rescale; "
                 "RMS direction error over 96 lights"),
        het_anchor=het_anchor,
        gt_comparison=dict(
            rms_direction_error_deg=rms_dir,
            max_direction_error_deg=float(dir_err_deg.max()),
            median_direction_error_deg=float(np.median(dir_err_deg)),
            rms_relative_intensity_error=rms_int,
            log_domain_scalar=float(np.exp(c))),
        family_placement=dict(
            nearest_grid_point=nearest_point,
            log10_distance=round(nearest_log_dist, 4),
            note="nearest point of the frozen S1 45-point family grid "
                 "in (log sig_logI, log sig_dir_deg) space"),
        direction_share_at_anchor=dict(
            per_object={k: round(v, 6) for k, v in shares.items()},
            median=round(med_share, 6),
            definition="[D(sI,sd) - D(sI,closed_dir)] / D(sI,sd) via the "
                       "frozen channel-decomposition functional on the "
                       "OpenIllumination cohort"),
        outcome=outcome,
        outcome_rule=cfg["outcome_rule"].strip(),
        manifest=dict(
            config_sha256=_sha(Path(config_path)),
            git_sha=sha_at_launch,
            elapsed_s=round(time.time() - t_start, 1)),
        note="Anchors Sigma_phi to the measured errors of a standard "
             "sphere photometric-stereo calibration on DiLiGenT ballPNG "
             "(96 lights, GT normals): alternating LSQ rank-1 "
             "factorization on unsaturated lit pixels; the estimator "
             "never sees GT (GT enters only the preregistered lit mask "
             "and the final comparison). The measured anchor is then "
             "read through the frozen family machinery for the "
             "direction-share (C7/C9-D) judgment.")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(json.dumps(summary, ensure_ascii=False,
                                          indent=1).encode("utf-8"))
    print(f"[ball] anchor: sig_logI={sig_logI:.4f}, "
          f"sig_dir_deg={sig_dir_deg:.3f}")
    print(f"[ball] GT check: RMS dir {rms_dir:.3f} deg (max "
          f"{dir_err_deg.max():.2f}), RMS rel int {rms_int:.4f}")
    print(f"[ball] direction share at anchor: median "
          f"{med_share * 100:.3f}% -> {outcome}")
    print(f"[ball] nearest family grid point: "
          f"({nearest_point['sig_logI']}, {nearest_point['sig_dir_deg']}) "
          f"@ log10 dist {nearest_log_dist:.2f}")
    print(f"[ball] wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/ball_anchor.yaml"))
    ap.add_argument("--out", default=str(REPO / "results/openillumination/"
                                           "ball_anchor.json"))
    args = ap.parse_args()
    run(args.config, args.out)
