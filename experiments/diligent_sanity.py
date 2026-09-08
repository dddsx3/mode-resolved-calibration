"""DiLiGenT sanity check and robustness ablation (CI05).

run_sanity(config, run_dir): external sanity on real DiLiGenT objects -
  weak-mode existence, gauge cross-observability, and failure taxonomy
  (model mismatch vs calibration propagation), reported within-scene.
run_ablation(config, run_dir): synthetic robustness ablations -
  whitening vs raw, lambda misspecification (4x, single-scene second order),
  light-count sweep, and noise-model sweep.
"""
import json
import numpy as np
from calibinfo.datasets.synthetic import make_scene, sh_basis
from calibinfo.information.gauge import gauge_response
from calibinfo.information.retention import retention_spectrum
from calibinfo.information.schur import delta_f
from pathlib import Path


from calibinfo.datasets.diligent import load_object
from calibinfo.datasets.synthetic import sh_basis

DATA_ROOT = "D:/data/DiLiGenT/pmsData"
N_LIGHTS_SUB = 48
N_PIXEL_SUB = 1200


def run_sanity(config, run_dir):
    rng = np.random.default_rng(config["seed"])
    data_root = config.get("data_root", DATA_ROOT)
    objects = config.get("objects") or sorted(
        d.name for d in Path(data_root).iterdir() if d.is_dir())
    rows = []
    for name in objects:
        obj = load_object(Path(data_root) / name)
        dirs = obj["dirs"]
        I = obj["I_norm"]                                   # (96, P_full)
        n_gt = obj["normals_gt"]
        sel = np.sort(rng.choice(96, N_LIGHTS_SUB, replace=False))
        P_full = I.shape[1]
        pidx = np.sort(rng.choice(P_full, min(P_full, N_PIXEL_SUB), replace=False))
        d_sub = dirs[sel]
        I_s = I[sel][:, pidx]
        n_s = n_gt[pidx]
        # 朗伯残差（exp8R calibrate 同口径：全 96 光）
        nl_full = np.clip(n_gt @ dirs.T, 0, None)
        rho_full = (I.T * nl_full).sum(1) / np.maximum((nl_full * nl_full).sum(1), 1e-12)
        I_hat = (rho_full[:, None] * nl_full).T
        resid = I - I_hat
        lambert_resid = float(np.linalg.norm(resid) / np.linalg.norm(I))
        r3 = np.abs(resid)
        outlier_frac = float((r3 > 3 * r3.std()).mean())
        # 弱模式结构（within-scene）：A=D(ŝ), B=ρ̂h·Y（geometry-known——DiLiGenT 有法线 GT，
        # 这是与 exp8R 的合法差别；GT 用在标定侧不作估计初始化）
        s_hat = np.clip(n_s @ d_sub.T, 0, None).T          # (L,P)
        A_st = np.vstack([np.diag(s_k) for s_k in s_hat])
        Y = sh_basis(n_s)
        h = (s_hat > 0).astype(float)
        B_blk = np.zeros((len(sel) * len(pidx), len(sel) * 9))
        rho_s = rho_full[pidx]
        for k in range(len(sel)):
            B_blk[k * len(pidx):(k + 1) * len(pidx), k * 9:(k + 1) * 9] = \
                (rho_s * h[k])[:, None] * Y
        # 弱模式谱（逐灯 Schur，避免 N·P 全矩阵——OOM 教训：M=eye(N·P) 24.7GB）
        # Λ→0 极限下 ΔF 的谱 = Σ_k D(ŝ_k) 投影掉 col(B_k) 后的谱（V1b 块对角恒等式）；
        # 全体非零特征值由各灯块非零谱并集近似其分布（floor 取全体像素白化谱的分位）：
        ev_all = []
        for k in range(len(sel)):
            Ak = s_hat[k]
            Bk = B_blk[k * len(pidx):(k + 1) * len(pidx), k * 9:(k + 1) * 9].copy()
            DFk, _, _ = delta_f(np.diag(Ak), Bk, np.full((9, 9), 1e-12))
            evk = np.linalg.eigvalsh(DFk)
            ev_all.append(evk[evk > 1e-10 * max(evk.max(), 1e-300)])
        ev_all = np.concatenate(ev_all)
        pos = ev_all[ev_all > 1e-10 * max(ev_all.max(), 1e-300)]
        floor = float(np.percentile(pos, 1)) if pos.size else float("nan")
        med = float(np.median(pos)) if pos.size else float("nan")
        # gauge 交叉（首灯，与 CI02 同口径）
        c0 = sh_basis(d_sub[0][None, :])[0]
        resp = gauge_response(B_blk[:, :9], -c0, 1.0)
        lam_star_pred = floor / resp["slope"] if resp["slope"] > 0 else float("nan")
        # 失败分类（量级对比，仅 taxonomy）
        # 模型失配方差：朗伯残差方差（截断 3σ 内的稳健口径）
        miss_var = float(np.median(r3[r3 <= 3 * r3.std()] ** 2))
        # 校准传播方差：弱模式在 λ⋆ 预设不确定度下的方差 σ²/λ_j —— 用 floor 的对偶
        calib_var = float(1.0 / floor) if floor > 0 else float("nan")
        taxonomy = ("model_mismatch_dominant" if miss_var > calib_var
                    else "calibration_propagation_comparable")
        rows.append(dict(object=name, P=int(len(pidx)), L=int(len(sel)),
                         lambert_residual=lambert_resid, outlier_frac=outlier_frac,
                         weak_mode_floor=floor, weak_mode_median=med,
                         floor_over_med=float(floor / med) if med else float("nan"),
                         gauge_slope=resp["slope"], saturation=resp["saturation"],
                         lam_star_pred=lam_star_pred,
                         miss_var=miss_var, calib_var=calib_var,
                         taxonomy=taxonomy))
    return dict(run_name=config.get("run_name", "sanity"), seed=config["seed"],
                n_objects=len(rows), rows=rows,
                note="DiLiGenT 只承担 real sanity（§4.5）：弱模式存在性 + gauge "
                     "交叉 + 失败分类（模型失配 vs 校准传播，量级对比，不混写因果）")


# --------------------------------------------------------------- ablation
import numpy as np

from calibinfo.information.gauge import gauge_response
from calibinfo.information.schur import delta_f


def _weak_spectra(sc, lam, use_whitening=True, Sigma_y=None):
    """逐灯 Schur → ΔF 与弱 5 模式 retention 谱（within-scene）。"""
    P = sc["P"]
    q = sc["B_blk"].shape[1]
    per_l = sc["meta"]["n_lights"]
    ss = np.asarray(sc["ss"])                       # (L,P) 工厂返回 list → 数组化
    DF = np.zeros((P, P))
    for k in range(per_l):
        sl = slice(k * P, (k + 1) * P)
        Bk = sc["B_blk"][sl, k * 9:(k + 1) * 9]
        Ak = np.diag(ss[k])
        if use_whitening and Sigma_y is not None:
            wk = 1.0 / np.sqrt(Sigma_y[sl])          # 白化行权
            Ak, Bk = wk[:, None] * Ak, wk[:, None] * Bk
        DFk, _, _ = delta_f(Ak, Bk, lam * np.eye(9))
        DF += DFk
    if Sigma_y is None:
        Finf = np.diag((ss ** 2).sum(0))
    else:
        Finf = np.diag(((1.0 / Sigma_y.reshape(per_l, P)) * ss ** 2).sum(0))
    out = retention_spectrum(DF, Finf)
    return DF, out


def run_ablation(config, run_dir):
    rng = np.random.default_rng(config["seed"])
    rows = {k: [] for k in ("whitening_vs_raw", "lambda_misspec",
                            "light_count", "noise_model")}
    for si in range(config.get("n_scenes", 4)):
        sc = make_scene(np.random.default_rng(rng.integers(0, 2**63 - 1)),
                        config.get("P", 300), 3, geometry=config.get("geometry", "bumpy"))
        P = sc["P"]
        m_obs = sc["A_st"].shape[0]
        # 公共噪声口径：异方差 a+bI
        sig = config.get("sigma", 0.02)
        Sigma_y = np.full(m_obs, sig ** 2)
        Sigma_y[:m_obs // 2] *= 4.0                    # 对角异方差（两档）

        # 1 whitening vs raw λI：同一 λ，白化 vs 不白化，弱模式 retention 差
        s_med2 = float(np.median(np.linalg.eigvalsh(
            sc["B_blk"].T @ sc["B_blk"])))
        lam0 = s_med2 / 100
        _, w_out = _weak_spectra(sc, lam0, True, Sigma_y)
        _, r_out = _weak_spectra(sc, lam0, False)
        rows["whitening_vs_raw"].append(dict(
            scene=si, lam=lam0,
            rho_whiten=w_out["rho"][:5].tolist(),
            rho_raw=r_out["rho"][:5].tolist(),
            max_abs_diff=float(np.abs(w_out["rho"][:5] - r_out["rho"][:5]).max())))

        # 2 Λ 4× misspec（N5 单场景二阶口径）
        def tr_weak(lam):
            DF, _ = _weak_spectra(sc, lam, True, Sigma_y)
            ev = np.linalg.eigvalsh(DF)
            pos = ev[ev > 1e-10 * max(ev.max(), 1e-300)]
            return float((1.0 / pos[pos <= np.percentile(pos, 20)]).sum())
        t_true, t_miss = tr_weak(lam0), tr_weak(4 * lam0)
        rows["lambda_misspec"].append(dict(
            scene=si, lam=lam0,
            weak_tr_pinv_true=t_true, weak_tr_pinv_4x=t_miss,
            rel_change_pct=float(100 * (t_miss - t_true) / t_true)))

        # 3 灯数扫描（within-scene：用每灯独立块，取前 N 灯）
        for N in config.get("light_counts", [1, 2, 3]):
            sc_n = make_scene(np.random.default_rng(rng.integers(0, 2**63 - 1)),
                              config.get("P", 300), N, geometry=config.get("geometry", "bumpy"))
            e = np.linalg.eigvalsh(sc_n["B_blk"].T @ sc_n["B_blk"])
            s2 = float(np.median(e[e > 1e-3 * e.max()])) if (e > 1e-3 * e.max()).any() else 1.0
            lam_n = s2 / 100
            DFn, o = _weak_spectra(sc_n, lam_n, True, np.full(sc_n["A_st"].shape[0], sig ** 2))
            resp = gauge_response(sc_n["B_blk"][:sc_n["P"], :9],
                                  sc_n["gauge_cbar"][:9], lam_n)
            evn = np.linalg.eigvalsh(DFn)
            pos = evn[evn > 1e-10 * max(evn.max(), 1e-300)]
            rows["light_count"].append(dict(
                scene=si, n_lights=N, rho5=o["rho"][:5].tolist(),
                weakest_mode=float(pos[0]) if pos.size else float("nan")))

        # 4 噪声模型口径（同方差 vs 异方差白化）——retention 谱稳定性
        _, het = _weak_spectra(sc, lam0, True, Sigma_y)
        _, hom = _weak_spectra(sc, lam0, True, np.full(m_obs, sig ** 2))
        rows["noise_model"].append(dict(
            scene=si, rho_het=het["rho"][:5].tolist(), rho_hom=hom["rho"][:5].tolist(),
            max_abs_diff=float(np.abs(het["rho"][:5] - hom["rho"][:5]).max())))

    # 汇总（效应量 median/IQR，不设胜负判据）
    summary = {}
    wr = rows["whitening_vs_raw"]
    d = [r["max_abs_diff"] for r in wr]
    summary["whitening_vs_raw"] = dict(
        median=float(np.median(d)), iqr=[float(np.percentile(d, 25)),
                                         float(np.percentile(d, 75))], n=len(d))
    lm = rows["lambda_misspec"]
    d = [r["rel_change_pct"] for r in lm]
    summary["lambda_misspec"] = dict(
        median_pct=float(np.median(d)), iqr_pct=[float(np.percentile(d, 25)),
                                                 float(np.percentile(d, 75))], n=len(d))
    lc = {}
    for r in rows["light_count"]:
        lc.setdefault(r["n_lights"], []).append(np.log10(max(r["weakest_mode"], 1e-300)))
    summary["light_count"] = {str(k): dict(median_log10=float(np.median(v)),
                                           n=len(v)) for k, v in lc.items()}
    nm = rows["noise_model"]
    d = [r["max_abs_diff"] for r in nm]
    summary["noise_model"] = dict(
        median=float(np.median(d)), iqr=[float(np.percentile(d, 25)),
                                         float(np.percentile(d, 75))], n=len(d))
    return dict(run_name=config.get("run_name", "ablation"), seed=config["seed"],
                summary=summary, rows=rows,
                note="ablation 全部 within-scene；N5 式 4× misspec 单场景二阶，禁泛化（单场景二阶 misspec）")
