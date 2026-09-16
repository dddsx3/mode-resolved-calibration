"""P-ANCHOR-MECHANISM · 为什么同一实测 σ 剖面下方向通道在 OI 承载 36%、
在 DiLiGenT ≈0?(验收 530ff991 §5 留下的唯一实质问题)。

两个队列共享实测锚点 (σ_logI 0.0159, σ_dir 2.96°),但方向份额 OI 中位
36.0%(4.8–71.4%)vs DiLiGenT ≈0(cat 22.1%/pot1 29.5% 是仅有的高值)。
差异必来自**场景**。本实验用**与 D 功能量无关**的标称场景统计量(无
循环论证)对逐物体份额做相关,检验哪个场景量解释队列间差异。

候选统计量(全部由标称场景算出):
  finf_median_log10   Fisher 信息水平(log10 中位 Finf)
  finf_spread         Finf p90/p10
  light_spread_deg    active 灯方向的平均两两夹角
  light_isotropy      Σ d_k d_kᵀ/L 的最小/最大特征值
  dir_int_weak_ratio  **核心候选**:方向 nuisance 与强度 nuisance 在
                      bottom-5 弱模式子空间上的白化投影能量之比
  gauge_cos           每灯白化强度列与其 2D 方向 nuisance 子空间的
                      平均 |cos|(关闭一个通道在多大程度上同时移除
                      另一个通道的杠杆)

预注册判读:某统计量 pooled |Spearman| ≥ 0.7 且两队列内符号一致 ⇒
mechanism-supported;否则 C11 停在物体级条件性(验收明确可接受)。

输出: results/openillumination/anchor_mechanism.json
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

from calibinfo.datasets.diligent_oi_adapter import (               # noqa: E402
    load_diligent_as_oi)
from calibinfo.datasets.openillumination import load_object        # noqa: E402
from experiments.openillumination_validation import (              # noqa: E402
    N_MODES, NominalScene)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, cwd=str(REPO)).stdout.strip()


def scene_stats(scen):
    """标称场景统计量(与 D 功能量无关)。"""
    finf = np.asarray(scen.Finf_diag, float)
    d = np.asarray(scen.dirs, float)
    # 弱模式子空间(白化 F∞^{-1/2} 基下最小特征方向的 5 维)
    Finv_h = 1.0 / np.sqrt(finf)
    W = np.diag(Finv_h)
    # 弱模式:取 F∞^{1/2} 白化下 nuisance 最弱的 5 个方向 —— 与冻结
    # 判据同源:predicted_degradation 的 bottom-5(dual 坐标)。
    # 这里不重算 ΔF(那是 D 功能量),改用**几何**弱模式:
    # 白化设计矩阵 U 的最小奇异方向。U_k = sqrt(w_k) B_phi[k]。
    L = scen.B_phi.shape[0]
    U = np.concatenate([np.sqrt(scen.w[k])[:, None] * scen.B_phi[k]
                        for k in range(L)], axis=1)          # (P, 3L)
    # 白化
    Uw = W @ U
    # 弱方向 = Uw Uwᵀ 的最小特征方向(等价于右奇异向量的最小奇异值)
    # 用 SVD 拿右奇异向量再映回参数空间
    try:
        _u, _s, Vt = np.linalg.svd(Uw, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    # 参数空间的弱方向:V = Uw 的右奇异向量 → 参数方向 = V
    Vw = Vt[:N_MODES].T                                      # (3L, 5)
    # 每灯的白化 B 列(3 列/灯),投影到弱子空间的能量
    e_int, e_dir = [], []
    for k in range(L):
        cols = Uw[:, 3 * k:3 * k + 3]                        # (P, 3)
        proj = Vw[3 * k:3 * k + 3]                           # (3, 5)
        # 强度列(第 0 列)与方向列(1,2)
        c_int = cols[:, 0]
        c_dir = cols[:, 1:3]
        # 白化空间中的弱模式能量:把每列投影到 Vw 定义的弱方向
        # (注意:Vw 是参数空间方向,需在参数空间比较)
        p_int = proj[0:1, :]                                 # (1,5)
        p_dir = proj[1:3, :]                                 # (2,5)
        e_int.append(float(np.sum(p_int ** 2)))
        e_dir.append(float(np.sum(p_dir ** 2)))
    e_int = np.asarray(e_int)
    e_dir = np.asarray(e_dir)
    dir_int_weak_ratio = float(e_dir.sum() / max(e_int.sum(), 1e-30))

    # gauge_cos:每灯白化强度列与其方向子空间(白化)的夹角
    cos_vals = []
    for k in range(L):
        cols = Uw[:, 3 * k:3 * k + 3]
        c0 = cols[:, 0]
        cd = cols[:, 1:3]
        # 方向子空间的正交投影
        Q, _ = np.linalg.qr(cd)
        n0 = np.linalg.norm(c0)
        if n0 < 1e-30:
            continue
        cos_vals.append(float(np.linalg.norm(Q.T @ c0) / n0))
    gauge_cos = float(np.mean(cos_vals)) if cos_vals else float("nan")

    # 几何量
    D = d / np.linalg.norm(d, axis=1, keepdims=True)
    G = D @ D.T
    iu = np.triu_indices(D.shape[0], k=1)
    light_spread_deg = float(np.degrees(np.arccos(
        np.clip(G[iu], -1, 1))).mean())
    M = (D[:, :, None] * D[:, None, :]).sum(0) / D.shape[0]
    ev = np.linalg.eigvalsh(M)
    light_isotropy = float(ev[0] / max(ev[-1], 1e-30))

    return dict(
        finf_median_log10=float(np.log10(np.median(finf))),
        finf_spread=float(np.percentile(finf, 90)
                          / max(np.percentile(finf, 10), 1e-30)),
        light_spread_deg=light_spread_deg,
        light_isotropy=light_isotropy,
        dir_int_weak_ratio=dir_int_weak_ratio,
        gauge_cos=gauge_cos,
        n_active=int(L))


def run(config_path=REPO / "configs/anchor_mechanism.yaml",
        out_path=REPO / "results/openillumination/anchor_mechanism.json"):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    t_start = time.time()
    sha_at_launch = _git_sha()

    # 份额(从两份已入库产物读,不重算)
    oi_art = json.loads((REPO / cfg["oi_shares_artifact"])
                        .read_text(encoding="utf-8"))
    dq_art = json.loads((REPO / cfg["diligent_shares_artifact"])
                        .read_text(encoding="utf-8"))
    oi_shares = oi_art["direction_share_at_anchor"]["per_object"]
    dq_shares = dq_art["ball_anchor_share"]["per_object"]

    rows = []
    for tag, root, cohort, K, seed in (
            ("oi", cfg["oi_data_root"], cfg["oi_cohort"], int(cfg["K_oi"]),
             [20260910]),
            ("dq", cfg["diligent_data_root"], cfg["diligent_cohort"],
             int(cfg["K_dq"]), [20260915])):
        for i, name in enumerate(cohort):
            if tag == "oi":
                obj = load_object(root, name, data_meta=cfg.get("oi_data_meta"))
                scen = NominalScene(obj, np.random.default_rng(seed + [i]),
                                    noise_fit_convention=cfg[
                                        "noise_fit_convention"])
            else:
                obj = load_diligent_as_oi(Path(root) / name)
                scen = NominalScene(obj, np.random.default_rng(seed + [i]),
                                    noise_fit_convention=cfg[
                                        "noise_fit_convention"],
                                    n_lights_total=K)
            st = scene_stats(scen)
            share = (oi_shares if tag == "oi" else dq_shares)[name]
            rows.append(dict(cohort=tag, object=name, share=float(share),
                             **st))
            print(f"[mech] {tag}/{name}: share={share*100:+.1f}% "
                  f"ratio={st['dir_int_weak_ratio']:.3f} "
                  f"gauge_cos={st['gauge_cos']:.3f} "
                  f"({time.time()-t_start:.0f}s)", flush=True)

    stats_names = ("finf_median_log10", "finf_spread", "light_spread_deg",
                   "light_isotropy", "dir_int_weak_ratio", "gauge_cos")
    shares = np.array([r["share"] for r in rows])
    corr = {}
    for sn in stats_names:
        vals = np.array([r[sn] for r in rows])
        pooled = spearmanr(vals, shares).statistic
        per = {}
        for tag in ("oi", "dq"):
            m = np.array([r["cohort"] == tag for r in rows])
            if m.sum() >= 4:
                per[tag] = float(spearmanr(vals[m], shares[m]).statistic)
            else:
                per[tag] = None
        corr[sn] = dict(
            pooled=round(float(pooled), 6),
            per_cohort={k: (round(v, 6) if v is not None else None)
                        for k, v in per.items()},
            consistent_sign=bool(
                per["oi"] is not None and per["dq"] is not None
                and np.sign(per["oi"]) == np.sign(per["dq"])
                and np.sign(pooled) == np.sign(per["oi"])))

    supported = {sn: c for sn, c in corr.items()
                 if abs(c["pooled"]) >= 0.7 and c["consistent_sign"]}
    outcome = ("mechanism-supported" if supported
               else "no-single-statistic")

    summary = dict(
        gate=cfg["gate"], analysis_status=cfg["analysis_status"],
        anchor=cfg["anchor"], n_objects=len(rows),
        rows=rows, correlations=corr,
        mechanism_supported=sorted(supported.keys()),
        outcome=outcome,
        outcome_rule=cfg["outcome_rule"].strip(),
        reading=(
            f"the cohort gap is explained by {sorted(supported.keys())}"
            if supported else
            "no single scene statistic reaches |rho| >= 0.7 with a "
            "consistent sign across both cohorts; the anchor share stays "
            "object-conditional (C11 keeps its object-level caveat)"),
        manifest=dict(
            config_sha256=_sha(Path(config_path)),
            git_sha=sha_at_launch,
            oi_shares_sha256=_sha(REPO / cfg["oi_shares_artifact"]),
            diligent_shares_sha256=_sha(REPO / cfg["diligent_shares_artifact"]),
            elapsed_s=round(time.time() - t_start, 1)),
        note="Scene statistics (all from the nominal scene, none from the "
             "D functional) correlated against the per-object direction "
             "share at the shared measured anchor, to test which scene "
             "property explains the OI (36% median) vs DiLiGenT (~0) gap. "
             "Preregistered rule: |Spearman| >= 0.7 pooled with consistent "
             "per-cohort sign => mechanism-supported.")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(json.dumps(summary, ensure_ascii=False,
                                          indent=1).encode("utf-8"))
    print(f"[mech] correlations:")
    for sn, c in corr.items():
        print(f"  {sn}: pooled {c['pooled']:+.3f} | "
              f"oi {c['per_cohort']['oi']} dq {c['per_cohort']['dq']} | "
              f"consistent {c['consistent_sign']}")
    print(f"[mech] outcome: {outcome} {summary['mechanism_supported']}")
    print(f"[mech] wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",
                    default=str(REPO / "configs/anchor_mechanism.yaml"))
    ap.add_argument("--out", default=str(REPO / "results/openillumination/"
                                   "anchor_mechanism.json"))
    args = ap.parse_args()
    run(args.config, args.out)
