"""P-RADIUS · 真实数据线性化有效半径（v2，度量域修正版）。

**勘误（2026-09-13 外部审计 P0-1）**：v1 把腐蚀注入到**估计器侧**
（`estimate_albedo` 用 corrupted d2/g 拟合），强度通道 g=exp(logs·σ·ℓ) 在
大 ℓ 时单灯可达 e^±16，**单个极端灯支配分母 Σw·m² ⇒ ρ̃→0**，E(ℓ) 塌缩
6 个数量级（log-log 斜率 −1.9，与协议要求的 +2 相反）——度量的是**估计器
数值崩塌**，不是线性化失效。v1 结果作废。

**v2 度量域（方案 a，名义几何估计）**：腐蚀注入移到**观测侧**——前向
生成真实观测 I'(k,p) = ρ̂_p·max(n̂_p·d2_k,0)·g_k（真实光来自旋转后的 d2、
强度 g，全非线性模型，含 backface 翻转），分析者按**名义几何**
（dirs、单位增益）估计。估计器分母固定为名义 ŝ²（与 ℓ 无关、正定），
数值崩塌从度量域中移除；E(ℓ) 只反映观测扰动经名义管线的传播，其偏离
一阶标度 = 真实的线性化失效。

问题：以名义标定处的雅可比 Bφ 构造的线性化预测（ΔF 与 pred_deg），
在多大的腐蚀水平 ℓ 上仍与经验估计误差的二阶矩一致？

口径（与 ci04 场景装配一致）：
  - 场景：P=1200 子采样、48 active 灯、corrected 口径、场景 rng 同冻结；
  - 腐蚀：joint（方向旋转 + 强度缩放），水平 ℓ = CorruptionGenerator("joint", ℓ)
    的物理单位（σ_logI = ℓ、σ_deg = ℓ 度）——预注册网格 [0.05 ... 8.0]；
    种子空间与 v1 相同（同 raw innovations，可比）；
  - 估计器：固定 n̂ 白化逐像素 GLS @ 名义几何；残差 e = ρ̃ − ρ̂，gauge
    对齐（同冻结）；经验二阶矩 = 逐 seed 的 e @ W_dual 平方（bottom-5
    dual 子空间能量）的均值 E(ℓ)；
  - 线性化预测：一阶传播给出 dev(ℓ) = E(ℓ)/E(ℓ_ref) ∝ (ℓ/ℓ_ref)²。

度量（v2 语义）：超出因子 q(ℓ) = dev(ℓ) / (ℓ/ℓ_ref)²——线性化生效时
q ≈ 1；q 偏离 1 = 二阶及更高阶项主导。半径 = q 首次超过 {2, 10} 的最小 ℓ
（v1 的「dev 超 2×参考能量」语义在一阶标度下会在 ℓ≈0.07 平凡触发，
随 v1 一并作废）。

输出：results/magnitude/linearization_radius.json（覆盖 v1 无效结果）
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
from calibinfo.datasets.openillumination import load_object            # noqa: E402
from calibinfo.models.corruption import CorruptionGenerator            # noqa: E402
from experiments.openillumination_validation import NominalScene       # noqa: E402

CROSSINGS = (2.0, 10.0)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, cwd=str(REPO)).stdout.strip()


def run(config_path=REPO / "configs/linearization_radius.yaml",
        out_path=REPO / "results/magnitude/linearization_radius.json"):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    levels = [float(lv) for lv in cfg["levels"]]
    seeds = int(cfg["seeds"])
    ref_level = float(cfg.get("reference_level", levels[0]))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    objects = {}
    t_start = time.time()
    for obj_idx, obj_name in enumerate(cfg["cohort"]):
        obj = load_object(cfg["data_root"], obj_name, data_meta=cfg.get("data_meta"))
        scen = NominalScene(obj, np.random.default_rng([20260910, obj_idx]),
                            noise_fit_convention=cfg["noise_fit_convention"])
        rows_obj = []
        for level in levels:
            gen = CorruptionGenerator("joint", level)
            sig_logI, sig_rad = gen.sig_logI, np.radians(gen.sig_deg)
            # 线性化预测：ΔF(ℓ) 的保留谱（Σ_φ = level 的物理单位方差）
            _deg, W_dual = scen.predicted_degradation(
                gen.sigma_phi_diag(), mode_coordinate="dual")
            rho_pred = np.sort(_deg)
            # 经验：seeds 个腐蚀实现（方向 + 强度，联合），观测侧注入 +
            # 名义几何估计（v2，方案 a）
            energies = np.empty(seeds)
            for s_i in range(seeds):
                rng = np.random.default_rng(
                    [20260916, 4242, obj_idx, int(level * 1000), s_i])
                raw = raw_innovations(rng, len(scen.sel))
                scales = np.ones(len(scen.sel))
                d2, g = apply_scaled_corruption(
                    scen.dirs, sig_logI, sig_rad, scales, raw)
                # 观测侧注入：真实光来自 d2/强度 g（全非线性前向，含翻转）
                I_obs = scen.rho[None, :] \
                    * np.maximum(scen.n @ d2.T, 0.0).T * g[:, None]
                # 名义几何估计：分母固定为名义 ŝ²，与 ℓ 无关
                scen_obs = NominalScene._from_arrays(scen, I_obs)
                rho_t = scen_obs.estimate_albedo(
                    scen.dirs, np.ones(len(scen.sel)))
                e = rho_t - scen.rho
                sg = (e * scen.rho).sum() / (scen.rho ** 2).sum()
                e = e - sg * scen.rho
                energies[s_i] = float(np.sum((e @ W_dual) ** 2))
            E_mean = float(energies.mean())
            rows_obj.append(dict(level=level, E_mean=E_mean,
                                 pred_deg_bottom5=[float(x) for x in
                                                   1.0 / rho_pred[:5]],
                                 n_seeds=seeds))
        # 参考能量（最小水平）与一阶标度对齐
        E_ref = rows_obj[0]["E_mean"]
        for r in rows_obj:
            scale2 = (r["level"] / ref_level) ** 2
            r["dev_from_ref"] = r["E_mean"] / E_ref if E_ref > 0 else float("nan")
            r["first_order_scaling"] = scale2
            r["excess_over_scaling"] = (r["dev_from_ref"] / scale2
                                        if E_ref > 0 else float("nan"))
        # 半径：超出因子 q 首次超过 {2, 10} 的最小 ℓ
        radius = {}
        for thr in CROSSINGS:
            hit = [r["level"] for r in rows_obj
                   if r["excess_over_scaling"] > thr]
            radius[f"cross_{thr:g}x"] = (min(hit) if hit else None)
        objects[obj_name] = dict(rows=rows_obj, radius=radius, E_ref=E_ref)
        print(f"[radius] {obj_name}: dev@0.75={rows_obj[5]['dev_from_ref']:.2f} "
              f"(scaling {rows_obj[5]['first_order_scaling']:.0f}) "
              f"q@0.75={rows_obj[5]['excess_over_scaling']:.2f}; "
              f"radius2x={radius['cross_2x']}, radius10x={radius['cross_10x']} "
              f"elapsed={time.time() - t_start:.0f}s", flush=True)

    def _agg(key):
        vals = [v["radius"][key] for v in objects.values()
                if v["radius"][key] is not None]
        return dict(
            median_over_crossed_subset=(float(np.median(vals)) if vals else None),
            n_crossed=len(vals),
            n_total=len(objects),
            per_object={k: v["radius"][key] for k, v in objects.items()})

    summary = dict(
        gate="P-RADIUS: linearization validity radius on real data",
        analysis_status="linearization_radius_v2",
        levels=levels, seeds=seeds,
        noise_fit_convention=cfg["noise_fit_convention"],
        n_objects=len(cfg["cohort"]),
        radius_2x=_agg("cross_2x"),
        radius_10x=_agg("cross_10x"),
        objects=objects,
        manifest=dict(config_sha256=_sha(Path(config_path)),
                      git_sha=_git_sha(), seeds=seeds),
        metric_domain=dict(
            injection="observation-side: I'(k,p) = rho_hat(p) * max(n(p)@d2_k,0)"
                      " * g(k) — full nonlinear forward, backface flips included",
            estimator="nominal-geometry whitened GLS (denominator is the fixed"
                      " nominal s^2, level-independent; estimator numerical"
                      " collapse removed from the metric domain, fixing audit"
                      " P0-1: v1 injected corruption estimator-side and a"
                      " single extreme gain dominated sum(w*m^2))",
            radius_semantics="excess factor q(level) = dev(level) / "
                             "(level/level_ref)^2 over the first-order "
                             "scaling; radius = first level with q > 2 / 10"),
        note="Linearization validity radius: for each corruption level l, "
             "E(l) = mean over seeds of the bottom-5 dual-coordinate energy "
             "of the gauge-aligned residual of the NOMINAL-geometry estimator "
             "on observation-side-corrupted data; dev(l) = E(l)/E(l_min); "
             "q(l) = dev(l) / (l/l_min)^2. First-order theory holds while "
             "q ~ 1; radius = first level where q crosses 2x / 10x. "
             "No sign-based gate; all levels reported.")
    out.write_bytes(json.dumps(summary, ensure_ascii=False, indent=1)
                    .encode("utf-8"))
    print(f"[radius] wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/linearization_radius.yaml"))
    ap.add_argument("--out", default=str(REPO / "results/magnitude/linearization_radius.json"))
    args = ap.parse_args()
    run(args.config, args.out)
