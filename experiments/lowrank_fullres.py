"""P-LOWRANK-FULLRES · 全分辨率 / 全 142 灯认证表（plan v2 M2）。

消除 P-CERT v1 的"只在 1200 子采样像素上成立"攻击面：
  - 像素 = 全部 masked 像素（无子采样）；灯 = 全部 142 盏（无分析子集，
    每盏都 Fisher-active）；
  - J_A/梯度走 exact low-rank（Woodbury）路线——稠密 P×P 在 P ≈ 3 万时
    不可行（22 GB），本路线 O(P(3L)²)；
  - nominal 朗伯 PS 用批量正规方程版（calibrated_ps_fullres，语义同
    calibrated_ps：数据驱动初值 + 交替最小二乘 + n 归一 + 奇异回退 +z）。

协议预注册于 configs/lowrank_fullres.yaml（run 前提交）。输出：
  results/certification/lowrank_fullres.json
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

from calibinfo.allocation.convex import CertificateProblem, budget_for_k     # noqa: E402
from calibinfo.allocation.blocks import LightBlocks                          # noqa: E402
from calibinfo.datasets.openillumination import load_object                  # noqa: E402
from calibinfo.models.corruption import CorruptionGenerator                  # noqa: E402
from experiments.openillumination_validation import noise_fit                # noqa: E402


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, cwd=str(REPO)).stdout.strip()


def calibrated_ps_fullres(I, dirs, iters=25):
    """全分辨率逐像素朗伯 PS（批量正规方程；语义同 calibrated_ps）。

    I: (L,P) 灰度；dirs: (L,3)。返回 (rho (P,), n (P,3))。
    与逐像素 lstsq 版的差异：正规方程批量求解 + det 失效回退 +z
    （预注册声明于 config 的 scene_rng_spec/说明）。"""
    L, P = I.shape
    n = np.tile(np.array([0.0, 0.0, 1.0]), (P, 1))
    rho = np.maximum(I.mean(0), 1e-6)
    eye3 = np.eye(3)
    for _ in range(iters):
        ndl = np.clip(n @ dirs.T, 0, None)                       # (P,L)
        denom = np.maximum((ndl ** 2).sum(1), 1e-12)
        rho = np.maximum((ndl * I.T).sum(1) / denom, 1e-6)
        h = (ndl > 0).astype(float)                              # (P,L)
        Mm = np.einsum("pk,ki,kj->pij", h, dirs, dirs)           # (P,3,3)
        rhs = (h * (I.T / rho[:, None])) @ dirs                  # (P,3)
        det = (Mm[:, 0, 0] * (Mm[:, 1, 1] * Mm[:, 2, 2] - Mm[:, 1, 2] * Mm[:, 2, 1])
               - Mm[:, 0, 1] * (Mm[:, 1, 0] * Mm[:, 2, 2] - Mm[:, 1, 2] * Mm[:, 2, 0])
               + Mm[:, 0, 2] * (Mm[:, 1, 0] * Mm[:, 2, 1] - Mm[:, 1, 1] * Mm[:, 2, 0]))
        scale = np.maximum(np.abs(Mm).max(axis=(1, 2)), 1e-300) ** 3
        ok = np.abs(det) > 1e-10 * scale
        n_new = np.tile(np.array([0.0, 0.0, 1.0]), (P, 1))
        if ok.any():
            n_new[ok] = np.linalg.solve(Mm[ok] + 1e-12 * scale[ok, None, None]
                                        * eye3, rhs[ok][..., None])[..., 0]
        nv = np.linalg.norm(n_new, axis=1)
        good = ok & (nv > 1e-9)
        n[good] = n_new[good] / nv[good, None]
        n[~good] = np.array([0.0, 0.0, 1.0])
    return rho, n


def build_fullres_state(obj, cfg):
    """全分辨率 nominal 资产 → CertificateProblem（woodbury 路线）。"""
    img = obj["images"][..., 0]
    mask = obj["mask"]
    I_all = img[:, mask]                                    # (142, P_full)
    dirs = obj["light_directions"]                          # (142,3)
    L, P = I_all.shape
    rho, n = calibrated_ps_fullres(I_all, dirs,
                                   iters=int(cfg.get("ps_iters", 25)))
    s_hat = np.clip(n @ dirs.T, 0, None).T                  # (L,P)
    a, b = noise_fit(I_all, s_hat * rho[None, :], convention="corrected")
    w = 1.0 / np.maximum(a + b * I_all, 1e-6)
    # 切向基（同 validation._tangents）
    t1 = np.zeros_like(dirs)
    t2 = np.zeros_like(dirs)
    for k, dk in enumerate(dirs):
        ref = np.array([0.0, 0.0, 1.0]) if abs(dk[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        v = np.cross(dk, ref)
        v /= np.linalg.norm(v)
        t1[k] = v
        t2[k] = np.cross(dk, v)
    h = (s_hat > 0).astype(float)
    B_phi = np.stack([s_hat * rho[None, :],
                      (n @ t1.T).T * rho[None, :] * h,
                      (n @ t2.T).T * rho[None, :] * h], axis=-1)   # (L,P,3)
    finf = (w * s_hat ** 2).sum(0)
    if finf.min() <= 0:
        raise ValueError("全分辨率下存在 F∞=0 像素——按纪律截断并记录，禁静默")
    u = (w * s_hat)[:, :, None] * B_phi
    M0 = np.einsum("kpi,kpj->kij", B_phi, w[:, :, None] * B_phi)
    gen = CorruptionGenerator("joint", float(cfg["level"]))
    lam0 = np.stack([np.linalg.inv(gen.sigma_phi_diag())] * L)
    blocks = LightBlocks(u, M0, lam0, finf, np.ones(L, bool))
    prob = CertificateProblem(blocks=blocks, kappa=float(cfg["kappa"]),
                              route="woodbury")
    meta = dict(P=P, L=L, finf_min=float(finf.min()),
                finf_max=float(finf.max()), noise_fit=(float(a), float(b)))
    return prob, meta


def _apply_t(prob, lights, kappa):
    t = np.ones(prob.blocks.L)
    for idx in lights:
        t[idx] = kappa
    return t


def _greedy_steepest_prefix(prob, kappa, k_max):
    """最陡下降前缀 greedy（全分辨率低成本变体）：每步取精确梯度最负的
    未选 active 灯，立即 t=κ。与 P-CERT v1 的 exact-gain greedy 不同口径，
    预注册声明于 config。"""
    t = np.ones(prob.blocks.L)
    chosen, out = [], {}
    for step in range(k_max):
        g = prob.grad_J_A(t)
        g_masked = np.where((~np.isin(np.arange(len(g)), chosen))
                            & prob.blocks.active, g, np.inf)
        pick = int(np.argmin(g_masked))
        t[pick] = kappa
        chosen.append(pick)
        out[step + 1] = (list(chosen), prob.J_A(t))
    return out


def run(config_path=REPO / "configs/lowrank_fullres.yaml",
        out_path=REPO / "results/certification/lowrank_fullres.json"):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    kappa = float(cfg["kappa"])
    budgets_k = list(cfg["budgets_k"])
    k_max = max(budgets_k)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows, scene_meta = [], {}
    t_start = time.time()
    for obj_idx, obj_name in enumerate(cfg["cohort"]):
        obj = load_object(cfg["data_root"], obj_name, data_meta=cfg.get("data_meta"))
        # 全分辨率协议无子采样：scene_rng_spec 的 rng 在此协议下不消耗
        #（nominal PS 是确定性的批量交替最小二乘）
        prob, meta = build_fullres_state(obj, cfg)
        scene_meta[obj_name] = meta
        active = list(range(prob.blocks.L))                  # 全 142 灯 active
        J_none = prob.J_A(np.ones(prob.blocks.L))
        J_all = prob.J_A(np.full(prob.blocks.L, kappa))
        greedy = _greedy_steepest_prefix(prob, kappa, k_max)
        rng = np.random.default_rng(int(cfg["random_seed"]) + obj_idx)
        for k in budgets_k:
            B = budget_for_k(k, kappa)
            fw = prob.frank_wolfe(B, iters=int(cfg["fw_iters"]))
            g_lights, J_greedy = greedy[k]
            rand_J = [prob.J_A(_apply_t(
                prob, list(rng.choice(active, size=k, replace=False)), kappa))
                for _ in range(int(cfg["n_random"]))]
            rows.append(dict(
                object=obj_name, k=k, B=B, P=meta["P"], L=meta["L"],
                J_none=J_none, J_all=J_all,
                fw_J_A=fw["J_A"], fw_gap=fw["gap"],
                fw_lower_bound=fw["J_A"] - fw["gap"],
                greedy_J_A=J_greedy, greedy_lights=g_lights,
                random_mean=float(np.mean(rand_J)),
                random_min=float(np.min(rand_J)),
            ))
        print(f"[fullres] {obj_name} (P={meta['P']}): dynamic range "
              f"{(J_none - J_all) / abs(J_none) * 100:.2f}% "
              f"elapsed={time.time() - t_start:.0f}s", flush=True)

    by_k = {}
    for k in budgets_k:
        rs = [r for r in rows if r["k"] == k]
        dyn = [(r["J_none"] - r["J_all"]) / abs(r["J_none"]) * 100 for r in rs]
        g = [(r["greedy_J_A"] - r["fw_lower_bound"]) / abs(r["J_none"]) * 100
             for r in rs]
        rm = [(r["random_mean"] - r["fw_lower_bound"]) / abs(r["J_none"]) * 100
              for r in rs]
        by_k[k] = dict(dynamic_range_pct=dict(median=float(np.median(dyn)),
                                              min=float(np.min(dyn)),
                                              max=float(np.max(dyn))),
                       greedy_above_lower_pct=dict(median=float(np.median(g)),
                                                   max=float(np.max(g))),
                       random_mean_above_lower_pct=dict(
                           median=float(np.median(rm)),
                           max=float(np.max(rm))))
    display = {str(k): dict(
        dynamic_range_median_pct=round(by_k[k]["dynamic_range_pct"]["median"], 2),
        greedy_above_lower_median_pct=round(by_k[k]["greedy_above_lower_pct"]["median"], 3),
        random_mean_above_lower_median_pct=round(
            by_k[k]["random_mean_above_lower_pct"]["median"], 3),
    ) for k in budgets_k}

    summary = dict(
        gate="P-LOWRANK-FULLRES v1: certified gaps at full resolution",
        analysis_status="lowrank_fullres_v1",
        objective=cfg["objective"], route="woodbury",
        level=float(cfg["level"]), kappa=kappa, budgets_k=budgets_k,
        noise_fit_convention=cfg["noise_fit_convention"],
        n_objects=len(cfg["cohort"]),
        by_k=by_k, display=display, rows=rows, scene_meta=scene_meta,
        manifest=dict(config_sha256=_sha(Path(config_path)),
                      git_sha=_git_sha(),
                      random_seed=int(cfg["random_seed"]),
                      elapsed_s=round(time.time() - t_start, 1)),
        note="Full-resolution (no pixel subsampling) / full-142-light "
             "certified gap table via the exact low-rank Woodbury route. "
             "No sign-based gate; every row reported.")
    out.write_bytes(json.dumps(summary, ensure_ascii=False, indent=1)
                    .encode("utf-8"))
    for k in budgets_k:
        print(f"[fullres] k={k}: dyn median "
              f"{by_k[k]['dynamic_range_pct']['median']:.2f}% | "
              f"greedy-above-lower median "
              f"{by_k[k]['greedy_above_lower_pct']['median']:.4f}%")
    print(f"[fullres] wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/lowrank_fullres.yaml"))
    ap.add_argument("--out", default=str(REPO / "results/certification/lowrank_fullres.json"))
    args = ap.parse_args()
    run(args.config, args.out)
