"""P-SIGMA-FAMILY (E 臂) · 缩减决策质量在 Σ_φ 参数族上的稳定性(S2-4 E)。

冻结结论 E(claim B9 informed-only):4 个 informed 策略的预测序 vs 实现
端点序逐对符号一致 678/987 = 0.687(全网格)。本实验把腐蚀形状换成
三轴族成员(anchor 控制 / dir_heavy / het),在缩减网格
(level 0.5, regime 10, budgets [14,28], 4 informed, 10 seeds)上重测
同一统计量。预算选择是功效性的:冻结产物中 dp≠0 只出现在 k=14/28
(k≥57 四序完全重合),[14,28] 承载 100% 可判别对;冻结子集对照
88/125 = 0.7040。

控制臂锚点:anchor 臂的行(orderings + seeds + 注入路径全部退化一致)
必须与冻结 decision_quality.json 子集**逐位一致**——运行时断言;且
pooled 统计量必须精确复现 88/125。

输出: results/openillumination/decision_quality_family.json
(不覆盖冻结的 decision_quality.json)
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

from calibinfo.allocation.corruption import (                        # noqa: E402
    raw_innovations_family, apply_corruption_family)
from calibinfo.allocation.convex import CertificateProblem          # noqa: E402
from calibinfo.allocation.policies import (                         # noqa: E402
    LightBlocks, SelectionState, budget_scales, select_ordering,
    select_ordering_mode_aware)
from calibinfo.datasets.openillumination import load_object         # noqa: E402
from calibinfo.models.corruption_family import CorruptionFamily     # noqa: E402
from experiments.openillumination_validation import NominalScene    # noqa: E402

FROZEN = REPO / "results/openillumination/decision_quality.json"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, cwd=str(REPO)).stdout.strip()


def arm_metrics_family(scales48, raw, scen, sig_I, sig_R, W_dual):
    """arm_metrics 的族版本:唯一差异是注入行换 apply_corruption_family
    (接受逐灯 σ 向量);其余逐语句相同。标量 σ + rho_c=0 的 raw 与
    arm_metrics 逐位一致(S0-2 钉死)。"""
    d2, g = apply_corruption_family(scen.dirs, sig_I, sig_R, scales48, raw)
    rho_t = scen.estimate_albedo(d2, g)
    # --- 一步法线重估(calibrated_ps 的 n-更新;掩码 (I>1e-3)&(n·d2>0))---
    ndl = np.clip(scen.n @ d2.T, 0, None).T                  # (L,P)
    act = (scen.I > 1e-3) & (ndl > 0)
    g_col = (np.broadcast_to(np.asarray(g, float).reshape(-1, 1), ndl.shape)
             if np.ndim(g) == 1 else np.asarray(g, float))
    Y = scen.I / np.maximum(rho_t[None, :] * g_col, 1e-12)   # (L,P) ≈ n·d2
    M = act.astype(float)
    A = np.einsum("lp,li,lj->pij", M, d2, d2)                # (P,3,3)
    b = np.einsum("lp,lp,li->pi", M, Y, d2)                  # (P,3)
    n_sol = np.linalg.pinv(A) @ b[..., None]                 # (P,3,1)
    nrm = np.linalg.norm(n_sol[..., 0], axis=1)
    ok = nrm > 1e-9
    n_est = np.zeros_like(scen.n)
    n_est[ok] = n_sol[ok, :, 0] / nrm[ok, None]
    n_est[~ok] = np.array([0.0, 0.0, 1.0])
    cosang = np.clip((n_est * scen.n).sum(1), -1.0, 1.0)
    ang = np.degrees(np.arccos(cosang))
    e = rho_t - scen.rho
    sg = (e * scen.rho).sum() / (scen.rho ** 2).sum()
    e_al = e - sg * scen.rho
    return dict(
        dual=((e_al @ W_dual) ** 2).tolist(),
        ang_mean_deg=float(ang.mean()),
        ang_median_deg=float(np.median(ang)),
        ang_p95_deg=float(np.percentile(ang, 95)),
        mse_aligned=float(e_al @ e_al / len(e_al)),
        mae_raw=float(np.abs(e).mean()))


def _run_arm_object(payload):
    """单对象 × 单臂(任务间零耦合;picklable,供 spawn Pool)。"""
    (obj_idx, obj_name, cfg, arm_name, arm_cfg) = payload
    K = int(cfg["K_lights"])
    regime = int(cfg["regimes"][0])
    budgets = list(cfg["budgets_k"])
    policies = list(cfg["policies"])
    n_seeds = int(cfg["seeds_per_level"])
    fam_seed = int(cfg.get("family_seed", 20260915))

    obj = load_object(cfg["data_root"], obj_name, data_meta=cfg.get("data_meta"))
    scen = NominalScene(obj, np.random.default_rng([20260910, obj_idx]),
                        noise_fit_convention=cfg["noise_fit_convention"])
    finf = scen.Finf_diag
    u_act = (scen.w * scen.s_hat)[:, :, None] * scen.B_phi
    M0_act = np.einsum("kpi,kpj->kij", scen.B_phi,
                       scen.w[:, :, None] * scen.B_phi)
    u142 = np.zeros((K, finf.shape[0], 3))
    M0142 = np.zeros((K, 3, 3))
    u142[scen.sel] = u_act
    M0142[scen.sel] = M0_act
    active142 = np.zeros(K, bool)
    active142[scen.sel] = True

    sI = float(arm_cfg["sig_logI"])
    sd = float(arm_cfg["sig_dir_deg"])
    het = float(arm_cfg["het_sigma"])
    fam = CorruptionFamily(sI, sd, het_sigma=het, rho_c=0.0,
                           seed=fam_seed, n_lights=K)
    # 预测侧状态:镜像冻结 build_structure(非 active 灯 eye(3);
    # active 灯位级一致——批量 inv == 单次 inv,S0 探针钉死)
    lam0142 = np.stack([np.eye(3)] * K)
    lam0142[scen.sel] = np.linalg.inv(fam.sigma_phi_block())[scen.sel]
    _deg, W_dual = scen.predicted_degradation(
        fam.sigma_phi_block()[scen.sel], mode_coordinate="dual")

    state = SelectionState(u=u142, M0=M0142, lam0=lam0142.copy(),
                           finf=finf, active=active142)
    blocks = LightBlocks(u142, M0142, lam0142.copy(), finf, active142)
    prob = CertificateProblem(blocks=blocks, kappa=float(regime))
    ords = {}
    for pol in policies:
        if pol == "mode_aware":
            ords[pol] = select_ordering_mode_aware(state, float(regime))[0]
        else:
            ords[pol] = select_ordering(state, pol, float(regime))[0]

    pred_J = {}
    for unit, ordr in ords.items():
        for k in budgets:
            t = np.ones(K)
            t[np.asarray(ordr[:k])] = float(regime)
            pred_J[(unit, k)] = prob.J_A(t)

    # 注入侧 σ:het 用逐灯向量;het=0 标量(控制臂与冻结逐位一致)
    if het == 0.0:
        sig_I, sig_R = sI, np.radians(sd)
    else:
        mult = fam.sigma_logI_vec()[scen.sel] / sI
        sig_I = sI * mult
        sig_R = np.radians(sd) * mult

    acc = {(u_, k): dict(ang=[], mse=[], dual=[])
           for u_ in ords for k in budgets}
    for s_i in range(n_seeds):
        z_rng = np.random.default_rng(
            [20260910, 7777, obj_idx, int(cfg["seed_level_index"]), s_i])
        raw = raw_innovations_family(z_rng, len(scen.sel), rho_c=0.0)
        for unit, ordr in ords.items():
            for k in budgets:
                scales = budget_scales(ordr, k, regime)[scen.sel]
                m = arm_metrics_family(scales, raw, scen, sig_I, sig_R, W_dual)
                acc[(unit, k)]["ang"].append(m["ang_mean_deg"])
                acc[(unit, k)]["mse"].append(m["mse_aligned"])
                acc[(unit, k)]["dual"].append(float(np.mean(m["dual"])))
    rows = []
    for unit in ords:
        for k in budgets:
            a = acc[(unit, k)]
            rows.append(dict(
                object=obj_name, arm=arm_name, unit=unit, k=k,
                pred_J_A=pred_J[(unit, k)],
                ang_mean_deg=float(np.mean(a["ang"])),
                mse_aligned=float(np.mean(a["mse"])),
                dual_mean=float(np.mean(a["dual"]))))
    return arm_name, obj_name, rows


def run(config_path=REPO / "configs/decision_quality_family.yaml",
        out_path=REPO / "results/openillumination/"
                       "decision_quality_family.json",
        workers_override=None):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    t_start = time.time()
    sha_at_launch = _git_sha()
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    frozen_rows = {(r["object"], r["unit"], r["k"]): r
                   for r in frozen["rows"]
                   if r["level"] == 0.5 and r["regime"] == 10}

    payloads = [(i, n, cfg, arm_name, arm_cfg)
                for i, n in enumerate(cfg["cohort"])
                for arm_name, arm_cfg in cfg["arms"].items()]
    all_rows = []
    import multiprocessing
    ctx = multiprocessing.get_context("spawn")
    workers = int(workers_override or cfg.get("workers", 2))
    if workers > 1:
        with ctx.Pool(processes=min(workers, len(payloads))) as pool:
            for arm_name, obj_name, rows in pool.imap_unordered(
                    _run_arm_object, payloads):
                all_rows.extend(rows)
                print(f"[famE] {arm_name}/{obj_name} done "
                      f"({time.time() - t_start:.0f}s)", flush=True)
    else:
        for pl in payloads:
            arm_name, obj_name, rows = _run_arm_object(pl)
            all_rows.extend(rows)
            print(f"[famE] {arm_name}/{obj_name} done "
                  f"({time.time() - t_start:.0f}s)", flush=True)

    # ---- 控制臂逐位锚点(orderings + seeds + 注入路径)----
    n_anchor = 0
    for r in all_rows:
        if r["arm"] != "anchor":
            continue
        f = frozen_rows.get((r["object"], r["unit"], r["k"]))
        if f is None:
            raise RuntimeError(f"anchor row missing in frozen subset: {r}")
        for field in ("pred_J_A", "ang_mean_deg", "mse_aligned", "dual_mean"):
            if r[field] != f[field]:
                raise RuntimeError(
                    f"anchor bit-identity broken: {r['object']} "
                    f"{r['unit']} k={r['k']} {field}: {r[field]!r} != "
                    f"frozen {f[field]!r}")
        n_anchor += 1
    print(f"[famE] control anchor: {n_anchor} rows bit-identical")

    # ---- 每臂 pooled informed 逐对符号一致(冻结 E 统计量)----
    from scipy.stats import binomtest, spearmanr
    units = list(cfg["policies"])
    budgets = list(cfg["budgets_k"])
    arms_out = {}
    for arm_name in cfg["arms"]:
        rows = [r for r in all_rows if r["arm"] == arm_name]
        byok = {}
        for r in rows:
            byok.setdefault((r["object"], r["k"]), {})[r["unit"]] = r
        agree = tot = 0
        per_obj = {}
        for (obj_name, k), us in sorted(byok.items()):
            if len(us) < len(units):
                continue
            a_o = t_o = 0
            for i, ua in enumerate(units):
                for ub in units[i + 1:]:
                    dp = us[ua]["pred_J_A"] - us[ub]["pred_J_A"]
                    da = us[ua]["ang_mean_deg"] - us[ub]["ang_mean_deg"]
                    if dp != 0:
                        tot += 1
                        t_o += 1
                        agree += int((dp > 0) == (da > 0))
                        a_o += int((dp > 0) == (da > 0))
            per_obj[f"{obj_name}@k{k}"] = dict(agree=a_o, total=t_o)
        ci = binomtest(agree, tot, 0.5).proportion_ci(0.95)
        # 单元级 informed Spearman(连续性;per (obj,k) n=4)
        spears = []
        for (obj_name, k), us in sorted(byok.items()):
            if len(us) < len(units):
                continue
            # 逐 (obj, k) 的 4 点 informed Spearman(pred vs ang)
            pred_k = [us[u]["pred_J_A"] for u in units]
            ang_k = [us[u]["ang_mean_deg"] for u in units]
            if len(set(pred_k)) > 1 and len(set(ang_k)) > 1:
                spears.append(float(spearmanr(pred_k, ang_k).statistic))
        arms_out[arm_name] = dict(
            informed_pairwise_sign=dict(agree=agree, total=tot,
                                        rate=round(agree / tot, 6) if tot
                                        else None,
                                        ci95=[round(float(ci.low), 6),
                                              round(float(ci.high), 6)]),
            per_object=per_obj,
            spearman_informed_per_cell=dict(
                n=len(spears),
                median=round(float(np.median(spears)), 6) if spears else None,
                min=round(min(spears), 6) if spears else None,
                max=round(max(spears), 6) if spears else None),
            arm_params=cfg["arms"][arm_name])

    # ---- 预注册判据 ----
    comparator = cfg["frozen_subset_comparator"]
    anchor_stat = arms_out["anchor"]["informed_pairwise_sign"]
    control_ok = (anchor_stat["agree"] == comparator["agree"]
                  and anchor_stat["total"] == comparator["total"])
    all_ge = all(arms_out[a]["informed_pairwise_sign"]["rate"] >= 0.60
                 for a in cfg["arms"])
    outcome = ("robust" if (control_ok and all_ge)
               else "parameterization-specific")

    summary = dict(
        gate=cfg["gate"], analysis_status=cfg["analysis_status"],
        n_objects=len(cfg["cohort"]),
        levels=cfg["levels"], regimes=cfg["regimes"],
        budgets_k=budgets, policies=units,
        seed_level_index=int(cfg["seed_level_index"]),
        seeds_per_level=int(cfg["seeds_per_level"]),
        family_seed=int(cfg.get("family_seed", 20260915)),
        noise_fit_convention=cfg["noise_fit_convention"],
        arms=arms_out,
        frozen_reference=dict(
            artifact="results/openillumination/decision_quality.json",
            pooled_all_cells=dict(agree=678, total=987, rate=0.68693,
                                  ci95=[0.656966, 0.715774]),
            subset_level05_regime10_k1428=dict(**comparator)),
        control_anchor=f"{n_anchor} anchor-arm rows BIT-IDENTICAL to the "
                       "frozen subset (pred_J_A / ang_mean_deg / "
                       "mse_aligned / dual_mean); runtime-asserted",
        control_reproduces_subset=control_ok,
        outcome=outcome,
        outcome_rule=cfg["outcome_rule"].strip(),
        rows=all_rows,
        manifest=dict(
            config_sha256=_sha(Path(config_path)),
            git_sha=sha_at_launch,
            scene_rng_spec=cfg["scene_rng_spec"],
            elapsed_s=round(time.time() - t_start, 1)),
        note="Reduced decision quality per family arm (S2-4 E): same "
             "informed-only pairwise sign statistic as the frozen E claim, "
             "on the power-maximal budget subset (k=14/28 carry 100% of "
             "dp!=0 pairs; k>=57 orderings coincide). Random baselines are "
             "out of scope (the E statistic is informed-only).")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(json.dumps(summary, ensure_ascii=False,
                                          indent=1).encode("utf-8"))
    for a, d in arms_out.items():
        st = d["informed_pairwise_sign"]
        print(f"[famE] {a}: {st['agree']}/{st['total']} = {st['rate']} "
              f"ci95 {st['ci95']}")
    print(f"[famE] control reproduces 88/125: {control_ok}")
    print(f"[famE] outcome: {outcome}")
    print(f"[famE] wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",
                    default=str(REPO / "configs/decision_quality_family.yaml"))
    ap.add_argument("--out",
                    default=str(REPO / "results/openillumination/"
                                "decision_quality_family.json"))
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    run(args.config, args.out, args.workers)
