"""P-BASELINE · 文献式选择基线在同一套灯上的对照(TCI 缺口 3)。

问题:纯几何的光照配置选择(Drbohlav & Chantler ICCV 2005 的
"良构配置"思想,操作化为方向球面上的贪心最远点采样)能否捕获
校准预测模型所捕获的分配价值?同一套灯、同一预算框架、同一实现
端点(法向角误差)——唯一差异是选择规则。

- DC05:种子 = 最接近相机轴的灯(max z);迭代加入与已选集合
  max 点积最小的灯(= 最大化最小大圆距离;平局取最低索引)。
  无腐蚀模型、无预测模型:纯几何。
- OI 队列:4 informed + DC05 + 3 random;**硬锚点**:4 个 informed
  单元的行必须与冻结 decision_quality_family.json 的 anchor 臂子集
  **逐位一致**(共享排序与种子的构造性后果)。
- DiLiGenT 队列:a_opt + DC05 + 3 random(最小可比三人组)。
- Gardi 2022 / ReLeaPS 2023 不重实现(objective differs;见 config
  的 comparability statement 与 docs)。

预注册判读:DC05-informative / geometry-insufficient(每队列×每预算;
描述性,无符号门)。

输出: results/baseline/baseline_comparison.json
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

from calibinfo.allocation.convex import CertificateProblem          # noqa: E402
from calibinfo.allocation.policies import (                         # noqa: E402
    LightBlocks, SelectionState, budget_scales, select_ordering,
    select_ordering_mode_aware)
from calibinfo.allocation.corruption import (                        # noqa: E402
    raw_innovations_family)
from calibinfo.models.corruption_family import CorruptionFamily     # noqa: E402
from experiments.decision_quality_family import arm_metrics_family  # noqa: E402
from experiments.openillumination_validation import NominalScene    # noqa: E402

FROZEN_E = REPO / "results/openillumination/decision_quality_family.json"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, cwd=str(REPO)).stdout.strip()


def dc05_ordering(dirs):
    """Drbohlav–Chantler 式良构配置:方向球面贪心最远点采样。

    种子 = argmax z(最接近相机轴);之后每次加入
    argmin_i max_{j∈sel} (d_i·d_j)(= 最大化最小角距离);
    平局取最低索引。返回全域排列。
    """
    D = np.asarray(dirs, float)
    L = D.shape[0]
    sel = [int(np.argmax(D[:, 2]))]
    unselected = set(range(L)) - {sel[0]}
    while unselected:
        best_i, best_dot = None, None
        for i in sorted(unselected):
            m = max(float(D[i] @ D[j]) for j in sel)
            if best_dot is None or m < best_dot - 1e-15:
                best_i, best_dot = i, m
        sel.append(best_i)
        unselected.remove(best_i)
    return sel


def run_object(payload):
    """单对象 × 单队列(任务间零耦合;picklable)。"""
    (cohort_tag, obj_idx, obj_name, cfg, K, scen, dir_args) = payload
    regime = float(cfg["regime"])
    budgets = list(cfg["budgets_k"])
    n_seeds = int(cfg["seeds"])
    level = float(cfg["level"])
    fam_seed = int(cfg.get("family_seed", 20260915))
    units = cfg["oi_units"] if cohort_tag == "oi" else cfg["dq_units"]

    # 预测侧状态(与 E 臂 anchor 构造一致:level 0.5 joint,均匀块)
    fam = CorruptionFamily(level, level, het_sigma=0.0, rho_c=0.0,
                           seed=fam_seed, n_lights=K)
    lam0142 = np.stack([np.eye(3)] * K)
    lam0142[scen.sel] = np.linalg.inv(fam.sigma_phi_block())[scen.sel]
    _deg, W_dual = scen.predicted_degradation(
        fam.sigma_phi_block()[scen.sel], mode_coordinate="dual")
    state = SelectionState(u=dir_args["u"], M0=dir_args["M0"],
                           lam0=lam0142.copy(), finf=dir_args["finf"],
                           active=dir_args["active"])
    blocks = LightBlocks(dir_args["u"], dir_args["M0"], lam0142.copy(),
                         dir_args["finf"], dir_args["active"])
    prob = CertificateProblem(blocks=blocks, kappa=regime)

    ords = {}
    for unit in units:
        if unit == "mode_aware":
            ords[unit] = select_ordering_mode_aware(state, regime)[0]
        elif unit == "dc05":
            ords[unit] = dc05_ordering(dir_args["dirs_all"])
        elif unit == "dc05_active":
            # v1.1:同一几何规则,候选池限制在 active(照到物体)灯
            act_ids = [int(i) for i in np.flatnonzero(dir_args["active"])]
            act_dirs = dir_args["dirs_all"][act_ids]
            ords[unit] = [act_ids[i]
                          for i in dc05_ordering(act_dirs)] +                 [i for i in range(K) if i not in set(act_ids)]
        elif unit.startswith("randomU_"):
            p = int(unit.split("_")[1])
            rng = np.random.default_rng([20260911, obj_idx, 0, p])
            ords[unit] = [int(i) for i in rng.permutation(K)]
        elif unit.startswith("randomA48_"):
            # v1.1:C8 规定的诚实基线——只在 active 集内置换
            # (rng spec 沿用冻结 DQ 的 active48 约定)
            p = int(unit.split("_")[1])
            rng = np.random.default_rng([20260911, obj_idx, 0, 100 + p])
            act_ids = np.asarray([int(i) for i in
                                  np.flatnonzero(dir_args["active"])])
            inact_ids = np.asarray([int(i) for i in
                                    np.flatnonzero(~dir_args["active"])])
            ords[unit] = ([int(i) for i in rng.permutation(act_ids)]
                          + [int(i) for i in rng.permutation(inact_ids)])
        else:
            ords[unit] = select_ordering(state, unit, regime)[0]

    pred_J = {}
    for unit, ordr in ords.items():
        for k in budgets:
            t = np.ones(K)
            t[np.asarray(ordr[:k])] = regime
            pred_J[(unit, k)] = prob.J_A(t)

    # 注入侧:level 标量(与 E 臂 anchor 路径逐位一致)
    acc = {(u_, k): [] for u_ in ords for k in budgets}
    for s_i in range(n_seeds):
        z_rng = np.random.default_rng([20260910, 7777, obj_idx, 1, s_i])
        raw = raw_innovations_family(z_rng, len(scen.sel), rho_c=0.0)
        for unit, ordr in ords.items():
            for k in budgets:
                scales = budget_scales(ordr, k, int(regime))[scen.sel]
                m = arm_metrics_family(scales, raw, scen, level,
                                       np.radians(level), W_dual)
                acc[(unit, k)].append(m["ang_mean_deg"])
    rows = []
    act_set = set(int(i) for i in np.flatnonzero(dir_args["active"]))
    for unit in ords:
        for k in budgets:
            overlap = sum(1 for i in ords[unit][:k] if i in act_set)
            rows.append(dict(cohort=cohort_tag, object=obj_name,
                             unit=unit, k=k,
                             pred_J_A=pred_J[(unit, k)],
                             ang_mean_deg=float(np.mean(acc[(unit, k)])),
                             active_overlap=overlap))
    return cohort_tag, obj_name, rows


def run(config_path=REPO / "configs/baseline_comparison.yaml",
        out_path=REPO / "results/baseline/baseline_comparison.json",
        workers_override=None):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    t_start = time.time()
    sha_at_launch = _git_sha()

    from calibinfo.datasets.diligent_oi_adapter import load_diligent_as_oi
    from calibinfo.datasets.openillumination import load_object

    payloads = []
    # ---- OI 队列 ----
    for obj_idx, obj_name in enumerate(cfg["oi_cohort"]):
        obj = load_object(cfg["oi_data_root"], obj_name,
                          data_meta=cfg.get("oi_data_meta"))
        scen = NominalScene(obj, np.random.default_rng([20260910, obj_idx]),
                            noise_fit_convention=cfg["noise_fit_convention"])
        K = int(cfg["K_oi"])
        u_act = (scen.w * scen.s_hat)[:, :, None] * scen.B_phi
        M0_act = np.einsum("kpi,kpj->kij", scen.B_phi,
                           scen.w[:, :, None] * scen.B_phi)
        u = np.zeros((K, scen.Finf_diag.shape[0], 3))
        M0 = np.zeros((K, 3, 3))
        u[scen.sel] = u_act
        M0[scen.sel] = M0_act
        active = np.zeros(K, bool)
        active[scen.sel] = True
        payloads.append(("oi", obj_idx, obj_name, cfg, K, scen,
                         dict(u=u, M0=M0, finf=scen.Finf_diag,
                              active=active,
                              dirs_all=obj["light_directions"])))
    # ---- DiLiGenT 队列 ----
    for obj_idx, obj_name in enumerate(cfg["diligent_cohort"]):
        obj = load_diligent_as_oi(Path(cfg["diligent_data_root"]) / obj_name)
        scen = NominalScene(obj, np.random.default_rng([20260915, obj_idx]),
                            noise_fit_convention=cfg["noise_fit_convention"],
                            n_lights_total=int(cfg["K_dq"]))
        K = int(cfg["K_dq"])
        u_act = (scen.w * scen.s_hat)[:, :, None] * scen.B_phi
        M0_act = np.einsum("kpi,kpj->kij", scen.B_phi,
                           scen.w[:, :, None] * scen.B_phi)
        u = np.zeros((K, scen.Finf_diag.shape[0], 3))
        M0 = np.zeros((K, 3, 3))
        u[scen.sel] = u_act
        M0[scen.sel] = M0_act
        active = np.zeros(K, bool)
        active[scen.sel] = True
        payloads.append(("dq", obj_idx, obj_name, cfg, K, scen,
                         dict(u=u, M0=M0, finf=scen.Finf_diag,
                              active=active,
                              dirs_all=obj["light_directions"])))

    all_rows = []
    import multiprocessing
    ctx = multiprocessing.get_context("spawn")
    workers = int(workers_override or cfg.get("workers", 2))
    if workers > 1:
        with ctx.Pool(processes=min(workers, len(payloads))) as pool:
            for cohort_tag, obj_name, rows in pool.imap_unordered(
                    run_object, payloads):
                all_rows.extend(rows)
                print(f"[base] {cohort_tag}/{obj_name} done "
                      f"({time.time() - t_start:.0f}s)", flush=True)
    else:
        for pl in payloads:
            cohort_tag, obj_name, rows = run_object(pl)
            all_rows.extend(rows)
            print(f"[base] {cohort_tag}/{obj_name} done "
                  f"({time.time() - t_start:.0f}s)", flush=True)

    # ---- OI 硬锚点:informed 4 单元行 == 冻结 E 臂 anchor 子集 ----
    frozen = json.loads(FROZEN_E.read_text(encoding="utf-8"))
    frozen_rows = {(r["object"], r["unit"], r["k"]): r
                   for r in frozen["rows"] if r["arm"] == "anchor"}
    n_anchor = 0
    for r in all_rows:
        if r["cohort"] != "oi" or r["unit"] not in (
                "mode_aware", "e_opt", "a_opt", "d_opt"):
            continue
        f = frozen_rows[(r["object"], r["unit"], r["k"])]
        assert r["pred_J_A"] == f["pred_J_A"], (r["object"], r["unit"],
                                                r["k"], "pred")
        assert r["ang_mean_deg"] == f["ang_mean_deg"], (r["object"],
                                                        r["unit"], r["k"],
                                                        "ang")
        n_anchor += 1
    assert n_anchor == 88, n_anchor
    print(f"[base] OI hard anchor: {n_anchor} rows bit-identical")

    # ---- 汇总:每队列 × 每预算,每单元中位 + 相对 random 均值的偏差 ----
    def summarize(cohort_tag, units_informed):
        out = {}
        for k in cfg["budgets_k"]:
            k = int(k)
            per_unit = {}
            for unit in ({u for c, u in [(r["cohort"], r["unit"])
                                         for r in all_rows
                                         if r["cohort"] == cohort_tag]}):
                vals = [r["ang_mean_deg"] for r in all_rows
                        if r["cohort"] == cohort_tag and r["unit"] == unit
                        and r["k"] == k]
                if vals:
                    per_unit[unit] = round(float(np.median(vals)), 6)
            objs = sorted({r["object"] for r in all_rows
                           if r["cohort"] == cohort_tag})
            # v1.1:两个参照系——universe-random(v1,仅参考)与
            # active48 random(v1.1 判读基础,C8 规定的诚实基线)
            devs = {"dc05": [], "dc05_active": [], "informed": []}
            overlap = {}
            wasted = {}
            for obj in objs:
                rU = [r["ang_mean_deg"] for r in all_rows
                      if r["cohort"] == cohort_tag and r["object"] == obj
                      and r["k"] == k and r["unit"].startswith("randomU")]
                rA = [r["ang_mean_deg"] for r in all_rows
                      if r["cohort"] == cohort_tag and r["object"] == obj
                      and r["k"] == k and r["unit"].startswith("randomA48")]
                if not rA:
                    continue
                rAm = float(np.mean(rA))
                rUm = float(np.mean(rU)) if rU else np.nan
                d_a = [r["ang_mean_deg"] - rAm for r in all_rows
                       if r["cohort"] == cohort_tag and r["object"] == obj
                       and r["k"] == k and r["unit"] == "dc05_active"]
                devs["dc05_active"].append(d_a[0] if d_a else np.nan)
                if rU:
                    d_u = [r["ang_mean_deg"] - rUm for r in all_rows
                           if r["cohort"] == cohort_tag
                           and r["object"] == obj and r["k"] == k
                           and r["unit"] == "dc05"]
                    devs["dc05"].append(d_u[0] if d_u else np.nan)
                di = [r["ang_mean_deg"] - rAm for r in all_rows
                      if r["cohort"] == cohort_tag and r["object"] == obj
                      and r["k"] == k and r["unit"] in units_informed]
                devs["informed"].append(float(np.median(di)) if di
                                        else np.nan)
                # v1 混淆自文档化:dc05(全域池)在 active 集内的重合度
                ov = [r["active_overlap"] for r in all_rows
                      if r["cohort"] == cohort_tag and r["object"] == obj
                      and r["k"] == k and r["unit"] == "dc05"]
                overlap[obj] = ov[0] if ov else None
                # P3:被浪费的预算份额 = 1 − overlap/k(由产物字段程序化
                # 算出,文档引用而不是手抄)
                if overlap[obj] is not None:
                    wasted[obj] = round(1.0 - overlap[obj] / k, 6)
            med_dca = float(np.nanmedian(devs["dc05_active"]))
            med_inf = float(np.nanmedian(devs["informed"]))
            has_u = any(not np.isnan(x) for x in devs["dc05"])
            med_dcu = (float(np.nanmedian(devs["dc05"])) if has_u
                       else None)
            # P8:两个正交判据 + 组合读法(标注驱动因素)
            geometry_vs_random = ("geometry-beats-randomA48"
                                  if med_dca < 0 else "geometry-at-randomA48")
            model_vs_random = ("model-beats-randomA48"
                               if med_inf < 0 else "model-loses-to-randomA48")
            if med_dca >= med_inf:
                reading = "geometry-insufficient"
            elif med_inf >= 0:
                reading = ("geometry-informative (driven by informed "
                           "failure, not geometry strength)")
            else:
                reading = "geometry-informative (both beat random)"
            out[str(k)] = dict(
                median_ang_by_unit=per_unit,
                dev_per_object=dict(
                    dc05_active_vs_randomA48=[
                        None if np.isnan(x) else round(float(x), 6)
                        for x in devs["dc05_active"]],
                    informed_vs_randomA48=[
                        None if np.isnan(x) else round(float(x), 6)
                        for x in devs["informed"]],
                    dc05_vs_randomU=[
                        None if np.isnan(x) else round(float(x), 6)
                        for x in devs["dc05"]]),
                median_dev_from=dict(
                    dc05_active_vs_randomA48=round(med_dca, 6),
                    informed_vs_randomA48=round(med_inf, 6),
                    dc05_vs_randomU=(round(med_dcu, 6)
                                     if med_dcu is not None else None)),
                dc05_allpool_overlap_at_k=overlap,
                wasted_budget_share=wasted,
                wasted_share_range=(
                    [round(min(wasted.values()), 6),
                     round(max(wasted.values()), 6)] if wasted else None),
                geometry_vs_randomA48=geometry_vs_random,
                model_vs_randomA48=model_vs_random,
                reading=reading)
        return out

    oi_summary = summarize("oi", ("mode_aware", "e_opt", "a_opt", "d_opt"))
    dq_summary = summarize("dq", ("a_opt",))

    summary = dict(
        gate=cfg["gate"], analysis_status=cfg["analysis_status"],
        level=float(cfg["level"]), regime=int(cfg["regime"]),
        budgets_k=list(cfg["budgets_k"]), seeds=int(cfg["seeds"]),
        oi=oi_summary, diligent=dq_summary,
        baseline_definition=dict(
            dc05="greedy farthest-point direction sampling (seed = "
                 "max-z light; add argmin max-dot to selected; ties -> "
                 "lowest index); corruption-model-free, "
                 "prediction-model-free",
            comparability="Gardi 2022 (OED light-position optimization) "
                          "and ReLeaPS 2023 (RL next-light selection) "
                          "optimize different objectives (positions / "
                          "sequences, not per-light calibration "
                          "precision) -- not reimplemented; not "
                          "numerically comparable. DC05 is the "
                          "implementable same-lights geometry baseline."),
        oi_anchor=f"{n_anchor} informed rows BIT-IDENTICAL to the frozen "
                  f"E-arm anchor subset (shared orderings + seeds)",
        manifest=dict(
            config_sha256=_sha(Path(config_path)),
            git_sha=sha_at_launch,
            elapsed_s=round(time.time() - t_start, 1)),
        note="Literature-style selection baselines on the same light "
             "sets: geometry-only (DC05) vs the calibrated prediction "
             "model vs random, at the frozen E-arm operating point "
             "(level 0.5, regime 10, budgets 14/28), realized endpoint "
             "= normal angular error. OI cohort carries the full frame "
             "with a bit-identity anchor; DiLiGenT the minimal "
             "comparable trio.")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(json.dumps(summary, ensure_ascii=False,
                                          indent=1).encode("utf-8"))
    for tag, s in (("oi", oi_summary), ("dq", dq_summary)):
        for k, d in s.items():
            m = d["median_dev_from"]
            print(f"[base] {tag} k={k}: {d['reading']} "
                  f"(dc05_active {m['dc05_active_vs_randomA48']}, "
                  f"informed {m['informed_vs_randomA48']}, "
                  f"v1 dc05-vs-U {m['dc05_vs_randomU']})")
    print(f"[base] wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",
                    default=str(REPO / "configs/baseline_comparison.yaml"))
    ap.add_argument("--out",
                    default=str(REPO / "results/baseline/"
                                "baseline_comparison.json"))
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    run(args.config, args.out, args.workers)
