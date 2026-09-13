"""P-ACTIVE-SET-ABLATION · formalize the active-set ablation (E4/M1).

Decomposes the "informed beats random" advantage on the certified
information functional J_A (identical P-CERT state: level 0.5, kappa 10,
142-light universe, 48 Fisher-active lights, corrected convention):

  (a) active-set effect   random_universe (any of 142; inactive = exact
                          no-op, u_k = 0) vs random_active48;
  (b) mode-ordering       mode_tail (mode-aware ordering prefix) vs
                          classical A-opt greedy on the same J_A;
  (c) rounding loss       rounded (top-k of the continuous fw_t) vs the
                          continuous optimum fw_J_A;
  (d) certified gap       greedy vs the FW duality-gap lower bound.

Existing policies are read from results/certification/certified_gaps.json;
random_universe / mode_tail / rounded are computed here on the rebuilt
state. Realized-value arm (M1-3, reframed after diagnosis): the weak-mode
endpoint is NON-MONOTONE in refinement coverage (the E(k) hump) -- partial
refinement of the information-best lights first WORSENS the gauge-aligned
weak-mode energy (removing mostly gauge-parallel error and redistributing
into the aligned weak modes) before recovering at full coverage. The arm
records the stable per-light realized-value ranking (100 seeds) vs the
single-light J_A gains, the unaligned-mae vs aligned-energy anchors
(base vs all-refined), and the E(k) curve along the a_opt ordering in
BOTH conventions -- the legacy frame reproduces the frozen per-run numbers
bit-exactly (the legacy/corrected convention split is documented).

AUC reference layer: medians/CIs copied from the existing AUC artifacts
with source labels (no recomputation).

Output: results/openillumination/active_set_ablation.json.

Usage:
  python experiments/active_set_ablation.py [--config configs/active_set_ablation.yaml]
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
    raw_innovations)
from calibinfo.allocation.policies import (                            # noqa: E402
    SelectionState, select_ordering_mode_aware)
from calibinfo.models.corruption import CorruptionGenerator            # noqa: E402
from experiments.allocation_mode_tail import arm_energy               # noqa: E402
from experiments.certified_gaps import _build_state, _apply_t         # noqa: E402
from scipy.stats import spearmanr                                       # noqa: E402


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, cwd=str(REPO)).stdout.strip()


def _oracle_arm(scen, obj_idx, level, regime, n_seeds):
    """M1-3 · per-light realized value + 机制锚点(单残差管线,读出对拍)。

    每种子:base(全 scale 1)与逐灯精化(k 单灯 scale 1/√regime)各跑一次
    corruption→固定 n̂ GLS→gauge 对齐;value_k = mean(E_base) − mean(E_k)。
    同时记录机制锚点:未对齐 mae 与对齐后 bottom-5 dual 能量(base vs 全精化)。
    能量读出与 arm_energy(冻结管线)逐值对拍(首个种子),保证单一管线口径。"""
    from calibinfo.allocation.corruption import apply_scaled_corruption
    gen = CorruptionGenerator("joint", level)
    sig_logI, sig_rad = gen.sig_logI, np.radians(gen.sig_deg)
    _deg, W_dual = scen.predicted_degradation(gen.sigma_phi_diag(),
                                              mode_coordinate="dual")
    n_act = len(scen.sel)
    vals = np.zeros((n_seeds, n_act))
    mae = dict(base=[], all48=[])
    energy = dict(base=[], all48=[])
    for s_i in range(n_seeds):
        rng = np.random.default_rng([20260920, 555, obj_idx, s_i])
        raw = raw_innovations(rng, n_act)

        def _run(scales):
            d2, g = apply_scaled_corruption(scen.dirs, sig_logI, sig_rad,
                                            scales, raw)
            rho_t = scen.estimate_albedo(d2, g)
            e = rho_t - scen.rho
            sg = (e * scen.rho).sum() / (scen.rho ** 2).sum()
            e = e - sg * scen.rho
            return float(np.mean(np.abs(rho_t - scen.rho))), \
                float(np.mean((e @ W_dual) ** 2))

        mae_b, en_b = _run(np.ones(n_act))
        mae_a, en_a = _run(np.full(n_act, 1.0 / np.sqrt(regime)))
        mae["base"].append(mae_b); mae["all48"].append(mae_a)
        energy["base"].append(en_b); energy["all48"].append(en_a)
        if s_i == 0:   # 单一管线对拍:能量读出必须与 arm_energy 一致
            ref = float(np.mean(arm_energy(np.ones(n_act), raw, scen,
                                           sig_logI, sig_rad, W_dual)))
            assert abs(en_b - ref) < 1e-9 * max(1.0, abs(ref)), (en_b, ref)
        for k in range(n_act):
            scales = np.ones(n_act)
            scales[k] = 1.0 / np.sqrt(regime)
            _m, e_k = _run(scales)
            vals[s_i, k] = en_b - e_k
    return vals, dict(
        mae_base=float(np.mean(mae["base"])),
        mae_all48=float(np.mean(mae["all48"])),
        dual_energy_base=float(np.mean(energy["base"])),
        dual_energy_all48=float(np.mean(energy["all48"])))


def _k_curve(scen, obj_idx, levels, n_seeds, regime, ks):
    """E(k) 曲线(沿 a_opt 排序;冻结种子规范)。

    口径由传入的 scen 决定:legacy scen(冻结基准口径)位级复现冻结
    per-run 数字;corrected scen 给本口径的对照曲线。"""
    from calibinfo.allocation.policies import select_ordering
    from calibinfo.allocation.corruption import apply_scaled_corruption
    sel = np.asarray(scen.sel)
    n_act = len(sel)
    out = {int(k): [] for k in ks}
    for lv_i, level in enumerate(levels):
        gen = CorruptionGenerator("joint", level)
        sig_logI, sig_rad = gen.sig_logI, np.radians(gen.sig_deg)
        _deg, W = scen.predicted_degradation(gen.sigma_phi_diag(),
                                             mode_coordinate="legacy")
        prob = _build_state(scen, level, 10.0, 142)
        state = SelectionState(u=prob.blocks.u, M0=prob.blocks.M0,
                               lam0=prob.blocks.lam0, finf=prob.blocks.finf,
                               active=prob.blocks.active)
        a_order = select_ordering(state, "a_opt", float(regime))[0]
        for s_i in range(n_seeds):
            rng = np.random.default_rng([20260910, 7777, obj_idx, lv_i, s_i])
            raw = raw_innovations(rng, n_act)
            for k in ks:
                scales = np.ones(n_act)
                if k > 0:
                    ids = np.asarray(a_order[:k])
                    pos = np.searchsorted(sel, ids[np.isin(ids, sel)])
                    scales[pos] = 1.0 / np.sqrt(regime)
                d2, g = apply_scaled_corruption(scen.dirs, sig_logI,
                                                sig_rad, scales, raw)
                rho_t = scen.estimate_albedo(d2, g)
                e = rho_t - scen.rho
                sg = (e * scen.rho).sum() / (scen.rho ** 2).sum()
                e = e - sg * scen.rho
                out[int(k)].append(float(np.sum((e @ W) ** 2)))
    return {int(k): round(float(np.mean(v)), 6) for k, v in out.items()}


def run(config_path=REPO / "configs/active_set_ablation.yaml",
        out_path=REPO / "results/openillumination/active_set_ablation.json"):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    kappa = float(cfg["kappa"])
    level = float(cfg["level"])
    budgets = list(cfg["budgets_k"])
    K = int(cfg["K_lights"])
    t_start = time.time()

    cg = json.loads((REPO / cfg["certified_gaps"]).read_text(encoding="utf-8"))
    cg_rows = {(r["object"], int(r["k"])): r for r in cg["rows"]}

    rows = []
    oracle_rows = []
    for obj_idx, obj_name in enumerate(cfg["cohort"]):
        scen = _load_scene(cfg, obj_idx, obj_name)
        prob = _build_state(scen, level, kappa, K)
        active = [int(i) for i in np.flatnonzero(prob.blocks.active)]

        # mode-aware ordering(142 空间,regime = kappa)
        state = SelectionState(u=prob.blocks.u, M0=prob.blocks.M0,
                               lam0=prob.blocks.lam0, finf=prob.blocks.finf,
                               active=prob.blocks.active)
        mt_order, _steps = select_ordering_mode_aware(state, kappa)

        # 单灯 J_A 增益(oracle 对照的信息论信号)
        t1 = np.ones(prob.blocks.L)
        J1 = prob.J_A(t1)
        jgains = np.zeros(len(active))
        for j, k in enumerate(active):
            t = t1.copy()
            t[k] = kappa
            jgains[j] = J1 - prob.J_A(t)

        rng = np.random.default_rng(int(cfg["random_seed"]) + obj_idx)
        for k in budgets:
            base = cg_rows[(obj_name, k)]
            # random_universe:全 142 域随机 k 子集(非活跃 = 精确 no-op)
            ru = []
            for _p in range(int(cfg["n_random_universe"])):
                t = t1.copy()
                pick = rng.choice(K, size=k, replace=False)
                t[pick] = kappa
                ru.append(prob.J_A(t))
            # mode_tail 前缀 / rounded(连续解 top-k)
            t_mt = _apply_t(prob, [int(x) for x in mt_order[:k]], kappa)
            fw_t = np.asarray(base["fw_t"], float)
            top = np.argsort(-fw_t, kind="stable")[:k]
            t_rd = _apply_t(prob, [int(x) for x in top], kappa)
            rows.append(dict(
                object=obj_name, k=k,
                J_none=base["J_none"], J_all=base["J_all"],
                continuous_fw_J_A=base["fw_J_A"],
                fw_lower_bound=base["fw_lower_bound"],
                classical_a_opt_J_A=base["greedy_J_A"],
                classical_a_opt_lights=base["greedy_lights"],
                random_active48_mean=base["random_mean"],
                random_active48_min=base["random_min"],
                random_universe_mean=float(np.mean(ru)),
                random_universe_min=float(np.min(ru)),
                mode_tail_J_A=float(prob.J_A(t_mt)),
                mode_tail_lights=[int(x) for x in mt_order[:k]],
                rounded_J_A=float(prob.J_A(t_rd)),
                rounded_lights=[int(x) for x in top],
                # 派生量(J_A 单位,正 = 该效应存在)
                rounding_loss=float(prob.J_A(t_rd) - base["fw_J_A"]),
                certified_gap=float(base["greedy_J_A"] - base["fw_lower_bound"]),
                mode_ordering_gap=float(prob.J_A(t_mt) - base["greedy_J_A"]),
                active_set_dilution=float(np.mean(ru) - base["random_mean"])))
        # realized-value arm(M1-3,仅配置的物体;诚实重构版)
        if obj_name in cfg["oracle"]["objects"]:
            vals, anchors = _oracle_arm(scen, obj_idx, level,
                                        float(cfg["oracle"]["regime"]),
                                        int(cfg["oracle"]["seeds"]))
            vm = vals.mean(0)
            rk_o = np.argsort(-vm, kind="stable")
            rk_j = np.argsort(-jgains, kind="stable")
            pos_o = {int(kk): i for i, kk in enumerate(rk_o)}
            pos_j = {int(kk): i for i, kk in enumerate(rk_j)}
            ko = np.array([pos_o[int(kk)] for kk in sorted(pos_o)])
            kj = np.array([pos_j[int(kk)] for kk in sorted(pos_j)])
            h1, h2 = vals[: vals.shape[0] // 2].mean(0), vals[vals.shape[0] // 2:].mean(0)
            # E(k) 驼峰:corrected(本口径)与 legacy(冻结基准锚定,位级复现)
            ks_curve = [int(x) for x in cfg["oracle"]["k_curve_ks"]]
            curve_corr = _k_curve(
                scen, obj_idx,
                [float(x) for x in cfg["oracle"]["k_curve_levels"]],
                int(cfg["oracle"]["k_curve_seeds_corrected"]),
                float(cfg["oracle"]["regime"]), ks_curve)
            scen_leg = _load_scene(cfg, obj_idx, obj_name, convention="legacy")
            curve_leg = _k_curve(scen_leg, obj_idx,
                                 [float(x) for x in cfg["oracle"]["k_curve_levels"]],
                                 int(cfg["oracle"]["k_curve_seeds"]),
                                 float(cfg["oracle"]["regime"]),
                                 ks_curve)
            oracle_rows.append(dict(
                object=obj_name, level=level,
                regime=float(cfg["oracle"]["regime"]),
                seeds=int(cfg["oracle"]["seeds"]),
                spearman_oracle_vs_Jgain=round(
                    float(spearmanr(ko, kj).statistic), 6),
                split_half_spearman=round(
                    float(spearmanr(h1, h2).statistic), 6),
                top5_overlap_oracle_vs_Jgain=int(len(
                    set(rk_o[:5].tolist()) & set(rk_j[:5].tolist()))),
                oracle_top10=[int(active[i]) for i in rk_o[:10]],
                jgain_top10=[int(active[i]) for i in rk_j[:10]],
                realized_value_seed_std=round(
                    float(vals.std(0).mean()), 6),
                mechanism_anchors={k: round(v, 6)
                                   for k, v in anchors.items()},
                k_curve_corrected=curve_corr,
                k_curve_legacy_frozen_anchor=curve_leg))
        print(f"[abl] {obj_name} done", flush=True)

    # ---- 汇总 ----
    def _agg(key):
        return dict(min=round(min(r[key] for r in rows), 6),
                    median=round(float(np.median([r[key] for r in rows])), 6),
                    max=round(max(r[key] for r in rows), 6))

    # AUC 引用层(复制不重算)
    al = json.loads((REPO / cfg["auc_reference"]["allocation_summary"])
                    .read_text(encoding="utf-8"))
    r48 = json.loads((REPO / cfg["auc_reference"]["random48_summary"])
                     .read_text(encoding="utf-8"))
    auc_ref = dict(
        source_allocation_summary=f"results/{cfg['auc_reference']['allocation_summary'].split('results/')[-1]}",
        source_random48_summary=f"results/{cfg['auc_reference']['random48_summary'].split('results/')[-1]}",
        mode_minus_random_auc={rg: dict(
            median=round(al["regimes"][rg]["policy_deltas"]["mode_aware"]["median"], 4),
            ci95=[round(x, 4) for x in al["regimes"][rg]["policy_deltas"]["mode_aware"]["ci95"]])
            for rg in al["regimes"]},
        aopt_minus_random_auc={rg: dict(
            median=round(al["regimes"][rg]["policy_deltas"]["a_opt"]["median"], 4),
            ci95=[round(x, 4) for x in al["regimes"][rg]["policy_deltas"]["a_opt"]["ci95"]])
            for rg in al["regimes"]},
        mode_minus_random_active48_auc={rg: dict(
            median=round(r48["regimes"][rg]["mode_minus_random_active48"]["median"], 4),
            ci95=[round(x, 4) for x in r48["regimes"][rg]["mode_minus_random_active48"]["ci95"]])
            for rg in r48["regimes"]})

    summary = dict(
        gate=cfg["gate"], analysis_status=cfg["analysis_status"],
        kappa=kappa, level=level, budgets_k=budgets,
        n_objects=len(cfg["cohort"]),
        noise_fit_convention=cfg["noise_fit_convention"],
        decomposition=dict(
            active_set_dilution_J_A=_agg("active_set_dilution"),
            mode_ordering_gap_J_A=_agg("mode_ordering_gap"),
            rounding_loss_J_A=_agg("rounding_loss"),
            certified_gap_J_A=_agg("certified_gap")),
        oracle=oracle_rows,
        auc_reference=auc_ref,
        rows=rows,
        manifest=dict(
            config_sha256=_sha(Path(config_path)),
            git_sha=_git_sha(),
            certified_gaps_sha256=_sha(REPO / cfg["certified_gaps"]),
            scene_rng_spec=cfg["scene_rng_spec"],
            elapsed_s=round(time.time() - t_start, 1)),
        note="active-set ablation on the certified J_A functional; existing "
             "policies read from certified_gaps.json (same state), "
             "random_universe/mode_tail/rounded computed here; oracle arm "
             "uses the single frozen corruption+GLS residual pipeline "
             "(arm_energy, readout cross-checked); k-curves use the "
             "legacy-coordinate projection for cross-convention "
             "comparability, the legacy curve reproduces the frozen "
             "per-run numbers bit-exactly; AUC numbers copied from the "
             "named artifacts. Descriptive, no sign gate.")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(json.dumps(summary, ensure_ascii=False,
                                          indent=1).encode("utf-8"))
    print(f"[abl] wrote {out_path}")


def _load_scene(cfg, obj_idx, obj_name, convention=None):
    from calibinfo.datasets.openillumination import load_object
    from experiments.openillumination_validation import NominalScene
    obj = load_object(cfg["data_root"], obj_name, data_meta=cfg.get("data_meta"))
    conv = convention or cfg["noise_fit_convention"]
    return NominalScene(obj, np.random.default_rng([20260910, obj_idx]),
                        noise_fit_convention=conv)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",
                    default=str(REPO / "configs/active_set_ablation.yaml"))
    ap.add_argument("--out",
                    default=str(REPO / "results/openillumination/active_set_ablation.json"))
    args = ap.parse_args()
    run(args.config, args.out)
