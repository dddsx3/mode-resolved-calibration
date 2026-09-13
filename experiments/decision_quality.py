"""P-DECISION-QUALITY · predicted vs realized decision quality (E3/M3).

The impact-facing question: does the information-theoretic allocation
predict REALIZED reconstruction quality? Controlled closed loop (no real
recalibration -- OpenIllumination is a calibration dataset):

  corruption injection (frozen grid) -> per-unit allocation (9 selection
  units: mode_aware / e/a/d-opt + 5 random perms) -> the SINGLE residual
  pipeline (arm_metrics: same corruption + fixed-n-hat GLS as arm_energy,
  extended with a one-step calibrated_ps normal refit) -> realized
  endpoints: normal angular error (deg), gauge-aligned albedo MSE,
  bottom-5 dual energy. Predicted side: J_A of the same allocation on the
  same corrected state.

Statistics: trapezoid AUC over the frozen budget fractions; dAUC
(policy - mean random) per object x level x regime with object-cluster
paired bootstrap; within-cell Spearman(predicted J_A, realized endpoint)
across the 9 units.

Output: results/openillumination/decision_quality.json +
docs/img/decision_quality.png

Usage:
  python experiments/decision_quality.py [--config configs/decision_quality.yaml]
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

from calibinfo.allocation.corruption import raw_innovations           # noqa: E402
from calibinfo.allocation.policies import (                           # noqa: E402
    LightBlocks, SelectionState, budget_scales, select_ordering,
    select_ordering_mode_aware)
from calibinfo.allocation.convex import CertificateProblem            # noqa: E402
from calibinfo.metrics.cluster_bootstrap import cluster_bootstrap     # noqa: E402
from calibinfo.models.corruption import CorruptionGenerator           # noqa: E402
from calibinfo.datasets.openillumination import load_object           # noqa: E402
from experiments.allocation_mode_tail import arm_metrics              # noqa: E402
from experiments.openillumination_validation import NominalScene      # noqa: E402

ENDPOINTS = ("ang_mean_deg", "mse_aligned", "dual_mean")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, cwd=str(REPO)).stdout.strip()


def _trapezoid_auc(Es, fracs):
    Es = np.asarray(Es, float)
    bs = np.asarray(fracs, float)
    return float(np.sum((Es[:-1] + Es[1:]) / 2.0 * np.diff(bs))
                 / (bs[-1] - bs[0]))


def run(config_path=REPO / "configs/decision_quality.yaml",
        out_path=REPO / "results/openillumination/decision_quality.json",
        img_path=REPO / "docs/img/decision_quality.png"):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    t_start = time.time()

    rows = []            # per (obj, level, regime, unit, k) means
    scatter = []         # per (obj, level, regime, unit, k): pred J_A + endpoints
    payloads = [(i, n, cfg) for i, n in enumerate(cfg["cohort"])]
    import multiprocessing
    ctx = multiprocessing.get_context("spawn")
    workers = int(cfg.get("workers", 1))
    if workers > 1 and len(payloads) > 1:
        with ctx.Pool(processes=min(workers, len(payloads))) as pool:
            for obj_name, rws, sc in pool.imap_unordered(_run_object, payloads):
                rows.extend(rws)
                scatter.extend(sc)
                print(f"[dq] {obj_name} done ({time.time()-t_start:.0f}s)",
                      flush=True)
    else:
        for pl in payloads:
            obj_name, rws, sc = _run_object(pl)
            rows.extend(rws)
            scatter.extend(sc)
            print(f"[dq] {obj_name} done ({time.time()-t_start:.0f}s)",
                  flush=True)
    _aggregate_and_write(cfg, rows, scatter, out_path, img_path, t_start,
                         Path(config_path))


def _run_object(payload):
    """单对象全网格(对象间零耦合;picklable,供 spawn Pool)。"""
    obj_idx, obj_name, cfg = payload
    levels = [float(x) for x in cfg["levels"]]
    regimes = [int(x) for x in cfg["regimes"]]
    budgets = list(cfg["budgets_k"])
    policies = list(cfg["policies"])
    n_rand = int(cfg["n_random_perms"])
    n_seeds = int(cfg["seeds_per_level"])
    rows = []
    scatter = []
    obj = load_object(cfg["data_root"], obj_name,
                      data_meta=cfg.get("data_meta"))
    scen = NominalScene(obj, np.random.default_rng([20260910, obj_idx]),
                        noise_fit_convention=cfg["noise_fit_convention"])
    K = int(cfg["K_lights"])
    finf = scen.Finf_diag
    u_act = (scen.w * scen.s_hat)[:, :, None] * scen.B_phi
    M0_act = np.einsum("kpi,kpj->kij", scen.B_phi,
                       scen.w[:, :, None] * scen.B_phi)
    u142 = np.zeros((K, finf.shape[0], 3))
    M0142 = np.zeros((K, 3, 3))
    lam0142 = np.stack([np.eye(3)] * K)
    u142[scen.sel] = u_act
    M0142[scen.sel] = M0_act
    active142 = np.zeros(K, bool)
    active142[scen.sel] = True

    for lv_i, level in enumerate(levels):
        gen = CorruptionGenerator("joint", level)
        sig_logI, sig_rad = gen.sig_logI, np.radians(gen.sig_deg)
        lam0142[scen.sel] = np.linalg.inv(gen.sigma_phi_diag())
        _deg, W_dual = scen.predicted_degradation(
            gen.sigma_phi_diag(), mode_coordinate="dual")
        for regime in regimes:
            state = SelectionState(u=u142, M0=M0142,
                                   lam0=lam0142.copy(), finf=finf,
                                   active=active142)
            blocks = LightBlocks(u142, M0142, lam0142.copy(), finf,
                                 active142)
            prob = CertificateProblem(blocks=blocks, kappa=float(regime))
            ords = {}
            for pol in policies:
                if pol == "mode_aware":
                    ords[pol] = select_ordering_mode_aware(
                        state, float(regime))[0]
                else:
                    ords[pol] = select_ordering(state, pol,
                                                float(regime))[0]
            for p_i in range(n_rand):
                rperm = np.random.default_rng(
                    [20260911, obj_idx, regimes.index(regime), p_i])
                ords[f"random_{p_i}"] = select_ordering(
                    state, "random", rng=rperm)[0]
            # ---- predicted J_A(同分配,同状态)----
            pred_J = {}
            for unit, ordr in ords.items():
                for k in budgets:
                    t = np.ones(K)
                    t[np.asarray(ordr[:k])] = float(regime)
                    pred_J[(unit, k)] = prob.J_A(t)
            # ---- realized(单残差管线,逐种子配对)----
            acc = {(u_, k): dict(ang=[], mse=[], dual=[])
                   for u_ in ords for k in budgets}
            for s_i in range(n_seeds):
                z_rng = np.random.default_rng(
                    [20260910, 7777, obj_idx, lv_i, s_i])
                raw = raw_innovations(z_rng, len(scen.sel))
                for unit, ordr in ords.items():
                    for k in budgets:
                        scales = budget_scales(ordr, k, regime)[scen.sel]
                        m = arm_metrics(scales, raw, scen, sig_logI,
                                        sig_rad, W_dual)
                        acc[(unit, k)]["ang"].append(m["ang_mean_deg"])
                        acc[(unit, k)]["mse"].append(m["mse_aligned"])
                        acc[(unit, k)]["dual"].append(
                            float(np.mean(m["dual"])))
            for unit in ords:
                for k in budgets:
                    a = acc[(unit, k)]
                    rows.append(dict(
                        object=obj_name, level=level, regime=regime,
                        unit=unit, k=k,
                        pred_J_A=pred_J[(unit, k)],
                        ang_mean_deg=float(np.mean(a["ang"])),
                        mse_aligned=float(np.mean(a["mse"])),
                        dual_mean=float(np.mean(a["dual"]))))
                    scatter.append(dict(
                        object=obj_name, level=level, regime=regime,
                        unit=unit, k=k, policy=unit.split("_")[0],
                        pred_J_A=pred_J[(unit, k)],
                        ang_mean_deg=float(np.mean(a["ang"]))))
    return obj_name, rows, scatter


def _aggregate_and_write(cfg, rows, scatter, out_path, img_path, t_start,
                        config_path=REPO / "configs/decision_quality.yaml"):
    levels = [float(x) for x in cfg["levels"]]
    regimes = [int(x) for x in cfg["regimes"]]
    budgets = list(cfg["budgets_k"])
    fracs = [float(x) for x in cfg["budget_fractions"]]
    policies = list(cfg["policies"])
    n_rand = int(cfg["n_random_perms"])
    # ---- AUC + dAUC ----
    units_det = list(policies)
    units_rand = [f"random_{i}" for i in range(n_rand)]
    auc_rows = []
    by_olr = {}
    for r in rows:
        by_olr.setdefault((r["object"], r["level"], r["regime"]), {})[
            (r["unit"], r["k"])] = r
    for (obj_name, level, regime), cell in by_olr.items():
        aucs = {}
        for unit in units_det + units_rand:
            aucs[unit] = {
                ep: _trapezoid_auc(
                    [cell[(unit, k)][ep] for k in budgets], fracs)
                for ep in ENDPOINTS}
        rand_auc = {ep: float(np.mean([aucs[u][ep] for u in units_rand]))
                    for ep in ENDPOINTS}
        # 9 单元内 predicted-vs-realized 排序一致(每端点)
        units_all = units_det + units_rand
        preds = [cell[(u, budgets[0])]["pred_J_A"] for u in units_all]
        # pred_J 逐 k 不同;用 AUC 层面的 pred? 不 —— 逐 k 的 Spearman 更细
        # (per-k cells below);此处用每单元 k 网格平均 pred 作为单元级信息量
        pred_unit = [float(np.mean([cell[(u, k)]["pred_J_A"] for k in budgets]))
                     for u in units_all]
        spearmans = {
            ep: float(spearmanr(pred_unit,
                                [aucs[u][ep] for u in units_all]).statistic)
            for ep in ENDPOINTS}
        auc_rows.append(dict(
            object=obj_name, level=level, regime=regime,
            aucs={u: {ep: round(v, 6) for ep, v in d.items()}
                  for u, d in aucs.items()},
            dAUC={u: {ep: round(aucs[u][ep] - rand_auc[ep], 6)
                      for ep in ENDPOINTS}
                  for u in units_det},
            spearman_pred_vs_realized={ep: round(v, 6)
                                       for ep, v in spearmans.items()}))

    # ---- 对象级配对 bootstrap(每 policy x regime x 端点)----
    boot = {}
    for regime in regimes:
        for pol in policies:
            for ep in ENDPOINTS:
                objs = [(r["object"], r["dAUC"][pol][ep]) for r in auc_rows
                        if r["regime"] == regime]
                payload = [(o, v) for o, v in objs]

                def stat(items, _ep=ep):
                    return float(np.median([v for _, v in items]))

                point, (lo, hi), _b = cluster_bootstrap(
                    payload, stat, int(cfg["bootstrap"]["B"]),
                    int(cfg["bootstrap"]["seed"]))
                boot[f"{pol}|{regime}|{ep}"] = dict(
                    median_dAUC=round(point, 6),
                    ci95=[round(lo, 6), round(hi, 6)],
                    n_objects=len(payload))

    # ---- 汇总 ----
    sp_all = [r["spearman_pred_vs_realized"] for r in auc_rows]
    summary_stats = {
        ep: dict(min=round(min(s[ep] for s in sp_all), 6),
                 median=round(float(np.median([s[ep] for s in sp_all])), 6),
                 max=round(max(s[ep] for s in sp_all), 6),
                 n_cells=len(sp_all))
        for ep in ENDPOINTS}
    summary = dict(
        gate=cfg["gate"], analysis_status=cfg["analysis_status"],
        n_objects=len(cfg["cohort"]), levels=levels, regimes=regimes,
        budgets_k=budgets, policies=policies, n_random_perms=n_rand,
        noise_fit_convention=cfg["noise_fit_convention"],
        endpoints=list(ENDPOINTS),
        spearman_pred_vs_realized=summary_stats,
        bootstrap_dAUC=boot,
        auc_rows=auc_rows,
        rows=rows,
        manifest=dict(
            config_sha256=_sha(Path(config_path)),
            git_sha=_git_sha(),
            scene_rng_spec=cfg["scene_rng_spec"],
            elapsed_s=round(time.time() - t_start, 1)),
        note="predicted-vs-realized closed loop on the corrected-convention "
             "frame; single residual pipeline (arm_metrics: same corruption "
             "+ fixed-n-hat GLS as arm_energy, plus the calibrated_ps "
             "one-step normal refit); predicted side is J_A of the same "
             "allocation on the same state. dAUC = policy AUC minus "
             "mean-random AUC (negative = policy better). Descriptive, no "
             "sign gate.")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(json.dumps(summary, ensure_ascii=False,
                                          indent=1).encode("utf-8"))

    # ---- 图:predicted J_A vs realized normal error(按 policy 着色)+ dAUC ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                             gridspec_kw={"width_ratios": [1.4, 1]})
    colors = {"mode_aware": "tab:purple", "e_opt": "tab:blue",
              "a_opt": "tab:green", "d_opt": "tab:orange", "random": "0.6"}
    for s in scatter:
        axes[0].scatter(s["pred_J_A"], s["ang_mean_deg"], s=10, alpha=0.45,
                        c=colors.get(s["policy"], "0.6"), edgecolors="none")
    for pol, c in colors.items():
        axes[0].scatter([], [], c=c, label=pol)
    axes[0].set_xlabel("predicted J_A = tr ΔF(t)^{-1} (allocation)")
    axes[0].set_ylabel("realized normal angular error (mean, deg)")
    axes[0].set_title("predicted vs realized (per object x level x regime "
                      "x budget x unit)", fontsize=10)
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].grid(alpha=0.3, lw=0.4)
    # dAUC forest(法线误差端点)
    ylabels, ypos = [], []
    for regime in regimes:
        for pol in policies:
            b = boot[f"{pol}|{regime}|ang_mean_deg"]
            yp = len(ylabels)
            axes[1].plot([b["ci95"][0], b["ci95"][1]], [yp, yp],
                         lw=2, color=colors[pol])
            axes[1].plot(b["median_dAUC"], yp, "o", color=colors[pol])
            ylabels.append(f"{pol} @{regime}x")
            ypos.append(yp)
    axes[1].axvline(0.0, color="k", ls="--", lw=0.8)
    axes[1].set_yticks(ypos)
    axes[1].set_yticklabels(ylabels, fontsize=8)
    axes[1].set_xlabel("Δ AUC (normal angular error) vs random\n"
                       "(negative = policy better)")
    axes[1].set_title("object-cluster bootstrap (B=10000)", fontsize=10)
    axes[1].grid(alpha=0.3, lw=0.4, axis="x")
    fig.tight_layout()
    Path(img_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(img_path, dpi=150)
    print(f"[dq] wrote {out_path}")
    print(f"[dq] wrote {img_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",
                    default=str(REPO / "configs/decision_quality.yaml"))
    ap.add_argument("--out",
                    default=str(REPO / "results/openillumination/decision_quality.json"))
    ap.add_argument("--img",
                    default=str(REPO / "docs/img/decision_quality.png"))
    a = ap.parse_args()
    run(a.config, a.out, a.img)
