"""P-CERT · certified optimality gaps for calibration-precision allocation.

Preregistered protocol (configs/certified_gaps.yaml, committed before the run).
For each of the 11 real OpenIllumination objects (corrected-interface
nominal scenes, the same pinned rng spec as the frozen allocation experiment):

  - the budget-constrained convex program
        min J_A(t) = tr DeltaF(t)^{-1}   s.t. sum_k (t_k - 1) <= B,
  is solved by Frank-Wolfe with an analytic LMO; the duality gap at the
  returned point is a GLOBAL optimality certificate (convex objective,
  convex compact feasible set), giving the per-k lower bound
        J*_k >= J_A(t_FW) - gap(t_FW).
  - discrete candidate allocations are evaluated exactly on the same
    functional: the J_A-greedy prefix (exact trace-kernel gains), 10 seeded
    random k-subsets of the 48 Fisher-active lights, all-refined, none.

Output: results/certification/certified_gaps.json (rows per object x k,
summary statistics, manifest). No sign-based gate: every computed row is
reported regardless of outcome.

Usage:
  python experiments/certified_gaps.py [--config configs/certified_gaps.yaml]
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

from calibinfo.allocation.blocks import LightBlocks, sym_inv               # noqa: E402
from calibinfo.allocation.convex import CertificateProblem, budget_for_k   # noqa: E402
from calibinfo.datasets.openillumination import load_object                 # noqa: E402
from calibinfo.models.corruption import CorruptionGenerator                 # noqa: E402
from experiments.openillumination_validation import NominalScene            # noqa: E402


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, cwd=str(REPO)).stdout.strip()


def _build_state(scen, level: float, kappa: float, K: int) -> CertificateProblem:
    """冻结 allocation 同款装配：142 灯全域，48 个 Fisher-active 灯。"""
    gen = CorruptionGenerator("joint", level)
    base_lam0 = np.linalg.inv(gen.sigma_phi_diag())
    u_act = (scen.w * scen.s_hat)[:, :, None] * scen.B_phi
    M0_act = np.einsum("kpi,kpj->kij", scen.B_phi, scen.w[:, :, None] * scen.B_phi)
    u142 = np.zeros((K, scen.Finf_diag.shape[0], 3))
    M0142 = np.zeros((K, 3, 3))
    lam0142 = np.stack([np.eye(3)] * K)
    u142[scen.sel] = u_act
    M0142[scen.sel] = M0_act
    lam0142[scen.sel] = base_lam0
    active142 = np.zeros(K, bool)
    active142[scen.sel] = True
    blocks = LightBlocks(u142, M0142, lam0142, scen.Finf_diag, active142)
    return CertificateProblem(blocks=blocks, kappa=kappa)


def _apply_t(prob: CertificateProblem, lights: list[int], kappa: float):
    t = np.ones(prob.blocks.L)
    for idx in lights:
        t[idx] = kappa
    return t


def _greedy_J_A(prob: CertificateProblem, kappa: float, k_max: int):
    """J_A-greedy：每步从剩余 active 灯中选 J_A 增益最大者（trace 核，解析）。
    返回 {k: (lights_prefix, J_A)}——前缀即离散分配（预算 = k(κ−1)）。"""
    blk = prob.blocks
    t = np.ones(blk.L)
    chosen = []
    out = {}
    J_cur = prob.J_A(t)
    for step in range(k_max):
        _val, DFinv = prob.J_A_with_inv(t)
        best_k, best_gain = None, -np.inf
        for k in range(blk.L):
            if not blk.active[k] or k in chosen:
                continue
            K = sym_inv(blk.M0[k] + t[k] * blk.lam0[k])
            M = K @ blk.lam0[k] @ K
            G = DFinv @ blk.u[k]
            gain = float(np.trace(M @ (G.T @ G)))     # J_A 下降量（解析）
            if gain > best_gain:
                best_k, best_gain = k, gain
        t[best_k] = kappa
        chosen.append(best_k)
        J_cur = prob.J_A(t)
        out[step + 1] = (list(chosen), J_cur)
    return out


def run(config_path=REPO / "configs/certified_gaps.yaml",
        out_path=REPO / "results/certification/certified_gaps.json"):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    kappa = float(cfg["kappa"])
    budgets_k = list(cfg["budgets_k"])
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    t_start = time.time()
    for obj_idx, obj_name in enumerate(cfg["cohort"]):
        obj = load_object(cfg["data_root"], obj_name, data_meta=cfg.get("data_meta"))
        scen = NominalScene(obj, np.random.default_rng([20260910, obj_idx]),
                            noise_fit_convention=cfg["noise_fit_convention"])
        prob = _build_state(scen, float(cfg["level"]), kappa, int(cfg["K_lights"])
                            if "K_lights" in cfg else 142)
        active = [int(i) for i in np.flatnonzero(prob.blocks.active)]

        t_one = np.ones(prob.blocks.L)
        t_all = _apply_t(prob, active, kappa)
        J_none = prob.J_A(t_one)
        J_all = prob.J_A(t_all)
        greedy = _greedy_J_A(prob, kappa, max(budgets_k))

        rng = np.random.default_rng(int(cfg["random_seed"]) + obj_idx)
        for k in budgets_k:
            B = budget_for_k(k, kappa)
            fw = prob.frank_wolfe(B, iters=int(cfg["fw_iters"]))
            g_lights, J_greedy = greedy[k]
            rand_J = []
            for p_i in range(int(cfg["n_random"])):
                subset = list(rng.choice(active, size=k, replace=False))
                rand_J.append(prob.J_A(_apply_t(prob, subset, kappa)))
            rows.append(dict(
                object=obj_name, k=k, B=B,
                J_none=J_none, J_all=J_all,
                fw_J_A=fw["J_A"], fw_gap=fw["gap"],
                fw_lower_bound=fw["J_A"] - fw["gap"],
                fw_t=[float(x) for x in fw["t"]],
                greedy_J_A=J_greedy, greedy_lights=g_lights,
                random_J_A=[float(x) for x in rand_J],
                random_mean=float(np.mean(rand_J)),
                random_min=float(np.min(rand_J)),
            ))
        print(f"[cert] {obj_name}: dynamic range "
              f"{(J_none - J_all) / abs(J_none) * 100:.3f}% "
              f"({J_none:.4f} -> {J_all:.4f}); "
              f"rows={len(rows)} elapsed={time.time() - t_start:.0f}s", flush=True)

    # ---- 汇总（无符号判据：全部行如实报告）----
    by_k = {}
    for k in budgets_k:
        rs = [r for r in rows if r["k"] == k]
        dyn = [(r["J_none"] - r["J_all"]) / abs(r["J_none"]) for r in rs]
        greedy_gap = [(r["greedy_J_A"] - r["fw_lower_bound"]) / abs(r["J_none"])
                      for r in rs]
        rand_gap = [(r["random_mean"] - r["fw_lower_bound"]) / abs(r["J_none"])
                    for r in rs]
        fw_close = [(r["fw_J_A"] - r["fw_lower_bound"]) / abs(r["J_none"])
                    for r in rs]
        by_k[k] = dict(
            dynamic_range_rel=dict(median=float(np.median(dyn)),
                                   min=float(np.min(dyn)), max=float(np.max(dyn))),
            greedy_minus_lower_rel=dict(median=float(np.median(greedy_gap)),
                                        max=float(np.max(greedy_gap))),
            random_mean_minus_lower_rel=dict(median=float(np.median(rand_gap))),
            fw_self_gap_rel=dict(median=float(np.median(fw_close)),
                                 max=float(np.max(fw_close))))

    summary = dict(
        gate="P-CERT v1: certified optimality gaps (preregistered)",
        analysis_status="certified_gaps_v1",
        objective=cfg["objective"],
        parameterization=cfg["parameterization"],
        level=float(cfg["level"]), kappa=kappa, budgets_k=budgets_k,
        noise_fit_convention=cfg["noise_fit_convention"],
        n_objects=len(cfg["cohort"]),
        dynamic_range_rel_all=dict(
            median=float(np.median([(r["J_none"] - r["J_all"]) / abs(r["J_none"])
                                    for r in rows if r["k"] == budgets_k[-1]])),
            note="t=1 vs t=kappa on every active light, at the largest k row"),
        by_k=by_k,
        rows=rows,
        manifest=dict(
            config_sha256=_sha(Path(config_path)),
            git_sha=_git_sha(),
            scene_rng_spec=cfg["scene_rng_spec"],
            fw_iters=int(cfg["fw_iters"]),
            random_seed=int(cfg["random_seed"]),
            elapsed_s=round(time.time() - t_start, 1)),
        note="Certified layer for the allocation problem: J_A lower bounds are "
             "global (FW dual gap on a convex program). No sign-based gate; "
             "every row is reported regardless of outcome. Discrete candidates "
             "compared on the same functional; the frozen policy comparison "
             "lives in results/openillumination/allocation/ and is not re-run.")
    out.write_bytes(json.dumps(summary, ensure_ascii=False, indent=1)
                    .encode("utf-8"))
    for k in budgets_k:
        b = by_k[k]
        print(f"[cert] k={k}: dynamic range median "
              f"{b['dynamic_range_rel']['median'] * 100:.3f}% | "
              f"greedy-above-lower median "
              f"{b['greedy_minus_lower_rel']['median'] * 100:.4f}% | "
              f"random-mean-above-lower median "
              f"{b['random_mean_minus_lower_rel']['median'] * 100:.4f}%")
    print(f"[cert] wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/certified_gaps.yaml"))
    ap.add_argument("--out", default=str(REPO / "results/certification/certified_gaps.json"))
    args = ap.parse_args()
    run(args.config, args.out)
