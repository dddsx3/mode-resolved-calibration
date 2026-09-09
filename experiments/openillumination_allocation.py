"""Allocation experiment on the fixed held-out cohort (preregistered).

Task-book A4 entry point. GATED: run only after the pre-run audit passes
(tag allocation-prerun). Reads configs/openillumination_allocation.yaml (frozen
at tag allocation-prereg-frozen); writes results/openillumination/allocation/ only.

Design (frozen in the config): five policies share one sequential selection frame
producing a full 142-light ordering per (object, level, regime); the five budgets
are prefixes. The empirical arm reuses the CI04 estimator and error convention
unchanged; the same raw corruption innovations are shared across policies/budgets
(paired), with selected lights carrying std scale 1/sqrt(regime). The primary
endpoint is the normalized trapezoidal AUC of the gauge-aligned mode-projected
error over the budget grid; the frozen statistics (paired object-level bootstrap,
all 11 paired differences, sign count) and result grades are computed by _analyze.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from calibinfo.allocation.corruption import raw_innovations, apply_scaled_corruption
from calibinfo.allocation.policies import (
    SelectionState,
    budget_scales,
    select_ordering,
    select_ordering_mode_aware,
)
from calibinfo.datasets.openillumination import load_object
from calibinfo.models.corruption import CorruptionGenerator
from experiments.openillumination_validation import NominalScene

REPO = Path(__file__).resolve().parents[1]
BUDGETS = [0.1, 0.2, 0.4, 0.6, 0.8]
BUDGET_COUNTS = [14, 28, 57, 85, 114]      # k = round(b * 142)
REGIMES = [10, 100]
POLICIES = ["mode_aware", "e_opt", "a_opt", "d_opt", "random"]
DET_POLICIES = POLICIES[:-1]
LEVELS = [0.2, 0.5, 1.0]                   # L-tier grid (tau-gated; see configs)
SEEDS_PER_LEVEL = 10


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _trapezoid_auc(Es):
    Es = np.asarray(Es, float)
    bs = np.asarray(BUDGETS, float)
    return float(np.sum((Es[:-1] + Es[1:]) / 2.0 * np.diff(bs)) / (0.8 - 0.1))


def run(config_path=REPO / "configs/openillumination_allocation.yaml",
        out_dir=REPO / "results/openillumination/allocation"):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    assert cfg["budget_counts"] == BUDGET_COUNTS and cfg["regimes"] == REGIMES
    assert cfg["n_objects"] == len(cfg["cohort"]) == 11
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    U_rows = {}          # (obj, regime, policy) -> {budget: E_osb}
    orderings = {}       # (obj, level, regime, policy) -> ordering (or perms)
    per_run = []         # provenance: one row per reconstruction run

    for obj_idx, obj_name in enumerate(cfg["cohort"]):
        obj = load_object(cfg["data_root"], obj_name, data_meta=cfg.get("data_meta"))
        # pinned scene-subset stream: choice(142, 48) then choice(P_full, 1200)
        scen = NominalScene(obj, np.random.default_rng([20260910, obj_idx]))
        L_analysis = scen.s_hat.shape[0]

        # prediction-side per-light blocks (level-independent assets)
        u = (scen.w * scen.s_hat)[:, :, None] * scen.B_phi
        M0 = np.einsum("kpi,kpj->kij", scen.B_phi, scen.w[:, :, None] * scen.B_phi)
        finf = scen.Finf_diag
        active = np.ones(L_analysis, bool)

        for lv_i, level in enumerate(LEVELS):
            gen = CorruptionGenerator("joint", level)
            sig_logI, sig_rad = gen.sig_logI, np.radians(gen.sig_deg)
            lam0 = np.linalg.inv(gen.sigma_phi_diag())
            state = SelectionState(u=u, M0=M0, lam0=np.stack([lam0] * L_analysis),
                                   finf=finf, active=active)
            _deg, V_level = scen.predicted_degradation(gen.sigma_phi_diag())

            for r_i, regime in enumerate(REGIMES):
                # ---- selection (prediction side only) ----
                ords = {}
                for pol in DET_POLICIES:
                    if pol == "mode_aware":
                        ords[pol] = select_ordering_mode_aware(state, regime)[0]
                    else:
                        ords[pol] = select_ordering(state, pol, regime)[0]
                for p_i in range(N_RANDOM_PERMS):
                    rperm = np.random.default_rng([20260911, obj_idx, r_i, p_i])
                    ords[f"random_{p_i}"] = select_ordering(state, "random", rng=rperm)[0]
                    orderings[f"{obj_name}|{level}|{regime}|random_{p_i}"] = ords[f"random_{p_i}"]
                for pol in DET_POLICIES:
                    orderings[f"{obj_name}|{level}|{regime}|{pol}"] = ords[pol]

                # ---- empirical runs (paired raw innovations) ----
                # units: 4 deterministic policies + 5 random permutations
                E_sum = {f"random_{p_i}": {b: 0.0 for b in BUDGETS}
                         for p_i in range(N_RANDOM_PERMS)}
                for pol in DET_POLICIES:
                    E_sum[pol] = {b: 0.0 for b in BUDGETS}
                for s_i in range(SEEDS_PER_LEVEL):
                    z_rng = np.random.default_rng([20260910, 7777, obj_idx, lv_i, s_i])
                    raw = raw_innovations(z_rng, L_analysis)
                    resid_pool = (scen.I - scen.s_hat * scen.rho[None, :]).ravel()
                    del resid_pool                     # control arm not in this design
                    for r_i, regime in enumerate(REGIMES):
                        for pol in POLICIES:
                            units = (DET_POLICIES if pol != "random"
                                     else [f"random_{p_i}" for p_i in range(N_RANDOM_PERMS)])
                            for unit in units:
                                ordr = orderings[f"{obj_name}|{level}|{regime}|{unit}"]
                                for b, k in zip(BUDGETS, BUDGET_COUNTS):
                                    slot_scales = budget_scales(ordr, k, regime)
                                    scales = slot_scales[scen.sel]   # analysis lights
                                    d2, g = apply_scaled_corruption(
                                        scen.dirs, sig_logI, sig_rad, scales, raw)
                                    rho_t = scen.estimate_albedo(d2, g)
                                    e = rho_t - scen.rho
                                    s_g = (e * scen.rho).sum() / (scen.rho ** 2).sum()
                                    e = e - s_g * scen.rho           # gauge alignment
                                    E_run = float((e @ V_level) ** 2).sum()
                                    E_sum[unit][b] += E_run
                                    per_run.append(dict(
                                        object=obj_name, level=level, seed=s_i,
                                        regime=regime, policy=unit, budget=b,
                                        E_run=E_run))
                # E_osb = mean over levels x seeds
                for unit in [p for p in POLICIES] + [f"random_{p_i}" for p_i in range(N_RANDOM_PERMS)]:
                    if unit in E_sum:
                        for b in BUDGETS:
                            E_sum[unit][b] /= len(LEVELS) * SEEDS_PER_LEVEL
                for pol in POLICIES[:-1]:
                    U_rows[(obj_name, regime, pol)] = {b: E_sum[pol][b] for b in BUDGETS}
                U_rows[(obj_name, regime, "random")] = {
                    b: float(np.mean([E_sum[f"random_{p_i}"][b]
                                      for p_i in range(N_RANDOM_PERMS)]))
                    for b in BUDGETS}

    summary = _analyze(U_rows)
    (out / "selection_orders.json").write_text(
        json.dumps(orderings, indent=1), encoding="utf-8")
    (out / "per_run_errors.csv").write_text(
        "\n".join(["object,level,seed,regime,policy,budget,E_run"]
                  + [f"{r['object']},{r['level']},{r['seed']},{r['regime']},"
                     f"{r['policy']},{r['budget']},{r['E_run']!r}"
                     for r in per_run]) + "\n", encoding="utf-8")
    (out / "allocation_summary.json").write_text(
        json.dumps(dict(config_sha256=_sha(REPO / "configs/openillumination_allocation.yaml"),
                        summary=summary), ensure_ascii=False, indent=1),
        encoding="utf-8")
    _write_tables(out, summary)
    return summary


def _analyze(U_rows):
    """Frozen statistics: paired object-level bootstrap per regime + grades."""
    out = {"regimes": {}}
    for regime in REGIMES:
        objs = sorted({o for (o, r, _p) in U_rows if r == regime})
        Uos = {o: {p: U_rows[(o, regime, p)] for p in POLICIES} for o in objs}

        def deltas(p):
            return np.array([Uos[o][p] - Uos[o]["random"] for o in objs])

        rng = np.random.default_rng(20260910)
        n = len(objs)
        res = {}
        # single shared resample stream across policies (same bootstrap draws)
        rng = np.random.default_rng(20260910)
        draws = [rng.integers(0, n, n) for _ in range(10000)]
        for p in POLICIES[:-1]:
            d = deltas(p)
            boots = [float(np.median(d[idx])) for idx in draws]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            res[p] = dict(median=float(np.median(d)), ci95=[float(lo), float(hi)],
                          benefit_significant=bool(hi < 0),
                          per_object=dict(zip(objs, d.tolist())),
                          improved=int((d < 0).sum()))
        strong = res["mode_aware"]["benefit_significant"] and any(
            res[p]["benefit_significant"] for p in ("e_opt", "a_opt", "d_opt"))
        neutral = res["mode_aware"]["benefit_significant"] and not strong
        grade = "strong" if strong else ("neutral" if neutral else "negative")
        out["regimes"][regime] = dict(policy_deltas=res, grade=grade, objects=objs)
    return out


def _write_tables(out, summary):
    lines = ["regime,policy,median_delta,ci_lo,ci_hi,benefit_significant,improved_of_11"]
    for regime, blk in summary["regimes"].items():
        for p, r in blk["policy_deltas"].items():
            lines.append(f"{regime},{p},{r['median']:.6g},{r['ci95'][0]:.6g},"
                         f"{r['ci95'][1]:.6g},{r['benefit_significant']},"
                         f"{r['improved']}/11")
    (out / "allocation_deltas.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/openillumination_allocation.yaml"))
    ap.add_argument("--out", default=str(REPO / "results/openillumination/allocation"))
    args = ap.parse_args()
    summary = run(args.config, args.out)
    for regime, blk in summary["regimes"].items():
        print(f"[allocation] regime {regime}x -> grade: {blk['grade']}")


if __name__ == "__main__":
    main()
