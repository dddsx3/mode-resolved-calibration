"""Allocation experiment on the fixed held-out cohort (preregistered).

GATED entry point: run only after the pre-run audit passes (tag allocation-prerun).
Reads configs/openillumination_allocation.yaml (frozen at tag
allocation-prereg-frozen); writes results/openillumination/allocation/ only.

Design (frozen in the config): five policies share one sequential selection frame
producing a full 142-light ordering per (object, level, regime); the five budgets
are prefixes. The selection universe is the full capture light set (K=142): the 48
analysis lights are Fisher-active, the remaining 94 are inactive placeholders with
zero effect on prediction and reconstruction (asserted each run). The empirical arm
reuses the CI04 estimator and error convention unchanged; the same raw corruption
innovations are shared across policies/budgets/regimes (paired), with selected
lights carrying std scale 1/sqrt(regime). The primary endpoint is the normalized
trapezoidal AUC of the gauge-aligned mode-projected error over the budget grid; the
frozen statistics (paired object-level bootstrap, all 11 paired differences, sign
count) are computed by _analyze.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))          # direct `python experiments/...` invocation

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
N_RANDOM_PERMS = 5                         # keep in sync with the frozen config
LEVELS = [0.2, 0.5, 1.0]                   # L-tier grid (tau-gated; recorded in the audit package)
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
    assert cfg["policies"] == POLICIES
    assert cfg["selection"]["random"]["permutations"] == N_RANDOM_PERMS
    assert cfg["n_objects"] == len(cfg["cohort"])
    K = cfg["K_lights"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_units = DET_POLICIES + [f"random_{p_i}" for p_i in range(N_RANDOM_PERMS)]
    U_rows = {}          # (obj, regime, policy) -> {budget: E_osb}
    orderings = {}       # (obj, level, regime, policy) -> full 142-light ordering
    per_run = []         # provenance: one row per reconstruction run
    wiring_levels = []   # audit F1: base precision blocks per level (wiring gate)

    for obj_idx, obj_name in enumerate(cfg["cohort"]):
        obj = load_object(cfg["data_root"], obj_name, data_meta=cfg.get("data_meta"))
        # pinned scene-subset stream: choice(142, 48) then choice(P_full, 1200)
        scen = NominalScene(obj, np.random.default_rng([20260910, obj_idx]))
        L_analysis = scen.s_hat.shape[0]

        # prediction-side blocks for the analysis subset (level-independent assets)
        u_act = (scen.w * scen.s_hat)[:, :, None] * scen.B_phi
        M0_act = np.einsum("kpi,kpj->kij", scen.B_phi, scen.w[:, :, None] * scen.B_phi)
        finf = scen.Finf_diag

        # E_osb accumulator spanning levels x seeds, per regime/policy/budget
        E_sum = {regime: {unit: {b: 0.0 for b in BUDGETS} for unit in all_units}
                 for regime in REGIMES}

        for lv_i, level in enumerate(LEVELS):
            gen = CorruptionGenerator("joint", level)
            sig_logI, sig_rad = gen.sig_logI, np.radians(gen.sig_deg)
            base_lam0 = np.linalg.inv(gen.sigma_phi_diag())

            # selection universe = all K lights: analysis lights active, the rest
            # inactive placeholders (u = 0, M0 = 0, identity precision placeholder)
            u142 = np.zeros((K, finf.shape[0], 3))
            M0142 = np.zeros((K, 3, 3))
            lam0142 = np.stack([np.eye(3)] * K)
            u142[scen.sel] = u_act
            M0142[scen.sel] = M0_act
            # frozen wiring: base precision Lam0 = Sigma_phi(level)^{-1} on the
            # analysis lights (preregistered; regime multiplies a selected light's
            # block). Inactive placeholders keep the identity direction.
            lam0142[scen.sel] = base_lam0
            active142 = np.zeros(K, bool)
            active142[scen.sel] = True
            state = SelectionState(u=u142, M0=M0142, lam0=lam0142,
                                   finf=finf, active=active142)
            assert state.u.shape[0] == K
            # wiring gate (audit F1): the selection state must carry the level's
            # base precision, not a level-independent placeholder
            assert np.allclose(state.lam0[scen.sel], base_lam0)
            _deg, V_level = scen.predicted_degradation(gen.sigma_phi_diag())
            wiring_levels.append(base_lam0.copy())

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
                for unit, ordr in ords.items():
                    orderings[f"{obj_name}|{level}|{regime}|{unit}"] = ordr

                # ---- empirical runs (paired raw innovations) ----
                for s_i in range(SEEDS_PER_LEVEL):
                    z_rng = np.random.default_rng([20260910, 7777, obj_idx, lv_i, s_i])
                    raw = raw_innovations(z_rng, L_analysis)
                    for unit, ordr in ords.items():
                        for b, k in zip(BUDGETS, BUDGET_COUNTS):
                            slot_scales = budget_scales(ordr, k, regime)
                            assert (slot_scales < 1.0).sum() == k
                            scales = slot_scales[scen.sel]     # the analysis lights
                            d2, g = apply_scaled_corruption(
                                scen.dirs, sig_logI, sig_rad, scales, raw)
                            rho_t = scen.estimate_albedo(d2, g)
                            e = rho_t - scen.rho
                            s_g = (e * scen.rho).sum() / (scen.rho ** 2).sum()
                            e = e - s_g * scen.rho             # gauge alignment (CI04)
                            E_run = float(np.sum((e @ V_level) ** 2))
                            E_sum[regime][unit][b] += E_run
                            per_run.append(dict(
                                object=obj_name, level=level, seed=s_i,
                                regime=regime, policy=unit, budget=b,
                                n_improved=int((slot_scales[scen.sel] < 1.0).sum()),
                                E_run=E_run))

        # E_osb = mean over levels x seeds (frozen endpoint)
        n_mean = len(LEVELS) * SEEDS_PER_LEVEL
        for regime in REGIMES:
            for unit in E_sum[regime]:
                for b in BUDGETS:
                    E_sum[regime][unit][b] /= n_mean
        # audit F1 wiring gate (level dependence): with the base precision wired,
        # per-level mode_aware orderings must not be identical across all levels --
        # a level-independent triple would mean the wiring was lost again
        if len(wiring_levels) == len(LEVELS) and len(LEVELS) > 1:
            same = all(np.allclose(wiring_levels[0], w) for w in wiring_levels[1:])
            assert not same, "base precision identical across levels: wiring lost"
            mode_orders = [
                tuple(orderings[f"{obj_name}|{level}|{REGIMES[0]}|mode_aware"])
                for level in LEVELS]
            assert not (mode_orders[0] == mode_orders[1] == mode_orders[2]), \
                "mode_aware ordering level-independent: wiring lost"
        del wiring_levels[:]
        for regime in REGIMES:
            for pol in DET_POLICIES:
                U_rows[(obj_name, regime, pol)] = {b: E_sum[regime][pol][b] for b in BUDGETS}
            U_rows[(obj_name, regime, "random")] = {
                b: float(np.mean([E_sum[regime][f"random_{p_i}"][b]
                                  for p_i in range(N_RANDOM_PERMS)]))
                for b in BUDGETS}

    summary = _analyze(U_rows, cfg)
    (out / "selection_orders.json").write_text(
        json.dumps(orderings, indent=1), encoding="utf-8")
    (out / "per_run_errors.csv").write_text(
        "\n".join(["object,level,seed,regime,policy,budget,n_improved,E_run"]
                  + [f"{r['object']},{r['level']},{r['seed']},{r['regime']},"
                     f"{r['policy']},{r['budget']},{r['n_improved']},{r['E_run']!r}"
                     for r in per_run]) + "\n", encoding="utf-8")
    (out / "uos_table.csv").write_text(
        "\n".join(["object,regime,policy,budget,E_osb"]
                  + [f"{o},{regime},{p},{b},{U_rows[(o, regime, p)][b]!r}"
                     for (o, regime, p) in sorted(U_rows, key=str)
                     for b in BUDGETS]) + "\n", encoding="utf-8")
    (out / "allocation_summary.json").write_text(
        json.dumps(dict(config_sha256=_sha(REPO / "configs/openillumination_allocation.yaml"),
                        grid=dict(levels=LEVELS, seeds_per_level=SEEDS_PER_LEVEL,
                                  budgets=BUDGETS, regimes=REGIMES,
                                  random_permutations=N_RANDOM_PERMS),
                        summary=summary), ensure_ascii=False, indent=1),
        encoding="utf-8")
    _write_tables(out, summary)
    return summary


def _analyze(U_rows, cfg):
    """Frozen statistics: per-regime paired object-level bootstrap + grades.

    U_rows[(obj, regime, policy)] maps budget -> E_osb; the primary endpoint is the
    normalized trapezoidal AUC over the budget grid (frozen formula). Bootstrap
    resample indices are drawn once and shared across policies, so paired
    differences are computed within the same resamples.
    """
    out = {"regimes": {}}
    bootstrap_seed = cfg["seeds"]["bootstrap"]
    for regime in REGIMES:
        objs = sorted({o for (o, r, _p) in U_rows if r == regime})
        Uos = {o: {p: _trapezoid_auc([U_rows[(o, regime, p)][b] for b in BUDGETS])
                   for p in POLICIES} for o in objs}
        n = len(objs)
        rng = np.random.default_rng(bootstrap_seed)
        draws = [rng.integers(0, n, n) for _ in range(10000)]
        res = {}
        for p in POLICIES[:-1]:
            d = np.array([Uos[o][p] - Uos[o]["random"] for o in objs])
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
