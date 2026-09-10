"""random_active48 — the single bounded post-closure attribution control.

Author-authorized (2026-09-10) as the ONLY experiment added after science
closure; everything else in results/ stays frozen. Question: how much of any
policy's advantage over the full-universe random baseline merely reflects
improving SOME Fisher-active light rather than the specific mode-derived
ordering? The control replaces the random baseline with uniform permutations
over the 48 Fisher-active analysis lights ONLY — same cohort, levels, seeds,
regimes, budgets, and paired raw innovations (identical rng spec and draw
order) as the frozen benchmark run.

No selection policy is re-run: U_mode and U_random(full universe) are read
from the committed uos_table.csv. The paired statistics use the identical
object-level bootstrap as A5 (np.random.default_rng(20260910), B=10000,
shared resamples, same draw-generation order). Permutation seed space is
[20260912, obj_idx, r_i, p_i] (disjoint from the benchmark's 20260911).

Saturation: a 48-light permutation prefix covers all 48 active lights once
k >= 48, so budgets {57, 85, 114} share one saturated reconstruction per
(level, seed, regime, perm) — recorded as repeated rows with n_improved = 48.

Outputs (labelled posthoc_attribution_control):
  allocation_random48_per_run.csv
  allocation_random48_summary.json

Usage: python experiments/openillumination_allocation_random48.py \
           --data D:/data/OpenIllumination --meta D:/data/OpenIllumination_meta
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from calibinfo.allocation.corruption import (          # noqa: E402
    apply_scaled_corruption, raw_innovations)
from calibinfo.allocation.policies import budget_scales  # noqa: E402
from calibinfo.datasets.openillumination import load_object  # noqa: E402
from calibinfo.models.corruption import CorruptionGenerator  # noqa: E402
from experiments.openillumination_validation import NominalScene  # noqa: E402

REGIMES = [10, 100]
LEVELS = [0.2, 0.5, 1.0]
SEEDS_PER_LEVEL = 10
BUDGETS = [0.1, 0.2, 0.4, 0.6, 0.8]
BUDGET_COUNTS = [14, 28, 57, 85, 114]
N_PERMS = 5
BOOTSTRAP_SEED = 20260910
B = 10000


def trapezoid_auc(Es):
    Es = np.asarray(Es, float)
    bs = np.asarray(BUDGETS, float)
    return float(np.sum((Es[:-1] + Es[1:]) / 2.0 * np.diff(bs)) / (0.8 - 0.1))


def _recon_error(scen, V_level, sig_logI, sig_rad, scales, raw):
    """One reconstruction: scaled corruption -> whitened GLS estimate ->
    gauge-aligned residual projected on the object's bottom modes (identical
    math to the frozen benchmark's cell_work per-run computation)."""
    d2, g = apply_scaled_corruption(scen.dirs, sig_logI, sig_rad, scales, raw)
    rho_t = scen.estimate_albedo(d2, g)
    e = rho_t - scen.rho
    s_g = (e * scen.rho).sum() / (scen.rho ** 2).sum()
    e = e - s_g * scen.rho
    return float(np.sum((e @ V_level) ** 2))


def run_control(data_root, data_meta):
    cfg = yaml.safe_load(
        (REPO / "configs/openillumination_allocation.yaml").read_text(
            encoding="utf-8"))
    cohort = list(cfg["cohort"])
    assert len(cohort) == 11

    per_run = ["object,level,seed,regime,policy,budget,n_improved,E_run"]
    E48 = {}       # (obj, regime, budget) -> mean over (level, seed, perm)
    for obj_idx, obj_name in enumerate(cohort):
        obj = load_object(data_root, obj_name, data_meta=data_meta)
        scen = NominalScene(obj, np.random.default_rng([20260910, obj_idx]))
        assert len(scen.sel) == 48, f"{obj_name}: expected 48 analysis lights"
        acc = {}   # (regime, budget) -> [E over (level, seed, perm)]
        for lv_i, level in enumerate(LEVELS):
            gen = CorruptionGenerator("joint", level)
            sig_logI, sig_rad = gen.sig_logI, np.radians(gen.sig_deg)
            _deg, V_level = scen.predicted_degradation(gen.sigma_phi_diag())
            for s_i in range(SEEDS_PER_LEVEL):
                raw = raw_innovations(
                    np.random.default_rng([20260910, 7777, obj_idx, lv_i, s_i]),
                    len(scen.sel))
                for r_i, regime in enumerate(REGIMES):
                    # saturation: any prefix with k >= 48 improves all 48
                    # active lights, so one reconstruction serves all such
                    # budgets and all permutations
                    sat_scales = np.full(48, 1.0 / np.sqrt(regime))
                    sat_E = None
                    for p_i in range(N_PERMS):
                        ordr48 = np.random.default_rng(
                            [20260912, obj_idx, r_i, p_i]).permutation(48)
                        for b, k in zip(BUDGETS, BUDGET_COUNTS):
                            if k >= 48:
                                if sat_E is None:
                                    sat_E = _recon_error(
                                        scen, V_level, sig_logI, sig_rad,
                                        sat_scales, raw)
                                E, n_imp = sat_E, 48
                            else:
                                scales = budget_scales(ordr48, k, regime)
                                E = _recon_error(
                                    scen, V_level, sig_logI, sig_rad,
                                    scales, raw)
                                n_imp = k
                            acc.setdefault((regime, b), []).append(E)
                            per_run.append(
                                f"{obj_name},{level},{s_i},{regime},"
                                f"random48_{p_i},{b},{n_imp},{E:.9g}")
        for (regime, b), vals in acc.items():
            assert len(vals) == len(LEVELS) * SEEDS_PER_LEVEL * N_PERMS
            E48[(obj_name, regime, b)] = float(np.mean(vals))

    return cohort, per_run, E48


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="D:/data/OpenIllumination")
    ap.add_argument("--meta", default="D:/data/OpenIllumination_meta")
    args = ap.parse_args()

    cohort, per_run, E48 = run_control(args.data, args.meta)

    # frozen benchmark Uos from the committed uos_table.csv
    frozen = {}
    with open(REPO / "results/openillumination/allocation/uos_table.csv",
              newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            frozen[(r["object"], int(r["regime"]), r["policy"],
                    float(r["budget"]))] = float(r["E_osb"])

    objs = cohort
    U_mode = {o: {r: trapezoid_auc([frozen[(o, r, "mode_aware", b)]
                                   for b in BUDGETS]) for r in REGIMES}
              for o in objs}
    U_rand = {o: {r: trapezoid_auc([frozen[(o, r, "random", b)]
                                   for b in BUDGETS]) for r in REGIMES}
              for o in objs}
    U_r48 = {o: {r: trapezoid_auc([E48[(o, r, b)] for b in BUDGETS])
                 for r in REGIMES} for o in objs}

    # anchor: recomputed frozen deltas must match allocation_summary exactly
    al = json.loads((REPO / "results/openillumination/allocation/"
                     "allocation_summary.json").read_text(encoding="utf-8"))
    for r in REGIMES:
        for o in objs:
            ref = al["regimes"][str(r)]["policy_deltas"]["mode_aware"]["per_object"][o]
            got = U_mode[o][r] - U_rand[o][r]
            assert abs(ref - got) < 1e-9, (o, r, ref, got)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(objs)
    draws = [rng.integers(0, n, n) for _ in range(B)]   # shared resamples

    def paired(get):
        d = np.array([get(o) for o in objs])
        boots = [float(np.median(d[idx])) for idx in draws]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return {"median": float(np.median(d)), "ci95": [float(lo), float(hi)],
                "per_object": {o: float(get(o)) for o in objs},
                "improved_negative": int((d < 0).sum())}

    summary = {"analysis_status": "posthoc_attribution_control",
               "note": "random_active48 replaces the full-universe random "
                       "baseline with uniform permutations over the 48 "
                       "Fisher-active analysis lights only; paired raw "
                       "innovations and identical bootstrap machinery as the "
                       "frozen benchmark. Delta < 0 = the first-named policy "
                       "has lower AUC (better).",
               "protocol": {"cohort": cohort, "levels": LEVELS,
                            "seeds_per_level": SEEDS_PER_LEVEL,
                            "regimes": REGIMES, "budgets": BUDGETS,
                            "budget_counts": BUDGET_COUNTS,
                            "n_perms": N_PERMS,
                            "perm_seed_space": "[20260912, obj_idx, r_i, p_i]",
                            "paired_innovation_seed_space":
                                "[20260910, 7777, obj_idx, lv_i, s_i]",
                            "bootstrap_seed": BOOTSTRAP_SEED, "B": B,
                            "saturation": "k >= 48 covers all 48 active "
                                          "lights; budgets 57/85/114 share "
                                          "the saturated reconstruction"},
               "regimes": {}}
    for r in REGIMES:
        d48 = paired(lambda o: U_mode[o][r] - U_r48[o][r])
        dfull = paired(lambda o: U_mode[o][r] - U_rand[o][r])
        split = paired(lambda o: U_r48[o][r] - U_rand[o][r])
        summary["regimes"][str(r)] = {
            "mode_minus_random_full": dfull,
            "mode_minus_random_active48": d48,
            "random_active48_minus_random_full": split,
            "uos_random_active48": {o: U_r48[o][r] for o in objs},
        }
        print(f"[r48] regime {r}x:")
        print(f"  mode-random_full  median {dfull['median']:+.4g} "
              f"CI [{dfull['ci95'][0]:+.4g}, {dfull['ci95'][1]:+.4g}]")
        print(f"  mode-random48     median {d48['median']:+.4g} "
              f"CI [{d48['ci95'][0]:+.4g}, {d48['ci95'][1]:+.4g}] "
              f"improved {d48['improved_negative']}/11")
        print(f"  random48-random   median {split['median']:+.4g} "
              f"CI [{split['ci95'][0]:+.4g}, {split['ci95'][1]:+.4g}]")

    out = REPO / "results/openillumination/allocation"
    (out / "allocation_random48_per_run.csv").write_bytes(
        ("\n".join(per_run) + "\n").encode("utf-8"))
    (out / "allocation_random48_summary.json").write_bytes(
        (json.dumps(summary, indent=1) + "\n").encode("utf-8"))
    print(f"[r48] wrote allocation_random48_per_run.csv "
          f"({len(per_run) - 1} rows) and allocation_random48_summary.json")


if __name__ == "__main__":
    main()
