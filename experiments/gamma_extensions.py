# status: experimental — NOT part of the published results
"""Run the independent M13/M15 exact proofs and optional 11-object comparison.

Writes ONLY results/theory_extension_20260918/gamma_extensions.json.
No Git commands, no image modifications, no rewriting historical artifacts.

Example (absolute invocation is supported):
  OPENBLAS_NUM_THREADS=1 python experiments/gamma_extensions.py --real
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh

REPO = Path(__file__).resolve().parents[1]
for folder in (REPO, REPO / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from calibinfo.allocation.blocks import LightBlocks, sym_inv  # noqa: E402
from experiments.theory_gamma import (  # noqa: E402
    alpha_value, exhaustive_gamma, factor_spectral_certificates,
    physical_exact_certificate, physical_lambertian_family, psd_factor,
    spectral_certificates, tight_two_candidate_family,
)

OUTPUT = REPO / "results/theory_extension_20260918/gamma_extensions.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save(report):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
                      + "\n", encoding="utf-8")


def mathematical_report():
    import sympy as sp

    exact = physical_exact_certificate()
    instance = physical_lambertian_family()
    numeric = exhaustive_gamma(instance.baseline, instance.updates)
    bounds = spectral_certificates(instance.baseline, instance.updates)
    physical_rows = []
    for q in ["1/10", "1/30", "1/100", "1/300", "1/1000"]:
        cert = physical_exact_certificate(q)
        p = physical_lambertian_family(float(Fraction(q)))
        row_bounds = spectral_certificates(p.baseline, p.updates)
        physical_rows.append({"q": q, "epsilon_exact": cert["epsilon"],
                              "epsilon": p.epsilon, "alpha": cert["alpha"],
                              "gamma": cert["gamma"], "gamma_exact": cert["gamma_exact"],
                              "witness_upper_ratio": float(Fraction(cert["strict_witness_ratio_exact"])),
                              "gamma_over_epsilon_squared": cert["gamma"]/p.epsilon**2,
                              "bounds": row_bounds,
                              "gamma_over_improved": cert["gamma"]/row_bounds["improved"]})
    tight_rows = []
    ar = sp.Rational(1, 4)
    for denominator in [4, 8, 16, 32, 64]:
        h = sp.Rational(1, denominator)
        a = sp.diag(1, ar)
        w0 = sp.diag(h**-4, 0)
        w1 = h**-4*sp.Matrix([[1, h], [h, h*h]])
        def d(mat, update):
            return sp.factor(sp.trace(mat.inv()-(mat+update).inv()))
        ratios = [sp.factor(d(a, w0)/d(a+w1, w0)),
                  sp.factor(d(a, w1)/d(a+w0, w1))]
        gamma = min(sp.Integer(1), *ratios)
        af, ws = tight_two_candidate_family(float(h), float(ar))
        bb = spectral_certificates(af, ws)
        tight_rows.append({"h_exact": str(h), "h": float(h), "a": float(ar),
                           "gamma_exact": str(gamma), "gamma": float(gamma),
                           "strict_ratios_exact": [str(r) for r in ratios],
                           "bounds": bb, "scaled_gap": float((gamma-ar)/h**2),
                           "scaled_gap_limit": 7/8,
                           "gamma_over_improved": float(gamma)/bb["improved"],
                           "gamma_over_leave_one": float(gamma)/bb["leave_one_spectral"]})
    original_a = np.diag([1., 0.01])
    original_ws = [np.diag([1., 0]), np.array([[0.5, 0.05], [0.05, 0.005]])]
    return {"physical_counterexample": {"exact": exact, "lightblocks": numeric,
                                          "bounds": bounds,
                                          "gamma_over_leave_one": exact["gamma"]/bounds["leave_one_spectral"],
                                          "gamma_over_improved": exact["gamma"]/bounds["improved"]},
            "physical_parameter_family": {
                "fixed_kappa": 10, "fixed_alpha_exact": exact["alpha_exact"],
                "upper_ratio_formula": "3025408256*epsilon^2/[77*(30614089*epsilon^2+531441)]",
                "upper_quadratic_coefficient_exact": "3025408256/40920957",
                "new_bound_order": "Theta(epsilon^2)",
                "upper_ratio_over_new_bound_limit_exact": "672018808864/15154394409",
                "rows": physical_rows},
            "tightness_two_candidates": {
                "scope": "dim=2, two nonzero candidates, fixed chi=1/a; update size unbounded",
                "infimum_gamma_exact": "a=1/chi",
                "family": "A=diag(1,a); W0=h^-4*diag(1,0); W1=h^-4*[[1,h],[h,h^2]]",
                "gamma_expansion": "a+(2*a^2-a+1)*h^2+O(h^4), 0<a<1",
                "new_bound_for_sufficiently_small_h": "a",
                "old_leave_one_expansion": "a*h^4+O(h^6)",
                "fixed_kappa_or_update_norm_infimum_solved": False,
                "pairwise_kantorovich_sharpness_claimed": False,
                "rows": tight_rows},
            "original_general_counterexample": {
                "gamma_exact": "14/109", "gamma": 14/109,
                "bounds": spectral_certificates(original_a, original_ws)}}


def random_exhaustive_report(count=40):
    rows, violations, triples = [], 0, 0
    for seed in range(count):
        rng = np.random.default_rng(20260918+seed)
        n, m = 2+seed % 3, 2+seed % 3
        q = np.linalg.qr(rng.normal(size=(n, n)))[0]
        a = (q*10**rng.uniform(-3, 1, n))@q.T
        fs = [rng.normal(size=(n, 1+(seed+i) % n))*10**rng.uniform(-1, 0.5)
              for i in range(m)]
        ws = [f@f.T for f in fs]
        truth = exhaustive_gamma(a, ws)
        bounds = spectral_certificates(a, ws)
        lowrank = factor_spectral_certificates(a, fs, top_rank=min(2, n))
        failed = truth["gamma"] < bounds["improved"]-1e-8
        violations += int(failed)
        triples += truth["triples"]
        rows.append({"seed": 20260918+seed, "dimension": n, "candidates": m,
                     "gamma": truth["gamma"], "triples": truth["triples"],
                     "bounds": bounds, "lowrank_bounds": lowrank,
                     "violation": bool(failed)})
    return {"purpose": "falsification and implementation testing, NOT tightness evidence",
            "instances": count, "triples": triples, "violations": violations, "rows": rows}


def _nominal_blocks(scene, level):
    from calibinfo.models.corruption import CorruptionGenerator

    precision = np.linalg.inv(CorruptionGenerator("joint", level).sigma_phi_diag())
    u = (scene.w*scene.s_hat)[:, :, None]*scene.B_phi
    m0 = np.einsum("kpi,kpj->kij", scene.B_phi, scene.w[:, :, None]*scene.B_phi)
    lam = np.repeat(precision[None], len(u), axis=0)
    blocks = LightBlocks(u, m0, lam, scene.Finf_diag, np.ones(len(u), bool))
    factors = []
    for k in range(len(u)):
        c = sym_inv(m0[k]+lam[k])-sym_inv(m0[k]+10*lam[k])
        factors.append(u[k]@psd_factor(c))
    return blocks, factors


def real_design_report(report, data_root, data_meta, top_rank=12):
    import yaml
    from calibinfo.datasets.openillumination import load_object
    from experiments.openillumination_validation import NominalScene

    config_path = REPO / "configs/alpha_bound.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    # Reuse only the historical scene parameters, NOT its refuted theorem prose.
    historical_path = REPO / "results/submodularity/alpha_bound.json"
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    previous = {(r["object"], float(r["level"])): r for r in historical["real"]}
    data_root = Path(data_root)
    if data_root.name.upper() == "OLAT":
        data_root = data_root.parent  # existing loader adds OLAT itself
    real = {"status": "running", "cohort_size": len(cfg["cohort"]), "rows": [],
            "levels": [float(x) for x in cfg["levels"]], "kappa": 10,
            "noise_fit_convention": "corrected",
            "scene_rng": "default_rng([20260910, object_index])",
            "pixels": 1200, "analysis_lights": 48, "universe_lights": 142,
            "inactive_placeholders_omitted": 94,
            "omission_reason": "W=0 cannot occur as x or strict-inclusion witness y",
            "data_root": str(data_root), "data_meta": str(data_meta),
            "source_config_sha256": sha(config_path),
            "historical_artifact_sha256_before": sha(historical_path),
            "spectral_method": "one top-subspace solve of total plus one baseline smallest-eigenvalue solve",
            "per_removal_dense_factorizations": 0,
            "old_candidate_is_a_guarantee": False,
            "exact_gamma_on_real_designs_computed": False}
    report["real_designs"] = real
    _save(report)
    for obj_index, name in enumerate(cfg["cohort"]):
        started = time.perf_counter()
        obj = load_object(data_root, name, data_meta=data_meta)
        scene = NominalScene(obj, np.random.default_rng([20260910, obj_index]),
                             noise_fit_convention="corrected")
        del obj
        for level in real["levels"]:
            blocks, factors = _nominal_blocks(scene, level)
            base = blocks.assemble(blocks.lam0)
            bound = factor_spectral_certificates(base, factors, top_rank=top_rank)
            chol = cho_factor(base, lower=True, check_finite=False)
            alpha = max(float(eigh(f.T@cho_solve(chol, f, check_finite=False),
                                   eigvals_only=True, check_finite=False)[-1])
                        for f in factors)
            old = previous[(name, level)]
            upper_old_valid = bound["leave_one_spectral_interval"][1]
            row = {"object": name, "level": level, "dimension": blocks.P,
                   "active_candidates": len(factors), "alpha": alpha,
                   "archived_alpha": old["alpha"],
                   "alpha_abs_difference_to_rounded_archive": abs(alpha-old["alpha"]),
                   "historical_candidate_expression": 1/(1+alpha),
                   "historical_candidate_not_validated": True,
                   "bounds": bound,
                   "improvement_factor_vs_valid_leave_one_at_least": bound["improved"]/upper_old_valid,
                   "leave_one_relative_bracket_width":
                       (upper_old_valid-bound["leave_one_spectral_interval"][0])/upper_old_valid,
                   "subset_lattice_not_enumerated": True}
            real["rows"].append(row)
            print(f"[gamma] {name} level={level:g} alpha={alpha:.6g} "
                  f"old-valid=[{bound['leave_one_spectral_interval'][0]:.8g},"
                  f"{upper_old_valid:.8g}] new={bound['improved']:.8g} "
                  f"factor>={row['improvement_factor_vs_valid_leave_one_at_least']:.5g}", flush=True)
        print(f"[gamma] scene elapsed {time.perf_counter()-started:.2f}s", flush=True)
        _save(report)
    real["status"] = "complete"
    real["design_points"] = len(real["rows"])
    real["historical_artifact_sha256_after"] = sha(historical_path)
    assert real["historical_artifact_sha256_after"] == real["historical_artifact_sha256_before"]
    real["summary"] = {
        "improvement_factor_min": min(r["improvement_factor_vs_valid_leave_one_at_least"] for r in real["rows"]),
        "improvement_factor_max": max(r["improvement_factor_vs_valid_leave_one_at_least"] for r in real["rows"]),
        "leave_one_bracket_relative_width_max": max(r["leave_one_relative_bracket_width"] for r in real["rows"]),
        "alpha_archive_abs_difference_max": max(r["alpha_abs_difference_to_rounded_archive"] for r in real["rows"]),
        "new_lower_bound_min": min(r["bounds"]["improved"] for r in real["rows"]),
        "new_lower_bound_max": max(r["bounds"]["improved"] for r in real["rows"])}
    real["by_level"] = {}
    for level in real["levels"]:
        selected = [r for r in real["rows"] if r["level"] == level]
        worst = min(selected, key=lambda r: r["bounds"]["improved"])
        real["by_level"][str(level)] = {
            "new_lower_bound_min": worst["bounds"]["improved"], "worst_object": worst["object"],
            "valid_leave_one_lower_min": min(r["bounds"]["leave_one_spectral_interval"][0] for r in selected),
            "improvement_factor_min": min(r["improvement_factor_vs_valid_leave_one_at_least"] for r in selected)}


def run(include_real=False, data_root="D:/data/OpenIllumination/OLAT",
        data_meta="D:/data/OpenIllumination_meta", top_rank=12):
    started = time.perf_counter()
    report = {"status": "experimental — NOT part of the published results",
              "date": "2026-09-18", "reserved_claim_ids": ["M13", "M15"],
              "objective": "unweighted full trace, fixed SPD space; S=T included",
              "certificate_formula": "max{b2,4*bfull/(1+bfull)^2}; b2 omits two nonzero candidates",
              "not_claimed": ["full sharp fixed-kappa/m/chi/update-size infimum",
                              "Kantorovich leave-one substitution",
                              "arbitrary rank-deficient task certificate",
                              "diagonal homogeneous physical prior counterexample"],
              "provenance": {"core_sha256": sha(REPO/"experiments/theory_gamma.py"),
                             "runner_sha256": sha(__file__),
                             "blocks_sha256": sha(REPO/"src/calibinfo/allocation/blocks.py"),
                             "nominal_sha256": sha(REPO/"experiments/openillumination_validation.py"),
                             "no_git_commands": True},
              "mathematics": mathematical_report(),
              "random_exhaustive": random_exhaustive_report(),
              "real_designs": {"status": "not_requested"}}
    _save(report)
    if include_real:
        real_design_report(report, data_root, data_meta, top_rank)
    report["elapsed_seconds"] = time.perf_counter()-started
    _save(report)
    print(f"[gamma] wrote {OUTPUT}", flush=True)
    print(f"[gamma] exact gamma={report['mathematics']['physical_counterexample']['exact']['gamma_exact']}; "
          f"random violations={report['random_exhaustive']['violations']}/"
          f"{report['random_exhaustive']['triples']}; elapsed={report['elapsed_seconds']:.2f}s", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", action="store_true", help="also recompute all 11 objects at both levels")
    parser.add_argument("--data-root", default="D:/data/OpenIllumination/OLAT")
    parser.add_argument("--data-meta", default="D:/data/OpenIllumination_meta")
    parser.add_argument("--top-rank", type=int, default=12)
    args = parser.parse_args()
    run(args.real, args.data_root, args.data_meta, args.top_rank)
