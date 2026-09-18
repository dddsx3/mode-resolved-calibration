# status: experimental — NOT part of the published results
"""Entry point for preregistered C13/M16 research; never writes old results.

Run S1 first:
  python experiments/matched_ensemble_bridge.py --stage s1
Then the nested experiment and independent bridge validation:
  python experiments/matched_ensemble_bridge.py --stage all
No command-line tuning of seeds, thresholds, populations or MC budgets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

# Bound dense BLAS work before importing numpy, including direct CLI execution.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

REPO = Path(__file__).resolve().parents[1]
for root in (REPO, REPO / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import numpy as np
import scipy

from calibinfo.datasets.openillumination import load_object as load_oi
from experiments.openillumination_validation import NominalScene, _tangents
from experiments.theory_ensemble import (
    matched_gls, matched_photometric, matched_mode_readout,
    sample_mode_ensembles, variance_summary, residual_bootstrap_baseline,
    multiplicative_attribution, finite_photometric_scores,
    gauge_projector, quadratic_risk, gaussian_quadratic_mean_radius,
    risk_error_radius, pairwise_margin_certificate, family_rank_diagnostics,
)

CONFIG = REPO / "configs/matched_ensemble_bridge_20260918.json"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array_hash(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        arr = np.ascontiguousarray(array)
        digest.update(str(arr.shape).encode())
        digest.update(arr.dtype.str.encode())
        digest.update(arr.tobytes())
    return digest.hexdigest()


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_new(path, artifact):
    path = Path(path)
    allowed = (REPO / "results/theory_extension_20260918").resolve()
    if path.resolve().parent != allowed:
        raise ValueError("only the exclusive new theory result directory is writable")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_jsonable(artifact), ensure_ascii=False, indent=2, allow_nan=False)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text + "\n")


def _summary(values):
    values = np.asarray(values, float)
    if not values.size or not np.all(np.isfinite(values)):
        raise ValueError("cannot summarize nonfinite/empty values")
    out = dict(n=int(values.size), minimum=float(values.min()),
               median=float(np.median(values)), maximum=float(values.max()))
    if np.all(values > 0):
        out["geometric_mean"] = float(np.exp(np.log(values).mean()))
    return out


def _manifest(config_path, started):
    sources = ["experiments/theory_ensemble.py", "experiments/matched_ensemble_bridge.py",
               "experiments/openillumination_validation.py",
               "src/calibinfo/information/schur.py",
               "src/calibinfo/datasets/openillumination.py"]
    return dict(config_absolute_path=str(Path(config_path).resolve()),
                config_sha256=_sha(config_path),
                config_mtime_utc=datetime.fromtimestamp(Path(config_path).stat().st_mtime,
                                                        timezone.utc).isoformat(),
                started_utc=started, completed_utc=datetime.now(timezone.utc).isoformat(),
                source_sha256={p: _sha(REPO / p) for p in sources},
                git_head=subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                        capture_output=True, text=True).stdout.strip(),
                git_branch=subprocess.run(["git", "branch", "--show-current"], cwd=REPO,
                                          capture_output=True, text=True).stdout.strip(),
                versions=dict(python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__),
                historical_outputs_written=False,
                experimental_not_published=True)


def _restricted_scene(obj, n_lights, n_pixels, seed, convention):
    rng = np.random.default_rng(seed)
    L = len(obj["light_directions"])
    P = int(np.count_nonzero(obj["mask"]))
    if L < n_lights or P < n_pixels:
        raise ValueError("dataset does not contain the preregistered scene budget")
    sel = np.sort(rng.choice(L, n_lights, replace=False))
    pidx = np.sort(rng.choice(P, n_pixels, replace=False))
    scene = NominalScene(obj, rng, noise_fit_convention=convention,
                         selections=(sel, pidx), n_lights_total=L)
    valid = np.isfinite(scene.Finf_diag) & (scene.Finf_diag > 0)
    excluded = pidx[~valid].tolist()
    # Deterministic exact-unidentifiable removal only, no relative conditioning cut.
    if not valid.all():
        for name in ("I", "s_hat", "h", "w", "B_phi"):
            setattr(scene, name, getattr(scene, name)[:, valid])
        for name in ("rho", "n", "Finf_diag", "pidx"):
            setattr(scene, name, getattr(scene, name)[valid])
    if len(scene.rho) < 6:
        raise ValueError("insufficient identifiable nominal pixels; not resampling")
    coordinates = np.argwhere(obj["mask"])[scene.pidx]
    meta = dict(n_lights_total=L, n_masked_pixels=P, selected_lights=sel.tolist(),
                requested_pixels=n_pixels, retained_pixels=len(scene.rho),
                selected_masked_pixel_indices=scene.pidx.tolist(),
                selected_pixel_coordinates_yx=coordinates.tolist(),
                exact_unidentifiable_pixel_exclusions=excluded,
                nominal_noise_coefficients=dict(intercept_a=scene.a, slope_b=scene.b),
                inverse_weight_variance_range=[float((1 / scene.w).min()),
                                               float((1 / scene.w).max())],
                finf_diagonal_range=[float(scene.Finf_diag.min()), float(scene.Finf_diag.max())],
                parameterization="linear albedo (A=diag(shading)); fixed nominal normals",
                nominal_array_sha256=_array_hash(scene.I, scene.rho, scene.n, scene.w,
                                                scene.B_phi, scene.Finf_diag, sel, scene.pidx))
    return scene, meta


def _oi_scene(cfg, obj_index):
    spec = cfg["openillumination"]
    name = spec["objects"][obj_index]
    olat_root = Path(spec["olat_root"])
    if olat_root.name != "OLAT":
        raise ValueError("olat_root must be the actual OLAT directory")
    obj = load_oi(olat_root.parent, name, camera=spec["camera"], data_meta=spec["data_meta"])
    scene, meta = _restricted_scene(
        obj, spec["n_lights"], spec["n_pixels"], [cfg["seeds"]["scene"], obj_index],
        spec["noise_fit_convention"])
    paths = [Path(spec["data_meta"]) / "light_pos.npy"]
    paths += [olat_root / name / "Lights" / f"{i:03d}" / "com_masked_thumbnail"
              / (spec["camera"] + ".png") for i in range(len(obj["light_directions"]))]
    meta.update(dataset="OpenIllumination", object=name, source_kind="actual_local_thumbnails",
                source_files=[dict(path=str(p.resolve()), sha256=_sha(p)) for p in paths],
                camera=spec["camera"], image_resolution_wh=list(obj["meta"]["resolution"]),
                scope="Real nominal asset only; ensemble observations are synthetic")
    return scene, meta


def _physical_model(scene, level, sigma):
    cov = np.diag([level ** 2, np.radians(level) ** 2, np.radians(level) ** 2])
    return matched_photometric(scene.s_hat, scene.B_phi, 1 / scene.w, cov, sigma)


def _seeds(cfg, index):
    return {name: [cfg["seeds"][name], int(index)] for name in
            ("measurement_noise", "nuisance", "ideal_baseline", "residual_bootstrap")}


def _evaluate_s1(model, gauge, cfg, case_index):
    spec = cfg["s1"]
    modes = matched_mode_readout(model, gauge, spec["n_modes"])
    if not modes["valid"].all():
        raise ValueError("preregistered S1 mode vanished after projection; no variance flooring")
    seeds = _seeds(cfg, case_index)
    samples, baseline = sample_mode_ensembles(
        model, modes["scores"], spec["trials"], spec["batch_size"],
        seeds["measurement_noise"], seeds["nuisance"], seeds["ideal_baseline"])
    numerator = variance_summary(samples, modes["variance"])
    base = variance_summary(baseline, modes["baseline_variance"])
    ratios = (numerator["empirical_over_predicted"] / base["empirical_over_predicted"])
    lo, hi = spec["gate_empirical_over_predicted"]
    gate_vectors = [numerator["empirical_over_predicted"], base["empirical_over_predicted"], ratios]
    identity = model.covariance_identity_diagnostics()
    numerical_pass = max(identity["relative_covariance_identity_error"],
                         identity["relative_baseline_identity_error"]) < spec["identity_relative_tolerance"]
    record = dict(
        trials=spec["trials"], batch_size=spec["batch_size"], seeds=seeds,
        generation="synthetic observation residual B phi + epsilon; phi and epsilon independently resampled",
        baseline="independent synthetic epsilon0, same Sigma and sigma, phi exactly zero",
        estimator="matched marginal GLS; ideal zero-nuisance GLS for baseline",
        sigma=model.sigma, model_metadata=model.metadata,
        covariance_identity=identity, numerator=numerator, ideal_baseline=base,
        primary_ratio_with_analytic_ideal_baseline=numerator["empirical_over_predicted"],
        empirical_degradation_over_prediction=ratios,
        predicted_degradation=modes["predicted_degradation"],
        unprojected_one_over_retention=modes["unprojected_prediction"],
        retention_eigenvalues=modes["retention"],
        projection=dict(formula="P=I-rho rho^T/(rho^T rho); score=P F_inf^(1/2) U",
                        baseline_projected_fraction=modes["baseline_projected_fraction"],
                        score_sha256=_array_hash(modes["scores"]),
                        all_modes_valid=bool(modes["valid"].all()), variance_floor_count=0),
        gate_interval=[lo, hi], numerical_identity_pass=bool(numerical_pass),
        gate_pass=bool(numerical_pass and all(np.all((v >= lo) & (v <= hi)) for v in gate_vectors)),
        variance_ratio_summary=_summary(numerator["empirical_over_predicted"]))
    return record, modes, samples, baseline


def _dense_cases(cfg):
    spec = cfg["dense_controls"]
    rng = np.random.default_rng(cfg["seeds"]["dense_design"])
    m, n, q = spec["m"], spec["n"], spec["q"]
    A = rng.normal(size=(m, n))
    B = rng.normal(size=(m, q))
    scale = np.linspace(0.5, 2.0, m)
    Sigma = spec["noise_ar1_correlation"] ** np.abs(np.arange(m)[:, None] - np.arange(m)[None, :])
    Sigma *= scale[:, None] * scale[None, :]
    U, _ = np.linalg.qr(rng.normal(size=(q, q)))
    Sigma_phi = (U * np.array(spec["nuisance_eigenvalues"])[None, :]) @ U.T
    gauge = rng.uniform(0.4, 1.4, n)
    for i, case in enumerate(spec["cases"]):
        cov = np.zeros_like(Sigma_phi) if case["zero_nuisance"] else Sigma_phi
        yield case["name"], matched_gls(A, B, Sigma, cov, case["sigma"]), gauge, i


def run_s1(cfg, config_path, output):
    if output.exists():
        raise FileExistsError(f"S1 result exists; refusing to overwrite {output}")
    started, clock = datetime.now(timezone.utc).isoformat(), time.perf_counter()
    initial_config_sha = _sha(config_path)
    cases, scenes = [], []
    for name, model, gauge, i in _dense_cases(cfg):
        rec, _, _, _ = _evaluate_s1(model, gauge, cfg, 1000 + i)
        rec.update(case_id=name, nominal_source="independent dense synthetic design", scene_index=None)
        cases.append(rec)
        print(f"[S1] {name}: pass={rec['gate_pass']} {rec['variance_ratio_summary']}", flush=True)
    for oi, name in enumerate(cfg["openillumination"]["objects"]):
        scene, meta = _oi_scene(cfg, oi)
        scenes.append(meta)
        for li, level in enumerate(cfg["openillumination"]["levels"]):
            index = 100 * oi + li
            model = _physical_model(scene, level, cfg["openillumination"]["sigma"])
            rec, _, _, _ = _evaluate_s1(model, scene.rho, cfg, index)
            rec.update(case_id=f"OI_{name}_level_{level:g}", scene_index=oi,
                       object=name, level=level, nominal_source="real OI thumbnails; synthetic observations")
            cases.append(rec)
            print(f"[S1] {name} level={level}: pass={rec['gate_pass']} {rec['variance_ratio_summary']}", flush=True)
    if _sha(config_path) != initial_config_sha:
        raise RuntimeError("preregistered config changed during S1")
    artifact = dict(
        protocol="P-MATCHED-ENSEMBLE", stage="S1", status="completed",
        claim_slot="C13 (reserved; coordinator integrates)", config=cfg,
        n_cases=len(cases), n_mode_checks=sum(len(c["predicted_degradation"]) for c in cases),
        all_pass=all(c["gate_pass"] for c in cases),
        numerator_ratio_summary=_summary(np.concatenate([c["numerator"]["empirical_over_predicted"] for c in cases])),
        baseline_ratio_summary=_summary(np.concatenate([c["ideal_baseline"]["empirical_over_predicted"] for c in cases])),
        degradation_ratio_summary=_summary(np.concatenate([c["empirical_degradation_over_prediction"] for c in cases])),
        maximum_numerical_identity_error=max(c["covariance_identity"]["relative_covariance_identity_error"] for c in cases),
        cases=cases, scenes=scenes, wall_seconds=time.perf_counter() - clock,
        interpretation="Conditional closure of the matched linear ensemble only; not a theorem test on new real observations or the historical estimator.",
        manifest=_manifest(config_path, started))
    _write_new(output, artifact)
    print(f"[S1] all_pass={artifact['all_pass']}; wrote {output}", flush=True)
    if not artifact["all_pass"]:
        raise RuntimeError("S1 preregistered gate failed: investigate implementation, do not tune config")
    return artifact


def run_nested(cfg, s1):
    """Rerun exactly frozen S1 draws; S2 denominator only; S3 forward only."""
    records = []
    by_id = {case["case_id"]: case for case in s1["cases"]}
    for oi, name in enumerate(cfg["openillumination"]["objects"]):
        scene, meta = _oi_scene(cfg, oi)
        if meta["nominal_array_sha256"] != s1["scenes"][oi]["nominal_array_sha256"]:
            raise ValueError("nominal arrays changed since S1; refuse a mixed nested experiment")
        for li, level in enumerate(cfg["openillumination"]["levels"]):
            index = 100 * oi + li
            model = _physical_model(scene, level, cfg["openillumination"]["sigma"])
            rec, modes, linear, baseline = _evaluate_s1(model, scene.rho, cfg, index)
            key = f"OI_{name}_level_{level:g}"
            prior = by_id[key]
            replay = float(np.max(np.abs(rec["numerator"]["empirical_variance"]
                                        / np.array(prior["numerator"]["empirical_variance"]) - 1)))
            if replay > cfg["s1"]["identity_relative_tolerance"]:
                raise ValueError("S1 numerator replay changed")
            seeds = _seeds(cfg, index)
            pool = (scene.I - scene.s_hat * scene.rho[None, :]).ravel()
            bootstrap, boot_meta = residual_bootstrap_baseline(
                model, pool, modes["scores"], cfg["s2"]["trials"], cfg["s1"]["batch_size"],
                seeds["residual_bootstrap"])
            finite, paired_linear, physical_meta = finite_photometric_scores(
                model, scene, modes["scores"], cfg["s3"]["trials"], cfg["s1"]["batch_size"],
                seeds["measurement_noise"], seeds["nuisance"], _tangents)
            paired_error = float(np.max(np.abs(linear - paired_linear)))
            if paired_error > 1e-10 * max(1.0, float(np.max(np.abs(linear)))):
                raise ValueError("S3 did not reuse the S1 linear samples")
            factors = multiplicative_attribution(
                linear.var(0, ddof=1), modes["variance"], modes["baseline_variance"],
                boot_meta["population_variance"], bootstrap.var(0, ddof=1), finite.var(0, ddof=1))
            row = dict(case_id=key, object=name, level=level,
                       scope="OI nominal design; synthetic S1/S3 numerators; real residual pool denominator",
                       replay_relative_variance_error=replay,
                       paired_s1_s3_linear_scores_max_abs_error=paired_error,
                       projection_score_sha256=_array_hash(modes["scores"]),
                       s1_ratio=factors["s1"],
                       s2=dict(bootstrap=boot_meta, empirical_baseline_variance=bootstrap.var(0, ddof=1),
                               empirical_baseline_mean=bootstrap.mean(0),
                               empirical_over_predicted=factors["s2"], numerator_unchanged=True),
                       s3=dict(empirical_variance=finite.var(0, ddof=1),
                               empirical_mean=finite.mean(0),
                               empirical_second_moment=np.mean(finite * finite, axis=0),
                               linear_second_moment=np.mean(linear * linear, axis=0),
                               empirical_over_predicted=factors["s3"],
                               numerator_variance_over_linear_prediction=finite.var(0, ddof=1) / modes["variance"],
                               denominator_unchanged=True, physical_draws=physical_meta),
                       factors=factors)
            records.append(row)
            print(f"[S2/S3] {key}: S2={_summary(factors['s2'])}; nonlinear={_summary(factors['s2_to_s3_nonlinearity_factor'])}", flush=True)
    keys = ["s1", "residual_pool_population_factor", "bootstrap_finite_sample_factor",
            "s1_to_s2_factor", "s2", "s2_to_s3_nonlinearity_factor", "s3"]
    aggregate = {key: _summary(np.concatenate([r["factors"][key] for r in records])) for key in keys}
    geometric_closure = float(aggregate["s3"]["geometric_mean"] / (
        aggregate["s1"]["geometric_mean"] * aggregate["residual_pool_population_factor"]["geometric_mean"]
        * aggregate["bootstrap_finite_sample_factor"]["geometric_mean"]
        * aggregate["s2_to_s3_nonlinearity_factor"]["geometric_mean"]) - 1)
    return dict(cases=records, aggregate=aggregate,
                geometric_mean_multiplicative_closure_error=geometric_closure,
                no_product_of_medians_claim=True,
                historical_53744_explained=False,
                historical_boundary=cfg["s3"]["historical_boundary"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["s1", "all"], required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    out_dir = REPO / cfg["output"]["directory"]
    checkpoint = out_dir / cfg["output"]["s1_checkpoint"]
    if args.stage == "s1":
        run_s1(cfg, args.config, checkpoint)
    else:
        run_all(cfg, args.config, checkpoint, out_dir / cfg["output"]["main"])


def _diligent_scene(cfg, obj_index):
    # Reuse the canonical intensity-normalized loader; do not create a second
    # PNG-normalization convention or the memory-heavy RGB adapter stack.
    from calibinfo.datasets.diligent import load_object as load_diligent
    from PIL import Image

    spec = cfg["bridge"]
    name = spec["objects"][obj_index]
    directory = Path(spec["diligent_root"]) / name
    loaded = load_diligent(directory)
    intensities = loaded["I_norm"]
    # NominalScene consumes images[...,0][:,mask]. Packing already-masked
    # intensities is lossless; original image coordinates are restored below.
    packed = dict(images=intensities[:, None, :, None],
                  mask=np.ones((1, intensities.shape[1]), dtype=bool),
                  light_directions=loaded["dirs"])
    scene, meta = _restricted_scene(
        packed, spec["n_lights"], spec["n_pixels"],
        [cfg["seeds"]["bridge_scene"], obj_index], spec["noise_fit_convention"])
    gt = loaded["normals_gt"][scene.pidx]
    gt = gt / np.linalg.norm(gt, axis=1, keepdims=True)
    angular = np.degrees(np.arccos(np.clip(np.sum(scene.n * gt, axis=1), -1, 1)))
    meta["selected_pixel_coordinates_yx"] = np.argwhere(loaded["mask"])[scene.pidx].tolist()
    # Absolute normalization check against the underlying pixels, not another
    # same-source relative drift endpoint. No GT normals enter the model.
    scale = np.loadtxt(directory / "light_intensities.txt")[:, 0]
    max_difference = 0.0
    for k in scene.sel:
        with Image.open(directory / f"{k + 1:03d}.png") as image:
            raw = np.asarray(image.convert("L"), dtype=float)[loaded["mask"]][scene.pidx]
        direct = raw / scale[k]
        max_difference = max(max_difference, float(np.max(np.abs(direct - scene.I[np.where(scene.sel == k)[0][0]]))))
    if max_difference != 0:
        raise ValueError("canonical DiLiGenT normalization check failed")
    if np.any(np.sum(scene.s_hat > 0, axis=1) == 0):
        raise ValueError("selected DiLiGenT light is entirely inactive; preregistration forbids replacement")
    paths = [directory / name for name in
             ("light_directions.txt", "light_intensities.txt", "Normal_gt.mat", "mask.png")]
    paths += [directory / f"{k + 1:03d}.png" for k in range(len(loaded["dirs"]))]
    meta.update(dataset="DiLiGenT", object=name,
                source_kind="actual independent local pmsData",
                source_files=[dict(path=str(p.resolve()), sha256=_sha(p)) for p in paths],
                canonical_normalization="PIL gray / light_intensities[R]; no /255 or peak rescaling",
                direct_pixel_normalization_max_abs_error=max_difference,
                nominal_absolute_gt_normal_error_deg=dict(mean=float(angular.mean()),
                                                         median=float(np.median(angular)),
                                                         p95=float(np.percentile(angular, 95))),
                gt_usage="absolute sanity only; nominal normals remain image-estimated and frozen",
                selected_light_visible_pixel_counts=np.sum(scene.s_hat > 0, axis=1).tolist(),
                scope=spec["scope"])
    return scene, meta


def _bridge_plan_cells(cfg, n_lights):
    spec = cfg["bridge"]["plan_cells"]
    ladder = [dict(name=f"all_precision_{value:g}", precision=np.full(n_lights, value),
                   selected_local_lights=list(range(n_lights)))
              for value in spec["precision_ladder"]["precision_multipliers"]]
    equal = []
    n_plans = spec["equal_budget_subsets"]["n_plans"]
    for i in range(n_plans):
        selected = np.flatnonzero(np.arange(n_lights) % n_plans == i)
        if len(selected) != spec["equal_budget_subsets"]["budget_k"]:
            raise ValueError("equal-budget plan construction changed")
        precision = np.ones(n_lights)
        precision[selected] = spec["equal_budget_subsets"]["precision_multiplier"]
        equal.append(dict(name=f"modulo_subset_{i}", precision=precision,
                          selected_local_lights=selected.tolist()))
    return [("precision_ladder", ladder), ("equal_budget_subsets", equal)]


def _gaussian_plan_risk(model, H, cfg, seed, simultaneous_count):
    spec = cfg["bridge"]
    N = spec["matched_trials_per_plan"]
    scores, _ = sample_mode_ensembles(
        model, H.T, N, spec["batch_size"], seed + [0], seed + [1], seed + [2])
    losses = np.sum(scores * scores, axis=1)
    predicted = quadratic_risk(model.covariance, H)
    radius = gaussian_quadratic_mean_radius(
        model.covariance, H, N, spec["confidence_failure_probability"], simultaneous_count)
    components = risk_error_radius(
        predicted, covariance_error_bound=0.0, bias_H_norm=0.0,
        remainder_H_rms_bound=0.0, endpoint_expectation_bound=0.0,
        sampling_radius=radius)
    return dict(predicted_expected_risk=predicted,
                empirical_mean_quadratic_loss=float(losses.mean()),
                empirical_over_predicted=float(losses.mean() / predicted),
                empirical_mean_score_squared_norm=float(np.sum(scores.mean(axis=0) ** 2)),
                sampling_radius=radius, radius_components=components,
                simultaneous_count=simultaneous_count,
                failure_probability=spec["confidence_failure_probability"], trials=N,
                matched_conditions=dict(noise_covariance=True, nuisance_covariance=True,
                                        estimator=True, endpoint_H=True, linear_model=True),
                numerical_covariance_identity=model.covariance_identity_diagnostics(),
                gaussian_zero_mean_generator=True,
                generated_observations_not_just_covariance_draws=True)


def _median_diagnostics(rows):
    keys = ["s_pred", "s_real", "fit_anc", "fit_het", "cross_ah", "cross_ha"]
    result = {}
    for key in keys:
        vals = [r[key] for r in rows if r[key] is not None]
        result[key] = dict(median=float(np.median(vals)) if vals else None,
                           n_valid=len(vals), n_constant_cells_skipped=len(rows) - len(vals))
    return result


def _high_precision_photometric_risk(scene, sigma_phi_blocks, sigma, H):
    """Independent 60-digit reference for ill-conditioned float64 cross-checks.

    This is a numerical audit, not threshold tuning. It recomputes the same
    proper full-rank Schur model from the frozen physical arrays and preserves
    the original Woodbury tolerance failure in the artifact. All bridge blocks
    are positive definite; no singular-covariance precision shortcut is used.
    mpmath is supplied with the available sympy installation.
    """
    import mpmath as mp

    with mp.workdps(60):
        pixels = len(scene.rho)
        F = mp.zeros(pixels)
        sigma2 = mp.mpf(float(sigma)) ** 2
        noise = 1 / scene.w
        for k in range(len(scene.dirs)):
            weight = mp.diag([1 / mp.mpf(float(v)) for v in noise[k]])
            A = mp.diag(scene.s_hat[k].tolist())
            B = mp.matrix(scene.B_phi[k].tolist())
            covariance = mp.matrix(sigma_phi_blocks[k].tolist())
            u = A * weight * B
            precision = sigma2 * covariance ** -1
            F += A * weight * A - u * (B.T * weight * B + precision) ** -1 * u.T
        inverse = F ** -1
        task = mp.matrix(H.tolist())
        task_weight = task.T * task
        risk = sigma2 * mp.fsum(task_weight[i, j] * inverse[j, i]
                               for i in range(pixels) for j in range(pixels))
        return dict(expected_risk=float(risk), expected_risk_decimal=mp.nstr(risk, 40),
                    decimal_precision=60,
                    source="same frozen physical arrays; direct per-light Schur assembled at 60 decimal digits")


def run_bridge(cfg):
    """Independent DiLiGenT synthetic proof check AND separately labelled real arm."""
    from calibinfo.allocation.corruption import raw_innovations
    from calibinfo.information.lowrank import woodbury_quad_risk
    from experiments.allocation_mode_tail import arm_metrics

    spec = cfg["bridge"]
    plan_cells = _bridge_plan_cells(cfg, spec["n_lights"])
    simultaneous_count = len(spec["objects"]) * len(spec["families"]) * sum(len(plans) for _, plans in plan_cells)
    cells, scene_metadata = [], []
    clock = time.perf_counter()
    for oi, name in enumerate(spec["objects"]):
        scene, meta = _diligent_scene(cfg, oi)
        scene_metadata.append(meta)
        P = gauge_projector(scene.rho)
        H = P / np.sqrt(len(scene.rho))
        operator_hash = _array_hash(H)
        n_lights = len(scene.dirs)
        het = np.exp(np.linspace(-0.7, 0.7, n_lights))
        het /= np.sqrt(np.mean(het ** 2))
        family_scales = dict(anchor=np.ones(n_lights), het=het)
        base_cov = np.diag([spec["sig_logI"] ** 2,
                            np.radians(spec["sig_dir_deg_per_tangent"]) ** 2,
                            np.radians(spec["sig_dir_deg_per_tangent"]) ** 2])
        reference_model = matched_photometric(scene.s_hat, scene.B_phi, 1 / scene.w,
                                               base_cov, spec["sigma"])
        W = matched_mode_readout(reference_model, scene.rho, 5)["W"]
        u = (scene.w * scene.s_hat)[:, :, None] * scene.B_phi
        M0 = np.einsum("kpi,kpj->kij", scene.B_phi, scene.w[:, :, None] * scene.B_phi)
        for ci, (cell_name, plans) in enumerate(plan_cells):
            families = {}
            for fi, family in enumerate(spec["families"]):
                scale = family_scales[family]
                cov_blocks = base_cov[None] * scale[:, None, None] ** 2
                plan_records = []
                for pi, plan in enumerate(plans):
                    model = matched_photometric(
                        scene.s_hat, scene.B_phi, 1 / scene.w,
                        cov_blocks / plan["precision"][:, None, None], spec["sigma"])
                    gaussian = _gaussian_plan_risk(
                        model, H, cfg, [cfg["seeds"]["bridge_gaussian"], oi, ci, fi, pi],
                        simultaneous_count)
                    # Exact cross-check against the ORIGINAL J_H implementation,
                    # with sigma^2 precision convention kept explicit.
                    J_H = woodbury_quad_risk(
                        scene.Finf_diag, u, M0, spec["sigma"] ** 2 * np.linalg.inv(cov_blocks),
                        np.ones(n_lights, dtype=bool), plan["precision"], H)
                    cross_error = abs(spec["sigma"] ** 2 * J_H / gaussian["predicted_expected_risk"] - 1)
                    cross_pass = cross_error <= cfg["s1"]["identity_relative_tolerance"]
                    numeric_audit = dict(original_woodbury_float64_check_pass=bool(cross_pass),
                                         unchanged_relative_tolerance=cfg["s1"]["identity_relative_tolerance"],
                                         high_precision_reference=None)
                    if not cross_pass:
                        audit = _high_precision_photometric_risk(
                            scene, cov_blocks / plan["precision"][:, None, None], spec["sigma"], H)
                        audit["new_gls_relative_error"] = abs(gaussian["predicted_expected_risk"] / audit["expected_risk"] - 1)
                        audit["original_woodbury_relative_error"] = abs(spec["sigma"] ** 2 * J_H / audit["expected_risk"] - 1)
                        audit["new_gls_pass_at_unchanged_tolerance"] = audit["new_gls_relative_error"] <= cfg["s1"]["identity_relative_tolerance"]
                        numeric_audit["high_precision_reference"] = audit
                        if not audit["new_gls_pass_at_unchanged_tolerance"]:
                            raise ValueError("GLS disagrees with independent high-precision risk at the unchanged tolerance")
                        print(f"[numeric audit] {name}/{family}/{plan['name']}: original Woodbury "
                              f"error={audit['original_woodbury_relative_error']:.3e}; "
                              f"GLS error={audit['new_gls_relative_error']:.3e}; tolerance unchanged", flush=True)
                    plan_records.append(dict(
                        name=plan["name"], precision=plan["precision"],
                        selected_local_lights=plan["selected_local_lights"],
                        matched_gaussian=gaussian, original_woodbury_J_H=J_H,
                        original_J_H_relative_crosscheck_error=cross_error,
                        numerical_audit=numeric_audit,
                        real_accumulator=dict(mse_aligned=[], ang_mean_deg=[], dual_mean=[])))
                families[family] = dict(sd_multipliers=scale, plans=plan_records)
            # The REAL diagnostic uses original fixed-image arm_metrics, so that
            # estimator-side corruption and the one-step normal refit are not
            # silently replaced by this task's synthetic marginal GLS.
            for si in range(spec["fixed_real_trials_per_plan"]):
                raw = raw_innovations(np.random.default_rng(
                    [cfg["seeds"]["bridge_real"], oi, ci, si]), n_lights)
                for family in spec["families"]:
                    scale = family_scales[family]
                    for pi, plan in enumerate(plans):
                        metrics = arm_metrics(
                            scale / np.sqrt(plan["precision"]), raw, scene,
                            spec["sig_logI"], np.radians(spec["sig_dir_deg_per_tangent"]), W)
                        accumulator = families[family]["plans"][pi]["real_accumulator"]
                        accumulator["mse_aligned"].append(metrics["mse_aligned"])
                        accumulator["ang_mean_deg"].append(metrics["ang_mean_deg"])
                        accumulator["dual_mean"].append(float(np.mean(metrics["dual"])))
            certificates = {}
            for family in spec["families"]:
                rows = families[family]["plans"]
                for row in rows:
                    acc = row.pop("real_accumulator")
                    row["fixed_real_image"] = dict(
                        endpoint_mean={ep: float(np.mean(vals)) for ep, vals in acc.items()},
                        endpoint_sample_standard_deviation={ep: float(np.std(vals, ddof=1))
                                                           for ep, vals in acc.items()},
                        trials=spec["fixed_real_trials_per_plan"],
                        expectation_or_sampling_certificate=None,
                        certification_status="not certified: marginal-GLS covariance/ensemble/tail conditions are not established for this fixed-image plug-in estimator",
                        normal_endpoint="angular drift vs image-estimated nominal normals, not GT error",
                        H_albedo_operator_sha256=operator_hash,
                        mse_empirical_over_predicted=float(np.mean(acc["mse_aligned"])
                            / row["matched_gaussian"]["predicted_expected_risk"]))
                pred = [row["matched_gaussian"]["predicted_expected_risk"] for row in rows]
                means = [row["matched_gaussian"]["empirical_mean_quadratic_loss"] for row in rows]
                radii = [row["matched_gaussian"]["radius_components"]["total_radius"] for row in rows]
                certificates[family] = dict(
                    expectation=pairwise_margin_certificate(pred, np.zeros(len(pred)), pred,
                                                              spec["spearman_target"]),
                    gaussian_empirical=pairwise_margin_certificate(pred, radii, means,
                                                                   spec["spearman_target"]))
            anchor, heterogeneous = families["anchor"]["plans"], families["het"]["plans"]
            ap = [r["matched_gaussian"]["predicted_expected_risk"] for r in anchor]
            hp = [r["matched_gaussian"]["predicted_expected_risk"] for r in heterogeneous]
            diagnostics = {}
            diagnostics["matched_population_quadratic_risk"] = family_rank_diagnostics(ap, hp, ap, hp)
            diagnostics["matched_empirical_quadratic_loss"] = family_rank_diagnostics(
                ap, hp, [r["matched_gaussian"]["empirical_mean_quadratic_loss"] for r in anchor],
                [r["matched_gaussian"]["empirical_mean_quadratic_loss"] for r in heterogeneous])
            for ep in ["mse_aligned", "ang_mean_deg", "dual_mean"]:
                diagnostics[f"fixed_real_{ep}"] = family_rank_diagnostics(
                    ap, hp, [r["fixed_real_image"]["endpoint_mean"][ep] for r in anchor],
                    [r["fixed_real_image"]["endpoint_mean"][ep] for r in heterogeneous])
            record = dict(
                object=name, plan_cell=cell_name, H_operator_sha256=operator_hash,
                H_definition="(I-rho rho^T/(rho^T rho))/sqrt(P), fixed for all plans/families within scene",
                equal_budget=cell_name == "equal_budget_subsets",
                selected_set_sizes=[len(plan["selected_local_lights"]) for plan in plans],
                active_set_size=n_lights,
                candidate_pool="selected active lights only; no universe-random comparison or random-advantage claim",
                budget_degeneracy=dict(
                    complete_set_saturation=cell_name == "precision_ladder",
                    interpretation="different precision levels, not a selection-advantage test"
                    if cell_name == "precision_ladder" else "4 of 16 active lights per plan; distinct equal-budget subsets",
                    all_plan_precision_vectors_distinct=len({tuple(plan["precision"]) for plan in plans}) == len(plans)),
                families=families, certificates=certificates, family_e_diag=diagnostics)
            cells.append(record)
            print(f"[bridge] {name}/{cell_name}: matched fit="
                  f"{diagnostics['matched_empirical_quadratic_loss']}; "
                  f"real MSE fit={diagnostics['fixed_real_mse_aligned']}", flush=True)
    diagnostic_names = list(cells[0]["family_e_diag"])
    summary = {name: _median_diagnostics([c["family_e_diag"][name] for c in cells])
               for name in diagnostic_names}
    gaussian_rows = [r["matched_gaussian"] for cell in cells for family in spec["families"]
                     for r in cell["families"][family]["plans"]]
    certificates = [cell["certificates"][family]["gaussian_empirical"]
                    for cell in cells for family in spec["families"]]
    return dict(
        protocol="M16 independent bridge validation", scope=spec["scope"],
        scenes=scene_metadata, cells=cells, family_e_diag_summary=summary,
        n_gaussian_plan_risks=len(gaussian_rows), n_certificate_cells=len(certificates),
        gaussian_risk_ratio_summary=_summary([r["empirical_over_predicted"] for r in gaussian_rows]),
        n_all_pairs_certified=sum(c["all_pairs_certified"] for c in certificates),
        n_target_certified=sum(c["target_certified"] for c in certificates),
        all_gaussian_risks_inside_simultaneous_radius=all(c["simultaneous_error_event_observed"] for c in certificates),
        all_certified_pairs_observed_agree=all(c["certified_pairs_observed_agree"] for c in certificates),
        fixed_real_certificate_claim=False,
        historical_direction_covariance_note="original random-axis plus scalar-angle generator has tangent covariance sigma_rad^2/2 per axis at first order, whereas the nominal prediction specifies sigma_rad^2 per tangent coordinate",
        interpretation="Matched Gaussian expectation identities and measured margins are tested independently; actual-image plug-in rankings remain an uncertified endpoint/ensemble diagnostic. S_pred/S_real are descriptive, not proof of causal exclusion.",
        wall_seconds=time.perf_counter() - clock)


def run_counterexamples(cfg):
    import math
    from fractions import Fraction
    from scipy.stats import spearmanr

    spec = cfg["counterexamples"]
    K = spec["n_policies"]
    N = spec["gaussian_trials"]
    exact, finite = [], []
    simultaneous_count = K * (len(spec["epsilon_values"]) + 1)
    for i, epsilon in enumerate(spec["epsilon_values"]):
        index = np.arange(K)
        pred = 1 + epsilon * index
        true = 1 + epsilon * index[::-1]
        radius = np.full(K, K * epsilon)  # strict interior; exact discrepancy <=(K-1)epsilon
        rational_epsilon = Fraction(str(epsilon))
        exact.append(dict(
            epsilon=epsilon, predicted_risk=pred, actual_expected_risk=true,
            covariance_error_radius=radius,
            exact_max_covariance_error=str((K - 1) * rational_epsilon),
            exact_predicted_risks=[str(1 + int(j) * rational_epsilon) for j in index],
            exact_actual_risks=[str(1 + int(j) * rational_epsilon) for j in index[::-1]],
            certificate=pairwise_margin_certificate(pred, radius, true, cfg["bridge"]["spearman_target"]),
            conditions=dict(same_H=True, unbiased=True, exactly_linear=True,
                            covariance_exactly_matched=False, covariance_error_tends_to_zero=True,
                            sufficient_margin=False)))
        # Different counterexample: covariance now EXACTLY matched, but finite
        # noisy means need a gap. No result-dependent resampling or failure gate.
        rng = np.random.default_rng([cfg["seeds"]["counterexample"], i, 1])
        draws = rng.normal(size=(N, K)) * np.sqrt(pred)[None, :]
        empirical = np.mean(draws ** 2, axis=0)
        rad = np.array([gaussian_quadratic_mean_radius(
            np.array([[v]]), np.ones((1, 1)), N,
            cfg["bridge"]["confidence_failure_probability"], simultaneous_count) for v in pred])
        finite.append(dict(
            epsilon=epsilon, predicted_and_expected_risk=pred,
            empirical_mean=empirical, sampling_radius=rad, trials=N,
            certificate=pairwise_margin_certificate(pred, rad, empirical, cfg["bridge"]["spearman_target"]),
            conditions=dict(same_H=True, covariance_exactly_matched=True,
                            unbiased=True, exactly_linear=True, finite_random_mean=True)))
    risks = np.array(spec["separated_synthetic_risks"], dtype=float)
    rng = np.random.default_rng([cfg["seeds"]["counterexample"], 99])
    draws = rng.normal(size=(N, len(risks))) * np.sqrt(risks)[None, :]
    means = np.mean(draws ** 2, axis=0)
    radii = np.array([gaussian_quadratic_mean_radius(
        np.array([[v]]), np.ones((1, 1)), N, cfg["bridge"]["confidence_failure_probability"],
        simultaneous_count) for v in risks])
    wrong_H = dict(
        covariance_plan0=[[1.0, 0.0], [0.0, 4.0]], covariance_plan1=[[2.0, 0.0], [0.0, 1.0]],
        information_task_H=[[1.0, 0.0]], actual_endpoint_G=[[0.0, 1.0]],
        predicted_H_risks=[1.0, 2.0], actual_G_risks=[4.0, 1.0], spearman=-1.0,
        conditions=dict(covariance_exactly_matched=True, unbiased=True, exactly_linear=True,
                        same_endpoint_H=False),
        failed_condition="The endpoint operator differs: H!=G; normal angular drift and aligned-albedo MSE likewise cannot silently share H.")
    return dict(
        covariance_near_match_without_margin=exact,
        exact_match_but_finite_samples_without_margin=finite,
        finite_sample_impossibility=dict(
            statement="For any finite N and distinct positive Gaussian variances, independent sample mean squared errors have positive densities on (0,infinity). Every reversed-rank open region therefore has positive probability even with exact matching.",
            near_tie_complete_reversal_probability_limit=f"1/{math.factorial(K)}",
            threshold_guarantee_without_margin=False,
            expectation_identity_not_refuted=True),
        wrong_H=wrong_H,
        independent_separated_synthetic_control=dict(
            predicted_and_expected_risk=risks, empirical_mean=means, sampling_radius=radii, trials=N,
            certificate=pairwise_margin_certificate(risks, radii, means, cfg["bridge"]["spearman_target"]),
            scope="independent scalar Gaussian synthetic systems; positive margin witness only, no dataset transfer claim"))


def read_only_historical_context(cfg):
    import zipfile

    source = cfg["archive_reference"]
    with zipfile.ZipFile(source["path"]) as archive:
        raw = archive.read(source["member"])
    amp = json.loads(raw)
    references = dict(
        archive_absolute_path=str(Path(source["path"]).resolve()),
        archive_member=source["member"], archive_member_sha256=hashlib.sha256(raw).hexdigest(),
        archived_source_revision=amp["source_revision"],
        exact_field_paths=dict(
            D_matched_median="variants.D.new_ratio_matched_prediction.pooled.median",
            D_original_prediction_median="variants.D.new_ratio_original_prediction.pooled.median"),
        D_matched_median=amp["variants"]["D"]["new_ratio_matched_prediction"]["pooled"]["median"],
        D_original_prediction_median=amp["variants"]["D"]["new_ratio_original_prediction"]["pooled"]["median"],
        interpretation=amp["interpretation"], historical_outputs_modified=False)
    file_paths = {
        "decision_quality": REPO / "results/openillumination/decision_quality.json",
        "family_e_diag": REPO / "results/openillumination/corruption_family_e_diag.json",
        "linearization_radius": REPO / "results/magnitude/linearization_radius.json"}
    for name, path in file_paths.items():
        art = json.loads(path.read_text(encoding="utf-8"))
        if name == "decision_quality":
            value = {key: art[key] for key in ["spearman_pred_vs_realized", "spearman_informed_only"]}
        elif name == "family_e_diag":
            value = dict(summary=art["summary"],
                         inference_boundary="S_pred/S_real reused descriptively; high S_pred alone does not logically exclude covariance misspecification or endpoint mismatch")
        else:
            value = dict(
                radius_2x_per_selected_object={name: art["radius_2x"]["per_object"][name]
                                              for name in cfg["openillumination"]["objects"]},
                metric_domain=art["metric_domain"],
                not_a_certificate_for_current_S3=True)
        references[name] = dict(absolute_path=str(path.resolve()), sha256=_sha(path), fields=value)
    return references


def run_all(cfg, config_path, checkpoint, output):
    if output.exists():
        raise FileExistsError(f"main result exists; refusing overwrite {output}")
    started, clock = datetime.now(timezone.utc).isoformat(), time.perf_counter()
    if not checkpoint.exists():
        raise FileNotFoundError("Run the preregistered S1 stage first")
    s1 = json.loads(checkpoint.read_text(encoding="utf-8"))
    initial_config_sha = _sha(config_path)
    if not s1["all_pass"] or s1["manifest"]["config_sha256"] != initial_config_sha:
        raise ValueError("S1 did not pass under exactly this preregistration")
    # Exclusive new checkpoint only. No historic amplitude result is loaded as
    # a replacement numerator/denominator in the nested experiment.
    nested_path = output.parent / "matched_ensemble_bridge_nested.json"
    if nested_path.exists():
        nested = json.loads(nested_path.read_text(encoding="utf-8"))
        if nested["config_sha256"] != initial_config_sha or nested["s1_sha256"] != _sha(checkpoint):
            raise ValueError("S2/S3 checkpoint provenance does not match S1")
    else:
        nested = run_nested(cfg, s1)
        nested.update(config_sha256=initial_config_sha, s1_sha256=_sha(checkpoint))
        _write_new(nested_path, nested)
    bridge = run_bridge(cfg)
    counterexamples = run_counterexamples(cfg)
    history = read_only_historical_context(cfg)
    if _sha(config_path) != initial_config_sha:
        raise ValueError("preregistered config changed during experiment")
    out = dict(
        protocol="P-MATCHED-ENSEMBLE and measurable information-risk bridge",
        status="experimental completed; not published results",
        provisional_claim_slots=cfg["provisional_claim_slots"], config=cfg,
        s1=dict(artifact_absolute_path=str(checkpoint.resolve()), artifact_sha256=_sha(checkpoint),
                all_pass=s1["all_pass"], n_cases=s1["n_cases"], n_mode_checks=s1["n_mode_checks"],
                numerator_ratio_summary=s1["numerator_ratio_summary"],
                baseline_ratio_summary=s1["baseline_ratio_summary"],
                degradation_ratio_summary=s1["degradation_ratio_summary"],
                maximum_numerical_identity_error=s1["maximum_numerical_identity_error"]),
        s2_s3=nested, bridge=bridge, counterexamples=counterexamples,
        historical_read_only=history,
        conclusions=dict(
            matched_linear_closure=s1["all_pass"],
            denominator_bias_attributed_per_mode=True,
            s3_only_forward_perturbation_changed=True,
            explains_historical_53744=False,
            expected_quadratic_identity="E||H e||^2 = tr(H Cov(e) H^T)+||H E e||^2; matching gives sigma^2 J_H",
            margin_required_for_empirical_spearman=True,
            no_unconditional_real_or_transfer_ordering_claim=True,
            no_joint_normal_estimator_claim=True),
        wall_seconds=time.perf_counter() - clock, manifest=_manifest(config_path, started))
    _write_new(output, out)
    print(f"[complete] wrote {output}; seconds={out['wall_seconds']:.2f}", flush=True)
    return out


if __name__ == "__main__":
    main()
