# status: experimental — NOT part of the published results
"""Contracts for the isolated C13/M16 research (no published src additions)."""
from __future__ import annotations

from itertools import permutations
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import block_diag
from scipy.stats import spearmanr

from calibinfo.information.schur import delta_f
from experiments.openillumination_validation import _tangents
from experiments.theory_ensemble import (
    proper_covariance_factor, matched_gls, matched_photometric,
    gauge_projector, matched_mode_readout, sample_mode_ensembles,
    residual_bootstrap_baseline, multiplicative_attribution,
    sphere_exp_directions, finite_photometric_scores,
    quadratic_risk, covariance_risk_radius, endpoint_operator_radius,
    risk_error_radius, gaussian_quadratic_mean_radius,
    pairwise_margin_certificate, family_rank_diagnostics,
)

REPO = Path(__file__).resolve().parents[1]


def _small_model(sigma=0.4):
    rng = np.random.default_rng(182031)
    A, B = rng.normal(size=(12, 4)), rng.normal(size=(12, 3))
    noise = 0.5 ** np.abs(np.arange(12)[:, None] - np.arange(12)[None, :])
    phi = np.diag([0.04, 0.008, 0.0])
    return matched_gls(A, B, noise, phi, sigma), A, B, noise, phi


def _physical_scene():
    dirs = np.array([[0.3, 0.1, 1.0], [-0.3, 0.5, 1.0], [0.1, -0.5, 1.0]])
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    normals = np.array([[0.3, 0.2, 1.0], [-0.2, 0.4, 1.0], [0.1, -0.2, 1.0],
                        [-0.3, -0.2, 1.0]])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    rho = np.array([0.4, 0.6, 0.8, 1.0])
    shade = np.maximum((normals @ dirs.T).T, 0.0)
    t1, t2 = _tangents(dirs)
    B = np.stack([shade * rho, (normals @ t1.T).T * rho,
                  (normals @ t2.T).T * rho], axis=-1)
    return SimpleNamespace(dirs=dirs, n=normals, rho=rho, s_hat=shade, B_phi=B)


def test_preregistration_is_frozen_and_experiment_is_quarantined():
    cfg = json.loads((REPO / "configs/matched_ensemble_bridge_20260918.json").read_text(encoding="utf-8"))
    assert cfg["s1"]["gate_empirical_over_predicted"] == [0.8, 1.25]
    assert cfg["s1"]["trials"] == cfg["s2"]["trials"] == cfg["s3"]["trials"] == 8192
    assert cfg["seeds"]["measurement_noise"] == 2026091832
    assert cfg["bridge"]["matched_trials_per_plan"] == 4096
    assert cfg["bridge"]["spearman_target"] == 0.7
    assert not (REPO / "src/calibinfo/information/ensemble_bridge.py").exists()
    for rel in ["experiments/theory_ensemble.py", "experiments/matched_ensemble_bridge.py"]:
        assert (REPO / rel).read_text(encoding="utf-8").startswith(
            "# status: experimental — NOT part of the published results")


@pytest.mark.parametrize("scale", [1.0, 1e-20, 1e12])
def test_proper_factor_preserves_small_singular_support(scale):
    cov = scale * np.diag([4.0, 1.0, 0.0])
    L, meta = proper_covariance_factor(cov)
    assert meta["rank"] == 2
    assert L.shape == (3, 2)
    np.testing.assert_allclose(L @ L.T, cov, rtol=1e-13, atol=scale * 1e-15)


def test_proper_zero_covariance_is_deterministic_not_flat_precision():
    model = matched_gls([[1.0]], [[1.0, 1.0]], [1.0], np.zeros((2, 2)), 1.0)
    np.testing.assert_allclose(model.information, [[1.0]])
    assert model.nuisance_factor.shape == (2, 0)
    assert np.count_nonzero(model.nuisance_draws(np.random.default_rng(1), 3)) == 0


def test_singular_covariance_known_answer_and_pseudoprecision_counterexample():
    model = matched_gls([[1.0]], [[1.0, 1.0]], [1.0], np.diag([1.0, 0.0]), 1.0)
    np.testing.assert_allclose(model.information, [[0.5]], atol=1e-14)
    wrong, _, _ = delta_f(np.ones((1, 1)), np.ones((1, 2)), np.diag([1.0, 0.0]))
    assert abs(wrong.item()) < 1e-14
    assert model.covariance.item() == pytest.approx(2.0)


@pytest.mark.parametrize("cov", [np.diag([1.0, -0.1]), [[1, 0.3], [0.0, 1]],
                                 [[1, np.nan], [np.nan, 1]], [[1j]]])
def test_invalid_covariance_is_rejected(cov):
    with pytest.raises(ValueError):
        proper_covariance_factor(cov)


@pytest.mark.parametrize("sigma", [0.17, 1.0, 2.3])
def test_dense_matched_gls_identity_and_sigma_scaling(sigma):
    model, A, B, noise, phi = _small_model(sigma)
    V = sigma ** 2 * noise + B @ phi @ B.T
    expected = np.linalg.inv(A.T @ np.linalg.solve(V, A))
    np.testing.assert_allclose(model.covariance, expected, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(model.baseline_covariance,
                               sigma ** 2 * np.linalg.inv(A.T @ np.linalg.solve(noise, A)),
                               rtol=1e-12, atol=1e-13)
    diag = model.covariance_identity_diagnostics()
    assert diag["relative_covariance_identity_error"] < 1e-12
    assert diag["relative_baseline_identity_error"] < 1e-12
    assert diag["unbiasedness_error"] < 1e-12


def test_full_rank_precision_has_sigma_squared_after_Sigma_only_whitening():
    rng = np.random.default_rng(16)
    A, B = rng.normal(size=(9, 3)), rng.normal(size=(9, 2))
    var = np.linspace(0.7, 2.1, 9)
    cov, sigma = np.diag([0.04, 0.015]), 0.23
    model = matched_gls(A, B, var, cov, sigma)
    correct, _, _ = delta_f(A / np.sqrt(var)[:, None], B / np.sqrt(var)[:, None],
                             sigma ** 2 * np.linalg.inv(cov))
    wrong, _, _ = delta_f(A / np.sqrt(var)[:, None], B / np.sqrt(var)[:, None],
                           np.linalg.inv(cov))
    np.testing.assert_allclose(model.information, correct, rtol=1e-12)
    assert np.linalg.norm(model.information - wrong) / np.linalg.norm(correct) > 0.01


def test_photometric_blocks_agree_with_general_route_and_zero_covariance():
    scene = _physical_scene()
    noise = np.arange(12).reshape(3, 4) / 100 + 0.02
    shared = np.diag([0.04, 0.001, 0.0])
    covs = np.stack([shared, np.zeros((3, 3)), shared * 2])
    block = matched_photometric(scene.s_hat, scene.B_phi, noise, covs, sigma=0.7)
    dense = matched_gls(block.A, block.B, noise.ravel(), block_diag(*list(covs)), sigma=0.7)
    for name in ["gain", "baseline_gain", "information", "covariance", "baseline_covariance"]:
        np.testing.assert_allclose(getattr(block, name), getattr(dense, name), rtol=1e-11, atol=1e-13)
    assert block.nuisance_factor.shape == (9, 4)


def test_projection_same_on_covariance_and_samples_not_one_over_retention():
    model, *_ = _small_model()
    rho = np.array([0.3, 0.7, 0.5, 1.0])
    modes = matched_mode_readout(model, rho, 3)
    P, W, Q = modes["projection"], modes["W"], modes["scores"]
    np.testing.assert_allclose(P @ P, P, atol=1e-14)
    np.testing.assert_allclose(P @ rho, 0, atol=1e-14)
    direct = np.diag(W.T @ P @ model.covariance @ P @ W) / np.diag(
        W.T @ P @ model.baseline_covariance @ P @ W)
    np.testing.assert_allclose(modes["predicted_degradation"], direct, rtol=1e-13)
    assert np.max(np.abs(direct - modes["unprojected_prediction"])) > 0.001
    E = np.arange(16).reshape(4, 4)
    np.testing.assert_allclose(E @ Q, E @ P @ W, atol=1e-14)


def test_exactly_projected_away_mode_invalid_without_floor():
    model = matched_gls(np.eye(2), np.eye(2), [1, 1], np.diag([3, 1]), 1.0)
    modes = matched_mode_readout(model, [1, 0], 2)
    assert not modes["valid"][0]
    assert modes["baseline_variance"][0] == 0
    assert np.isnan(modes["predicted_degradation"][0])


def test_actual_observation_draws_and_ideal_gls_close_in_small_mc():
    model, *_ = _small_model()
    readout = matched_mode_readout(model, [1, 0.6, 0.7, 0.5], 3)
    E, E0 = sample_mode_ensembles(model, readout["scores"], 4096, 128, 100, 101, 102)
    assert np.all((E.var(0, ddof=1) / readout["variance"] > 0.8)
                  & (E.var(0, ddof=1) / readout["variance"] < 1.25))
    assert np.all((E0.var(0, ddof=1) / readout["baseline_variance"] > 0.8)
                  & (E0.var(0, ddof=1) / readout["baseline_variance"] < 1.25))


def test_residual_pool_population_covariance_includes_nonzero_mean_no_recentering():
    model, *_ = _small_model()
    Q = np.eye(4)
    pool = np.array([-2.0, -1.0, 2.0, 4.0, 8.0])
    samples, meta = residual_bootstrap_baseline(model, pool, Q, 4096, 128, 111)
    expected = pool.var(ddof=0) * model.baseline_gain @ model.baseline_gain.T
    np.testing.assert_allclose(meta["population_variance"], np.diag(expected))
    np.testing.assert_allclose(meta["population_mean"], pool.mean() * model.baseline_gain.sum(axis=1))
    assert not meta["centered_before_sampling"]
    assert np.all(np.abs(samples.var(0, ddof=1) / np.diag(expected) - 1) < 0.15)


def test_exact_multiplicative_factors_and_geomean_not_medians():
    arrays = [np.array(v, float) for v in ([2, 5, 12], [2, 4, 10], [1, 2, 3],
                                          [2, 6, 1], [2.1, 6.1, 1.1], [3, 7, 20])]
    factors = multiplicative_attribution(*arrays[:5], nonlinear_variance=arrays[5])
    product = factors["s1"] * factors["residual_pool_population_factor"] \
        * factors["bootstrap_finite_sample_factor"] * factors["s2_to_s3_nonlinearity_factor"]
    np.testing.assert_allclose(product, factors["s3"])
    assert factors["s3_closure_relative_error"] < 1e-14
    gm = lambda a: np.exp(np.mean(np.log(a)))
    assert gm(product) == pytest.approx(gm(factors["s3"]))
    assert np.median(factors["s1"]) * np.median(factors["s1_to_s2_factor"]) \
        != pytest.approx(np.median(factors["s2"]))


def test_finite_forward_derivative_matches_nominal_Bphi_not_random_axis_law():
    scene = _physical_scene()
    t1, t2 = _tangents(scene.dirs)
    h = 1e-6
    for channel in range(3):
        phi = np.zeros((2, len(scene.dirs), 3))
        phi[0, :, channel] = h
        phi[1, :, channel] = -h
        d2 = sphere_exp_directions(scene.dirs, t1, t2, phi)
        np.testing.assert_allclose(np.linalg.norm(d2, axis=-1), 1, atol=1e-14)
        signal = np.maximum(np.einsum("blc,pc->blp", d2, scene.n), 0)
        signal *= scene.rho[None, None] * np.exp(phi[:, :, 0, None])
        derivative = (signal[0] - signal[1]) / (2 * h)
        np.testing.assert_allclose(derivative, scene.B_phi[:, :, channel], rtol=1e-8, atol=1e-10)


def test_s3_zero_perturbation_reuses_noise_and_s2_denominator_is_external():
    scene = _physical_scene()
    model = matched_photometric(scene.s_hat, scene.B_phi, np.full((3, 4), 0.01),
                                np.zeros((3, 3)))
    Q = gauge_projector(scene.rho)
    nonlinear, paired, meta = finite_photometric_scores(model, scene, Q, 128, 32, 71, 72, _tangents)
    direct, _ = sample_mode_ensembles(model, Q, 128, 32, 71, 72, 73)
    np.testing.assert_allclose(paired, direct, atol=1e-14)
    np.testing.assert_allclose(nonlinear, paired, atol=1e-14)
    assert not meta["estimator_isomorphic_to_historical"]
    assert meta["shadow_state_change_fraction"] == 0


def test_exact_expected_quadratic_risk_includes_bias_and_operator_semantics():
    C = np.array([[2, 0.2], [0.2, 1]])
    H = np.array([[1, 2], [-1, 0.5]])
    bias = np.array([0.3, -0.4])
    expected = np.trace(H @ C @ H.T) + (H @ bias) @ (H @ bias)
    assert quadratic_risk(C, H, bias) == pytest.approx(expected)
    assert quadratic_risk(C, H) != pytest.approx(quadratic_risk(C, H, bias))


def test_covariance_nuclear_radius_and_endpoint_operator_radius_are_bounds():
    C1, C2 = np.diag([1, 3]), np.diag([2, 1])
    H, G = np.array([[1, 0.5]]), np.array([[0, 2]])
    bias = np.array([0.1, 0.2])
    difference = abs(quadratic_risk(C1, H) - quadratic_risk(C2, H))
    assert difference <= covariance_risk_radius(C1, C2, H) + 1e-14
    endpoint_difference = abs(quadratic_risk(C1, H, bias) - quadratic_risk(C1, G, bias))
    assert endpoint_difference <= endpoint_operator_radius(C1, bias, H, G) + 1e-14


def test_nonlinear_remainder_radius_reaches_cauchy_equality():
    # e_linear=z, r=0.1 z, endpoint is exactly squared error.
    rad = risk_error_radius(1.0, covariance_error_bound=0.0, bias_H_norm=0.0,
                            remainder_H_rms_bound=0.1, endpoint_expectation_bound=0.0,
                            sampling_radius=0.0)
    assert rad["expectation_radius"] == pytest.approx(1.1 ** 2 - 1)


def test_gaussian_quadratic_bound_uses_union_factor_and_noise_units():
    C, H, N, alpha, k = np.diag([2.0, 1.0]), np.eye(2), 2000, 0.01, 4
    t = np.log(2 * k / alpha)
    expected = 2 * np.sqrt(5) * np.sqrt(t / N) + 4 * t / N
    assert gaussian_quadratic_mean_radius(C, H, N, alpha, k) == pytest.approx(expected)
    assert gaussian_quadratic_mean_radius(C * 4, H, N, alpha, k) == pytest.approx(4 * expected)


def test_all_pairwise_margins_force_spearman_one():
    pred = np.array([1, 2, 4, 8.0])
    rad = np.array([0.1, 0.2, 0.3, 0.4])
    real = pred + np.array([0.1, -0.2, 0.3, -0.4])
    certificate = pairwise_margin_certificate(pred, rad, real)
    assert certificate["all_pairs_certified"]
    assert certificate["spearman_lower_bound"] == 1
    assert certificate["observed_spearman"] == 1
    assert certificate["target_certified"]


def test_partial_rank_displacement_bound_by_exhaustive_permutations():
    pred = np.arange(6, dtype=float)
    rad = np.array([0.7, 0.7, 0.1, 0.7, 0.7, 0.1])
    cert = pairwise_margin_certificate(pred, rad)
    count = 0
    for permutation in permutations(range(6)):
        rank = np.argsort(permutation) + 1
        if any(rank[pair["upper_plan"]] <= rank[pair["lower_plan"]]
               for pair in cert["pairs"] if pair["certified"]):
            continue
        rho = float(spearmanr(pred, rank).statistic)
        assert rho >= cert["spearman_lower_bound"] - 1e-14
        count += 1
    assert count > 1


def test_strict_margin_boundary_and_ties_do_not_create_a_certificate():
    boundary = pairwise_margin_certificate([1, 2], [0.5, 0.5])
    assert not boundary["all_pairs_certified"]
    ties = pairwise_margin_certificate([1, 1, 2], [0.0, 0.0, 0.0])
    assert ties["predicted_ties"]
    assert ties["spearman_lower_bound"] is None
    assert not ties["target_certified"]


@pytest.mark.parametrize("epsilon", [1e-2, 1e-4, 1e-6])
def test_no_margin_arbitrarily_small_error_can_reverse_all_ranks(epsilon):
    index = np.arange(6)
    pred = 1 + epsilon * index
    real = 1 + epsilon * index[::-1]
    # An O(epsilon) radius with strict interior avoids a floating equality at 5*epsilon.
    radius = np.full(6, 6 * epsilon)
    cert = pairwise_margin_certificate(pred, radius, real)
    assert cert["simultaneous_error_event_observed"]
    assert cert["observed_spearman"] == -1
    assert cert["n_certified_pairs"] == 0
    assert not cert["target_certified"]


def test_wrong_H_endpoint_reverses_even_perfectly_matched_covariances():
    C0, C1 = np.diag([1.0, 4.0]), np.diag([2.0, 1.0])
    H, G = np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])
    assert quadratic_risk(C0, H) < quadratic_risk(C1, H)
    assert quadratic_risk(C0, G) > quadratic_risk(C1, G)


def test_family_diagnostics_reuse_existing_definitions_and_skip_constant_only():
    values = family_rank_diagnostics([1, 2, 3, 4], [1, 2, 3, 4],
                                     [1, 2, 3, 4], [4, 3, 2, 1])
    assert values == dict(s_pred=1.0, s_real=-1.0, fit_anc=1.0, fit_het=-1.0,
                          cross_ah=-1.0, cross_ha=1.0)
    constant = family_rank_diagnostics([1, 1], [1, 2], [1, 2], [1, 2])
    assert constant["s_pred"] is None
    assert constant["s_real"] == pytest.approx(1)


def test_saved_s1_preregistered_contract_if_artifact_present():
    path = REPO / "results/theory_extension_20260918/matched_ensemble_bridge_s1.json"
    if not path.exists():
        pytest.skip("S1 experiment has not yet produced its immutable new artifact")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    assert artifact["all_pass"]
    assert artifact["n_cases"] == 7
    assert artifact["n_mode_checks"] == 35
    assert artifact["maximum_numerical_identity_error"] < 1e-9
    for case in artifact["cases"]:
        assert case["gate_interval"] == [0.8, 1.25]
        assert case["trials"] == 8192
        assert case["projection"]["variance_floor_count"] == 0
        assert case["projection"]["all_modes_valid"]


def test_60_digit_reference_is_same_fixed_physical_model():
    from experiments.matched_ensemble_bridge import _high_precision_photometric_risk

    scene = _physical_scene()
    scene.w = np.arange(12).reshape(3, 4) + 2.0
    covariance = np.broadcast_to(np.diag([0.01, 0.002, 0.001]), (3, 3, 3))
    H = gauge_projector(scene.rho) / 2
    sigma = 0.4
    model = matched_photometric(scene.s_hat, scene.B_phi, 1 / scene.w, covariance, sigma)
    exact = _high_precision_photometric_risk(scene, covariance, sigma, H)
    assert exact["decimal_precision"] == 60
    assert quadratic_risk(model.covariance, H) == pytest.approx(exact["expected_risk"], rel=1e-12)


def test_original_projected_prediction_and_new_covariance_units_agree():
    try:
        from calibinfo.metrics.gauge import matched_linear_prediction
    except ImportError:
        pytest.skip("coordinator has not yet imported the archived projection helper")
    scene = _physical_scene()
    model = matched_photometric(scene.s_hat, scene.B_phi, np.full((3, 4), 0.02),
                                np.diag([0.02, 0.001, 0.0003]), sigma=0.7)
    modes = matched_mode_readout(model, scene.rho, 3)
    f = np.diag(model.finf)
    invroot = np.diag(1 / np.sqrt(f))
    R = invroot @ model.information @ invroot
    retention, U = np.linalg.eigh((R + R.T) / 2)
    old = matched_linear_prediction(f, retention, U, modes["W"], scene.rho)
    np.testing.assert_allclose(old["pred_deg"], modes["predicted_degradation"], rtol=1e-11)
    np.testing.assert_allclose(0.7 ** 2 * old["variance"], modes["variance"], rtol=1e-11)
    np.testing.assert_allclose(0.7 ** 2 * old["baseline_variance"], modes["baseline_variance"], rtol=1e-11)


@pytest.fixture(scope="module")
def final_artifact():
    path = REPO / "results/theory_extension_20260918/matched_ensemble_bridge.json"
    if not path.exists():
        pytest.skip("full frozen experiment not yet present")
    return json.loads(path.read_text(encoding="utf-8"))


def test_artifact_preregistration_hash_predates_S1_and_is_unchanged(final_artifact):
    import hashlib
    from datetime import datetime

    cfg_path = REPO / "configs/matched_ensemble_bridge_20260918.json"
    cfg_hash = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
    s1path = REPO / "results/theory_extension_20260918/matched_ensemble_bridge_s1.json"
    s1 = json.loads(s1path.read_text(encoding="utf-8"))
    assert cfg_hash == s1["manifest"]["config_sha256"] == final_artifact["manifest"]["config_sha256"]
    assert cfg_hash == "bfb61839b002c02a896a64be5cebca03098db2c38aa0e682cb5bb0f8fb46368f"
    assert datetime.fromisoformat(s1["manifest"]["config_mtime_utc"]) < datetime.fromisoformat(s1["manifest"]["started_utc"])
    assert hashlib.sha256(s1path.read_bytes()).hexdigest() == final_artifact["s1"]["artifact_sha256"]
    assert not final_artifact["manifest"]["historical_outputs_written"]


def test_stored_nested_steps_change_only_the_registered_component(final_artifact):
    nested = final_artifact["s2_s3"]
    assert len(nested["cases"]) == 4
    for case in nested["cases"]:
        assert case["replay_relative_variance_error"] == 0
        assert case["paired_s1_s3_linear_scores_max_abs_error"] == 0
        assert case["s2"]["numerator_unchanged"]
        assert case["s3"]["denominator_unchanged"]
        f = {k: np.asarray(v) for k, v in case["factors"].items()}
        np.testing.assert_allclose(f["s2"], f["s1"] * f["residual_pool_population_factor"]
                                   * f["bootstrap_finite_sample_factor"], rtol=1e-13)
        np.testing.assert_allclose(f["s3"], f["s2"] * f["s2_to_s3_nonlinearity_factor"], rtol=1e-13)
        assert not case["s3"]["physical_draws"]["estimator_isomorphic_to_historical"]
    assert abs(nested["geometric_mean_multiplicative_closure_error"]) < 1e-13
    assert not nested["historical_53744_explained"]


def test_stored_paired_nonlinear_sample_radius_is_valid_not_population_bound(final_artifact):
    for case in final_artifact["s2_s3"]["cases"]:
        nonlinear = np.array(case["s3"]["empirical_second_moment"])
        linear = np.array(case["s3"]["linear_second_moment"])
        meta = case["s3"]["physical_draws"]
        d = np.array(meta["observed_remainder_score_rms"])
        radius = np.array(meta["paired_sample_second_moment_radius"])
        np.testing.assert_allclose(radius, 2 * np.sqrt(linear) * d + d ** 2, rtol=1e-12)
        assert np.all(np.abs(nonlinear - linear) <= radius + 1e-12)
        assert "not a population" in meta["radius_status"]


def test_independent_DiLiGenT_scope_margins_and_endpoints_remain_separate(final_artifact):
    bridge = final_artifact["bridge"]
    assert bridge["n_gaussian_plan_risks"] == 32
    assert bridge["n_certificate_cells"] == 8
    assert bridge["all_gaussian_risks_inside_simultaneous_radius"]
    assert bridge["all_certified_pairs_observed_agree"]
    assert bridge["n_target_certified"] == 2
    assert bridge["n_all_pairs_certified"] == 0
    assert not bridge["fixed_real_certificate_claim"]
    assert {s["object"] for s in bridge["scenes"]} == {"ballPNG", "bearPNG"}
    for scene in bridge["scenes"]:
        assert scene["direct_pixel_normalization_max_abs_error"] == 0
        assert scene["requested_pixels"] == scene["retained_pixels"] == 64
        assert len(scene["selected_lights"]) == 16
    for cell in bridge["cells"]:
        assert cell["budget_degeneracy"]["all_plan_precision_vectors_distinct"]
        for family in ["anchor", "het"]:
            plans = cell["families"][family]["plans"]
            for plan in plans:
                assert plan["fixed_real_image"]["H_albedo_operator_sha256"] == cell["H_operator_sha256"]
                assert plan["fixed_real_image"]["expectation_or_sampling_certificate"] is None
                assert plan["matched_gaussian"]["trials"] == 4096
                assert plan["fixed_real_image"]["trials"] == 256
                assert plan["matched_gaussian"]["simultaneous_count"] == 32
            if cell["equal_budget"]:
                assert [len(p["selected_local_lights"]) for p in plans] == [4, 4, 4, 4]


def test_numeric_audit_preserves_failed_old_float_checks_without_tolerance_tuning(final_artifact):
    audits = [plan["numerical_audit"] for cell in final_artifact["bridge"]["cells"]
              for family in ["anchor", "het"] for plan in cell["families"][family]["plans"]]
    failed = [a for a in audits if not a["original_woodbury_float64_check_pass"]]
    assert len(failed) == 5
    for audit in failed:
        assert audit["unchanged_relative_tolerance"] == 1e-9
        reference = audit["high_precision_reference"]
        assert reference["decimal_precision"] == 60
        assert reference["new_gls_pass_at_unchanged_tolerance"]
        assert reference["new_gls_relative_error"] < 1e-9
        assert reference["original_woodbury_relative_error"] > 1e-9


def test_stored_family_e_diag_matches_definitions_not_another_metric(final_artifact):
    for cell in final_artifact["bridge"]["cells"]:
        a, h = cell["families"]["anchor"]["plans"], cell["families"]["het"]["plans"]
        ap = [p["matched_gaussian"]["predicted_expected_risk"] for p in a]
        hp = [p["matched_gaussian"]["predicted_expected_risk"] for p in h]
        for endpoint in ["mse_aligned", "ang_mean_deg", "dual_mean"]:
            expected = family_rank_diagnostics(
                ap, hp, [p["fixed_real_image"]["endpoint_mean"][endpoint] for p in a],
                [p["fixed_real_image"]["endpoint_mean"][endpoint] for p in h])
            assert expected == cell["family_e_diag"][f"fixed_real_{endpoint}"]


def test_saved_counterexamples_do_not_claim_rank_guarantees_without_margin(final_artifact):
    examples = final_artifact["counterexamples"]
    for row in examples["covariance_near_match_without_margin"]:
        assert row["certificate"]["observed_spearman"] == -1
        assert row["certificate"]["simultaneous_error_event_observed"]
        assert not row["certificate"]["target_certified"]
    for row in examples["exact_match_but_finite_samples_without_margin"]:
        assert not row["certificate"]["target_certified"]
        assert row["conditions"]["covariance_exactly_matched"]
    positive = examples["independent_separated_synthetic_control"]["certificate"]
    assert positive["n_certified_pairs"] == 15
    assert positive["spearman_lower_bound"] == positive["observed_spearman"] == 1
    assert examples["wrong_H"]["spearman"] == -1
