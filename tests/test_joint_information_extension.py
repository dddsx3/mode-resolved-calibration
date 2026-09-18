"""Joint-model known answers; no raw data or historical result rewrites."""
import numpy as np
import pytest
from scipy.linalg import block_diag

from calibinfo.information.schur import delta_f, delta_f_marginal
from calibinfo.metrics.spectral_criteria import identifiable_subspace
from experiments.theory_joint import build_joint_linearization, synthetic_joint_scene


def test_joint_physical_jacobian_matches_central_differences():
    model = synthetic_joint_scene()
    z = np.zeros(3 * (model.P + model.L))
    _, A, B = model.physical(z[:3*model.P], z[3*model.P:], jacobian=True)
    J = np.c_[A, B]
    numerical = np.empty_like(J)
    step = 1e-6
    for j in range(len(z)):
        delta = np.zeros_like(z)
        delta[j] = step
        plus = model.physical(delta[:3*model.P], delta[3*model.P:])
        minus = model.physical(-delta[:3*model.P], -delta[3*model.P:])
        numerical[:, j] = ((plus - minus) / (2*step)).ravel()
    np.testing.assert_allclose(numerical, J, rtol=2e-8, atol=2e-10)
    np.testing.assert_allclose(model.A, np.sqrt(model.weights).ravel()[:, None] * A)
    np.testing.assert_allclose(model.B, np.sqrt(model.weights).ravel()[:, None] * B)


def test_additive_albedo_and_log_albedo_are_congruent_not_identical():
    model = synthetic_joint_scene()
    albedo_cols = model.A[:, ::3]
    additive = albedo_cols / model.rho[None, :]
    np.testing.assert_allclose(albedo_cols, additive @ np.diag(model.rho))
    assert not np.allclose(albedo_cols, additive)
    cosine = np.maximum(model.directions @ model.normals.T, 0)
    expected = np.zeros_like(additive)
    for k in range(model.L):
        expected[k*model.P:(k+1)*model.P] = np.diag(
            np.sqrt(model.weights[k]) * model.intensity[k] * cosine[k])
    np.testing.assert_allclose(additive, expected)


def test_nine_uncalibrated_gauges_and_scale_response_reuse():
    model = synthetic_joint_scene()
    Gx, Gc = model.gauge_generators()
    np.testing.assert_allclose(model.A @ Gx + model.B @ Gc, 0, atol=5e-15)
    assert np.linalg.matrix_rank(Gx) == 9
    assert np.linalg.matrix_rank(model.A) == 3*model.P
    assert np.linalg.matrix_rank(model.B) == 3*model.L
    assert np.linalg.matrix_rank(np.c_[model.A, model.B]) == 3*(model.P+model.L)-9
    F0, _, _ = delta_f(model.A, model.B, 0)
    rank, basis, _ = identifiable_subspace(F0, 1e-10)
    assert rank == 3*model.P-9
    np.testing.assert_allclose(basis.T @ Gx, 0, atol=2e-13)
    report = model.scale_gauge_report()
    assert report['alignment_residual'] == 0
    assert report['gauge_response_max_abs_error'] < 1e-12
    assert np.linalg.eigvalsh(model.fisher(np.ones(model.L)))[0] > 0


def test_joint_degenerate_parallel_lights_have_only_one_scene_axis_per_pixel():
    normals = np.array([[0.0, 0.0, 1.0], [.6, 0, .8]])
    dirs = np.tile([0.0, 0.0, 1.0], (4, 1))
    model = build_joint_linearization([1, 2], normals, dirs)
    basis, metadata = model.fixed_basis()
    assert metadata['rank'] == model.P
    assert basis.shape == (3*model.P, model.P)
    F = model.fisher(np.ones(model.L))
    assert np.linalg.matrix_rank(F, tol=1e-10) == model.P
    Fq = model.fisher(np.ones(model.L), basis)
    assert np.linalg.eigvalsh(Fq)[0] > 0


def test_joint_full_schur_matches_blocks_lowrank_and_marginal():
    model = synthetic_joint_scene()
    t = np.linspace(1, 4, model.L)
    F = model.fisher(t)
    assembled = model.D.copy()
    for k in range(model.L):
        assembled -= model.u[k] @ np.linalg.solve(
            model.M0[k]+t[k]*model.lam0[k], model.u[k].T)
    np.testing.assert_allclose(F, assembled, rtol=1e-11, atol=2e-14)
    sigma = .03
    Sigma = sigma**2 * np.linalg.inv(model.precision(t))
    marginal, _ = delta_f_marginal(model.A, model.B, Sigma, sigma)
    np.testing.assert_allclose(F, marginal, rtol=1e-11, atol=2e-14)
    rng = np.random.default_rng(18)
    H = rng.normal(size=(5, 3*model.P))
    exact = np.trace(H @ np.linalg.solve(F, H.T))
    assert model.task_risk_lowrank(t, H) == pytest.approx(exact, rel=1e-12)
    assert model.risk_and_gradient(t, H)[0] == pytest.approx(exact, rel=1e-12)


def test_joint_precision_ceiling_fixed_kernel_and_concavity():
    model = synthetic_joint_scene()
    rng = np.random.default_rng(20260918)
    F1 = model.fisher(np.ones(model.L))
    kappa = 10
    for _ in range(20):
        t, s = rng.uniform(1, kappa, size=(2, model.L))
        Ft, Fs = model.fisher(t), model.fisher(s)
        assert np.linalg.eigvalsh(Ft-F1)[0] >= -1e-12
        assert np.linalg.eigvalsh(kappa*F1-Ft)[0] >= -1e-12
        assert np.linalg.eigvalsh(model.fisher((t+s)/2)-(Ft+Fs)/2)[0] >= -1e-12
        H = rng.normal(size=(2, 3*model.P))
        j1, _ = model.risk_and_gradient(np.ones(model.L), H)
        jt, _ = model.risk_and_gradient(t, H)
        assert 0 <= 1-jt/j1 <= 1-1/kappa + 1e-12


def test_joint_task_gradient_and_discrete_fw_certificate():
    model = synthetic_joint_scene()
    rng = np.random.default_rng(20260919)
    H = rng.normal(size=(4, 3*model.P))
    kappa, k = 3.0, 2
    budget = k*(kappa-1)
    t = np.full(model.L, 1+budget/model.L)
    value, gradient = model.risk_and_gradient(t, H)
    step = 1e-5
    for j in range(model.L):
        d = np.zeros(model.L)
        d[j] = step
        fd = (model.risk_and_gradient(t+d, H)[0]
              - model.risk_and_gradient(t-d, H)[0])/(2*step)
        assert fd == pytest.approx(gradient[j], rel=1e-6, abs=2e-8)
    cert = model.certificate(t, H, budget, kappa)
    assert cert['risk'] == pytest.approx(value)
    assert cert['gap'] >= -1e-12
    import itertools
    for S in itertools.combinations(range(model.L), k):
        discrete = np.ones(model.L)
        discrete[list(S)] = kappa
        risk, _ = model.risk_and_gradient(discrete, H)
        assert risk >= cert['lower_bound']-1e-10


def test_joint_linear_profile_and_matched_covariance_identity():
    model = synthetic_joint_scene()
    t = np.linspace(1, 3, model.L)
    Lambda = model.precision(t)
    V = np.eye(len(model.A)) + model.B @ np.linalg.solve(Lambda, model.B.T)
    F = model.fisher(t)
    M = np.eye(len(model.A)) - model.B @ np.linalg.solve(
        model.B.T @ model.B + Lambda, model.B.T)
    K = np.linalg.solve(F, model.A.T @ M)
    np.testing.assert_allclose(K @ V @ K.T, np.linalg.inv(F),
                               rtol=2e-12, atol=3e-14)
    observations = np.random.default_rng(4).normal(size=(6, len(model.A)))
    direct = model.linear_profile(observations, t)
    np.testing.assert_allclose(direct, observations @ K.T,
                               rtol=2e-12, atol=2e-13)


def _one_pixel_scene(weights=1.0, prior_scale=1.0, heterogeneous=False):
    directions = np.array([[.6, 0, .8], [0, .6, .8], [-.6, 0, .8]])
    prior = np.tile(prior_scale*np.eye(3), (3, 1, 1))
    if heterogeneous:
        prior[0] = 1e16*np.eye(3)
    return build_joint_linearization(
        [1.0], [[0, 0, 1.0]], directions,
        weights=np.full((3, 1), weights), lam0=prior)


def test_heterogeneous_spd_priors_preserve_risk_gradient_and_certificate():
    model = _one_pixel_scene(heterogeneous=True)
    t = np.ones(3)
    exact = np.array([[1.28, .24, -.24], [.24, .18, 0], [-.24, 0, .54]])
    np.testing.assert_allclose(model.fisher(t), exact, rtol=1e-14, atol=1e-15)
    value, grad = model.risk_and_gradient(t, np.eye(3))
    assert value == pytest.approx(10.894097222222221, rel=1e-13)
    assert model.task_risk_lowrank(t, np.eye(3)) == pytest.approx(value, rel=1e-13)
    step = 1e-5
    direction = np.array([0, step, 0])
    fd = (model.risk_and_gradient(t+direction, np.eye(3))[0]
          - model.risk_and_gradient(t-direction, np.eye(3))[0])/(2*step)
    assert fd == pytest.approx(grad[1], rel=1e-8)
    certificate = model.certificate(t, np.eye(3), 4.0, 3.0)
    improved = model.risk_and_gradient(np.array([1, 3, 3]), np.eye(3))[0]
    assert value-improved <= certificate['gap'] + 1e-12
    observations = np.random.default_rng(14).normal(size=(4, 3))
    np.testing.assert_allclose(model.linear_profile(observations, t),
                               np.linalg.solve(model.A, observations.T).T,
                               rtol=1e-12, atol=1e-12)


def test_joint_lowrank_is_invariant_to_information_units():
    reference = _one_pixel_scene()
    scale = 1e-14
    model = _one_pixel_scene(weights=scale, prior_scale=scale)
    t, H = np.ones(3), np.eye(3)
    np.testing.assert_allclose(model.fisher(t), model.D/2, rtol=1e-13, atol=1e-30)
    exact = reference.risk_and_gradient(t, H)[0]/scale
    assert model.risk_and_gradient(t, H)[0] == pytest.approx(exact, rel=1e-13)
    assert model.task_risk_lowrank(t, H) == pytest.approx(exact, rel=1e-13)


def test_joint_rejects_shadow_boundary_and_nonunit_vectors():
    with pytest.raises(ValueError, match='shadow'):
        build_joint_linearization([1], [[1, 0, 0]], [[0, 0, 1]])
    with pytest.raises(ValueError, match='unit'):
        build_joint_linearization([1], [[0, 0, 2]], [[0, 0, 1]])
