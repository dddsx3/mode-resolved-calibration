# status: experimental — NOT part of the published results
"""M14: exact identities/proofs, non-circular intervals, and artifact replay gates.

Random checks below are regression checks, not proofs; symbolic identities and
explicit rational examples accompany the full analytical proof in the TeX file.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import block_diag
import sympy as sp

from experiments.theory_remainder import (
    BinnedRemainder, GaugeSpectrum, SpectralBin, analyze_prior,
    compress_spectrum, endpoint_certificate, prepare_geometry, value_certificate,
)

REPO = Path(__file__).resolve().parents[1]
ART = REPO / 'results/theory_extension_20260918/two_term_remainder.json'


def _problem(seed=19, p=5, n=14, m=7):
    rng = np.random.default_rng(seed)
    Q = np.linalg.qr(rng.normal(size=(n, p)))[0]
    d = rng.uniform(0.4, 3.0, p)
    A = Q * np.sqrt(d)[None, :]
    a = rng.normal(size=p)
    a /= np.linalg.norm(a)
    B = rng.normal(size=(n, m))
    B[:, 0] = A @ a
    c = np.eye(m)[0]
    U, M = A.T @ B, B.T @ B
    X = rng.normal(size=(m, m))
    sigma = X @ X.T + np.eye(m) * 0.25
    geo = prepare_geometry(d, U, M, a, c)
    return A, B, a, c, sigma, geo


def _risk(A, B, a, sigma, t):
    F = A.T @ A - A.T @ B @ np.linalg.solve(B.T @ B + t * np.linalg.inv(sigma), B.T @ A)
    return float(a @ np.linalg.solve(F, a))


@pytest.mark.parametrize('seed', [4, 19, 81, 203])
@pytest.mark.parametrize('t', [0.02, 0.7, 1.0, 10.0, 27.0])
def test_exact_remainder_identity_matches_independent_dense_fisher(seed, t):
    A, B, a, c, sigma, geo = _problem(seed)
    spec = analyze_prior(geo, sigma)
    got = 1 / (t * spec.q) + spec.r + spec.remainder_for_validation(t)
    expected = _risk(A, B, a, sigma, t)
    assert got == pytest.approx(expected, rel=2e-9, abs=1e-10)
    assert geo.delta_diagonal >= 0
    assert spec.remainder_for_validation(t) >= geo.delta_diagonal
    assert spec.diagnostics['prior_residual_squared'] == pytest.approx(
        geo.g @ sigma @ geo.g - 1 / spec.q, rel=1e-10, abs=1e-12)
    # q uses the full prior, not an intensity marginal or a single light.
    assert spec.q == pytest.approx(c @ np.linalg.solve(sigma, c), rel=1e-11)


def test_exact_symbolic_physical_direction_only_counterexample():
    # A=sqrt(w)*s=60; B=sqrt(w)*rho*(s,u,0)=(60,80,0),
    # w=10000, rho=1, n.d=3/5, n.t1=4/5, sigma_dir=1/100 rad.
    t, eps = sp.symbols('t eps', positive=True)
    A = sp.Matrix([[60]])
    B = sp.Matrix([[60, 80, 0]])
    sigma = sp.diag(eps**2, sp.Rational(1, 10000), sp.Rational(1, 10000))
    F = A.T * A - A.T * B * (B.T * B + t * sigma.inv()).inv() * B.T * A
    J = sp.simplify(F.inv()[0, 0])
    main = eps**2 / t + sp.Rational(1, 3600)
    rem = sp.Rational(4, 22500) / t
    assert sp.simplify(J - main - rem) == 0
    assert A * sp.Matrix([1]) == B * sp.Matrix([1, 0, 0])
    v = sp.factor(1 - J.subs(t, 10) / J.subs(t, 1))
    assert sp.limit(v, eps, 0) == sp.Rational(72, 205)
    assert sp.limit(1 - main.subs(t, 10) / main.subs(t, 1), eps, 0) == 0


def test_symbolic_diagonal_residual_and_value_kernel():
    d1, d2, p = sp.symbols('d1 d2 p', positive=True)
    mu = p * d1 + (1 - p) * d2
    delta = p / d1 + (1 - p) / d2 - 1 / mu
    positive_sum = (p * (d1 - mu)**2 / d1 + (1 - p) * (d2 - mu)**2 / d2) / mu**2
    assert sp.factor(delta - positive_sum) == 0
    x, k, qr = sp.symbols('x k qr', positive=True)
    beta = (1 / k + qr) / (1 + qr)
    kernel = 1 / (k + x) - beta / (1 + x)
    assert sp.factor(kernel - (1 - beta) * (x - k * qr) / ((1 + x) * (k + x))) == 0


def test_gauge_alone_can_have_arbitrarily_large_relative_risk_error():
    # Same physical one-pixel construction; arbitrary positive noise precision w.
    for weight in (1.0, 1e2, 1e6):
        A = np.array([[np.sqrt(weight) * 0.6]])
        B = np.array([[np.sqrt(weight) * 0.6, np.sqrt(weight) * 0.8, 0.0]])
        geo = prepare_geometry(np.diag(A.T @ A), A.T @ B, B.T @ B,
                               np.ones(1), np.array([1.0, 0.0, 0.0]))
        spec = analyze_prior(geo, np.diag([1e-20, 1e-4, 1e-4]))
        cert = endpoint_certificate(compress_spectrum(spec), 1.0)
        assert cert['remainder_upper'] >= weight * 0.64e-4 * spec.r * (1 - 1e-10)
        if weight == 1e6:
            assert cert['relative_to_prediction_upper'] > 60


def test_direction_only_certificate_excludes_small_value_error_without_actual_input():
    geo = prepare_geometry(np.array([3600.0]), np.array([[3600., 4800., 0.]]),
                           np.outer([60., 80., 0.], [60., 80., 0.]),
                           np.ones(1), np.array([1., 0., 0.]))
    sigma = np.diag([1e-12, 1e-4, 1e-4])
    bins = compress_spectrum(analyze_prior(geo, sigma))
    cert = value_certificate(bins, 10.0)
    A, B = np.array([[60.]]), np.array([[60., 80., 0.]])
    actual = 1 - _risk(A, B, np.ones(1), sigma, 10) / _risk(A, B, np.ones(1), sigma, 1)
    error = actual - cert['prediction']
    assert cert['correction_lower'] > 0.05
    assert cert['correction_lower'] <= error <= cert['correction_upper']
    assert actual == pytest.approx(float(sp.Rational(72, 205)), abs=2e-8)


def test_two_term_exact_case_and_zero_mass_are_retained():
    geo = prepare_geometry(np.array([4.0]), np.array([[4.0]]), np.array([[4.0]]),
                           np.ones(1), np.ones(1))
    spec = analyze_prior(geo, np.array([[0.09]]))
    assert len(spec.eigenvalues) == 0
    bins = compress_spectrum(spec)
    assert endpoint_certificate(bins, 1)['remainder_upper'] == 0
    assert value_certificate(bins, 10)['absolute_error_upper'] == 0
    assert spec.r == 0.25


def test_additional_null_modes_are_not_discarded_as_the_named_gauge():
    A = np.eye(2)
    a = np.array([1., 0.])
    B = np.array([[1., 2., 0.], [0., 0., 1.]])
    geo = prepare_geometry(np.ones(2), B, B.T @ B, a, np.array([1., 0., 0.]))
    spec = analyze_prior(geo, np.eye(3))
    assert np.all(spec.eigenvalues == 0)
    assert spec.diagnostics['prior_residual_squared'] == pytest.approx(4)
    assert spec.remainder_for_validation(10) == pytest.approx(0.4)
    assert spec.r + 1 / (10 * spec.q) + spec.remainder_for_validation(10) == pytest.approx(
        _risk(A, B, a, np.eye(3), 10))


def test_exact_zero_projection_gram_with_roundoff_uses_parent_scale():
    # Reviewer reproduction: B=AW => T=0 exactly, including multiple gauges.
    # The same RNG must be consumed continuously to reproduce cancellation.
    rng = np.random.default_rng(314159)
    A = np.linalg.qr(rng.normal(size=(7, 3)))[0] * np.sqrt([0.4, 1.2, 3.1])
    a = np.array([1.0, 0.0, 0.0])
    W = rng.normal(size=(3, 4))
    W[:, 0] = a
    B = A @ W
    c = np.array([1.0, 0.0, 0.0, 0.0])
    geo = prepare_geometry(np.diag(A.T @ A), A.T @ B, B.T @ B, a, c)
    for sigma in (np.eye(4), np.diag([0.3, 7., 0.2, 1.2])):
        spec = analyze_prior(geo, sigma)
        expected_scale = np.linalg.norm(B.T @ B, ord='fro') * np.linalg.eigvalsh(sigma)[-1]
        assert spec.diagnostics['spectrum_parent_gram_propagated_scale'] == pytest.approx(expected_scale)
        assert spec.diagnostics['spectrum_psd_guard_tolerance'] == pytest.approx(1e-10 * expected_scale)
        assert spec.diagnostics['spectrum_dimension'] == 3
        for t in (1.0, 10.0):
            got = spec.r + 1 / (t * spec.q) + spec.remainder_for_validation(t)
            assert got == pytest.approx(_risk(A, B, a, sigma, t), rel=2e-11)


def test_scalar_zero_projection_gram_roundoff_does_not_become_indefinite_model():
    A = np.array([[0.1]])
    B = np.array([[0.1, 1.5]])
    geo = prepare_geometry(np.diag(A.T @ A), A.T @ B, B.T @ B,
                           np.ones(1), np.array([1.0, 0.0]))
    assert np.linalg.norm(geo.projection_gram) < 1e-14
    spec = analyze_prior(geo, np.eye(2))
    assert spec.diagnostics['spectrum_psd_guard_tolerance'] > np.linalg.norm(geo.projection_gram)
    assert spec.remainder_for_validation(1) == pytest.approx(225.0, rel=1e-12)
    assert spec.r + 1 / spec.q + spec.remainder_for_validation(1) == pytest.approx(
        _risk(A, B, np.ones(1), np.eye(2), 1), rel=1e-12)


def test_general_bins_do_not_claim_dyadic_factor_two_without_checking_width():
    broad = BinnedRemainder(1., 1., 0., (SpectralBin(0., 100., 1., 1),))
    cert = endpoint_certificate(broad, 1)
    assert not cert['dyadic_factor_two_guarantee']
    exact_at_right = 1.0 / 101.0
    assert cert['remainder_upper'] / exact_at_right == pytest.approx(101.0)
    # At t=100 the same interval DOES imply factor two; t alone is not the gate.
    assert endpoint_certificate(broad, 100)['dyadic_factor_two_guarantee']
    dyadic = BinnedRemainder(1., 1., 0., (SpectralBin(1., 3., 1., 1),))
    assert endpoint_certificate(dyadic, 1)['dyadic_factor_two_guarantee']
    assert not endpoint_certificate(dyadic, 0.5)['dyadic_factor_two_guarantee']


@pytest.mark.parametrize('seed', [0, 11, 82])
def test_binned_endpoint_and_joint_value_bounds_without_observed_error(seed):
    rng = np.random.default_rng(seed)
    eigs = np.r_[0.0, 10**rng.uniform(-4, 5, 50)]
    weights = 10**rng.uniform(-4, 2, len(eigs))
    spec = GaugeSpectrum(3.0, 0.1, 0.07, eigs, weights, {})
    bins = compress_spectrum(spec)
    for t in (0.1, 0.7, 1., 10., 100.):
        cert = endpoint_certificate(bins, t)
        exact = spec.remainder_for_validation(t)
        assert cert['remainder_lower'] <= exact <= cert['remainder_upper']
        if t >= 1:
            assert cert['remainder_upper'] <= 2 * exact
        assert cert['relative_to_actual_upper'] == pytest.approx(
            cert['remainder_upper'] / cert['risk_upper'])
    for kappa in (1.001, 2.0, 10.0, 1000.0):
        cert = value_certificate(bins, kappa)
        J1 = bins.prediction(1) + spec.remainder_for_validation(1)
        Jk = bins.prediction(kappa) + spec.remainder_for_validation(kappa)
        corr = 1 - Jk / J1 - cert['prediction']
        assert cert['correction_lower'] <= corr <= cert['correction_upper']


def test_certificate_is_unchanged_if_exact_eigenvalues_move_inside_the_same_bins():
    # The exact remainder changes, but the certificate cannot see that change.
    first = GaugeSpectrum(2., 0.3, 0.05, np.array([1.1, 4.0]), np.array([2., 3.]), {})
    second = GaugeSpectrum(2., 0.3, 0.05, np.array([2.9, 6.9]), np.array([2., 3.]), {})
    b1, b2 = compress_spectrum(first), compress_spectrum(second)
    assert asdict(b1) == asdict(b2)
    assert first.remainder_for_validation(1) != second.remainder_for_validation(1)
    assert endpoint_certificate(b1, 1) == endpoint_certificate(b2, 1)
    assert value_certificate(b1, 10) == value_certificate(b2, 10)


def test_value_kernel_interior_maximum_not_just_endpoints():
    bins = BinnedRemainder(1.0, 0.1, 0.0, (SpectralBin(0., 100., 1., 1),))
    cert = value_certificate(bins, 10)
    x = cert['kernel_maximizer']
    beta = cert['beta']
    f = lambda z: 1 / (10 + z) - beta / (1 + z)
    assert 0 < x < 100
    assert cert['numerator_upper'] == pytest.approx(f(x))
    assert f(x) > max(f(0), f(100))


def test_no_universal_value_relative_tightness_in_presence_of_cancellation():
    # q=r=1, kappa=10 -> kernel zero at lambda=10; delta=0.
    # A realizable model may put residual mass here. Observed V error is zero,
    # while the intentionally coarsened [7,15] bin still has a positive bound.
    spec = GaugeSpectrum(1., 1., 0., np.array([10.]), np.array([1.]), {})
    bins = compress_spectrum(spec)
    cert = value_certificate(bins, 10)
    actual_v = 1 - (bins.prediction(10) + spec.remainder_for_validation(10)) / (
        bins.prediction(1) + spec.remainder_for_validation(1))
    assert actual_v == pytest.approx(cert['prediction'], abs=1e-15)
    assert cert['absolute_error_upper'] > 0


def test_nuisance_change_of_coordinates_preserves_the_decomposition():
    A, B, a, c, sigma, geo = _problem(42)
    rng = np.random.default_rng(73)
    R = rng.normal(size=(len(c), len(c))) + 4 * np.eye(len(c))
    Ri = np.linalg.inv(R)
    c2, B2, sigma2 = Ri @ c, B @ R, Ri @ sigma @ Ri.T
    geo2 = prepare_geometry(np.diag(A.T @ A), A.T @ B2, B2.T @ B2, a, c2)
    s1, s2 = analyze_prior(geo, sigma), analyze_prior(geo2, sigma2)
    assert s1.q == pytest.approx(s2.q, rel=1e-9)
    for t in (1., 10.):
        assert s1.remainder_for_validation(t) == pytest.approx(s2.remainder_for_validation(t), rel=1e-9)


def test_heterogeneous_coupled_physical_q_and_prior_residual():
    from calibinfo.models.corruption_family import CorruptionFamily
    A = np.array([[1.0], [2.0]])
    B = block_diag(np.array([[2.0, 0.3, -0.4]]), np.array([[4.0, 0.7, 0.1]]))
    c = np.tile([0.5, 0., 0.], 2)
    geo = prepare_geometry(np.array([5.0]), A.T @ B, B.T @ B, np.ones(1), c)
    family = CorruptionFamily(0.5, 2.0, het_sigma=0.5, rho_c=-0.5, seed=20260915, n_lights=2)
    sigma = block_diag(*family.sigma_phi_block())
    spec = analyze_prior(geo, sigma)
    expected_q = np.sum(1 / (family.sigma_logI_vec()**2 * (1 - family.rho_c**2))) / 4.0
    assert spec.q == pytest.approx(expected_q)
    residual = geo.g - np.linalg.solve(sigma, c) / spec.q
    assert residual @ sigma @ residual == pytest.approx(spec.diagnostics['prior_residual_squared'])
    assert 1 / spec.q + spec.r + spec.remainder_for_validation(1) == pytest.approx(
        _risk(A, B, np.ones(1), sigma, 1), rel=1e-10)


def test_all_channels_closed_leaves_diagonal_information_residual():
    A, B, a, c, sigma, geo = _problem(3)
    spec = analyze_prior(geo, sigma * 1e-14)
    assert spec.remainder_for_validation(1) == pytest.approx(geo.delta_diagonal, abs=1e-11)
    assert geo.delta_diagonal > 0


@pytest.mark.parametrize('kind', ['nonunit', 'wrong_gauge', 'nonpositive_D', 'nonfinite', 'inconsistent_gram'])
def test_bad_geometry_rejected_without_silent_repair(kind):
    A, B, a, c, sigma, geo = _problem()
    d, U, M = np.diag(A.T @ A).copy(), A.T @ B, B.T @ B
    if kind == 'nonunit':
        a = 2 * a
    elif kind == 'wrong_gauge':
        c = 2 * c
    elif kind == 'nonpositive_D':
        d[0] = 0
    elif kind == 'nonfinite':
        d[0] = np.nan
    else:
        M = M.copy()
        M[-1, -1] -= 1e6
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        prepare_geometry(d, U, M, a, c)


def test_invalid_prior_and_invalid_certificate_parameters_rejected():
    *_, geo = _problem()
    with pytest.raises(np.linalg.LinAlgError):
        analyze_prior(geo, np.zeros((len(geo.c), len(geo.c))))
    valid = BinnedRemainder(1., 1., 0., ())
    with pytest.raises(ValueError):
        endpoint_certificate(valid, 0)
    with pytest.raises(ValueError):
        value_certificate(valid, 1)
    with pytest.raises(ValueError):
        SpectralBin(-1, 1, 1, 1)
    with pytest.raises(ValueError):
        BinnedRemainder(1., 1., -0.1, ())


def _artifact():
    if not ART.exists():
        pytest.skip('run the authorized research experiment to create the new artifact')
    return json.loads(ART.read_text(encoding='utf-8'))


def test_artifact_all_frozen_nominal_cells_and_both_endpoints():
    out = _artifact()
    goal = json.loads((REPO / 'results/goal_oriented/goal_orientation.json').read_text(encoding='utf-8'))
    expected = {(r['object'], r['level']): r for r in goal['rows']}
    nominal = [r for r in out['rows'] if r['cohort'] == 'nominal']
    assert len(nominal) == len(expected) == 44
    for row in nominal:
        old = expected[(row['object'], row['parameters']['level'])]
        assert row['endpoints']['t1']['observed_risk'] == old['J_rho_mean_1']
        assert row['endpoints']['tkappa']['observed_risk'] == old['J_rho_mean_kappa']
        assert row['value']['observed'] == old['V_rho_mean']
    summary = out['summary']['nominal']
    assert summary['n_endpoints'] == 88
    assert summary['endpoint_bound_violations'] == summary['value_bound_violations'] == 0
    assert summary['endpoint_bound_over_observed']['max'] <= 2 + 1e-4
    assert summary['endpoint_tightness_le_10_count'] == 88
    assert summary['value_tightness_le_10_count'] == 44
    # Explicitly lock the nontrivial lowest-corruption level, not just 8.0.
    low = out['summary']['nominal_by_level']['0.1']
    assert low['endpoint_tightness_le_10_count'] == 22
    assert low['value_tightness_le_10_count'] == 11


def test_artifact_full_family_and_direction_only_failure_are_not_selected_posthoc():
    out = _artifact()
    family = json.loads((REPO / 'results/openillumination/corruption_family_sensitivity.json').read_text(encoding='utf-8'))
    assert out['protocol']['family_grid'] == family['grid']
    overlap = out['protocol']['overlap_scope']
    assert overlap['n_records'] == 539
    assert overlap['n_distinct_object_parameter_conditions'] == 528
    assert overlap['n_repeated_anchor_records'] == 11
    assert out['summary']['family']['n_cells'] == len(family['rows']) == 495
    tags = out['summary']['family_by_tag']
    assert tags['dir_only']['n_cells'] == 88
    assert tags['het_sweep']['n_cells'] == 33
    assert tags['rho_sweep']['n_cells'] == 22
    worst = out['summary']['direction_only_worst']['value']
    assert worst['observed_absolute_error'] == pytest.approx(0.464137, abs=1e-6)
    assert worst['certificate_rules_out_0_05_accuracy']
    assert worst['correction_lower'] <= worst['observed_correction'] <= worst['correction_upper']
    assert out['summary']['overall']['endpoint_bound_violations'] == 0
    assert out['summary']['overall']['value_bound_violations'] == 0


def test_artifact_certificates_recompute_from_bins_without_reading_observed_errors():
    out = _artifact()
    for row in out['rows']:
        summary = BinnedRemainder(row['q'], row['r'], row['delta_diagonal'],
                                  tuple(SpectralBin(**b) for b in row['spectral_bins']))
        for key, t in (('t1', 1.), ('tkappa', out['protocol']['kappa'])):
            cert = endpoint_certificate(summary, t)
            assert cert['remainder_upper'] == row['endpoints'][key]['remainder_upper']
        value = value_certificate(summary, out['protocol']['kappa'])
        assert value['absolute_error_upper'] == row['value']['absolute_error_upper']


def test_artifact_75_historical_results_are_byte_identical():
    out = _artifact()
    manifest = out['manifest']
    assert manifest['frozen_results_count_before'] == manifest['frozen_results_count_after'] == 75
    assert manifest['frozen_tree_sha256_before'] == manifest['frozen_tree_sha256_after']
    assert manifest['changed_frozen_paths'] == []
    for path, expected in manifest['frozen_results_sha256'].items():
        assert hashlib.sha256((REPO / path).read_bytes()).hexdigest() == expected
    # Pin the producing implementation/configs without pinning editable prose.
    for path in ('experiments/theory_remainder.py', 'experiments/two_term_remainder.py',
                 'configs/goal_orientation.yaml', 'configs/corruption_family_sensitivity.yaml'):
        assert hashlib.sha256((REPO / path).read_bytes()).hexdigest() == manifest['source_sha256'][path]
