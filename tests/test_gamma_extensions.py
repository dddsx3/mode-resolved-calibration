# status: experimental — NOT part of the published results
"""Independent exact and exhaustive regressions for M13/M15 research."""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
import sympy as sp
from scipy.linalg import eigh

from experiments.theory_gamma import (
    DowndateEnvelope, alpha_value, exhaustive_gamma, factor_spectral_certificates,
    full_trace_marginal, physical_exact_certificate, physical_lambertian_family,
    spectral_certificates, tight_two_candidate_family,
)


def _rational_matrix(encoded):
    return np.array([[float(Fraction(x)) for x in row] for row in encoded])


def test_physical_exact_rational_counterexample():
    result = physical_exact_certificate()
    assert result["verified_physical_membership"]
    assert result["verified_prior_spd"]
    assert result["epsilon"] == "200/10001"
    assert result["gamma_exact"] == "121016330240000/4187205554180957"
    assert result["witness_marginal_empty"] == "1/7"
    assert result["witness_marginal_refined"] == "598172222025851/121016330240000"
    assert result["exact_triples"] == 27
    assert result["gamma"] < 0.03 < result["historical_candidate"]
    assert result["coupling_rank"] == 2
    assert result["unaffected_retention_modes"] == 0  # not the false P-3L equality


@pytest.mark.parametrize("q", [0.1, 0.01, 0.001])
def test_lightblocks_reconstructs_all_exact_physical_subsets(q):
    p = physical_lambertian_family(q)
    exact = physical_exact_certificate(str(q))
    assert np.all(np.isfinite(p.normals))
    np.testing.assert_allclose(np.sum(p.normals**2, axis=1), 1, atol=1e-15)
    np.testing.assert_allclose(np.sum(p.directions**2, axis=1), 1, atol=1e-15)
    np.testing.assert_allclose(p.b_phi[:, :, 0], p.shading * p.rho, atol=1e-15)
    np.testing.assert_allclose(p.baseline, _rational_matrix(exact["baseline"]),
                               atol=3e-13, rtol=2e-11)
    assert np.all(p.weights > 0)
    assert np.all(np.linalg.eigvalsh(p.blocks.lam0) > 0)
    for w, encoded in zip(p.updates, exact["updates"]):
        np.testing.assert_allclose(w, _rational_matrix(encoded), atol=2e-13, rtol=2e-10)
    for mask in range(8):
        lam = p.blocks.lam0.copy()
        expected = p.baseline.copy()
        for k in range(3):
            if mask & (1 << k):
                lam[k] *= p.kappa
                expected += p.updates[k]
        np.testing.assert_allclose(p.blocks.assemble(lam), expected, atol=3e-13)
    np.testing.assert_allclose(exhaustive_gamma(p.baseline, p.updates)["gamma"],
                               exact["gamma"], rtol=3e-10)
    assert alpha_value(p.baseline, p.updates) == pytest.approx(exact["alpha"], rel=2e-11)
    slack = (p.kappa-1)*p.baseline-sum(p.updates)
    assert np.linalg.eigvalsh(slack)[0] > 0


def test_physical_fixed_alpha_and_quadratic_decay_symbolically():
    e = sp.symbols("e", positive=True)
    a = sp.diag(sp.Rational(3, 2), e**2)
    w0 = sp.diag(sp.Rational(9, 22), 0)
    w1 = sp.Matrix([[sp.Rational(99, 404), sp.Rational(729, 4444)*e],
                    [sp.Rational(729, 4444)*e, sp.Rational(99, 404)*e**2]])
    d0 = sp.trace(a.inv()-(a+w0).inv())
    d1 = sp.trace((a+w1).inv()-(a+w1+w0).inv())
    ratio = sp.factor(d0/d1)
    expected = 3025408256*e**2/(77*(30614089*e**2+531441))
    assert sp.factor(ratio-expected) == 0
    assert sp.limit(ratio/e**2, e, 0) == sp.Rational(3025408256, 40920957)
    h = sp.diag(sp.sqrt(sp.Rational(2, 3)), 1/e)
    assert not (h*w0*h).has(e)
    assert not (h*w1*h).has(e)
    c = sp.Rational(3025408256, 40920957)
    assert sp.factor(c*e**2-ratio).is_positive


def test_physical_geometry_has_a_common_open_camera_hemisphere():
    p = physical_lambertian_family()
    e, c = p.epsilon, -p.normals[1, 0]
    camera = np.array([e/(2*c), 1., 0.01])
    camera /= np.linalg.norm(camera)
    assert np.all(p.normals@camera > 0)
    assert np.all(p.directions@camera > 0)


def test_naive_intensity_only_diagonal_baseline_route_is_excluded():
    # With s>=0, rho,w>0 and a scalar logI nuisance, u=w*rho*s^2 >=0.
    # Every baseline offdiagonal is -sum_k u_kp*u_kq/(M0+lambda).
    # If that sum is zero every product vanishes, so all scalar updates
    # are diagonal.  They commute and cannot reproduce the mixed W2.
    u = [sp.Matrix([1, 0]), sp.Matrix([sp.Rational(1, 2), sp.Rational(1, 20)])]
    residual = -sum((v*v.T/sp.Rational(3, 2) for v in u), sp.zeros(2))
    assert residual[0, 1] < 0
    assert (u[1]*u[1].T)[0, 1] > 0


def test_spectral_strict_nested_improvement_on_original_counterexample():
    a = np.diag([1., 0.01])
    ws = [np.diag([1., 0]), np.array([[0.5, 0.05], [0.05, 0.005]])]
    result = spectral_certificates(a, ws)
    assert exhaustive_gamma(a, ws)["gamma"] == pytest.approx(14/109)
    assert result["leave_one_spectral"] == pytest.approx(1/200)
    assert result["leave_two_spectral"] == pytest.approx(1/100)
    assert result["improved"] > result["leave_two_spectral"]
    assert result["improved"] <= 14/109


@pytest.mark.parametrize("seed", range(16))
def test_random_small_instances_exhaustive_improved_spectral_bounds(seed):
    rng = np.random.default_rng(20260918+seed)
    n, m = 2+seed % 3, 2+seed % 3
    rotation = np.linalg.qr(rng.normal(size=(n, n)))[0]
    a = (rotation*10**rng.uniform(-3, 1, n))@rotation.T
    factors = [rng.normal(size=(n, 1+(seed+i) % n))*10**rng.uniform(-1, 0.5)
               for i in range(m)]
    ws = [f@f.T for f in factors]
    result = spectral_certificates(a, ws)
    actual = exhaustive_gamma(a, ws)
    assert actual["triples"] == m*3**(m-1)
    assert result["improved"] >= result["leave_one_spectral"]-1e-11
    assert actual["gamma"] >= result["improved"]-1e-8
    lowrank = factor_spectral_certificates(a, factors, top_rank=min(2, n))
    assert lowrank["leave_one_spectral_interval"][0] <= result["leave_one_spectral"]+1e-10
    assert lowrank["leave_one_spectral_interval"][1] >= result["leave_one_spectral"]-1e-10
    assert lowrank["leave_two_spectral_interval"][0] <= result["leave_two_spectral"]+1e-10
    assert lowrank["leave_two_spectral_interval"][1] >= result["leave_two_spectral"]-1e-10
    assert lowrank["improved"] <= actual["gamma"]+1e-8


def test_inverse_square_kantorovich_comparison_not_inverse_square_order():
    rng = np.random.default_rng(1831)
    for _ in range(20):
        z, v = rng.normal(size=(4, 4)), rng.normal(size=(4, 3))
        x = np.eye(4)+z@z.T
        y = x+v@v.T
        ev = np.linalg.eigvalsh(x)
        k = (ev[0]+ev[-1])**2/(4*ev[0]*ev[-1])
        # This is the proved replacement, not the false X^2 <= Y^2.
        assert np.linalg.eigvalsh(k*(y@y)-x@x)[0] >= -1e-9


def test_two_candidate_exact_infimum_and_convergence_rate_symbolically():
    h, a = sp.symbols("h a", positive=True)
    m = sp.diag(1, a)
    wx, wy = sp.diag(h**-4, 0), h**-4*sp.Matrix([[1, h], [h, h*h]])
    def d(mat, update):
        return sp.factor(sp.trace(mat.inv()-(mat+update).inv()))
    r0, r1 = sp.factor(d(m, wx)/d(m+wy, wx)), sp.factor(d(m, wy)/d(m+wx, wy))
    assert sp.limit(r0, h, 0) == a
    assert sp.limit(r1, h, 0) == a
    c0, c1 = sp.limit((r0-a)/h**2, h, 0), sp.limit((r1-a)/h**2, h, 0)
    assert sp.simplify(c0-(2*a*a-a+1)) == 0
    assert sp.factor(c1-c0) == (a-1)**2/a
    # This proves the sharp fixed-chi, two-candidate lower bound; no fixed
    # update-norm or fixed-kappa claim is inferred from this diverging family.


@pytest.mark.parametrize("h", [0.125, 0.0625, 0.03125])
def test_tight_family_new_bound_equals_inverse_condition_number(h):
    a, ws = tight_two_candidate_family(h)
    result = spectral_certificates(a, ws)
    actual = exhaustive_gamma(a, ws)["gamma"]
    assert result["improved"] == pytest.approx(0.25)
    assert actual > result["improved"]
    assert (actual-0.25)/h**2 == pytest.approx(7/8, abs=0.02)
    assert result["leave_one_spectral"]/h**4 == pytest.approx(0.25, abs=0.01)


def test_large_update_does_not_erase_leave_two_baseline():
    a, ws = tight_two_candidate_family(1e-5)
    result = spectral_certificates(a, ws)
    assert result["leave_two_denominator"] == pytest.approx(1.)
    assert result["leave_two_spectral"] == pytest.approx(0.25)
    assert result["improved"] == pytest.approx(0.25)
    assert np.isfinite(result["full_spectral"])
    assert result["leave_one_spectral"] > 0


def test_isotropic_two_candidate_endpoint_has_gamma_one():
    a, ws = tight_two_candidate_family(0.2, a=1.)
    assert spectral_certificates(a, ws)["improved"] == pytest.approx(1.)
    assert exhaustive_gamma(a, ws)["gamma"] == pytest.approx(1.)


def test_top_subspace_downdate_encloses_dense_largest_eigenvalue():
    rng = np.random.default_rng(6102)
    factors = [rng.normal(size=(13, 2)) for _ in range(5)]
    total = np.diag(np.linspace(1, 3, 13))+sum(f@f.T for f in factors)
    for rank in [1, 3, 13]:
        envelope = DowndateEnvelope.build(total, top_rank=rank)
        for removed in [factors[0], np.concatenate(factors[:2], axis=1)]:
            lo, hi = envelope.interval(removed)
            exact = eigh(total-removed@removed.T, eigvals_only=True)[-1]
            assert lo <= exact+1e-10 <= hi+1e-10
            if rank == 13:
                assert hi-lo < 1e-9


def test_padded_zero_denominator_lower_endpoint_remains_safe():
    a = np.eye(2)
    f = 1e7*np.array([[1.], [0.]])
    with np.errstate(divide="raise", invalid="raise"):
        result = factor_spectral_certificates(a, [f, f], top_rank=1)
    assert result["leave_two_denominator_interval"][0] == 0
    assert result["leave_two_spectral_interval"][1] == 1.
    assert 0 <= result["improved"] <= 1


def test_zero_update_conventions_and_fixed_spd_space():
    a = np.diag([1., 2.])
    zeros = [np.zeros_like(a)]
    assert exhaustive_gamma(a, zeros) == {"gamma": 1., "triples": 0, "witness": None}
    assert spectral_certificates(a, zeros)["improved"] == 1.
    assert spectral_certificates(a, [np.ones((2, 2))])["improved"] == 1.
    assert full_trace_marginal(a, np.zeros_like(a)) == 0.
    assert factor_spectral_certificates(a, [], top_rank=1)["improved"] == 1.
    with pytest.raises(ValueError, match="positive definite"):
        spectral_certificates(np.diag([1., 0.]), zeros)
    with pytest.raises(ValueError, match="semidefinite"):
        spectral_certificates(a, [-np.eye(2)])
    with pytest.raises(ValueError, match="symmetric"):
        spectral_certificates(a, [np.array([[1., 2.], [0., 1.]])])
    with pytest.raises(ValueError, match="between"):
        physical_lambertian_family(0)
