# status: experimental — NOT part of the published results
"""M14 (reserved): non-circular certificates for the gauge two-term risk law.

This research module is deliberately outside ``src/calibinfo``.  For a unit
``a`` with ``A @ a == B @ c``, D=A' A>0, U=A' B, M=B' B, and Sigma0>0,

    J_a(t) = 1/(t*q) + r + delta_D + e' (t*I + S_perp)^(-1) e,
    q = c' Sigma0^(-1) c, r = 1/(a' D a), delta_D >= 0.

``prepare_geometry`` specializes the general theorem to diagonal D (the
published fixed-normal albedo model).  ``analyze_prior`` uses a nuisance-size
matrix, not the p-by-p Fisher inverse.  ``compress_spectrum`` discards exact
eigenvalues within fixed dyadic bins.  The certificate functions ONLY accept
these bins and their nonnegative spectral masses: neither observed J nor an
observed approximation error is an input.

The theorem is exact arithmetic.  This implementation uses floating point,
with explicit gauge/PSD guards and reported Gram-eigenvalue roundoff; it is
not an interval-arithmetic verification.  No prior or Fisher ridge is added.
Full proofs, physical conditions, and scope are in
``docs/theory/section_two_term_remainder.tex`` and ``two_term_remainder.md``.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isfinite, sqrt

import numpy as np
from scipy.linalg import eigh, null_space, solve_triangular

GAUGE_RTOL = 1e-10
PSD_RTOL = 1e-10


def _array(value, ndim, name):
    out = np.asarray(value, dtype=float)
    if out.ndim != ndim or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    return out


def _symmetric(value, name):
    out = _array(value, 2, name)
    if out.shape[0] != out.shape[1]:
        raise ValueError(f"{name} must be square")
    scale = max(float(np.linalg.norm(out, ord='fro')), np.finfo(float).tiny)
    if np.linalg.norm(out - out.T, ord='fro') > GAUGE_RTOL * scale:
        raise ValueError(f"{name} must be symmetric")
    return (out + out.T) * 0.5


def _positive(value, name):
    value = float(value)
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return value


@dataclass(frozen=True)
class GaugeGeometry:
    """Sigma-independent ingredients; all nuisance columns use the same order."""

    projection_gram: np.ndarray  # T = B' (I - P_A) B
    g: np.ndarray              # U' D^-1 a
    c: np.ndarray
    r: float
    delta_diagonal: float
    diagnostics: dict


def prepare_geometry(finf_diag, cross, nuisance_gram, a, c):
    """Build T and g from diagonal D, U (p,m), M (m,m), unit a and exact c.

    Gram-level gauge equations Uc=Da and Mc=U'a are both checked.  They are
    necessary consequences of Aa=Bc; for a realizable PSD joint Gram matrix
    they also imply that its squared observation-space gauge residual is zero.
    The runner separately checks Aa-Bc on the actual observation blocks.
    """
    d = _array(finf_diag, 1, 'finf_diag')
    u = _array(cross, 2, 'cross')
    m = _symmetric(nuisance_gram, 'nuisance_gram')
    a = _array(a, 1, 'a')
    c = _array(c, 1, 'c')
    if len(d) == 0 or len(c) == 0 or u.shape != (len(d), len(c)):
        raise ValueError('inconsistent or empty observation/nuisance dimensions')
    if a.shape != d.shape or m.shape != (len(c), len(c)):
        raise ValueError('inconsistent a or nuisance_gram dimensions')
    if np.any(d <= 0):
        raise ValueError('D must be strictly positive; no pixel clipping is applied')
    if abs(float(a @ a) - 1.0) > GAUGE_RTOL:
        raise ValueError('a must have unit Euclidean norm')
    scale1 = max(float(np.linalg.norm(d * a)), np.finfo(float).tiny)
    scale2 = max(float(np.linalg.norm(u.T @ a)), np.finfo(float).tiny)
    gauge1 = float(np.linalg.norm(u @ c - d * a) / scale1)
    gauge2 = float(np.linalg.norm(m @ c - u.T @ a) / scale2)
    if max(gauge1, gauge2) > GAUGE_RTOL:
        raise ValueError('exact gauge Aa=Bc fails the two Gram-level checks')
    t = m - u.T @ (u / d[:, None])
    t = (t + t.T) * 0.5
    min_t = float(eigh(t, eigvals_only=True, subset_by_index=[0, 0])[0])
    gram_scale = max(float(np.linalg.norm(m, ord='fro')), np.finfo(float).tiny)
    if min_t < -PSD_RTOL * gram_scale:
        raise np.linalg.LinAlgError('joint Gram matrix is not PSD: T has negative spectrum')
    g = u.T @ (a / d)
    if abs(float(g @ c) - 1.0) > GAUGE_RTOL:
        raise ValueError('gauge normalization g^T c = 1 failed')
    mu = float((a * a) @ d)
    r = 1.0 / mu
    # Positive sum rather than cancellation in a' D^-1 a - 1/(a' D a).
    delta = float(np.sum(a * a * ((d - mu) / mu) ** 2 / d))
    spread = float(np.max(np.abs(d[a != 0] / mu - 1.0)))
    spread_bound = r * spread**2 / (1.0 - spread) if spread < 1 else None
    return GaugeGeometry(t, g, c.copy(), r, delta, dict(
        gram_gauge_Uc_relative=gauge1,
        gram_gauge_Mc_relative=gauge2,
        g_dot_c=float(g @ c),
        projection_gram_min_eigenvalue=min_t,
        projection_gram_scale=gram_scale,
        ideal_task_risk=float((a * a) @ (1.0 / d)),
        diagonal_information_mean=mu,
        diagonal_relative_spread=spread,
        diagonal_spread_sufficient_bound=spread_bound,
    ))


@dataclass(frozen=True)
class GaugeSpectrum:
    """Prior-specific exact decomposition data (up to reported roundoff).

    Exact spectral denominators are used ONLY by the validation helper below.
    The production certificates take a BinnedRemainder instead of this object.
    """

    q: float
    r: float
    delta_diagonal: float
    eigenvalues: np.ndarray
    masses: np.ndarray
    diagnostics: dict

    def remainder_for_validation(self, t):
        """Evaluate the identity for independent checks, NOT an error bound."""
        t = _positive(t, 't')
        return self.delta_diagonal + float(np.sum(self.masses / (t + self.eigenvalues)))


def analyze_prior(geometry, sigma0):
    """Whiten T by Sigma0=C C', remove ONLY the specified gauge, and diagonalize.

    h=C^-1 c/sqrt(q), v=C' g, e=v-h/sqrt(q), h'e=0.  Q spans h-perp;
    S_perp=Q' C' T C Q.  Additional zero modes are retained, not assumed away.
    Negative Gram eigenvalues within the propagated parent-Gram tolerance are
    recorded and treated as roundoff zeros; larger negatives raise.  If
    T >= -PSD_RTOL*||M||_F I (prepare_geometry's contract), then
    Q'C'TCQ >= -PSD_RTOL*||M||_F*||C||_2^2 I.  Scaling by ||C'TC|| instead
    would spuriously reject exact T=0 models after Gram cancellation.
    """
    sigma = _symmetric(sigma0, 'sigma0')
    m = len(geometry.c)
    if sigma.shape != (m, m):
        raise ValueError('sigma0 must match the nuisance dimension')
    chol = np.linalg.cholesky(sigma)
    d = solve_triangular(chol, geometry.c, lower=True)
    q = _positive(float(d @ d), 'q')
    h = d / sqrt(q)
    v = chol.T @ geometry.g
    residual = v - d / q
    orthogonal_residual = float(h @ residual)
    Q = null_space(h[None, :])
    s_full = chol.T @ geometry.projection_gram @ chol
    s_full = (s_full + s_full.T) * 0.5
    s = Q.T @ s_full @ Q
    s = (s + s.T) * 0.5
    e = Q.T @ residual
    scale = max(float(np.linalg.norm(s_full, ord='fro')), np.finfo(float).tiny)
    covariance_max = float(eigh(sigma, eigvals_only=True, subset_by_index=[m - 1, m - 1])[0])
    propagated_scale = geometry.diagnostics['projection_gram_scale'] * covariance_max
    psd_tolerance = PSD_RTOL * propagated_scale
    if not isfinite(psd_tolerance):
        raise ValueError('propagated parent-Gram tolerance overflows')
    if m > 1:
        raw, vectors = eigh(s)
        if float(raw[0]) < -psd_tolerance:
            raise np.linalg.LinAlgError('prior-whitened projection Gram is not PSD')
        values = np.maximum(raw, 0.0)
        masses = (vectors.T @ e) ** 2
        negative_count = int(np.count_nonzero(raw < 0))
        negative_max = float(max(0.0, -raw[0]))
    else:
        values, masses = np.empty(0), np.empty(0)
        negative_count, negative_max = 0, 0.0
    return GaugeSpectrum(q, geometry.r, geometry.delta_diagonal, values, masses, dict(
        prior_residual_squared=float(e @ e),
        prior_residual_orthogonality=orthogonal_residual,
        removed_gauge_null_relative=float(np.linalg.norm(s_full @ h) / scale),
        spectrum_min=float(values.min()) if len(values) else None,
        spectrum_max=float(values.max()) if len(values) else None,
        spectrum_dimension=int(len(values)),
        spectrum_negative_roundoff_count=negative_count,
        spectrum_negative_roundoff_max=negative_max,
        spectrum_scale=scale,
        spectrum_parent_gram_propagated_scale=propagated_scale,
        spectrum_psd_guard_tolerance=psd_tolerance,
        removed_gauge_null_parent_relative=float(np.linalg.norm(s_full @ h) / propagated_scale),
        covariance_condition=float(np.linalg.cond(sigma)),
    ))


@dataclass(frozen=True)
class SpectralBin:
    lower: float
    upper: float
    mass: float
    multiplicity: int

    def __post_init__(self):
        if (not all(isfinite(x) for x in (self.lower, self.upper, self.mass))
                or self.lower < 0 or self.upper < self.lower or self.mass < 0
                or self.multiplicity < 1):
            raise ValueError('invalid nonnegative spectral bin')


@dataclass(frozen=True)
class BinnedRemainder:
    q: float
    r: float
    delta_diagonal: float
    bins: tuple[SpectralBin, ...]

    def __post_init__(self):
        _positive(self.q, 'q')
        _positive(self.r, 'r')
        if not isfinite(self.delta_diagonal) or self.delta_diagonal < 0:
            raise ValueError('delta_diagonal must be finite and nonnegative')

    @property
    def residual_mass(self):
        return fsum(b.mass for b in self.bins)

    def prediction(self, t):
        return 1.0 / (_positive(t, 't') * self.q) + self.r


def compress_spectrum(spectrum):
    """Use fixed bins [2^j-1, 2^(j+1)-1], j=floor(log2(1+lambda)).

    No bin-width fitting or error-dependent refinement is performed.  For
    t>=1, every bin denominator ratio is <=2, so U_t <= 2*R_t in exact
    arithmetic (including the nonnegative delta_D term).
    """
    vals = _array(spectrum.eigenvalues, 1, 'eigenvalues')
    masses = _array(spectrum.masses, 1, 'masses')
    if vals.shape != masses.shape or np.any(vals < 0) or np.any(masses < 0):
        raise ValueError('spectrum and masses must be nonnegative and have equal length')
    indices = np.floor(np.log2(1.0 + vals)).astype(int)
    bins = []
    for j in np.unique(indices):
        lo, hi = float(2.0**int(j) - 1.0), float(2.0**(int(j) + 1) - 1.0)
        take = indices == j
        bins.append(SpectralBin(lo, hi, fsum(masses[take].tolist()), int(take.sum())))
    return BinnedRemainder(spectrum.q, spectrum.r, spectrum.delta_diagonal, tuple(bins))


def endpoint_certificate(summary, t):
    """Bound R_t=J_t-Jpred_t from bins, never from an observed risk/error."""
    t = _positive(t, 't')
    lower = summary.delta_diagonal + fsum(b.mass / (t + b.upper) for b in summary.bins)
    upper = summary.delta_diagonal + fsum(b.mass / (t + b.lower) for b in summary.bins)
    prediction = summary.prediction(t)
    return dict(
        t=t, prediction=prediction,
        remainder_lower=lower, remainder_upper=upper,
        risk_lower=prediction + lower, risk_upper=prediction + upper,
        relative_to_prediction_upper=upper / prediction,
        relative_to_actual_upper=upper / (prediction + upper),
        coarse_residual_upper=summary.delta_diagonal + summary.residual_mass / t,
        # Public summaries may carry general, non-dyadic intervals.  The
        # factor-two claim requires their actual denominator ratios, not
        # merely t>=1 (which suffices only for compress_spectrum's bins).
        dyadic_factor_two_guarantee=all((t + b.upper) / (t + b.lower) <= 2.0
                                       for b in summary.bins if b.mass > 0),
    )


def _value_kernel(x, kappa, value_pred):
    # Algebraically 1/(kappa+x)-beta/(1+x), but stable when beta is near 1.
    return (value_pred - (kappa - 1.0) / (kappa + x)) / (1.0 + x)


def value_certificate(summary, kappa):
    """Joint-endpoint enclosure of V-Vpred using the SAME spectral masses.

    N=(1-beta)*delta_D + integral [1/(kappa+x)-beta/(1+x)] dmu(x),
    V-Vpred=-N/(Jpred(1)+R_1).  The kernel has a single interior maximum;
    endpoints plus that maximum give its exact range on each interval.
    This avoids falsely treating correlated endpoint remainders as independent.
    """
    kappa = _positive(kappa, 'kappa')
    if kappa <= 1:
        raise ValueError('kappa must be greater than one')
    qr = summary.q * summary.r
    if not isfinite(qr):
        raise ValueError('q*r overflows')
    value_pred = (1.0 - 1.0 / kappa) / (1.0 + qr)
    beta = 1.0 - value_pred
    root = (kappa * sqrt(beta) - 1.0) * (1.0 + sqrt(beta)) / value_pred
    nlo = nhi = value_pred * summary.delta_diagonal
    for b in summary.bins:
        ends = [_value_kernel(b.lower, kappa, value_pred),
                _value_kernel(b.upper, kappa, value_pred)]
        blo, bhi = min(ends), max(ends)
        if b.lower <= root <= b.upper:
            bhi = max(bhi, _value_kernel(root, kappa, value_pred))
        nlo += b.mass * blo
        nhi += b.mass * bhi
    one = endpoint_certificate(summary, 1.0)
    dlo, dhi = one['risk_lower'], one['risk_upper']
    quotients = [n / d for n in (nlo, nhi) for d in (dlo, dhi)]
    correction_lo, correction_hi = -max(quotients), -min(quotients)
    return dict(
        prediction=value_pred, beta=beta,
        correction_lower=correction_lo, correction_upper=correction_hi,
        absolute_error_upper=max(abs(correction_lo), abs(correction_hi)),
        value_lower=value_pred + correction_lo,
        value_upper=value_pred + correction_hi,
        numerator_lower=nlo, numerator_upper=nhi,
        denominator_lower=dlo, denominator_upper=dhi,
        kernel_zero=kappa * qr, kernel_maximizer=root,
    )
