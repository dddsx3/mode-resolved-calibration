# status: experimental — NOT part of the published results
"""Isolated matched-ensemble and quadratic-risk bridge research.

No published interfaces are changed.  ``delta_f`` is reused on *supported*
proper-covariance factors; zero covariance is never interpreted as zero
profiling precision.  H is a task OPERATOR, as in ``woodbury_quad_risk``,
not a quadratic-weight matrix.  Scene parameters here are linear albedo
increments with fixed normals (the actual NominalScene Jacobian), not a
new joint normal/albedo model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve, solve_triangular
from scipy.stats import chi2, spearmanr

from calibinfo.information.schur import delta_f


def _real(value, name, ndim=None):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    out = np.asarray(value, dtype=float)
    if (ndim is not None and out.ndim != ndim) or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} has invalid dimensions or nonfinite entries")
    return out


def _symmetric(value, name):
    out = _real(value, name, 2)
    if out.shape[0] != out.shape[1] or out.shape[0] == 0:
        raise ValueError(f"{name} must be a nonempty square matrix")
    scale = max(float(np.max(np.abs(out))), np.finfo(float).tiny)
    if np.max(np.abs(out - out.T)) > 1e-11 * scale:
        raise ValueError(f"{name} must be symmetric")
    return (out + out.T) / 2.0


def proper_covariance_factor(covariance, rtol=1e-12):
    """Return L with covariance = L L.T, including a (q,0) zero factor.

    Relative rank tolerance is with respect to the covariance itself, NOT
    max(scale,1): tiny physical radian covariances must retain their support.
    Eigenvalues below the declared relative tolerance are numerical zero;
    the reconstruction discrepancy is exposed in metadata. Indefinite input
    is rejected, rather than laundered through a pseudoinverse or clipping.
    """
    if not np.isfinite(rtol) or rtol < 0 or rtol >= 1:
        raise ValueError("rtol must be in [0,1)")
    cov = _symmetric(covariance, "proper covariance")
    ev, vec = np.linalg.eigh(cov)
    scale = float(np.max(np.abs(ev)))
    tol = rtol * scale
    if ev[0] < -tol:
        raise ValueError("proper covariance must be positive semidefinite")
    keep = ev > tol
    factor = vec[:, keep] * np.sqrt(ev[keep])[None, :]
    residual = np.linalg.norm(factor @ factor.T - cov)
    return factor, dict(
        rank=int(keep.sum()), dimension=int(cov.shape[0]),
        eigenvalues=ev.tolist(), relative_rank_tolerance=rtol,
        relative_reconstruction_error=float(residual / np.linalg.norm(cov))
        if scale else 0.0,
        semantics="proper Gaussian support; zero variance means deterministic zero")


def _inverse_pd(matrix, name):
    matrix = _symmetric(matrix, name)
    try:
        inverse = cho_solve(cho_factor(matrix, lower=True), np.eye(len(matrix)))
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive definite on the declared space") from exc
    return (inverse + inverse.T) / 2.0


@dataclass
class MatchedGLS:
    """y=A x+B phi+eps, Cov(eps)=sigma^2 Sigma, Cov(phi)=Sigma_phi.

    With A_w=Sigma^-1/2 A and C=B_w L/sigma, M=(I+CC.T)^-1,
    DeltaF=A_w.T M A_w; K=DeltaF^-1 A_w.T M Sigma^-1/2.
    Thus Cov(Ky-x)=sigma^2 DeltaF^-1 and C0=sigma^2 F_inf^-1.
    ``sigma`` is a STANDARD deviation; Sigma_phi is in physical units and
    is not multiplied by sigma^2 a second time.
    """
    A: np.ndarray
    B: np.ndarray
    noise_covariance: np.ndarray
    noise_factor: np.ndarray
    nuisance_factor: np.ndarray
    sigma: float
    finf: np.ndarray
    information: np.ndarray
    gain: np.ndarray
    baseline_gain: np.ndarray
    covariance: np.ndarray
    baseline_covariance: np.ndarray
    metadata: dict

    def noise_draws(self, rng, count):
        z = rng.standard_normal((count, self.A.shape[0]))
        if self.noise_factor.ndim == 1:
            return self.sigma * z * self.noise_factor[None, :]
        return self.sigma * (z @ self.noise_factor.T)

    def nuisance_draws(self, rng, count):
        z = rng.standard_normal((count, self.nuisance_factor.shape[1]))
        return z @ self.nuisance_factor.T

    def covariance_identity_diagnostics(self):
        if self.noise_covariance.ndim == 1:
            actual = self.sigma ** 2 * (
                (self.gain * self.noise_covariance[None, :]) @ self.gain.T)
            actual0 = self.sigma ** 2 * (
                (self.baseline_gain * self.noise_covariance[None, :])
                @ self.baseline_gain.T)
        else:
            actual = self.sigma ** 2 * (
                self.gain @ self.noise_covariance @ self.gain.T)
            actual0 = self.sigma ** 2 * (
                self.baseline_gain @ self.noise_covariance @ self.baseline_gain.T)
        propagated = self.gain @ self.B @ self.nuisance_factor
        actual += propagated @ propagated.T
        return dict(
            relative_covariance_identity_error=float(
                np.linalg.norm(actual - self.covariance) / np.linalg.norm(self.covariance)),
            relative_baseline_identity_error=float(
                np.linalg.norm(actual0 - self.baseline_covariance)
                / np.linalg.norm(self.baseline_covariance)),
            unbiasedness_error=float(np.linalg.norm(self.gain @ self.A - np.eye(self.A.shape[1]))),
            baseline_unbiasedness_error=float(np.linalg.norm(
                self.baseline_gain @ self.A - np.eye(self.A.shape[1]))),
            condition_information=float(np.linalg.cond(self.information)))


def matched_gls(A, B, noise_covariance, nuisance_covariance, sigma):
    """General small dense control, supporting non-diagonal measurement noise."""
    A, B = _real(A, "A", 2), _real(B, "B", 2)
    if A.shape[0] != B.shape[0] or not A.size:
        raise ValueError("A and B must have equal nonzero observation dimensions")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be a positive standard deviation")
    L, factor_meta = proper_covariance_factor(nuisance_covariance)
    if L.shape[0] != B.shape[1]:
        raise ValueError("nuisance covariance dimension differs from B")
    nc = _real(noise_covariance, "noise covariance")
    m = A.shape[0]
    if nc.ndim == 1:
        if nc.shape != (m,) or np.any(nc <= 0):
            raise ValueError("measurement variances must be positive")
        noise_factor = np.sqrt(nc)
        whitener = np.diag(1.0 / noise_factor)
    else:
        nc = _symmetric(nc, "noise covariance")
        if nc.shape != (m, m):
            raise ValueError("measurement covariance dimension differs from A")
        try:
            noise_factor = np.linalg.cholesky(nc)
        except np.linalg.LinAlgError as exc:
            raise ValueError("measurement covariance must be positive definite") from exc
        whitener = solve_triangular(noise_factor, np.eye(m), lower=True)
    Aw, Bw = whitener @ A, whitener @ B
    finf = Aw.T @ Aw
    finv = _inverse_pd(finf, "F_infinity")
    if L.shape[1]:
        information, M, _ = delta_f(Aw, Bw @ L / sigma, np.eye(L.shape[1]))
    else:
        information, M = finf.copy(), np.eye(m)
    dinv = _inverse_pd(information, "DeltaF")
    return MatchedGLS(
        A=A, B=B, noise_covariance=nc, noise_factor=noise_factor,
        nuisance_factor=L, sigma=float(sigma), finf=finf, information=information,
        gain=dinv @ Aw.T @ M @ whitener,
        baseline_gain=finv @ Aw.T @ whitener,
        covariance=sigma ** 2 * dinv, baseline_covariance=sigma ** 2 * finv,
        metadata=dict(route="published delta_f on supported covariance factor",
                      nuisance=factor_meta, sigma=float(sigma),
                      full_rank_precision="Lambda = sigma^2 inverse(Sigma_phi) AFTER whitening by Sigma only"))


def matched_photometric(shading, B_phi, noise_variance, sigma_phi, sigma=1.0):
    """Per-light Schur route, avoiding an (L*P)^2 observation matrix.

    A_k=diag(shading[k]); B_k=B_phi[k]. Normal geometry and active-mask
    linearization are fixed. noise_variance is Sigma, not sigma^2 Sigma.
    """
    s = _real(shading, "shading", 2)
    b = _real(B_phi, "B_phi", 3)
    nv = _real(noise_variance, "noise variance", 2)
    Lc, pixels = s.shape
    if b.shape[:2] != s.shape or nv.shape != s.shape or np.any(nv <= 0):
        raise ValueError("photometric block shapes or variances are invalid")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be positive")
    cov = _real(sigma_phi, "Sigma_phi")
    q = b.shape[2]
    if cov.shape == (q, q):
        cov = np.broadcast_to(cov, (Lc, q, q))
    if cov.shape != (Lc, q, q):
        raise ValueError("Sigma_phi must be a shared block or active-light-ordered blocks")
    factors, metas = [], []
    information = np.zeros((pixels, pixels))
    numerator = []
    numerator0 = []
    for k in range(Lc):
        factor, meta = proper_covariance_factor(cov[k])
        factors.append(factor)
        metas.append(meta)
        weight_root = 1.0 / np.sqrt(nv[k])
        Aw = np.diag(weight_root * s[k])
        if factor.shape[1]:
            C = weight_root[:, None] * (b[k] @ factor) / sigma
            d, M, _ = delta_f(Aw, C, np.eye(factor.shape[1]))
        else:
            d, M = Aw.T @ Aw, np.eye(pixels)
        information += d
        numerator.append(Aw.T @ M * weight_root[None, :])
        numerator0.append(np.diag(s[k] / nv[k]))
    finf = np.diag(np.sum(s * s / nv, axis=0))
    dinv = _inverse_pd(information, "DeltaF")
    finv = _inverse_pd(finf, "F_infinity")
    # scipy.block_diag supports individual zero-column factors.
    factor_all = block_diag(*factors)
    return MatchedGLS(
        A=np.vstack([np.diag(row) for row in s]), B=block_diag(*list(b)),
        noise_covariance=nv.ravel(), noise_factor=np.sqrt(nv.ravel()),
        nuisance_factor=factor_all, sigma=float(sigma), finf=finf,
        information=information, gain=dinv @ np.hstack(numerator),
        baseline_gain=finv @ np.hstack(numerator0), covariance=sigma ** 2 * dinv,
        baseline_covariance=sigma ** 2 * finv,
        metadata=dict(route="per-light published delta_f on supported covariance factors",
                      nuisance_blocks=metas, n_lights=Lc, n_pixels=pixels,
                      sigma=float(sigma), noise_units="Cov(eps)=sigma^2 Sigma",
                      scene_parameter="linear albedo increment; normals held fixed"))


def gauge_projector(rho):
    rho = _real(rho, "gauge direction", 1)
    norm = np.linalg.norm(rho)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("gauge direction must have positive finite norm")
    unit = rho / norm
    return np.eye(len(rho)) - np.outer(unit, unit)


def matched_mode_readout(model, rho, n_modes=5):
    """The SAME score Q=P W on both ensembles and both covariance sides.

    After P, the prediction is diag(Q.T C Q)/diag(Q.T C0 Q), generally
    NOT 1/retention. A projected zero mode is invalid, not denominator-floored.
    """
    n = model.finf.shape[0]
    if n_modes < 1 or n_modes > n:
        raise ValueError("invalid requested mode count")
    ev, vec = np.linalg.eigh(model.finf)
    if ev[0] <= 0:
        raise ValueError("F_infinity must be positive definite")
    root = (vec * np.sqrt(ev)[None, :]) @ vec.T
    inverse_root = (vec / np.sqrt(ev)[None, :]) @ vec.T
    R = inverse_root @ model.information @ inverse_root
    retention, U = np.linalg.eigh((R + R.T) / 2)
    if retention[0] <= 0:
        raise ValueError("positive retention required")
    W = root @ U[:, :n_modes]
    P = gauge_projector(rho)
    Q = P @ W
    pred = np.diag(Q.T @ model.covariance @ Q)
    baseline = np.diag(Q.T @ model.baseline_covariance @ Q)
    reference = np.diag(W.T @ model.baseline_covariance @ W)
    valid = (baseline > (64 * np.finfo(float).eps) ** 2 * reference) & (pred > 0)
    ratio = np.full(n_modes, np.nan)
    np.divide(pred, baseline, out=ratio, where=valid)
    return dict(W=W, projection=P, scores=Q, retention=retention,
                variance=pred, baseline_variance=baseline, valid=valid,
                predicted_degradation=ratio,
                unprojected_prediction=1.0 / retention[:n_modes],
                baseline_projected_fraction=baseline / reference)


def sample_mode_ensembles(model, scores, trials, batch_size, noise_seed,
                          nuisance_seed, baseline_seed):
    """Actually generate observations, then apply GLS (not covariance draws)."""
    Q = _real(scores, "scores", 2)
    if Q.shape[0] != model.A.shape[1] or trials < 2 or batch_size < 1:
        raise ValueError("invalid score or Monte Carlo dimensions")
    noise_rng = np.random.default_rng(noise_seed)
    phi_rng = np.random.default_rng(nuisance_seed)
    base_rng = np.random.default_rng(baseline_seed)
    transfer = model.gain.T @ Q
    base_transfer = model.baseline_gain.T @ Q
    out = np.empty((trials, Q.shape[1]))
    base = np.empty_like(out)
    for start in range(0, trials, batch_size):
        count = min(batch_size, trials - start)
        eps = model.noise_draws(noise_rng, count)
        phi = model.nuisance_draws(phi_rng, count)
        out[start:start + count] = (eps + phi @ model.B.T) @ transfer
        base[start:start + count] = model.noise_draws(base_rng, count) @ base_transfer
    return out, base


def variance_summary(samples, prediction):
    samples = _real(samples, "samples", 2)
    pred = _real(prediction, "prediction", 1)
    if samples.shape[1] != len(pred) or len(samples) < 2 or np.any(pred <= 0):
        raise ValueError("positive predictions and at least two samples required")
    var = samples.var(axis=0, ddof=1)
    df = len(samples) - 1
    return dict(empirical_variance=var, predicted_variance=pred,
                empirical_over_predicted=var / pred, mean=samples.mean(axis=0),
                second_moment=np.mean(samples * samples, axis=0),
                gaussian_pointwise_ratio_reference95=(chi2.ppf([0.025, 0.975], df) / df),
                confidence_scope="chi-square reference only for Gaussian samples; not simultaneous")


def residual_bootstrap_baseline(model, residual_pool, scores, trials,
                                batch_size, seed):
    """Historical raw flattened-pool bootstrap, with an exact conditional law.

    Each resampled observation is iid from the FINITE pool: its variance is
    pool.var(ddof=0), not ddof=1, regardless of its nonzero mean. This function
    changes only the denominator and keeps the zero-nuisance GLS baseline.
    """
    pool = _real(residual_pool, "residual pool").ravel()
    Q = _real(scores, "scores", 2)
    if not pool.size or trials < 2 or batch_size < 1:
        raise ValueError("nonempty pool and valid sample sizes required")
    transfer = model.baseline_gain.T @ Q
    tau2 = float(pool.var(ddof=0))
    if tau2 <= 0:
        raise ValueError("degenerate residual pool; no variance floor is permitted")
    population = tau2 * np.sum(transfer ** 2, axis=0)
    rng = np.random.default_rng(seed)
    out = np.empty((trials, Q.shape[1]))
    for start in range(0, trials, batch_size):
        count = min(batch_size, trials - start)
        draws = rng.choice(pool, size=(count, model.A.shape[0]), replace=True)
        out[start:start + count] = draws @ transfer
    return out, dict(population_variance=population,
                     population_mean=float(pool.mean()) * transfer.sum(axis=0),
                     pool_mean=float(pool.mean()), pool_variance_ddof0=tau2,
                     pool_size=int(pool.size), iid_raw_pool=True,
                     centered_before_sampling=False, degrees_of_freedom_correction=False)


def multiplicative_attribution(linear_variance, predicted_variance,
                                ideal_baseline, bootstrap_population,
                                bootstrap_empirical, nonlinear_variance=None):
    """Per-mode exact factors. Geometric means compose; medians do not."""
    vals = [_real(v, "variance", 1) for v in (
        linear_variance, predicted_variance, ideal_baseline,
        bootstrap_population, bootstrap_empirical)]
    if any(v.shape != vals[0].shape or np.any(v <= 0) for v in vals):
        raise ValueError("same-shaped strictly positive variance vectors required")
    v, pv, b, bp, be = vals
    s1 = v / pv
    pool_factor = b / bp
    sampling_factor = bp / be
    s2 = (v / be) / (pv / b)
    out = dict(s1=s1, residual_pool_population_factor=pool_factor,
               bootstrap_finite_sample_factor=sampling_factor,
               s1_to_s2_factor=b / be, s2=s2,
               s2_closure_relative_error=float(np.max(np.abs(
                   s2 / (s1 * pool_factor * sampling_factor) - 1.0))))
    if nonlinear_variance is not None:
        vn = _real(nonlinear_variance, "nonlinear variance", 1)
        if vn.shape != v.shape or np.any(vn <= 0):
            raise ValueError("invalid nonlinear variances")
        s3 = (vn / be) / (pv / b)
        out.update(s2_to_s3_nonlinearity_factor=vn / v, s3=s3,
                   s3_closure_relative_error=float(np.max(np.abs(
                       s3 / (s2 * vn / v) - 1.0))))
    return out


def sphere_exp_directions(dirs, t1, t2, phi):
    """phi[...,0] is log gain; phi[...,1:3] are Gaussian tangent coordinates.

    Unlike the historical random-axis/scalar-angle law, EACH tangent SD
    agrees exactly with Sigma_phi. Gaussian draws are not radially truncated.
    """
    dirs, t1, t2 = (_real(x, "direction frame", 2) for x in (dirs, t1, t2))
    phi = _real(phi, "physical perturbations", 3)
    if dirs.shape != t1.shape or dirs.shape != t2.shape or phi.shape[1:] != dirs.shape:
        raise ValueError("direction-frame or perturbation dimensions disagree")
    tangent = phi[:, :, 1, None] * t1[None] + phi[:, :, 2, None] * t2[None]
    radius = np.linalg.norm(tangent, axis=-1)
    return (np.cos(radius)[..., None] * dirs[None]
            + np.sinc(radius / np.pi)[..., None] * tangent)


def finite_photometric_scores(model, scene, scores, trials, batch_size,
                              noise_seed, nuisance_seed, tangent_function):
    """S3: only replace B phi by the physical observation-side forward delta.

    GLS, Gaussian measurement noise, phi samples, projection and score modes
    are frozen. This is explicitly NOT the historical fixed-real-image
    estimator-side plug-in fit. Returns paired linear scores as a reuse check.
    """
    Q = _real(scores, "scores", 2)
    transfer = model.gain.T @ Q
    noise_rng, phi_rng = np.random.default_rng(noise_seed), np.random.default_rng(nuisance_seed)
    t1, t2 = tangent_function(scene.dirs)
    n_lights = len(scene.dirs)
    n_pixels = len(scene.rho)
    nominal = scene.s_hat * scene.rho[None, :]
    out = np.empty((trials, Q.shape[1]))
    linear = np.empty_like(out)
    logs_sq = radius_sq = 0.0
    max_log = max_radius = 0.0
    flips = total_rows = 0
    for start in range(0, trials, batch_size):
        count = min(batch_size, trials - start)
        eps = model.noise_draws(noise_rng, count)
        phi_flat = model.nuisance_draws(phi_rng, count)
        phi = phi_flat.reshape(count, n_lights, 3)
        d2 = sphere_exp_directions(scene.dirs, t1, t2, phi)
        dot = np.einsum("blc,pc->blp", d2, scene.n, optimize=True)
        forward = np.maximum(dot, 0.0) * scene.rho[None, None, :] * np.exp(phi[:, :, 0, None])
        perturbation = (forward - nominal[None]).reshape(count, n_lights * n_pixels)
        out[start:start + count] = (eps + perturbation) @ transfer
        linear[start:start + count] = (eps + phi_flat @ model.B.T) @ transfer
        logs = phi[:, :, 0]
        radii = np.linalg.norm(phi[:, :, 1:], axis=2)
        logs_sq += float(np.sum(logs * logs))
        radius_sq += float(np.sum(radii * radii))
        max_log = max(max_log, float(np.max(np.abs(logs))))
        max_radius = max(max_radius, float(np.max(radii)))
        flips += int(np.count_nonzero((dot > 0) != (scene.s_hat[None] > 0)))
        total_rows += int(dot.size)
    remainder = out - linear
    return out, linear, dict(
        log_intensity_rms=float(np.sqrt(logs_sq / (trials * n_lights))),
        log_intensity_abs_max=max_log,
        angular_radius_rms_deg=float(np.degrees(np.sqrt(radius_sq / (trials * n_lights)))),
        angular_radius_max_deg=float(np.degrees(max_radius)),
        shadow_state_change_fraction=float(flips / total_rows),
        observed_remainder_score_rms=np.sqrt(np.mean(remainder * remainder, axis=0)),
        paired_sample_second_moment_radius=(2 * np.sqrt(np.mean(linear * linear, axis=0))
                                            * np.sqrt(np.mean(remainder * remainder, axis=0))
                                            + np.mean(remainder * remainder, axis=0)),
        radius_status="samplewise Cauchy bound only; not a population or tail certificate",
        estimator_isomorphic_to_historical=False)


def quadratic_risk(covariance, H, bias=None):
    """EXACT E||H e||^2 = tr(H Cov(e) H.T) + ||H E e||^2."""
    cov = _symmetric(covariance, "error covariance")
    proper_covariance_factor(cov)
    H = _real(H, "task operator", 2)
    if H.shape[1] != len(cov):
        raise ValueError("H acts on a different parameter space")
    b = np.zeros(len(cov)) if bias is None else _real(bias, "bias", 1)
    if b.shape != (len(cov),):
        raise ValueError("bias dimension differs from H")
    return float(np.trace(H @ cov @ H.T) + np.sum((H @ b) ** 2))


def covariance_risk_radius(covariance_true, covariance_predicted, H):
    """Measurable nuclear-norm bound on the task-weighted trace discrepancy."""
    ct = _symmetric(covariance_true, "true covariance")
    cp = _symmetric(covariance_predicted, "predicted covariance")
    H = _real(H, "task operator", 2)
    if ct.shape != cp.shape or H.shape[1] != len(ct):
        raise ValueError("covariance/task dimensions disagree")
    proper_covariance_factor(ct)
    proper_covariance_factor(cp)
    difference = H @ (ct - cp) @ H.T
    return float(np.sum(np.abs(np.linalg.eigvalsh((difference + difference.T) / 2))))


def endpoint_operator_radius(covariance, bias, H, G):
    """Bound an explicit linear endpoint mismatch G vs information task H."""
    cov = _symmetric(covariance, "covariance")
    proper_covariance_factor(cov)
    b = _real(bias, "bias", 1)
    H, G = _real(H, "H", 2), _real(G, "G", 2)
    if H.shape[1] != len(cov) or G.shape[1] != len(cov) or b.shape != (len(cov),):
        raise ValueError("endpoint operators must share a parameter space")
    return float(np.linalg.norm(G.T @ G - H.T @ H, ord=2)
                 * (np.trace(cov) + b @ b))


def risk_error_radius(predicted_risk, *, covariance_error_bound, bias_H_norm,
                      remainder_H_rms_bound, endpoint_expectation_bound,
                      sampling_radius):
    """Conditional measurable radius; every uncertainty bound is explicit.

    For e=e_linear+r, E||H r||^2 <= d^2, and linear risk <= p+c+b^2,
    |E endpoint-p| <= c+b^2+2 sqrt(p+c+b^2)d+d^2+endpoint_bound.
    Add a VALID simultaneous empirical-mean radius for empirical ordering.
    Estimated RMS/SEM alone is not such a certificate; caller must establish
    each bound on a specified event/ensemble (or return 'not certified').
    """
    args = [predicted_risk, covariance_error_bound, bias_H_norm,
            remainder_H_rms_bound, endpoint_expectation_bound, sampling_radius]
    if not all(np.isfinite(v) and v >= 0 for v in args):
        raise ValueError("risk and every radius component must be finite and nonnegative")
    linear_upper = predicted_risk + covariance_error_bound + bias_H_norm ** 2
    nonlinear = 2 * np.sqrt(linear_upper) * remainder_H_rms_bound + remainder_H_rms_bound ** 2
    expectation = covariance_error_bound + bias_H_norm ** 2 + nonlinear + endpoint_expectation_bound
    return dict(covariance=covariance_error_bound, bias=bias_H_norm ** 2,
                nonlinear=float(nonlinear), endpoint=endpoint_expectation_bound,
                sampling=sampling_radius, expectation_radius=float(expectation),
                total_radius=float(expectation + sampling_radius),
                linear_risk_upper=float(linear_upper))


def gaussian_quadratic_mean_radius(covariance, H, trials, failure_probability,
                                   simultaneous_count=1):
    """Two-sided Gaussian quadratic concentration, union-bound simultaneous.

    For ZERO-MEAN iid Gaussian errors, T=H C H.T and t=log(2K/alpha):
      |sample mean ||H e||^2 - tr(T)|
        <= 2 ||T||_F sqrt(t/N) + 2 ||T||_op t/N
    with probability >=1-alpha for K plan risks. Cross-plan independence is
    NOT required. Non-Gaussian bootstrap/nonlinear/real endpoints may not use
    this formula without a separate tail argument.
    """
    cov = _symmetric(covariance, "covariance")
    proper_covariance_factor(cov)
    H = _real(H, "H", 2)
    if H.shape[1] != len(cov):
        raise ValueError("H covariance dimensions differ")
    if trials < 1 or simultaneous_count < 1 or not 0 < failure_probability < 1:
        raise ValueError("invalid sample size or confidence level")
    T = H @ cov @ H.T
    t = float(np.log(2 * simultaneous_count / failure_probability))
    return float(2 * np.linalg.norm(T, ord="fro") * np.sqrt(t / trials)
                 + 2 * np.linalg.norm(T, ord=2) * t / trials)


def pairwise_margin_certificate(predictions, radii, realized=None, target=0.7):
    """Strict interval separation implies risk/empirical ordering.

    Full separation implies Spearman=1 (distinct ranks automatically).
    For partial separation, possible rank intervals give displacement d_i,
    hence rho >= max(-1, 1-6 sum d_i^2/[K(K^2-1)]), CONDITIONAL on
    distinct realized ranks and simultaneous error bounds. Predicted ties
    invalidate the no-tie formula; no arbitrary tie-breaking is advertised.
    """
    pred, rad = _real(predictions, "predictions", 1), _real(radii, "radii", 1)
    k = len(pred)
    if k < 2 or rad.shape != pred.shape or np.any(rad < 0) or not -1 <= target <= 1:
        raise ValueError("invalid predictions, radii or target")
    ties = len(np.unique(pred)) != k
    lower, upper = pred - rad, pred + rad
    order = np.argsort(pred, kind="stable")
    ranks = np.empty(k, dtype=int)
    ranks[order] = np.arange(1, k + 1)
    earliest = np.array([1 + np.count_nonzero(upper < lower[i]) for i in range(k)])
    latest = np.array([k - np.count_nonzero(lower > upper[i]) for i in range(k)])
    displacements = np.maximum(ranks - earliest, latest - ranks)
    pairs = []
    for a in range(k):
        for b in range(a + 1, k):
            i, j = int(order[a]), int(order[b])
            gap = float(pred[j] - pred[i])
            radius_sum = float(rad[i] + rad[j])
            pairs.append(dict(lower_plan=i, upper_plan=j, predicted_gap=gap,
                              radius_sum=radius_sum, slack=gap - radius_sum,
                              certified=bool(gap > radius_sum)))
    all_pairs = not ties and all(row["certified"] for row in pairs)
    bound = (None if ties else max(-1.0, float(
        1 - 6 * np.sum(displacements.astype(float) ** 2) / (k * (k * k - 1)))))
    output = dict(n_plans=k, n_pairs=len(pairs), pairs=pairs,
                  n_certified_pairs=sum(int(row["certified"]) for row in pairs),
                  all_pairs_certified=all_pairs, predicted_ties=ties,
                  possible_rank_min=earliest.tolist(), possible_rank_max=latest.tolist(),
                  rank_displacement_bound=displacements.tolist(),
                  spearman_lower_bound=bound,
                  target=target, target_certified=bool(bound is not None and bound >= target),
                  assumptions="simultaneous error event; partial Spearman formula additionally requires no realized ties")
    if realized is not None:
        obs = _real(realized, "realized risks", 1)
        if obs.shape != pred.shape:
            raise ValueError("realized risk dimensions disagree")
        observed_ties = len(np.unique(obs)) != k
        output.update(simultaneous_error_event_observed=bool(np.all(np.abs(obs - pred) <= rad)),
                      realized_ties=observed_ties,
                      observed_spearman=_spearman(pred, obs),
                      certified_pairs_observed_agree=all(
                          obs[row["upper_plan"]] > obs[row["lower_plan"]]
                          for row in pairs if row["certified"]))
        if observed_ties and not all_pairs:
            output["spearman_lower_bound"] = None
            output["target_certified"] = False
    return output


def _spearman(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
        return None
    return float(spearmanr(a, b).statistic)


def family_rank_diagnostics(anchor_pred, het_pred, anchor_real, het_real):
    """Reuse family_e_diag's S_pred/S_real and crossed-fit definitions.

    One fixed (object, plan-cell, endpoint) at a time. Aggregate per-cell
    correlations by median, skipping only constant columns with a count.
    These are descriptive ranks, NOT an identification theorem excluding
    covariance misspecification as a causal mechanism.
    """
    arrays = [_real(v, "family values", 1) for v in
              (anchor_pred, het_pred, anchor_real, het_real)]
    if len(arrays[0]) < 2 or any(v.shape != arrays[0].shape for v in arrays):
        raise ValueError("family diagnostics require the same plan set")
    ap, hp, ar, hr = arrays
    return dict(s_pred=_spearman(ap, hp), s_real=_spearman(ar, hr),
                fit_anc=_spearman(ap, ar), fit_het=_spearman(hp, hr),
                cross_ah=_spearman(ap, hr), cross_ha=_spearman(hp, ar))
