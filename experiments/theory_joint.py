# status: experimental — NOT part of the published results
"""Local joint log-albedo/normal information with explicit calibration nuisance.

Scene order is [log rho_p, normal tangent 1, normal tangent 2] per pixel.
Calibration order is [log intensity_k, light tangent 1, light tangent 2].
Jacobians and visibility are fixed when evaluating an allocation certificate.
The historical fixed-normal ``joint_map`` estimator is not used here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve
from scipy.optimize import least_squares

from calibinfo.allocation.blocks import LightBlocks
from calibinfo.allocation.convex import CertificateProblem
from calibinfo.estimators.diagnostics import tangent_basis
from calibinfo.information.gauge import gauge_response
from calibinfo.information.lowrank import woodbury_quad_risk
from calibinfo.information.schur import delta_f
from calibinfo.information.whitening import whiten_system
from calibinfo.metrics.spectral_criteria import identifiable_subspace


def _unit_rows(values, name):
    values = np.asarray(values, dtype=float)
    if (values.ndim != 2 or values.shape[1] != 3
            or not np.all(np.isfinite(values))):
        raise ValueError(f"{name} must be finite (n,3) unit vectors")
    if not np.allclose(np.linalg.norm(values, axis=1), 1.0,
                       rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must contain unit vectors")
    return values.copy()


def _spd_solve(matrix, rhs):
    matrix = np.asarray(matrix, dtype=float)
    scale = float(np.max(np.abs(matrix)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("SPD system must have finite nonzero scale")
    return cho_solve(cho_factor(matrix / scale, lower=True),
                     np.asarray(rhs, float) / scale)


def _spd_inverse(matrix):
    matrix = np.asarray(matrix, dtype=float)
    return _spd_solve(matrix, np.eye(matrix.shape[0]))


def _chart(base, tangents, coordinates):
    raw = base + np.einsum("pij,pj->pi", tangents, coordinates)
    length = np.sqrt(1.0 + np.sum(coordinates**2, axis=1))
    values = raw / length[:, None]
    derivative = (tangents / length[:, None, None]
                  - raw[:, :, None] * coordinates[:, None, :]
                  / length[:, None, None]**3)
    return values, derivative


@dataclass
class JointLinearization:
    rho: np.ndarray
    normals: np.ndarray
    directions: np.ndarray
    intensity: np.ndarray
    weights: np.ndarray
    normal_tangents: np.ndarray
    light_tangents: np.ndarray
    A: np.ndarray
    B: np.ndarray
    D: np.ndarray
    u: np.ndarray
    M0: np.ndarray
    lam0: np.ndarray
    visibility: np.ndarray
    nominal: np.ndarray

    @property
    def P(self):
        return len(self.rho)

    @property
    def L(self):
        return len(self.directions)

    def precision(self, t):
        t = np.asarray(t, dtype=float)
        if t.shape != (self.L,) or np.any(t <= 0) or not np.all(np.isfinite(t)):
            raise ValueError("t must be a finite positive per-light vector")
        return block_diag(*(t[k] * self.lam0[k] for k in range(self.L)))

    def fisher(self, t, basis=None):
        t = np.asarray(t, float)
        self.precision(t)
        basis = np.eye(3*self.P) if basis is None else np.asarray(basis, float)
        F = np.zeros((basis.shape[1], basis.shape[1]))
        for k in range(self.L):
            Ak = self.A[k*self.P:(k+1)*self.P] @ basis
            Bk = self.B[k*self.P:(k+1)*self.P, 3*k:3*k+3]
            lam = t[k] * self.lam0[k]
            fitted = _spd_solve(self.M0[k] + lam, Bk.T @ Ak)
            residual = Ak - Bk @ fitted
            prior_residual = np.linalg.cholesky(lam).T @ fitted
            F += residual.T @ residual + prior_residual.T @ prior_residual
        return (F + F.T) / 2

    def fixed_basis(self, t=None):
        reference = self.D if t is None else self.fisher(t)
        rank, basis, values = identifiable_subspace(reference, 1e-10)
        try:
            np.linalg.cholesky(reference / np.max(np.abs(reference)))
            basis = np.eye(reference.shape[0])
            rank = reference.shape[0]
            values = np.linalg.eigvalsh(reference)
        except np.linalg.LinAlgError:
            pass
        return basis, {"rank": rank, "positive_eigenvalues": values.tolist(),
                       "reference": "F_infinity" if t is None else "F_baseline"}

    def whitened_blocks(self, basis=None):
        if basis is None:
            basis, _ = self.fixed_basis()
        Dq = basis.T @ self.D @ basis
        values, vectors = np.linalg.eigh(Dq)
        if values.size == 0 or values[0] <= 0:
            raise ValueError("a nonempty identifiable fixed basis is required")
        transform = basis @ ((vectors / np.sqrt(values)) @ vectors.T)
        uw = np.einsum("pn,lpk->lnk", transform, self.u)
        scale = np.maximum(np.max(np.abs(self.M0), axis=(1, 2)),
                           np.max(np.abs(self.lam0), axis=(1, 2)))
        blocks = LightBlocks(uw / np.sqrt(scale)[:, None, None],
                             self.M0 / scale[:, None, None],
                             self.lam0 / scale[:, None, None],
                             np.ones(transform.shape[1]),
                             np.any(np.abs(uw) > 0, axis=(1, 2)))
        return blocks, transform

    def task_risk_lowrank(self, t, H, basis=None):
        blocks, transform = self.whitened_blocks(basis)
        for k in range(self.L):
            ev = np.linalg.eigvalsh(blocks.M0[k] + t[k]*blocks.lam0[k])
            if ev[0] <= max(float(ev[-1]), 1.0)*1e-12:
                raise ValueError("legacy low-rank kernel would truncate positive precision; use the dense SPD route")
        return float(woodbury_quad_risk(
            blocks.finf, blocks.u, blocks.M0, blocks.lam0, blocks.active,
            np.asarray(t, float), np.atleast_2d(H) @ transform))

    def risk_and_gradient(self, t, H, basis=None):
        if basis is None:
            basis, _ = self.fixed_basis()
        Hq = np.atleast_2d(H) @ basis
        Fq = self.fisher(t, basis)
        Z = cho_solve(cho_factor(Fq, lower=True), Hq.T)
        risk = float(np.trace(Hq @ Z))
        gradient = np.empty(self.L)
        for k in range(self.L):
            K = _spd_inverse(self.M0[k] + t[k] * self.lam0[k])
            UZ = self.u[k].T @ basis @ Z
            gradient[k] = -float(np.trace(
                K @ self.lam0[k] @ K @ UZ @ UZ.T))
        return risk, gradient

    def certificate(self, t, H, budget, kappa, basis=None):
        t = np.asarray(t, float)
        if (kappa < 1 or budget < 0 or np.any(t < 1) or np.any(t > kappa)
                or np.sum(t - 1) > budget + 1e-10):
            raise ValueError("allocation is outside its boxed budget polytope")
        risk, gradient = self.risk_and_gradient(t, H, basis)
        blocks, _ = self.whitened_blocks(basis)
        vertex = CertificateProblem(blocks, kappa=kappa).lmo(gradient, budget)
        gap = float(gradient @ (t - vertex))
        return {"risk": risk, "gap": gap, "lower_bound": risk - gap,
                "gradient": gradient, "lmo": vertex}

    def linear_profile(self, observations, t, sigma=1.0, basis=None):
        """Solve the joint augmented least-squares system, then retain scene x."""
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if basis is None:
            basis, _ = self.fixed_basis()
        A = self.A @ basis
        Lambda = self.precision(t)
        root = np.linalg.cholesky(Lambda).T
        augmented = np.block([[A, self.B],
                              [np.zeros((3*self.L, A.shape[1])), root]])
        obs = np.atleast_2d(np.asarray(observations, float))
        if obs.shape[1] != len(self.A):
            raise ValueError("observations must be rows of whitened residuals")
        rhs = np.concatenate([obs, np.zeros((len(obs), 3*self.L))], axis=1)
        scales = np.sqrt(np.sum(augmented**2, axis=0))
        if np.any(scales <= 0) or not np.all(np.isfinite(scales)):
            raise ValueError("profile system has unidentifiable or nonfinite columns")
        normalized = augmented / scales[None, :]
        solution, _, rank, _ = np.linalg.lstsq(normalized, rhs.T, rcond=None)
        if rank != normalized.shape[1]:
            raise ValueError("profile system rank is not numerically resolved")
        estimates = (solution / scales[:, None]).T
        return estimates[:, :A.shape[1]] @ basis.T

    def physical(self, x, c, jacobian=False):
        x = np.asarray(x, float).reshape(self.P, 3)
        c = np.asarray(c, float).reshape(self.L, 3)
        normals, Dn = _chart(self.normals, self.normal_tangents, x[:, 1:])
        directions, Dl = _chart(self.directions, self.light_tangents, c[:, 1:])
        rho = self.rho * np.exp(x[:, 0])
        intensity = self.intensity * np.exp(c[:, 0])
        cosine = directions @ normals.T
        lit = cosine > 0
        factor = intensity[:, None] * rho[None, :]
        signal = factor * np.maximum(cosine, 0)
        if not jacobian:
            return signal
        As = np.zeros((self.L*self.P, 3*self.P))
        Bs = np.zeros((self.L*self.P, 3*self.L))
        for k in range(self.L):
            rows = k*self.P + np.arange(self.P)
            cols = 3*np.arange(self.P)
            As[rows, cols] = signal[k]
            As[rows, cols+1] = factor[k] * lit[k] * (Dn[:, :, 0] @ directions[k])
            As[rows, cols+2] = factor[k] * lit[k] * (Dn[:, :, 1] @ directions[k])
            Bs[rows, 3*k] = signal[k]
            Bs[rows, 3*k+1] = factor[k] * lit[k] * (normals @ Dl[k, :, 0])
            Bs[rows, 3*k+2] = factor[k] * lit[k] * (normals @ Dl[k, :, 1])
        return signal, As, Bs

    def nonlinear_profile(self, observations, t, sigma, max_nfev=50):
        """Small-noise joint MAP in local charts; normals are estimated, not fixed."""
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        observations = np.asarray(observations, float)
        if observations.shape != (self.L, self.P):
            raise ValueError("observations must have shape (L,P)")
        root = np.linalg.cholesky(self.precision(t)).T
        sqrtw = np.sqrt(self.weights).ravel()
        nscene = 3*self.P

        def residual(z):
            physical = self.physical(z[:nscene], z[nscene:])
            return np.concatenate([(physical - observations).ravel() * sqrtw,
                                   root @ z[nscene:]]) / sigma

        def jac(z):
            _, A, B = self.physical(z[:nscene], z[nscene:], jacobian=True)
            return np.block([[sqrtw[:, None] * A, sqrtw[:, None] * B],
                             [np.zeros((3*self.L, nscene)), root]]) / sigma

        fit = least_squares(residual, np.zeros(3*(self.P+self.L)), jac=jac,
                            ftol=1e-11, xtol=1e-11, gtol=1e-9,
                            max_nfev=max_nfev)
        return fit.x[:nscene], {"success": bool(fit.success),
                               "nfev": int(fit.nfev),
                               "optimality": float(fit.optimality)}

    def gauge_generators(self):
        """Nine local GL(3) generators of unconstrained photometric factorization."""
        scene, calibration = [], []
        for i in range(3):
            for j in range(3):
                E = np.zeros((3, 3))
                E[i, j] = 1
                dn = self.normals @ E.T
                dl = -self.directions @ E
                xs = np.empty((self.P, 3))
                cs = np.empty((self.L, 3))
                xs[:, 0] = np.sum(self.normals * dn, axis=1)
                xs[:, 1:] = np.einsum("pij,pi->pj", self.normal_tangents, dn)
                cs[:, 0] = np.sum(self.directions * dl, axis=1)
                cs[:, 1:] = np.einsum("kij,ki->kj", self.light_tangents, dl)
                scene.append(xs.ravel())
                calibration.append(cs.ravel())
        return np.column_stack(scene), np.column_stack(calibration)

    def scale_gauge_report(self, lambdas=(0.1, 1.0, 10.0)):
        a = np.zeros(3*self.P)
        c = np.zeros(3*self.L)
        a[::3] = 1
        c[::3] = 1
        response = gauge_response(self.B, c, np.asarray(lambdas))
        direct = []
        for lam in lambdas:
            F, _, _ = delta_f(self.A, self.B, lam)
            direct.append(float(a @ F @ a))
        return {"alignment_residual": float(np.linalg.norm(self.A @ a - self.B @ c)),
                "gauge_response_max_abs_error": float(np.max(
                    np.abs(response["exact"] - direct))),
                "scene_generator": a, "nuisance_generator": c}


def build_joint_linearization(rho, normals, directions, intensity=None,
                              weights=None, lam0=None):
    """Construct the finite-SPD experiment; flat-prior quotients use schur separately."""
    rho = np.asarray(rho, float)
    normals = _unit_rows(normals, "normals")
    directions = _unit_rows(directions, "directions")
    P, L = len(normals), len(directions)
    if rho.shape != (P,) or np.any(rho <= 0) or not np.all(np.isfinite(rho)):
        raise ValueError("rho must be a positive finite per-pixel vector")
    intensity = np.ones(L) if intensity is None else np.asarray(intensity, float)
    weights = np.ones((L, P)) if weights is None else np.asarray(weights, float)
    if (intensity.shape != (L,) or np.any(intensity <= 0)
            or not np.all(np.isfinite(intensity))):
        raise ValueError("intensity must be positive and finite")
    if (weights.shape != (L, P) or np.any(weights <= 0)
            or not np.all(np.isfinite(weights))):
        raise ValueError("weights must be positive finite (L,P)")
    lam0 = np.tile(np.eye(3), (L, 1, 1)) if lam0 is None else np.asarray(lam0, float)
    if lam0.shape != (L, 3, 3) or not np.all(np.isfinite(lam0)):
        raise ValueError("lam0 must contain finite per-light (3,3) SPD blocks")
    for block in lam0:
        if not np.allclose(block, block.T, rtol=1e-12, atol=1e-14):
            raise ValueError("precision blocks must be symmetric")
        np.linalg.cholesky(block)
    nt = np.stack(tangent_basis(normals), axis=-1)
    lt = np.stack(tangent_basis(directions), axis=-1)
    cosine = directions @ normals.T
    if np.any(np.abs(cosine) <= 1e-10):
        raise ValueError("linearization lies on a shadow boundary")
    shape = (L*P, 3*P)
    model = JointLinearization(
        rho.copy(), normals, directions, intensity.copy(), weights.copy(), nt, lt,
        np.empty(shape), np.empty((L*P, 3*L)), np.empty((3*P, 3*P)),
        np.empty((L, 3*P, 3)), np.empty((L, 3, 3)), lam0.copy(),
        cosine > 0, np.empty((L, P)))
    model.nominal, Araw, Braw = model.physical(
        np.zeros(3*P), np.zeros(3*L), jacobian=True)
    model.A, model.B, _ = whiten_system(Araw, Braw, 1.0 / weights.ravel())
    model.D = model.A.T @ model.A
    for k in range(L):
        Ak = model.A[k*P:(k+1)*P]
        Bk = model.B[k*P:(k+1)*P, 3*k:3*k+3]
        model.u[k] = Ak.T @ Bk
        model.M0[k] = Bk.T @ Bk
    return model


def synthetic_joint_scene():
    normals = np.array([[-.4, -.25, 1], [-.2, .35, 1], [.3, -.3, 1],
                        [.45, .2, 1], [.02, .04, 1], [-.12, .2, 1]])
    directions = np.array([[-.6, .1, 1], [.6, -.1, 1], [-.2, -.6, 1],
                           [.1, .6, 1], [.45, .45, 1], [-.45, -.45, 1],
                           [.3, -.5, 1], [-.5, .3, 1], [.03, .02, 1],
                           [.2, .4, 1]])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    rho = np.array([.6, .8, 1.0, 1.2, .9, .7])
    intensity = np.linspace(.8, 1.2, len(directions))
    weights = 1.0 + np.arange(len(directions)*len(normals)).reshape(-1, len(normals)) / 80
    lam = np.tile(np.diag([2.0, 3.0, 4.0]), (len(directions), 1, 1))
    return build_joint_linearization(rho, normals, directions, intensity, weights, lam)
