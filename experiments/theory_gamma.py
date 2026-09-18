# status: experimental — NOT part of the published results
"""Physical gamma counterexamples and additional full-trace certificates.

M13/M15 research only.  Nothing here changes allocation.alpha_bound or its
historical gamma_lower_bound interface.  All inverses below are on a fixed
strictly positive-definite space; singular baselines are rejected, not trimmed.
The physical Jacobian follows NominalScene (additive albedo coordinates).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh

from calibinfo.allocation.blocks import LightBlocks, sym_inv


def _symmetric(value, name="matrix"):
    a = np.asarray(value, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not a.shape[0]:
        raise ValueError(f"{name} must be a nonempty square matrix")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite")
    scale = max(float(np.max(np.abs(a))), np.finfo(float).tiny)
    if np.max(np.abs(a - a.T)) > 1e-10 * scale:
        raise ValueError(f"{name} must be symmetric")
    return (a + a.T) / 2


def _spd(value, name="baseline"):
    a = _symmetric(value, name)
    try:
        cho_factor(a, lower=True, check_finite=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be strictly positive definite") from exc
    return a


def psd_factor(value):
    """W=LL^T, retaining every positive eigenvalue (no positive-rank cutoff)."""
    w = _symmetric(value, "update")
    ev, q = eigh(w, check_finite=False)
    scale = max(float(np.max(np.abs(ev))), np.finfo(float).tiny)
    if ev[0] < -1e-10 * scale:
        raise ValueError("update must be positive semidefinite")
    positive = ev > 0
    return q[:, positive] * np.sqrt(ev[positive])


def _problem(baseline, updates):
    a = _spd(baseline)
    ws = []
    for value in updates:
        w = _symmetric(value, "update")
        if w.shape != a.shape:
            raise ValueError("baseline and updates must have the same shape")
        psd_factor(w)
        ws.append(w)
    return a, ws


def full_trace_marginal(baseline, update):
    """Stable Woodbury trace, without subtracting two large inverse traces."""
    a = _spd(baseline)
    l = psd_factor(update)
    if l.shape[0] != a.shape[0]:
        raise ValueError("baseline and update must have the same shape")
    if not l.shape[1]:
        return 0.0
    z = cho_solve(cho_factor(a, lower=True, check_finite=False), l,
                  check_finite=False)
    small = np.eye(l.shape[1]) + l.T @ z
    return float(np.trace(cho_solve(cho_factor(small, lower=True), z.T @ z)))


def exhaustive_gamma(baseline, updates, max_candidates=10):
    """All S subset T, including S=T.  Tests only, not a large-design solver."""
    a, ws = _problem(baseline, updates)
    m = len(ws)
    if m > max_candidates:
        raise ValueError("exhaustive gamma is restricted to small instances")
    ms = {0: a}
    for mask in range(1, 1 << m):
        bit = mask & -mask
        ms[mask] = ms[mask ^ bit] + ws[bit.bit_length() - 1]
    best, witness, count = 1.0, None, 0
    for x, w in enumerate(ws):
        if not np.any(w):
            continue
        ds = {t: full_trace_marginal(mat, w) for t, mat in ms.items()
              if not t & (1 << x)}
        for t, den in ds.items():
            if den <= 0:
                raise ArithmeticError("nonzero PSD update has nonpositive marginal")
            s = t
            while True:
                ratio = ds[s] / den
                count += 1
                if ratio < best:
                    best, witness = ratio, {"x": x, "S_mask": s, "T_mask": t}
                if s == 0:
                    break
                s = (s - 1) & t
    return {"gamma": float(best), "triples": count, "witness": witness}


def alpha_value(baseline, updates):
    a, ws = _problem(baseline, updates)
    return max((float(eigh(w, a, eigvals_only=True, check_finite=False)[-1])
                for w in ws), default=0.0)


def spectral_certificates(baseline, updates):
    """Exact dense certificates (small matrices).

    b1 = amin / max_x lambda_max(M(U\\{x})) is the existing valid bound.
    b2 omits TWO nonzero candidates.  Strict S<T omits x and at least one y,
    hence gamma >= b2 >= b1.  S=T gives 1.  For b0=amin/lambda_max(M(U)),
    the inverse-square Kantorovich comparison integrated along each update
    gives gamma >= 4*b0/(1+b0)^2.  The latter must use the FULL upper matrix,
    not the leave-one matrix.  No arbitrary task weight is accepted here.
    """
    a, ws = _problem(baseline, updates)
    active = [i for i, w in enumerate(ws) if np.any(w)]
    total = a + sum(ws, np.zeros_like(a))
    amin = float(eigh(a, eigvals_only=True, check_finite=False)[0])
    top = float(eigh(total, eigvals_only=True, check_finite=False)[-1])
    base = amin / top
    if active:
        # Reassemble from A rather than subtracting large updates: e.g. the
        # tight h^-4 family otherwise erases A by catastrophic cancellation.
        hi1 = max(float(eigh(a + sum((w for j, w in enumerate(ws) if j != i),
                                    np.zeros_like(a)), eigvals_only=True,
                             check_finite=False)[-1]) for i in active)
        b1 = amin / hi1
    else:
        hi1, b1 = top, 1.0
    if len(active) >= 2:
        hi2 = max(float(eigh(a + sum((w for k, w in enumerate(ws) if k not in (i, j)),
                                    np.zeros_like(a)), eigvals_only=True,
                             check_finite=False)[-1])
                  for i, j in combinations(active, 2))
        b2 = amin / hi2
    else:
        hi2, b2 = top, 1.0
    kant = 4 * base / (1 + base) ** 2
    return {"lambda_min_baseline": amin, "lambda_max_total": top,
            "active_candidates": len(active),
            "leave_one_denominator": hi1, "leave_two_denominator": hi2,
            "full_spectral": base, "leave_one_spectral": b1,
            "leave_two_spectral": min(1.0, b2), "kantorovich_full": kant,
            "improved": min(1.0, max(b1, b2, kant))}


@dataclass
class PhysicalInstance:
    normals: np.ndarray
    directions: np.ndarray
    rho: np.ndarray
    weights: np.ndarray
    shading: np.ndarray
    tangents: np.ndarray
    b_phi: np.ndarray
    blocks: LightBlocks
    kappa: float
    epsilon: float
    baseline: np.ndarray
    updates: list[np.ndarray]


def physical_lambertian_family(q=0.01):
    """Three physical lights, positive weights/priors, fixed kappa=10.

    epsilon=2q/(1+q^2), c=(1-q^2)/(1+q^2), 0<q<1.
    n1=(3/5,4/5,0), n2=(-c,epsilon,0), lights=(ex,ey,ey), rho=1.
    Exact membership and all rational calculations are independently checked
    by physical_exact_certificate; this path actually assembles LightBlocks.
    """
    q = float(q)
    if not np.isfinite(q) or not 0 < q < 1:
        raise ValueError("q must lie strictly between zero and one")
    e, c = 2 * q / (1 + q * q), (1 - q * q) / (1 + q * q)
    n = np.array([[3 / 5, 4 / 5, 0], [-c, e, 0]])
    dirs = np.array([[1., 0, 0], [0, 1., 0], [0, 1., 0]])
    t1 = np.cross(dirs, [0., 0, 1.])
    t2 = np.cross(dirs, t1)
    shading = np.maximum(n @ dirs.T, 0).T
    lit = shading > 0
    rho = np.ones(2)
    bphi = np.stack([shading, (n @ t1.T).T * lit,
                     (n @ t2.T).T * lit], axis=-1)
    w = np.array([[25 / 9, 1], [25 / 16, 1], [25 / 16, 1]])
    lam = np.zeros((3, 3, 3))
    lam[0] = np.diag([25 / 9, 25 / 9, 1])
    h = np.array([[1., 3 / 4], [e, -c]])
    for k, sign in [(1, 1), (2, -1)]:
        covariance = np.array([[101., sign * 99], [sign * 99, 101]]) / 20
        lam[k, :2, :2] = h.T @ cho_solve(cho_factor(covariance), h)
        lam[k, 2, 2] = 1
    u = (w * shading)[:, :, None] * bphi
    m0 = np.einsum("kpi,kpj->kij", bphi, w[:, :, None] * bphi)
    blocks = LightBlocks(u, m0, lam, np.sum(w * shading**2, axis=0),
                         np.ones(3, dtype=bool))
    baseline = blocks.assemble(lam)
    updates = [(u[k] @ (sym_inv(m0[k] + lam[k])
                       - sym_inv(m0[k] + 10 * lam[k]))) @ u[k].T
               for k in range(3)]
    return PhysicalInstance(n, dirs, rho, w, shading, np.stack([t1, t2], axis=1),
                            bphi, blocks, 10., e, baseline, updates)


def physical_exact_certificate(q="1/100"):
    """Independent SymPy rational reconstruction, with exact exhaustive gamma.

    This does not rationalize a floating-point matrix.  n,l,w,B,Lambda,u,M0
    and all inverses are built from integers/rationals before conversion.
    """
    import sympy as sp

    q = sp.Rational(str(q))
    if not 0 < q < 1:
        raise ValueError("q must lie strictly between zero and one")
    e, c = 2*q/(1+q*q), (1-q*q)/(1+q*q)
    n = sp.Matrix([[sp.Rational(3, 5), sp.Rational(4, 5), 0], [-c, e, 0]])
    dirs = sp.Matrix([[1, 0, 0], [0, 1, 0], [0, 1, 0]])
    w = [sp.diag(sp.Rational(25, 9), 1),
         sp.diag(sp.Rational(25, 16), 1), sp.diag(sp.Rational(25, 16), 1)]
    shading = [sp.Matrix([sp.Rational(3, 5), 0]),
               sp.Matrix([sp.Rational(4, 5), e]),
               sp.Matrix([sp.Rational(4, 5), e])]
    bphi, tangents = [], []
    for k in range(3):
        light = dirs.row(k).T
        t1 = light.cross(sp.Matrix([0, 0, 1]))
        t2 = light.cross(t1)
        tangents.append(sp.Matrix.hstack(t1, t2))
        bphi.append(sp.Matrix([[shading[k][p],
                               (n.row(p)*t1)[0] if shading[k][p] > 0 else 0,
                               (n.row(p)*t2)[0] if shading[k][p] > 0 else 0]
                              for p in range(2)]))
    h = sp.Matrix([[1, sp.Rational(3, 4)], [e, -c]])
    lam = [sp.diag(sp.Rational(25, 9), sp.Rational(25, 9), 1)]
    for sign in [1, -1]:
        covariance = sp.Matrix([[101, sign*99], [sign*99, 101]])/20
        precision = sp.eye(3)
        precision[:2, :2] = h.T * covariance.inv() * h
        lam.append(precision)
    u = [sp.diag(*shading[k])*w[k]*bphi[k] for k in range(3)]
    m0 = [bphi[k].T*w[k]*bphi[k] for k in range(3)]
    finf = sum((sp.diag(*shading[k])**2*w[k] for k in range(3)), sp.zeros(2))
    base = finf - sum((u[k]*(m0[k]+lam[k]).inv()*u[k].T for k in range(3)),
                     sp.zeros(2))
    ws = [sp.simplify(u[k]*((m0[k]+lam[k]).inv()
                          - (m0[k]+10*lam[k]).inv())*u[k].T)
          for k in range(3)]
    expected_base = sp.diag(sp.Rational(3, 2), e*e)
    v, z = sp.Rational(99, 404), sp.Rational(729, 4444)
    expected_ws = [sp.diag(sp.Rational(9, 22), 0),
                   sp.Matrix([[v, z*e], [z*e, v*e*e]]),
                   sp.Matrix([[v, -z*e], [-z*e, v*e*e]])]
    assert base == expected_base and ws == expected_ws
    assert all(all(mat[:i, :i].det() > 0 for i in range(1, 4)) for mat in lam)
    assert all((n.row(p)*n.row(p).T)[0] == 1 for p in range(2))
    all_m = {mask: base+sum((ws[k] for k in range(3) if mask & (1 << k)),
                           sp.zeros(2)) for mask in range(8)}
    def marginal(mat, update):
        return sp.factor(sp.trace(mat.inv()-(mat+update).inv()))
    candidates = []
    for x in range(3):
        for t in range(8):
            if t & (1 << x):
                continue
            s = t
            while True:
                ratio = sp.factor(marginal(all_m[s], ws[x])
                                  / marginal(all_m[t], ws[x]))
                candidates.append((ratio, x, s, t))
                if not s:
                    break
                s = (s-1) & t
    best = min(candidates, key=lambda item: item[0])
    alpha = sp.Rational(165, 808) + 3*sp.sqrt(172105)/8888
    whitened = sp.diag(sp.sqrt(sp.Rational(2, 3)), 1/e)
    max_alphas = [max((whitened*mat*whitened).eigenvals()) for mat in ws]
    assert sp.simplify(max(max_alphas)-alpha) == 0
    ratio_formula = (3025408256*e*e/(77*(30614089*e*e+531441)))
    assert sp.factor(marginal(base, ws[0])/marginal(base+ws[1], ws[0])
                     - ratio_formula) == 0
    encode = lambda mat: [[str(x) for x in mat.row(i)] for i in range(mat.rows)]
    return {"q": str(q), "epsilon": str(e), "c": str(c), "kappa": "10",
            "normals": encode(n), "light_directions": encode(dirs),
            "rho": ["1", "1"], "weights": [encode(mat) for mat in w],
            "shading": [[str(x) for x in vec] for vec in shading],
            "tangents": [encode(mat) for mat in tangents],
            "B_phi": [encode(mat) for mat in bphi],
            "Lambda0": [encode(mat) for mat in lam],
            "u": [encode(mat) for mat in u], "M0": [encode(mat) for mat in m0],
            "Finf": encode(finf), "baseline": encode(base),
            "updates": [encode(mat) for mat in ws],
            "alpha_exact": str(alpha), "alpha": float(alpha),
            "historical_candidate": float(1/(1+alpha)),
            "gamma_exact": str(best[0]), "gamma_decimal_60": str(sp.N(best[0], 60)),
            "gamma": float(best[0]), "witness": {"x": best[1], "S_mask": best[2],
                                                  "T_mask": best[3]},
            "strict_witness_ratio_exact": str(ratio_formula),
            "witness_marginal_empty": str(marginal(base, ws[0])),
            "witness_marginal_refined": str(marginal(base+ws[1], ws[0])),
            "asymptotic_upper_coefficient": "3025408256/40920957",
            "exact_triples": len(candidates), "verified_prior_spd": True,
            "verified_physical_membership": True,
            "coupling_rank": sp.Matrix.hstack(*u).rank(),
            "unaffected_retention_modes": 2-sp.Matrix.hstack(*u).rank()}


def tight_two_candidate_family(h, a=0.25):
    """For fixed 0<a<1, chi=1/a and gamma=a+(2a^2-a+1)h^2+O(h^4).

    At a=1 the exact coefficient is gamma=1 for every h (S=T included);
    the displayed nonconstant expansion does not apply to that endpoint.
    This family has unbounded update size as h->0.  It does NOT establish
    an infimum when kappa or update norms are fixed in addition to chi.
    """
    h, a = float(h), float(a)
    if not np.isfinite(h) or h <= 0 or not np.isfinite(a) or not 0 < a <= 1:
        raise ValueError("require h>0 and 0<a<=1")
    v = np.array([1., h])
    return np.diag([1., a]), [np.diag([h**-4, 0]), h**-4*np.outer(v, v)]


@dataclass
class DowndateEnvelope:
    """Certified-in-exact-arithmetic top-subspace bounds for E-VV^T.

    The top eigenspace of E is computed ONCE.  Each subsequent removal needs
    only small Gram matrices, including the cross-block spectral norm.  No
    unjustified subtraction of a nonzero low-rank eigenvalue is used.
    Ordinary LAPACK rounding is not interval arithmetic; callers report the
    numerical bracket width and a floating-point safety pad.
    """
    top_values: np.ndarray
    top_vectors: np.ndarray
    complement_ceiling: float
    dimension: int
    safety_pad: float

    @classmethod
    def build(cls, matrix, top_rank=12):
        a = _symmetric(matrix)
        p = a.shape[0]
        rank = min(max(int(top_rank), 1), p)
        start = max(0, p-rank-1)
        values, vectors = eigh(a, subset_by_index=[start, p-1], check_finite=False)
        pad = 64*np.finfo(float).eps*p*max(abs(float(values[-1])), 1.)
        if rank == p:
            return cls(values, vectors, float("-inf"), p, pad)
        return cls(values[1:], vectors[:, 1:], float(values[0]), p, pad)

    def interval(self, factor):
        v = np.asarray(factor, float)
        if v.ndim != 2 or v.shape[0] != self.dimension or not np.all(np.isfinite(v)):
            raise ValueError("factor has invalid shape or entries")
        z = self.top_vectors.T @ v
        gram = v.T @ v
        return self.interval_from_gram(z, gram)

    def interval_from_gram(self, projected_factor, factor_gram):
        z, gram = np.asarray(projected_factor, float), np.asarray(factor_gram, float)
        small = np.diag(self.top_values)-z@z.T
        top = float(eigh(small, eigvals_only=True, check_finite=False)[-1])
        if len(self.top_values) == self.dimension:
            return max(0., top-self.safety_pad), top+self.safety_pad
        residual = (gram-z.T@z)
        residual = (residual+residual.T)/2
        cross_sq = z@residual@z.T
        b2 = max(0., float(eigh((cross_sq+cross_sq.T)/2, eigvals_only=True,
                                check_finite=False)[-1]))
        c = self.complement_ceiling
        block_upper = (top+c+np.sqrt((top-c)**2+4*b2))/2
        upper = min(float(self.top_values[-1]), block_upper)
        return max(0., top-self.safety_pad), upper+self.safety_pad


def factor_spectral_certificates(baseline, factors, top_rank=12):
    """Low-rank large-design variant, with explicit spectral enclosures.

    Per candidate / pair no P-by-P factorization is performed.  Lower/upper
    entries for b1,b2 bracket the dense formula; only the LOWER entries are
    guarantees.  A strict gamma minimum over the subset lattice is NOT run.
    """
    a = _spd(baseline)
    fs = []
    for f in factors:
        f = np.asarray(f, float)
        if f.ndim != 2 or f.shape[0] != a.shape[0] or not np.all(np.isfinite(f)):
            raise ValueError("factor has invalid shape or entries")
        if np.any(f):
            fs.append(f)
    p = a.shape[0]
    total = a.copy()
    for f in fs:
        total += f@f.T
    env = DowndateEnvelope.build(total, top_rank)
    amin = float(eigh(a, eigvals_only=True, subset_by_index=[0, 0],
                      check_finite=False)[0])
    scale = max(float(np.max(np.abs(a))), 1.)
    amin_pad = 64*np.finfo(float).eps*p*scale
    lo_min, hi_min = max(0., amin-amin_pad), amin+amin_pad
    top_lo, top_hi = env.top_values[-1]-env.safety_pad, env.top_values[-1]+env.safety_pad
    if fs:
        joined = np.concatenate(fs, axis=1)
        projected = env.top_vectors.T@joined
        gram = joined.T@joined
        cuts = np.cumsum([0]+[f.shape[1] for f in fs])
        slots = [np.arange(cuts[i], cuts[i+1]) for i in range(len(fs))]
        def bracket(ids):
            index = np.concatenate([slots[i] for i in ids])
            return env.interval_from_gram(projected[:, index], gram[np.ix_(index, index)])
        intervals1 = [bracket([i]) for i in range(len(fs))]
        d1lo = max(pair[0] for pair in intervals1)
        d1hi = max(pair[1] for pair in intervals1)
        one = [lo_min/d1hi, 1. if d1lo <= 0 else min(1., hi_min/d1lo)]
    else:
        one, d1lo, d1hi = [1., 1.], top_lo, top_hi
    if len(fs) >= 2:
        intervals2 = [bracket([i, j]) for i, j in combinations(range(len(fs)), 2)]
        d2lo = max(pair[0] for pair in intervals2)
        d2hi = max(pair[1] for pair in intervals2)
        two = [lo_min/d2hi, 1. if d2lo <= 0 else min(1., hi_min/d2lo)]
    else:
        two, d2lo, d2hi = [1., 1.], top_lo, top_hi
    full = lo_min/top_hi
    kant = 4*full/(1+full)**2
    return {"active_candidates": len(fs), "lambda_min_baseline": amin,
            "lambda_max_total": float(env.top_values[-1]),
            "leave_one_denominator_interval": [d1lo, d1hi],
            "leave_two_denominator_interval": [d2lo, d2hi],
            "leave_one_spectral_interval": one, "leave_two_spectral_interval": two,
            "full_spectral": full, "kantorovich_full": kant,
            "improved": min(1., max(one[0], two[0], kant)),
            "top_subspace_rank": len(env.top_values),
            "rounding_pad": max(amin_pad, env.safety_pad),
            "spectral_enclosure_is_exact_arithmetic_theorem": True,
            "floating_point_not_interval_arithmetic": True,
            "gamma_exhaustively_evaluated": False}
