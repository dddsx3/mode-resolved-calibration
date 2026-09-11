"""E4: calibration-budget allocation is exactly convex -> certified optimality gap.

Model (same algebra as the repo's per-light whitened blocks):
    DeltaF(t) = diag(Finf) - sum_k u_k (M0_k + t_k Lam0_k)^{-1} u_k^T
Design: t_k in [1, regime]; budget sum_k (t_k - 1) <= kb*(regime-1).
Objective f(t) = tr(DeltaF^{-1})  (total MSE / sigma^2, A-optimality).
DeltaF is matrix-monotone & matrix-concave in t  =>  f convex  => Frank-Wolfe
gives a VALID lower bound (duality gap) and every FW vertex is a feasible
discrete allocation.  ClawsGO Science Harness probe.
"""
import numpy as np


def sym_inv(M, tol_rel=1e-12):
    w, V = np.linalg.eigh(M)
    tol = max(float(w.max()), 1.0) * tol_rel
    wi = np.where(w > tol, 1.0 / np.where(w > tol, w, 1.0), 0.0)
    return (V * wi) @ V.T


def sqrtm_psd(M):
    w, V = np.linalg.eigh(M)
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.T


def build(P=1200, Ltot=142, Lact=48, seed=1, sig_logI=0.1, sig_deg=0.1):
    rng = np.random.default_rng(seed)
    n = rng.normal(size=(P, 3)); n[:, 2] = np.abs(n[:, 2]) + 0.3
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    alb = np.exp(rng.uniform(np.log(0.2), 0.0, P))
    d = rng.normal(size=(Ltot, 3)); d[:, 2] = np.abs(d[:, 2]) + 0.2
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    s = np.clip(n @ d.T, 0, None).T
    h = (s > 0).astype(float)
    w = 1.0 / np.maximum(1e-4 + 0.05 * s * alb[None, :], 1e-6)
    t1 = np.zeros_like(d); t2 = np.zeros_like(d)
    for k, dk in enumerate(d):
        ref = np.array([0., 0., 1.]) if abs(dk[2]) < 0.9 else np.array([1., 0., 0.])
        v = np.cross(dk, ref); v /= np.linalg.norm(v)
        t1[k] = v; t2[k] = np.cross(dk, v)
    Bphi = np.stack([s * alb[None, :], (n @ t1.T).T * alb[None, :] * h,
                     (n @ t2.T).T * alb[None, :] * h], -1)
    active = np.zeros(Ltot, bool)
    active[np.sort(rng.choice(Ltot, Lact, replace=False))] = True
    u = np.zeros((Ltot, P, 3)); M0 = np.zeros((Ltot, 3, 3))
    for k in np.where(active)[0]:
        Ak = np.sqrt(w[k]) * s[k]; Bk = np.sqrt(w[k])[:, None] * Bphi[k]
        u[k] = Ak[:, None] * Bk; M0[k] = Bk.T @ Bk
    Finf = (w[active] * s[active] ** 2).sum(0)
    Sig = np.diag([sig_logI ** 2, np.radians(sig_deg) ** 2, np.radians(sig_deg) ** 2])
    Lam0 = np.tile(np.linalg.inv(Sig), (Ltot, 1, 1))
    return dict(u=u, M0=M0, Lam0=Lam0, Finf=Finf, active=active,
                Ltot=Ltot, P=P, idx_act=np.where(active)[0])


# ---------------------------------------------------------------- oracle f, grad
def f_and_grad(pr, t, want_grad=True):
    """f(t) = tr(DeltaF^{-1}) and df/dt_k, via the exact low-rank (Woodbury) route."""
    u, M0, Lam0, Finf = pr["u"], pr["M0"], pr["Lam0"], pr["Finf"]
    ia = pr["idx_act"]
    Ainv = 1.0 / Finf
    Ks, Vs = {}, []
    for k in ia:
        K = sym_inv(M0[k] + t[k] * Lam0[k])
        Ks[k] = K
        Vs.append(u[k] @ sqrtm_psd(K))
    V = np.concatenate(Vs, axis=1)                     # (P, 3|act|)
    Z = Ainv[:, None] * V
    G = np.eye(V.shape[1]) - V.T @ Z
    Gc = np.linalg.cholesky(G)
    Ginv = np.linalg.inv(G)
    # tr(DF^{-1}) = tr(D^{-1}) + tr(Ginv @ V^T D^{-2} V)
    Z2 = (Ainv ** 2)[:, None] * V
    f = float(Ainv.sum() + np.trace(Ginv @ (V.T @ Z2)))
    if not want_grad:
        return f, None
    U = np.concatenate([u[k] for k in ia], axis=1)      # (P, 3|act|)
    Aall = Ainv[:, None] * U
    Yall = Aall + Z @ (Ginv @ (Z.T @ U))                # DF^{-1} u_k blocks
    grad = np.zeros(pr["Ltot"])
    for j, k in enumerate(ia):
        Y = Yall[:, 3 * j:3 * j + 3]
        K = Ks[k]
        # d f / d t_k = -tr(K Lam0 K (u^T DF^{-2} u))
        grad[k] = -float(np.trace(K @ Lam0[k] @ K @ (Y.T @ Y)))
    return f, grad


def f_dense(pr, t):
    """Independent dense cross-check of f."""
    DF = np.diag(pr["Finf"])
    for k in pr["idx_act"]:
        DF -= (pr["u"][k] @ sym_inv(pr["M0"][k] + t[k] * pr["Lam0"][k])) @ pr["u"][k].T
    return float(np.trace(np.linalg.inv(DF)))


# ---------------------------------------------------------------- Frank-Wolfe
def lmo(grad, cand, budget, hi):
    """argmin <grad, z> over {0<=z_k<=hi (k in cand), sum z <= budget}."""
    z = np.zeros(len(grad))
    order = [k for k in cand if grad[k] < 0]
    order.sort(key=lambda k: grad[k])
    left = budget
    for k in order:
        if left <= 0:
            break
        take = min(hi, left)
        z[k] = take
        left -= take
    return z


def frank_wolfe(pr, cand, kb, regime, iters=120, verbose=False):
    hi = regime - 1.0
    budget = kb * hi
    z = np.zeros(pr["Ltot"])
    best_lb = -np.inf
    hist = []
    for it in range(iters):
        f, g = f_and_grad(pr, 1.0 + z)
        s = lmo(g, cand, budget, hi)
        gap = float(g @ (z - s))                       # >= 0
        best_lb = max(best_lb, f - gap)
        hist.append((f, best_lb))
        if verbose and it % 20 == 0:
            print(f"    it{it:3d}  f={f:.6e}  LB={best_lb:.6e}  gap={gap:.3e}")
        gamma = 2.0 / (it + 2.0)
        z = (1 - gamma) * z + gamma * s
    f, _ = f_and_grad(pr, 1.0 + z, want_grad=False)
    return dict(f_relaxed=min(f, min(h[0] for h in hist)), lb=best_lb, hist=hist)


# ---------------------------------------------------------------- discrete policies
def eval_subset(pr, sel, regime):
    t = np.ones(pr["Ltot"]); t[list(sel)] = regime
    return f_and_grad(pr, t, want_grad=False)[0]


def greedy_aopt(pr, cand, k, regime):
    sel = []
    for _ in range(k):
        best, bi = None, None
        for c in cand:
            if c in sel:
                continue
            v = eval_subset(pr, sel + [c], regime)
            if best is None or v < best:
                best, bi = v, c
        sel.append(bi)
    return sel


def greedy_mode_aware(pr, cand, k, regime, n_modes=5):
    """Repo's adaptive normalized weak-Fisher-mode sensitivity heuristic."""
    u, M0, Lam0, Finf = pr["u"], pr["M0"], pr["Lam0"], pr["Finf"]
    t = np.ones(pr["Ltot"]); sel = []
    for _ in range(k):
        DF = np.diag(Finf)
        for kk in pr["idx_act"]:
            DF -= (u[kk] @ sym_inv(M0[kk] + t[kk] * Lam0[kk])) @ u[kk].T
        _w, Vm = np.linalg.eigh(DF)
        modes = Vm[:, :n_modes]
        rem = [c for c in cand if c not in sel]
        g = np.zeros((len(rem), n_modes))
        for i, c in enumerate(rem):
            K = sym_inv(M0[c] + t[c] * Lam0[c])
            Kv = K @ (modes.T @ u[c]).T
            for j in range(n_modes):
                g[i, j] = Kv[:, j] @ (Lam0[c] @ Kv[:, j])
        Gs = (g / (g.sum(0) + 1e-12)).mean(1)
        pick = rem[int(np.argmax(Gs))]
        sel.append(pick); t[pick] = regime
    return sel
