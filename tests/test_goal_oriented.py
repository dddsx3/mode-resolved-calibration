"""N4/T10 · goal-oriented calibration value:quad-risk 低秩路线的数学门禁。

`woodbury_quad_risk(finf_diag, u, M0, lam, active, t, H)` =
`J_H(t) = tr(H ΔF(t)^{-1} Hᵀ)`(push-through 恒等式,3|active| 小矩阵路线)
与其解析梯度 `∇_{t_k} J_H = −tr(K_k Λ0k K_k · (u_kᵀ Z)(Zᵀ u_k))`,
Z = ΔF⁻¹Hᵀ。绑定验收(N4-1/N4-2):
  1. H = I 与 woodbury_trace_inv 相对误差 < 1e-10;
  2. 任意 H (m < P) 与稠密 tr(H ΔF⁻¹ Hᵀ) 相对误差 < 1e-10;
  3. H 行的正交混合不变性(tr((HQ)ΔF⁻¹(HQ)ᵀ) = tr(HΔF⁻¹Hᵀ));
  4. 解析梯度与 FD 对拍 ≤ 1e-6。
"""

import numpy as np
import pytest

from calibinfo.allocation.blocks import sym_inv
from calibinfo.information.lowrank import (
    woodbury_trace_inv, woodbury_trace_inv_grad,
    woodbury_quad_risk, woodbury_quad_risk_grad)


def _state(seed=20260913, L=6, P=30, q=3):
    rng = np.random.default_rng(seed)
    w = rng.uniform(0.5, 2.0, size=(L, P))
    s = rng.uniform(0.2, 1.5, size=(L, P))
    B = rng.normal(0.0, 0.5, size=(L, P, q))
    u = (w * s)[:, :, None] * B
    M0 = np.einsum("kpi,kpj->kij", B, w[:, :, None] * B)
    lam0 = np.stack([np.diag([30.0, 800.0, 800.0])] * L)
    finf = (w * s ** 2).sum(0)
    active = np.ones(L, bool)
    t = np.ones(L)
    return u, M0, lam0, finf, active, t


def _delta_f(u, M0, lam0, finf, active, t):
    DF = np.diag(finf)
    for k in np.flatnonzero(active):
        K = sym_inv(M0[k] + t[k] * lam0[k])
        DF -= (u[k] @ K) @ u[k].T
    return DF


# ------------------------------------------------------------- N4-1 验收
def test_quad_risk_H_identity_matches_trace_inv():
    """H = I 精确退化为 tr ΔF⁻¹(与 woodbury_trace_inv 对拍 < 1e-10)。"""
    u, M0, lam0, finf, active, t = _state()
    for tk in (1.0, 3.7, 10.0):
        tt = np.full(u.shape[0], tk)
        ref = woodbury_trace_inv(finf, u, M0, lam0, active, tt)
        got = woodbury_quad_risk(finf, u, M0, lam0, active, tt, np.eye(finf.size))
        assert got == pytest.approx(ref, rel=1e-10), (tk, got, ref)
    # H=None 走委托路径
    got_none = woodbury_quad_risk(finf, u, M0, lam0, active, t, None)
    assert got_none == pytest.approx(woodbury_trace_inv(finf, u, M0, lam0, active, t),
                                     rel=1e-12)


def test_quad_risk_matches_dense_arbitrary_H():
    """任意 (m < P) 任务算子:与稠密 tr(H ΔF⁻¹ Hᵀ) 对拍 < 1e-10。"""
    rng = np.random.default_rng(77)
    u, M0, lam0, finf, active, _ = _state(seed=20260914)
    P = finf.size
    for m in (1, 3, 11):
        H = rng.normal(size=(m, P))
        for tk in (1.0, 10.0):
            tt = np.full(u.shape[0], tk)
            DF = _delta_f(u, M0, lam0, finf, active, tt)
            ref = float(np.trace(H @ np.linalg.inv(DF) @ H.T))
            got = woodbury_quad_risk(finf, u, M0, lam0, active, tt, H)
            assert got == pytest.approx(ref, rel=1e-10), (m, tk, got, ref)


def test_quad_risk_orthogonal_row_mixing_invariant():
    """J_H 对 H 的正交行混合不变(任务空间的等距变换不改变风险)。"""
    rng = np.random.default_rng(78)
    u, M0, lam0, finf, active, _ = _state(seed=20260915)
    P = finf.size
    H = rng.normal(size=(5, P))
    Qr = np.linalg.qr(rng.normal(size=(5, 5)))[0]
    tt = np.full(u.shape[0], 3.0)
    a = woodbury_quad_risk(finf, u, M0, lam0, active, tt, H)
    b = woodbury_quad_risk(finf, u, M0, lam0, active, tt, Qr @ H)
    assert a == pytest.approx(b, rel=1e-10)


# ------------------------------------------------------------- N4-2 验收
def test_goal_oriented_gradient_fd():
    """∇_{t_k} J_H 解析式与中心差分对拍 ≤ 1e-6(逐灯、非均匀 t)。"""
    rng = np.random.default_rng(79)
    u, M0, lam0, finf, active, _ = _state(seed=20260916)
    P = finf.size
    H = rng.normal(size=(4, P))
    t0 = rng.uniform(1.0, 4.0, size=u.shape[0])
    f0, grad = woodbury_quad_risk_grad(finf, u, M0, lam0, active, t0, H)
    assert f0 == pytest.approx(
        woodbury_quad_risk(finf, u, M0, lam0, active, t0, H), rel=1e-12)
    h = 1e-6
    for k in np.flatnonzero(active):
        tp = t0.copy(); tp[k] += h
        tm = t0.copy(); tm[k] -= h
        fd = (woodbury_quad_risk(finf, u, M0, lam0, active, tp, H)
              - woodbury_quad_risk(finf, u, M0, lam0, active, tm, H)) / (2 * h)
        assert grad[k] == pytest.approx(fd, abs=1e-6), (k, grad[k], fd)


def test_goal_oriented_gradient_H_identity_matches_trace_grad():
    """H = I 时梯度退化为 woodbury_trace_inv_grad 的梯度。"""
    u, M0, lam0, finf, active, _ = _state(seed=20260917)
    t0 = np.linspace(1.0, 5.0, u.shape[0])
    _, g_ref = woodbury_trace_inv_grad(finf, u, M0, lam0, active, t0)
    _, g_H = woodbury_quad_risk_grad(finf, u, M0, lam0, active, t0,
                                     np.eye(finf.size))
    np.testing.assert_allclose(g_H, g_ref, rtol=1e-9, atol=1e-12)
