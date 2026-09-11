"""Math foundations lock（P-CERT/P-LOWRANK 的地基，M0；plan v2 §1）。

全部为小规模合成已知答案（无原始数据依赖，CI 可跑）。钉死四件事：
  1. **方向参数化**（v1→v2 修订的核心教训）：ΔF(t) = diag(F∞) − Σ u(M0+tΛ0)⁻¹uᵀ，
     t 大 = 精度高 = 信息多。J_A(κ·1) ≤ J_A(1)、ΔF(κ·1) ⪰ ΔF(1) 必须成立；
     反向读法（Λ0/t）不被任何证书支持。
  2. 逐灯 Loewner 单调（减项递减 ⇒ ΔF 递增；entrywise 不保证，明说）。
  3. PD 域上 J_A/J_E/J_D 的中点凸性（各 400 次随机方向，0 违例）
     与 ΔF 的联合算子凹性（中点残差 ≥ −tol）。
  4. 低秩恒等式 R = I − VVᵀ（V = F∞^{-1/2}[u_k K_k^{1/2}]_k，对称平方根）：
     谱偏差 ≤1e-10、ρ≡1 计数 = P − 3L；
     以及 J_A 梯度（对**装配后** ΔF）：∇_{t_k} = −tr(ΔF⁻¹ ∂_kΔF ΔF⁻¹)，
     FD 对拍 ≤1e-6——逐灯独立的错误实现会差约 P 倍。
"""

import numpy as np
import pytest

from calibinfo.allocation.blocks import sym_inv

SEED = 20260911


def _blocks(seed=SEED, L=4, P=40, q=3):
    """CI04 whitened per-light 形态的随机状态（同 allocation 家族）。"""
    rng = np.random.default_rng(seed)
    w = rng.uniform(0.5, 2.0, size=(L, P))
    s = rng.uniform(0.2, 1.5, size=(L, P))
    B = rng.normal(0.0, 0.5, size=(L, P, q))
    u = (w * s)[:, :, None] * B
    M0 = np.einsum("kpi,kpj->kij", B, w[:, :, None] * B)
    lam0 = np.stack([np.diag([30.0, 800.0, 800.0])] * L)
    finf = (w * s ** 2).sum(0)
    return u, M0, lam0, finf


def _assemble(u, M0, lam0, finf, t):
    """方向参数化（钉死）：ΔF(t) = diag(F∞) − Σ u_k (M0k + t_k Λ0k)⁻¹ u_kᵀ。"""
    DF = np.diag(finf)
    for k in range(u.shape[0]):
        K = sym_inv(M0[k] + t[k] * lam0[k])
        DF = DF - (u[k] @ K) @ u[k].T
    return DF


def _J_A(DF):
    ev = np.linalg.eigvalsh(DF)
    pos = ev > 1e-12 * max(ev.max(), 1.0)
    return float(np.sum(1.0 / ev[pos]))


def _J_E(DF):
    return -float(np.linalg.eigvalsh(DF)[0])


def _J_D(DF):
    ev = np.linalg.eigvalsh(DF)
    pos = ev > 1e-12 * max(ev.max(), 1.0)
    return float(-np.sum(np.log(ev[pos])))


# ------------------------------------------------------- 1 方向参数化
def test_direction_pinned_more_precision_is_more_information():
    """J_A(κ·1) ≤ J_A(1) 且 ΔF(κ·1) ⪰ ΔF(1)：t 大 = 精度高（方向 known-answer）。

    这是 plan v1→v2 的核心教训：反向读法（Λ0/t）下任何证书都不成立。"""
    u, M0, lam0, finf = _blocks()
    one, full = np.ones(4), np.full(4, 25.0)
    D = _assemble(u, M0, lam0, finf, one) - _assemble(u, M0, lam0, finf, full)
    assert np.linalg.eigvalsh(D).max() <= 1e-10 * max(1.0, float(np.abs(D).max()))
    assert _J_A(_assemble(u, M0, lam0, finf, full)) <= _J_A(_assemble(u, M0, lam0, finf, one))


# ------------------------------------------------------- 2 逐灯单调
def test_per_light_loewner_monotone_decreasing_kernel():
    """t_k ↦ u(M0+tΛ0)⁻¹uᵀ Loewner 递减（ΔF 递增的逐灯依据）。"""
    u, M0, lam0, _finf = _blocks(seed=SEED + 1)
    for k in range(u.shape[0]):
        for _ in range(20):
            t1 = float(np.random.default_rng().uniform(1, 20))
            t2 = t1 + float(np.random.default_rng().uniform(0.5, 5))
            D = ((u[k] @ sym_inv(M0[k] + t1 * lam0[k])) @ u[k].T
                 - (u[k] @ sym_inv(M0[k] + t2 * lam0[k])) @ u[k].T)
            assert np.linalg.eigvalsh(D).min() >= -1e-12 * max(
                1.0, float(np.abs(D).max()))


# ------------------------------------------------------- 3 凸性与算子凹
@pytest.mark.parametrize("name,J", [("J_A", _J_A), ("J_E", _J_E), ("J_D", _J_D)])
def test_midpoint_convexity_on_pd_region(name, J):
    """PD 域上三泛函的中点凸性：400 次随机方向 0 违例（方向钉死 + ΔF≻0 假设下）。"""
    u, M0, lam0, finf = _blocks(seed=SEED + 2)
    rng = np.random.default_rng(SEED + 3)
    violations = 0
    for _ in range(400):
        t1 = rng.uniform(1.0, 5.0, size=4)
        t2 = rng.uniform(1.0, 5.0, size=4)
        rhs = 0.5 * (J(_assemble(u, M0, lam0, finf, t1))
                     + J(_assemble(u, M0, lam0, finf, t2)))
        lhs = J(_assemble(u, M0, lam0, finf, 0.5 * (t1 + t2)))
        if lhs > rhs + 1e-9 * max(1.0, abs(rhs)):
            violations += 1
    assert violations == 0, f"{name}: {violations}/400 中点凸性违例"


def test_joint_operator_concavity_of_delta_f():
    """ΔF 联合算子凹：min eig(ΔF(mid) − avg) ≥ −tol（L2 的机器可查形式）。"""
    u, M0, lam0, finf = _blocks(seed=SEED + 4)
    rng = np.random.default_rng(SEED + 5)
    worst = np.inf
    for _ in range(100):
        t1 = rng.uniform(1.0, 5.0, size=4)
        t2 = rng.uniform(1.0, 5.0, size=4)
        diff = (_assemble(u, M0, lam0, finf, 0.5 * (t1 + t2))
                - 0.5 * (_assemble(u, M0, lam0, finf, t1)
                         + _assemble(u, M0, lam0, finf, t2)))
        worst = min(worst, float(np.linalg.eigvalsh(diff).min()))
    assert worst >= -1e-10 * max(1.0, abs(worst))


# ------------------------------------------------------- 4 低秩恒等式
def test_lowrank_retention_identity():
    """R = I − VVᵀ（V = F∞^{-1/2}[u K^{1/2}]_k，对称平方根）：
    谱偏差 ≤1e-10；ρ≡1 计数精确 = P − 3L（L5）。"""
    u, M0, lam0, finf = _blocks(seed=SEED + 6)
    L, P = u.shape[0], u.shape[1]
    t = np.full(L, 3.0)
    Ks = [sym_inv(M0[k] + t[k] * lam0[k]) for k in range(L)]
    DF = _assemble(u, M0, lam0, finf, t)
    # 稠密路线
    Fh_inv = np.diag(1.0 / np.sqrt(finf))
    R_dense = Fh_inv @ DF @ Fh_inv
    rho_dense = np.sort(np.linalg.eigvalsh(0.5 * (R_dense + R_dense.T)))
    # 低秩路线：V = F∞^{-1/2}[u_k K_k^{1/2}]_k（P × 3L）
    V = np.empty((P, 3 * L))
    for k in range(L):
        ev, W = np.linalg.eigh(0.5 * (Ks[k] + Ks[k].T))
        Khalf = (W * np.sqrt(np.clip(ev, 0.0, None))) @ W.T   # 对称平方根
        V[:, 3 * k:3 * k + 3] = Fh_inv @ (u[k] @ Khalf)
    mu = np.linalg.eigvalsh(V.T @ V)                       # (3L,) ⪯ 1
    rho_low = np.sort(np.concatenate([np.full(P - 3 * L, 1.0), 1.0 - mu]))
    n_flat = int(np.sum(rho_low > 1.0 - 1e-9))
    assert n_flat == P - 3 * L
    assert np.max(np.abs(rho_dense - rho_low)) <= 1e-10 * max(
        1.0, float(np.max(np.abs(rho_dense))))


# ------------------------------------------------------- 5 梯度 FD 对拍
def test_ja_gradient_on_assembled_delta_f_matches_fd():
    """J_A 梯度必须对**装配后**的 P×P ΔF 求：
    ∇_{t_k} J_A = −tr(ΔF⁻¹ · u_k K Λ0 K u_kᵀ · ΔF⁻¹)，FD 对拍 ≤1e-6。
    （逐灯独立实现的错误口径会差约 P 倍。）"""
    rng = np.random.default_rng(SEED + 7)
    L, P = 3, 20
    w = rng.uniform(0.5, 2.0, size=(L, P))
    s = rng.uniform(0.2, 1.5, size=(L, P))
    B = rng.normal(0.0, 0.5, size=(L, P, 3))
    u = (w * s)[:, :, None] * B
    M0 = np.einsum("kpi,kpj->kij", B, w[:, :, None] * B)
    lam0 = np.stack([np.diag([30.0, 800.0, 800.0])] * L)
    finf = (w * s ** 2).sum(0)

    t = rng.uniform(1.0, 4.0, size=L)
    DF = _assemble(u, M0, lam0, finf, t)
    DFinv = np.linalg.inv(DF)
    grad = np.empty(L)
    for k in range(L):
        K = sym_inv(M0[k] + t[k] * lam0[k])
        dD = (u[k] @ (K @ lam0[k] @ K)) @ u[k].T      # ∂ΔF/∂t_k ⪰ 0
        grad[k] = -float(np.trace(DFinv @ dD @ DFinv))
    # 中心差分
    h = 1e-6
    fd = np.empty(L)
    for k in range(L):
        tp, tm = t.copy(), t.copy()
        tp[k] += h
        tm[k] -= h
        fd[k] = (_J_A(_assemble(u, M0, lam0, finf, tp))
                 - _J_A(_assemble(u, M0, lam0, finf, tm))) / (2 * h)
    assert np.max(np.abs(grad - fd)) <= 1e-6 * max(1.0, float(np.max(np.abs(fd))))
    assert np.all(grad <= 1e-12)                         # J_A 关于 t 递减
