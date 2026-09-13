"""α-近似次模界:J_A 选择函数在标定精化集合上的先验可算下界(plan v2 H5)。

问题:F(S) = tr M(S)^{-1} 是"精化灯集合 S"的 A-opt 剩余代价,
    M(S) = ΔF(t_S) = A + Σ_{k∈S} W_k,   W_k = u_k[K_k(1) − K_k(κ)] u_k^T ⪰ 0,
    A := ΔF(1),  K_k(t) = (M0_k + t Λ0_k)^{-1}。
    G(S) = F(∅) − F(S)(增益,递增)。选择函数 A-opt 增益的次模结构:

    Δ_x G(S) = tr [ M(S)^{-1} W_x (M(S)+W_x)^{-1} ]         (Woodbury 恒等)
    特征值夹逼:  (1/(1+ρ)) tr[W_x M(S)^{-2}] ≤ Δ_x G(S) ≤ tr[W_x M(S)^{-2}],
      ρ(S,x) := λmax(M(S)^{-1} W_x)。
    S ⊆ T ⇒ M(S) ⪯ M(T) ⇒ Δ_x G(S) ≥ Δ_x G(T)(边际递减方向成立),
    且 ρ(S,x) ≤ ρ_max = max_x λmax(A^{-1}W_x) ⇒

        γ ≥ 1/(1+α),   α := max_x λmax(A^{-1} W_x)。

    α 只需名义设计(A、B、Λ0)即可先验算出,不需要真值或实测数据。
    Λ0_x → ∞(标定极好)或 Λ0_x → 0(标定极差)时 W_x → 0 ⇒ α → 0,γ → 1
    (精确次模);α 在中间档位取最大。这是 Chamon–Ribeiro (NeurIPS 2017)
    近似超模框架在"标定精度作为设计变量"下的实例化,本文不声称首先给出。

实现约定:全部 dense 装配走 blocks.LightBlocks / sym_inv;所有 P×P 求逆
只出现一次(A^{-1})或以 3×3 特征值形式出现,保证现实规模(P ~ 1200、
L ~ 142)下 α 仍可在桌面机秒级算完。
"""

from __future__ import annotations

import numpy as np

from calibinfo.allocation.blocks import LightBlocks, sym_inv


def decompose(blocks: LightBlocks, kappa: float):
    """名义设计分解:M(S) = A + Σ_{k∈S} u_k C_k u_k^T。

    返回 (A, C):
      A : (P, P) —— ΔF(1)(t=1 处的装配);
      C : (L, 3, 3) —— C_k = K_k(1) − K_k(κ) ⪰ 0(K 在 t 上 Loewner 递减)。
    W_k = u_k C_k u_k^T 不显式形成(W_k 是 P×P 秩 3);所有下游量通过
    u_k、C_k 与 3×3 特征值计算。
    """
    blk = blocks
    t1 = np.ones(blk.L)
    lam1 = np.stack([t1[k] * blk.lam0[k] for k in range(blk.L)])
    A = blk.assemble(lam1)
    C = np.empty((blk.L, 3, 3))
    for k in range(blk.L):
        if not blk.active[k]:
            C[k] = np.zeros((3, 3))
            continue
        K1 = sym_inv(blk.M0[k] + blk.lam0[k])
        Kk = sym_inv(blk.M0[k] + kappa * blk.lam0[k])
        C[k] = K1 - Kk
    return A, C


def alpha_of(A: np.ndarray, blocks: LightBlocks, C: np.ndarray) -> float:
    """α = max_x λmax(A^{-1} W_x) = max_x λmax(C_x^{1/2} u_x^T A^{-1} u_x C_x^{1/2})。

    只对 active 灯取 max(inactive 灯 u=0 ⇒ W=0);A^{-1} 只算一次,
    每个候选是 3×3 特征值问题。P、L 均为任意规模(桌面机可行)。"""
    Ainv = np.linalg.inv(A)
    blk = blocks
    best = 0.0
    for k in range(blk.L):
        if not blk.active[k]:
            continue
        w, V = np.linalg.eigh(C[k])
        c_sqrt = (V * np.sqrt(np.maximum(w, 0.0))) @ V.T     # C_k^{1/2} ⪰ 0
        X = c_sqrt @ (blk.u[k].T @ Ainv @ blk.u[k]) @ c_sqrt  # (3,3)
        rho = float(np.linalg.eigvalsh(0.5 * (X + X.T))[-1])
        best = max(best, rho)
    return best


def gamma_lower_bound(alpha: float) -> float:
    """γ ≥ 1/(1+α)。"""
    return 1.0 / (1.0 + alpha)


# ---------------------------------------------------------------- 数值原语
def _m_s_inv(A, blocks, C, S, kappa):
    """M(S)^{-1}(dense;P 小或单点验证用=toy/测试路径)。"""
    M = A.copy()
    for k in S:
        M += (blocks.u[k] @ C[k]) @ blocks.u[k].T
    return np.linalg.inv(M), M


def delta_gain_interval(A, blocks, C, S, x, kappa):
    """(S, x) 对的两端不等式与真值:
    返回 (lb, ub, val),val = Δ_x G(S),lb/ub 为任务书 C0(b) 的夹逼。"""
    blk = blocks
    Minv, M = _m_s_inv(A, blocks, C, S, kappa)
    Cx = C[x]
    ux = blk.u[x]
    # ρ = λmax(M^{-1} W_x) = λmax(C^{1/2} u^T M^{-1} u C^{1/2})
    w, V = np.linalg.eigh(Cx)
    c_sqrt = (V * np.sqrt(np.maximum(w, 0.0))) @ V.T
    Y = c_sqrt @ (ux.T @ Minv @ ux) @ c_sqrt
    rho = float(np.linalg.eigvalsh(0.5 * (Y + Y.T))[-1])
    # tr[W_x M^{-2}] = tr[C_x^{1/2} u_x^T M^{-2} u_x C_x^{1/2}]
    Minv2 = Minv @ Minv
    Z = c_sqrt @ (ux.T @ Minv2 @ ux) @ c_sqrt
    trW_M2 = float(np.trace(Z))
    # Δ_x G(S) = tr[M^{-1} W_x (M + W_x)^{-1}]
    val = float(np.trace(Minv @ (ux @ Cx) @ ux.T @ np.linalg.inv(M + (ux @ Cx) @ ux.T)))
    lb = trW_M2 / (1.0 + rho)
    ub = trW_M2
    return lb, ub, val