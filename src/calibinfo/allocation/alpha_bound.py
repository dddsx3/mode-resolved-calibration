"""全迹 A-opt 边际的有效谱界，以及仅供历史复现的 α 候选表达式。

在固定正定参数空间上，M(S) = A + Σ_{k∈S} W_k，A ≻ 0、W_k ⪰ 0，
G(S) = tr(A^{-1}) − tr(M(S)^{-1})。γ 对 S ⊆ T、x ∉ T、W_x ≠ 0
取边际比值 d_x(S)/d_x(T) 的下确界；允许 S=T。没有非零更新时约定 γ=1。

注意：M(S) ⪯ M(T) 并不推出边际递减，平方逆也不保持 Loewner 反序。
历史候选 γ ≥ 1/(1+α) 已被二维有理反例否决（α=1，γ=14/109），
不是尚待补证的定理，亦不能归为 Chamon–Ribeiro 的已证结论。
有效的局部夹逼 tr(W M^{-2})/(1+ρ) ≤ d(M,W) ≤ tr(W M^{-2}) 仍保留。

新入口 spectral_gamma_lower_bound 使用不依赖 α 的 leave-one-out 全局界：
    γ ≥ λmin(A) / max_{x:W_x≠0} λmax(A + Σ_{i≠x} W_i)。
它只面向未加权全迹、固定正定空间，不直接适用于任意线性任务 H 或伪逆。
一般共享线性高斯模型也有上述旧界反例；本轮独立的物理 Lambertian 构造
使用逐灯正定耦合先验，见 docs/theory/physical_gamma_and_tightness.md。
此处不把一般线性构造直接当作物理反例；谱界的完整证明见 docs/methods.md §9。

原 decompose/alpha_of 保留低秩名义设计的历史计算路径；新增谱界是独立的
稠密矩阵 API，不改写历史结果或幅值实验快照，也不新增运行时依赖。
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
    """返回已被反例否决的历史候选表达式 1/(1+α)，不是有效 γ 下界。

    仅为旧结果的数值复现保留原名、原计算及原输入行为；不发出新警告、
    不增加校验。一般 SPD+PSD 类及一般共享线性高斯类均已有精确反例。
    有效的未加权全迹保证请使用 spectral_gamma_lower_bound。
    """
    return 1.0 / (1.0 + alpha)


def _gamma_matrix(value, name, shape=None, *, positive_definite=False):
    """校验实数有限对称矩阵；仅修复相对机器精度量级的舍入误差。"""
    if np.iscomplexobj(value):
        raise ValueError(f"{name} 必须是实数矩阵")
    try:
        matrix = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} 必须是实数矩阵") from exc
    if (matrix.ndim != 2 or matrix.shape[0] == 0
            or matrix.shape[0] != matrix.shape[1]):
        raise ValueError(f"{name} 必须是非空方阵")
    if shape is not None and matrix.shape != shape:
        raise ValueError(f"{name} 的形状必须与 A 一致")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} 必须只含有限值")
    scale = float(np.max(np.abs(matrix)))
    if scale == 0.0:
        if positive_definite:
            raise ValueError("A 必须正定 (SPD)")
        return matrix
    # 按矩阵自身尺度校验，避免把微小但真正不定的更新当作零矩阵。
    normalized = matrix / scale
    tol = 64.0 * np.finfo(float).eps * matrix.shape[0]
    if np.max(np.abs(normalized - normalized.T)) > tol:
        raise ValueError(f"{name} 必须对称")
    matrix += 0.5 * (matrix.T - matrix)
    normalized = matrix / scale
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(normalized)
        if positive_definite:
            np.linalg.cholesky(normalized)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} 无法在浮点精度下通过正定/半正定校验") from exc
    if not np.all(np.isfinite(eigenvalues)):
        raise ValueError(f"{name} 的谱无法用有限浮点数表示")
    if positive_definite:
        if eigenvalues[0] <= 0.0:
            raise ValueError("A 必须正定 (SPD)，不接受奇异矩阵或伪逆")
    elif eigenvalues[0] < -tol * float(np.max(np.abs(eigenvalues))):
        raise ValueError(f"{name} 必须半正定 (PSD)")
    elif eigenvalues[0] < 0.0:
        # 秩亏 Gram 矩阵可产生极小负特征值；只对已通过容差的输入修复。
        normalized = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
        matrix = (0.5 * normalized + 0.5 * normalized.T) * scale
    return matrix


def spectral_gamma_lower_bound(A: np.ndarray, updates) -> float:
    """返回一般 SPD 基线 + PSD 更新的 leave-one-out 全局有效 γ 谱下界。

    对 M(S)=A+Σ_{i∈S}W_i、G(S)=tr(A^{-1})−tr(M(S)^{-1})，返回
        λmin(A) / max_{x:W_x≠0} λmax(A + Σ_{i≠x}W_i)。
    γ 取 S⊆T、x∉T 的正分母边际比值下确界，允许 S=T。零更新不参加
    分母的 max；空序列或全零更新约定返回 1.0，不计算 0/0。

    参数 A 为非空实对称正定 (n,n) 数组；updates 为有限可迭代对象，
    元素均为同形状实对称 PSD 矩阵，亦支持 (m,n,n) 数组和生成器。
    非有限、非对称、非 SPD/PSD、形状错误的输入抛出 ValueError。
    对称性与 PSD 的舍入容差为 64*n*eps（相对于各自矩阵尺度）；
    容差内先对称化，再将更新的微小负特征值截零。返回值针对这些校验后
    的矩阵；不修改调用方输入，不加 ridge，也不把微小非零更新当作零。
    超出浮点可分辨范围的正定性或正谱比值同样抛出 ValueError。

    只适用于固定正定参数空间上的未加权全迹；不可直接用于任意任务 H、
    奇异基线或随 S 变化的子空间。该界可能保守，即使可交换时真实 γ=1，
    也不保证返回 1。证明见 docs/methods.md §9；浮点值不是区间算术证书。
    本函数采用稠密特征值计算，不替代历史低秩 alpha_of 计算路径。
    """
    baseline = _gamma_matrix(A, "A", positive_definite=True)
    try:
        iterator = iter(updates)
    except TypeError as exc:
        raise ValueError("updates 必须是矩阵的有限可迭代对象") from exc
    matrices = [
        _gamma_matrix(update, f"updates[{i}]", baseline.shape)
        for i, update in enumerate(iterator)
    ]
    nonzero = [matrix for matrix in matrices if np.any(matrix != 0.0)]
    if not nonzero:
        return 1.0

    # 共同标量缩放不改变迹边际比或谱比，并避免大尺度矩阵求和溢出。
    scale = max(float(np.max(np.abs(matrix))) for matrix in [baseline, *nonzero])
    baseline = baseline / scale
    nonzero = [matrix / scale for matrix in nonzero]
    smallest = float(np.linalg.eigvalsh(baseline)[0])
    largest = 0.0
    for x in range(len(nonzero)):
        # 不用 total - W_x：大更新相消可能把正定基线完全丢掉。
        leave_one_out = baseline.copy()
        for i, update in enumerate(nonzero):
            if i != x:
                leave_one_out += update
        largest = max(largest, float(np.linalg.eigvalsh(leave_one_out)[-1]))
    bound = smallest / largest if largest > 0.0 else 0.0
    if not np.isfinite(bound) or bound <= 0.0:
        raise ValueError("谱比值超出浮点精度范围；请检查基线条件数与更新尺度")
    return float(min(1.0, bound))


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