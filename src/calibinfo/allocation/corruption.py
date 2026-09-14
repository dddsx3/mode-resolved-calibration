"""Paired corruption draws with per-light precision scaling.

Functional forms are exactly those of calibinfo.models.corruption (rotate_dirs,
scale_intensity); the difference is that the raw standard-normal innovations can be
drawn once and scaled per light, so that every policy/budget sees the SAME raw
innovations (paired comparison, frozen seed construction).

With scales == 1, `paired_corruption` is bit-identical to
CorruptionGenerator.apply on the same rng state (binding-tested), and
`apply_scaled_corruption` with pre-drawn innovations equals the same realization.
"""

from __future__ import annotations

import numpy as np


def raw_innovations(rng, n):
    """Shared raw standard-normal innovations for one corruption realization.

    Draw order mirrors calibinfo.models.corruption exactly: per light an axis
    vector (3 draws) then a raw angle, interleaved, then the n raw log-intensity
    innovations as one vector.
    """
    axes = np.empty((n, 3))
    angles = np.empty(n)
    for i in range(n):
        axes[i] = rng.normal(size=3)
        angles[i] = rng.normal(size=1)[0]
    logs = rng.normal(size=n)
    return axes, angles, logs


def apply_scaled_corruption(dirs, sig_logI, sig_rad, scales, raw):
    """Apply the corruption functional form with per-light std scales.

    dirs: (n, 3); scales: (n,) multiplicative factors on the per-light standard
    deviation (1 = no allocation, 1/sqrt(regime) = selected); raw = the shared
    (axes, angles, logs) innovations. Returns (dirs_corrupted (n, 3), gains (n,)).
    """
    dirs = np.atleast_2d(np.asarray(dirs, float))
    n = dirs.shape[0]
    scales = np.broadcast_to(np.asarray(scales, float), (n,))
    axes, angles, logs = raw
    if sig_rad > 0:
        d2 = np.empty_like(dirs)
        for i in range(n):
            v = axes[i].copy()
            v -= dirs[i] * (v @ dirs[i])
            nv = np.linalg.norm(v)
            v = v / nv if nv > 1e-12 else np.array([1.0, 0.0, 0.0])
            ang = angles[i] * sig_rad * scales[i]
            out = dirs[i] * np.cos(ang) + np.cross(v, dirs[i]) * np.sin(ang) \
                + dirs[i] * (dirs[i] @ v) * (1 - np.cos(ang)) * 0.0
            d2[i] = out / np.linalg.norm(out)
    else:
        d2 = dirs.copy()
    if sig_logI > 0:
        g = np.exp(logs * sig_logI * scales)
    else:
        g = np.ones(n)
    return d2, g


def paired_corruption(rng, dirs, sig_logI, sig_rad, scales):
    """Convenience wrapper: draw the shared innovations and apply them."""
    dirs = np.atleast_2d(np.asarray(dirs, float))
    raw = raw_innovations(rng, dirs.shape[0])
    return apply_scaled_corruption(dirs, sig_logI, sig_rad, scales, raw)


# ------------------------------------------------- S0-2: 参数族扩展
# 不改 apply_scaled_corruption 本体(既有绑定测试依赖它);新增两个函数:
# raw_innovations_family 在 log-强度与角度两个通道间引入相关 rho_c;
# apply_corruption_family 接受逐灯标准差向量。退化锚点(标量 σ、rho_c=0)
# 与上面两个函数逐位一致(tests/test_corruption_family_degeneracy.py 钉死)。

def raw_innovations_family(rng, n, rho_c=0.0):
    """带通道耦合的 raw innovations。

    与 raw_innovations 相同的抽取顺序(axis→angle 逐灯,再 logs 向量),
    但 (angles, logs) 从 2×2 相关矩阵 [[1, rho_c], [rho_c, 1]] 抽取。
    rho_c=0 时各通道独立;返回 (axes, angles, logs) 与 raw_innovations
    **逐位一致**的约定只在 rho_c=0 时成立(耦合时 angles/logs 走 Cholesky
    路径,数值上不再与逐通道独立抽取相同)。
    """
    axes = np.empty((n, 3))
    z_a = np.empty(n)
    for i in range(n):
        axes[i] = rng.normal(size=3)
        z_a[i] = rng.normal(size=1)[0]
    z_l = rng.normal(size=n)
    if rho_c == 0.0:
        return axes, z_a, z_l
    # Cholesky 耦合:angles 取 z_a,logs = rho*z_a + sqrt(1-rho^2)*z_l
    r = float(np.clip(rho_c, -0.999, 0.999))
    logs = r * z_a + np.sqrt(1.0 - r * r) * z_l
    return axes, z_a, logs


def apply_corruption_family(dirs, sig_logI_vec, sig_rad_vec, scales, raw):
    """apply_scaled_corruption 的族版本:逐灯标准差向量。

    sig_logI_vec / sig_rad_vec: (n,) 或标量(广播);scales 仍是逐灯分配
    乘子(与 family 正交:异质性改基线 σ,分配改精化倍率)。paired 纪律
    不变:raw 只抽一次,由调用方跨策略/预算复用。退化锚点:两个 σ 向量
    为标量、raw 来自 raw_innovations(无耦合)时,输出与
    apply_scaled_corruption 逐位一致。
    """
    dirs = np.atleast_2d(np.asarray(dirs, float))
    n = dirs.shape[0]
    sig_I = np.broadcast_to(np.asarray(sig_logI_vec, float), (n,))
    sig_R = np.broadcast_to(np.asarray(sig_rad_vec, float), (n,))
    scales = np.broadcast_to(np.asarray(scales, float), (n,))
    axes, angles, logs = raw
    if np.any(sig_R > 0):
        d2 = np.empty_like(dirs)
        for i in range(n):
            v = axes[i].copy()
            v -= dirs[i] * (v @ dirs[i])
            nv = np.linalg.norm(v)
            v = v / nv if nv > 1e-12 else np.array([1.0, 0.0, 0.0])
            ang = angles[i] * sig_R[i] * scales[i]
            out = dirs[i] * np.cos(ang) + np.cross(v, dirs[i]) * np.sin(ang) \
                + dirs[i] * (dirs[i] @ v) * (1 - np.cos(ang)) * 0.0
            d2[i] = out / np.linalg.norm(out)
    else:
        d2 = dirs.copy()
    if np.any(sig_I > 0):
        g = np.exp(logs * sig_I * scales)
    else:
        g = np.ones(n)
    return d2, g
